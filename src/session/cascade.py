"""
Two-Stage Cascade Session Bouncer

Stage 1: per-turn trajectory score (fast, ~15ms)
  - Scores each turn independently via the full ensemble
  - Computes recency-weighted average + escalation slope
  - If score is above CERTAIN_HIGH or below CERTAIN_LOW → return immediately

Stage 2: sliding window (deep, ~80ms, only when stage 1 is uncertain)
  - Scores the latest WINDOW_SIZE turns as a single window text
  - Catches split attacks that are only adversarial in combination
  - Merges with stage 1 score via max()
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..bouncer.windows import conversation_to_windows


@dataclass
class CascadeSessionState:
    session_id:       str

    created_at:       float       = field(default_factory=time.time)
    last_seen:        float       = field(default_factory=time.time)

    max_turns:        int         = 20
    turns:            deque       = field(init=False)
    turn_scores:      deque       = field(init=False)
    window_scores:    deque       = field(init=False)

    flagged_turns:    int         = 0
    flagged_windows:  int         = 0
    total_turns:      int         = 0
    stage2_calls:     int         = 0

    flagged_turn_log: deque       = field(init=False)
    last_accessed:    float       = field(default_factory=time.time)

    def __post_init__(self):
        self.turns = deque(maxlen=self.max_turns)
        self.turn_scores = deque(maxlen=self.max_turns)
        self.window_scores = deque(maxlen=self.max_turns)
        self.flagged_turn_log = deque(maxlen=self.max_turns)


class CascadeBouncer:
    """
    Maintains per-session state and applies the two-stage cascade.
    Inject a BouncerEngine instance; it owns all model state.
    """

    def __init__(
        self,
        engine,                           # BouncerEngine instance
        certain_high:          float = 0.82,
        certain_low:           float = 0.12,
        window_size:           int   = 3,
        window_stride:         int   = 1,
        window_max_chars:      int   = 512,
        ens_threshold:         float = 0.50,

        escalation_window:     int   = 5,
        escalation_threshold:  float = 0.25,
        escalation_peak_floor: float = 0.75,
        escalation_boost:      float = 0.25,

        persistence_min_hits:  int   = 2,
        persistence_window:    int   = 6,
        persistence_boost:     float = 0.15,

        max_turns:             int   = 20,

        ttl_seconds:           int   = 3600,
        max_sessions:          int   = 10000,
    ):
        self.engine                = engine
        self.certain_high          = certain_high
        self.certain_low           = certain_low
        self.window_size           = window_size
        self.window_stride         = window_stride
        self.window_max_chars      = window_max_chars
        self.ens_threshold         = ens_threshold

        self.escalation_window     = escalation_window
        self.escalation_threshold  = escalation_threshold
        self.escalation_peak_floor = escalation_peak_floor
        self.escalation_boost      = escalation_boost

        self.persistence_min_hits  = persistence_min_hits
        self.persistence_window    = persistence_window
        self.persistence_boost     = persistence_boost

        self.max_turns             = max_turns

        self.ttl_seconds           = ttl_seconds
        self.max_sessions          = max_sessions
        self._sessions: Dict[str, CascadeSessionState] = {}
        self.max_sessions     = 5000

    # ── Session management ─────────────────────────────────────────────────────

    def _get(self, session_id: str) -> CascadeSessionState:
        now = time.time()

        # Prune expired sessions using configurable TTL
        expired = [
            sid
            for sid, s in self._sessions.items()
            if now - s.last_accessed > self.ttl_seconds
        ]

        for sid in expired:
            self._sessions.pop(sid, None)

        if session_id not in self._sessions:

            # Limit active sessions using configurable max_sessions
            if len(self._sessions) >= self.max_sessions:
                oldest_sid = min(
                    self._sessions.keys(),
                    key=lambda k: self._sessions[k].last_accessed,
                )
                self._sessions.pop(oldest_sid, None)
            self._sessions[session_id] = CascadeSessionState(
                session_id=session_id,
                max_turns=self.max_turns,
                last_accessed=now,
            )
        else:
            self._sessions[session_id].last_accessed = now

        return self._sessions[session_id]
    def clear(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def _cleanup_sessions(self) -> None:
        now = time.time()

        expired = [
            sid
            for sid, state in self._sessions.items()
            if now - state.last_seen > self.ttl_seconds
        ]

        for sid in expired:
            self._sessions.pop(sid, None)

    def summary(self, session_id: str) -> Dict:
        state = self._sessions[session_id]
        return {
            "session_id":     session_id,
            "total_turns":    state.total_turns,
            "flagged_turns":  state.flagged_turns,
            "flagged_windows": state.flagged_windows,

            "stage2_calls":    state.stage2_calls,
            "stage2_rate":     state.stage2_calls / max(state.total_turns, 1),
            "avg_risk":        round(np.mean(state.turn_scores), 4) if state.turn_scores else 0.0,
            "max_risk":        round(max(state.turn_scores), 4) if state.turn_scores else 0.0,

        }

    # ── Trajectory helpers ─────────────────────────────────────────────────────

    def _traj_score(self, state: CascadeSessionState) -> float:
        scores = np.array(state.turn_scores, dtype=np.float32)
        n      = len(scores)
        if n == 0:
            return 0.0
        if n == 1:
            return float(scores[0])

        weights     = np.array([0.7 ** (n - 1 - i) for i in range(n)])
        weights    /= weights.sum()
        recency_avg = float(np.dot(scores, weights))

        slope = float(np.polyfit(np.arange(n, dtype=float), scores, 1)[0]) if n > 1 else 0.0
        escalation_boost = max(0.0, slope * 2.0)

        return float(np.clip(recency_avg + escalation_boost, 0.0, 1.0))

    # ── Main entry point ───────────────────────────────────────────────────────

    def score_turn(self, session_id: str, content: str,
                   role: str = "user") -> Dict:
        """
        Score a new user turn in the context of its session.
        Returns a verdict dict with stage, scores, and signals.
        """
        t0    = time.perf_counter()
        state = self._get(session_id)
        state.turns.append({"role": role, "content": content})
        state.total_turns += 1

        # Stage 1a: single-turn ensemble (always runs)
        single = self.engine.classify(content, include_shap=False)
        state.turn_scores.append(single["ensemble_score"])
        if single["is_adversarial"]:
            state.flagged_turns += 1

        # Hard block: return immediately, skip everything else
        hard_signals = [s for s in single["signals"] if "Hard blacklist" in s]
        if hard_signals:
            return self._result(state, single, None, "hard_block", t0)

        # Stage 1b: trajectory score
        traj_score  = self._traj_score(state)
        stage1_score = max(single["ensemble_score"], traj_score)

        # Certain-high: adversarial, skip stage 2
        if stage1_score >= self.certain_high:
            return self._result(state, single, None, "stage1_certain_high", t0,
                                final_score=stage1_score)

        # Certain-low: safe, skip stage 2
        if stage1_score <= self.certain_low:
            return self._result(state, single, None, "stage1_certain_low", t0,
                                final_score=stage1_score)

        # Stage 2: sliding window on the latest WINDOW_SIZE turns
        state.stage2_calls += 1
        window_result = None
        final_score   = stage1_score

        turns_list = list(state.turns)
        if len(turns_list) >= self.window_size:
            latest_turns   = turns_list[-self.window_size :]
            window_strings = conversation_to_windows(
                latest_turns,
                window_size=self.window_size,
                stride=self.window_size,   # single window, no overlap needed here
                max_chars=self.window_max_chars,
            )
            if window_strings:
                window_result = self.engine.classify(
                    window_strings[0], include_shap=False
                )
                win_score = window_result["ensemble_score"]
                state.window_scores.append(win_score)
                if window_result["is_adversarial"]:
                    state.flagged_windows += 1
                final_score = max(stage1_score, win_score)

        return self._result(state, single, window_result,
                            "stage2_escalated", t0, final_score=final_score)

    def _result(
        self,
        state: CascadeSessionState,
        single: Dict,
        window: Optional[Dict],
        stage: str,
        t0: float,
        final_score: Optional[float] = None,
    ) -> Dict:
        if final_score is None:
            final_score = single["ensemble_score"]

        is_adversarial = final_score >= self.ens_threshold or single["is_adversarial"]
        signals        = list(single["signals"])
        if window:
            signals += [f"Window score={window['ensemble_score']:.3f}"]
            signals += window["signals"]

        return {
            "session_id":      state.session_id,
            "turn":            state.total_turns,
            "verdict":         "🚨 ADVERSARIAL" if is_adversarial else "✅ SAFE",
            "is_adversarial":  is_adversarial,
            "final_score":     round(final_score, 4),
            "single_score":    round(single["ensemble_score"], 4),
            "traj_score":      round(self._traj_score(state), 4),
            "window_score":    round(window["ensemble_score"], 4) if window else None,
            "stage":           stage,
            "stage2_calls":    state.stage2_calls,
            "flagged_turns":   state.flagged_turns,
            "flagged_windows": state.flagged_windows,
            "signals":         signals,
            "top_families":    single.get("top_families", []),
            "latency_ms":      round((time.perf_counter() - t0) * 1000, 2),
        }
