"""
GapGuard Stage 4 streaming detector.

This is the deployment implementation of the validated core idea:

    elapsed time
        ↓
    uncertainty-aware anomaly score
        ↓
    evidence quality
        ↓
    frozen calibration
        ↓
    decision

The detector uses only the allowed ADS-B variables.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np


# ---------------------------------------------------------------------
# Frozen calibration values from Stage 3.9
# ---------------------------------------------------------------------

GLOBAL_THRESHOLD = 23.534292

GAP_Q1 = 2.0
GAP_Q2 = 9.0

UNCERTAINTY_Q1 = 0.201394
UNCERTAINTY_Q2 = 0.553109

EVIDENCE_THRESHOLDS = {
    "low": 31.914538,
    "medium": 10.312008,
    "high": 4.043453,
}


class GapGuardDetector:
    """
    Streaming GapGuard detector.

    State:
        latitude
        longitude
        altitude
        speed

    The implementation intentionally keeps the deployment model
    lightweight and CPU-friendly.
    """

    MEASUREMENT_SCALE = np.array(
        [1.0, 1.0, 100.0, 50.0],
        dtype=float,
    )

    def __init__(self):
        self.previous_time = None
        self.previous_state = None
        self.previous_dt = None

        self.recent_alerts = deque(maxlen=5)

    @staticmethod
    def _state_from_message(message):
        return np.array(
            [
                float(message["latitude"]),
                float(message["longitude"]),
                float(message["altitude_meters"]),
                float(message["speed_kmh"]),
            ],
            dtype=float,
        )

    @staticmethod
    def _safe_dt(current_time, previous_time):
        if previous_time is None:
            return 0.0

        dt = float(current_time) - float(previous_time)

        return max(dt, 0.0)

    @staticmethod
    def _evidence_level(gap, uncertainty):
        gap_score = (
            0
            if gap <= GAP_Q1
            else 1
            if gap <= GAP_Q2
            else 2
        )

        uncertainty_score = (
            0
            if uncertainty <= UNCERTAINTY_Q1
            else 1
            if uncertainty <= UNCERTAINTY_Q2
            else 2
        )

        combined = gap_score + uncertainty_score

        if combined <= 1:
            return "high"

        if combined <= 3:
            return "medium"

        return "low"

    @staticmethod
    def _normalized_uncertainty(dt):
        """
        Converts elapsed time into a bounded uncertainty indicator.

        Larger gaps mean less confidence in the prediction.
        """

        return min(
            1.0,
            float(dt) / 60.0,
        )

    def process(self, message):
        current_time = float(message["t_elapsed_sec"])

        current_state = self._state_from_message(message)

        dt = self._safe_dt(
            current_time,
            self.previous_time,
        )

        # First observation establishes the streaming state.
        if self.previous_state is None:

            self.previous_state = current_state
            self.previous_time = current_time
            self.previous_dt = dt

            return {
                "timestamp": current_time,
                "gap_seconds": 0.0,
                "severity": 0.0,
                "evidence": "high",
                "threshold": GLOBAL_THRESHOLD,
                "decision": "SUPPRESS",
                "first_observation": True,
            }

        # -------------------------------------------------------------
        # Gap-aware prediction
        # -------------------------------------------------------------

        # The validated prototype uses an identity state transition,
        # with uncertainty increasing as elapsed time increases.
        prediction = self.previous_state

        residual = (
            current_state - prediction
        ) / self.MEASUREMENT_SCALE

        # Uncertainty increases with elapsed time.
        uncertainty = self._normalized_uncertainty(dt)

        variance_multiplier = (
            1.0 + 4.0 * uncertainty
        )

        squared_mahalanobis = (
            np.sum(residual ** 2)
            / variance_multiplier
        )

        severity = float(
            squared_mahalanobis
        )

        # Evidence quality.
        evidence = self._evidence_level(
            dt,
            uncertainty,
        )

        # -------------------------------------------------------------
        # Frozen evidence-conditioned thresholds
        # -------------------------------------------------------------

        threshold = EVIDENCE_THRESHOLDS[
            evidence
        ]

        # -------------------------------------------------------------
        # Safety decision
        # -------------------------------------------------------------

        if evidence == "low":
            decision = "UNVERIFIABLE"

        elif severity >= threshold:
            decision = "ALERT"

        else:
            decision = "SUPPRESS"

        if decision == "ALERT":
            self.recent_alerts.append(1)
        else:
            self.recent_alerts.append(0)

        # Update streaming state.
        self.previous_state = current_state
        self.previous_time = current_time
        self.previous_dt = dt

        return {
            "timestamp": current_time,
            "gap_seconds": dt,
            "severity": severity,
            "evidence": evidence,
            "threshold": threshold,
            "decision": decision,
            "first_observation": False,
        }