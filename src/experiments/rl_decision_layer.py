"""
Stage 4.6
RL decision layer for GapGuard.

The RL policy is trained on detector outputs and labels.
It does not alter the detector itself.
"""

from pathlib import Path
import json
import numpy as np

from src.policy.rl_policy import GapGuardRLPolicy


RESULT_DIR = Path("results/rl_policy")


def make_training_samples(n=100000, seed=42):
    """
    Temporary controlled training distribution.

    These are normalized detector-output states,
    not raw ADS-B measurements.
    """

    rng = np.random.default_rng(seed)

    scores = rng.uniform(0.1, 2.5, n)

    threshold = 1.0

    evidence = rng.uniform(0.0, 1.0, n)

    # Low attack prevalence, similar to ADS-B experiment scale.
    labels = (
        rng.random(n) < 0.08
    ).astype(int)

    return [
        (float(s), threshold, float(e), int(y))
        for s, e, y in zip(
            scores,
            evidence,
            labels,
        )
    ]


def main():

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("STAGE 4.6 — GAPGUARD RL DECISION LAYER")
    print("=" * 80)

    policy = GapGuardRLPolicy(
        alpha=0.10,
        gamma=0.90,
        epsilon=0.20,
        epsilon_decay=0.995,
        epsilon_min=0.02,
    )

    samples = make_training_samples()

    print(f"Training samples : {len(samples):,}")
    print("State space      : 27")
    print("Actions           : SUPPRESS / ALERT / UNVERIFIABLE")

    policy.train(
        samples,
        episodes=10,
    )

    output_path = (
        RESULT_DIR /
        "gapguard_rl_policy.npy"
    )

    policy.save(output_path)

    # Demonstration states
    demo_states = [
        ("Low severity / high evidence", 0.40, 0.90, 0),
        ("High severity / high evidence", 2.00, 0.90, 0),
        ("High severity / low evidence", 2.00, 0.10, 0),
        ("Medium severity / medium evidence", 1.00, 0.50, 1),
    ]

    print()
    print("RL POLICY DEMONSTRATION")
    print("-" * 80)

    for name, score, evidence, fatigue in demo_states:

        action = policy.predict(
            score,
            1.0,
            evidence,
            fatigue,
        )

        print(
            f"{name:35s} -> {action}"
        )

    summary = {
        "state_space": 27,
        "actions": [
            "SUPPRESS",
            "ALERT",
            "UNVERIFIABLE",
        ],
        "training_samples": len(samples),
        "episodes": 10,
        "policy_file": str(output_path),
    }

    with open(
        RESULT_DIR / "rl_policy_result.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    print()
    print(f"Saved: {output_path}")
    print(
        "STAGE 4.6 COMPLETE — RL POLICY READY"
    )


if __name__ == "__main__":
    main()