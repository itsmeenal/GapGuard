"""
GapGuard Stage 4.6
Tabular Q-learning decision layer.

The RL layer does NOT change the anomaly score.
It only learns how to act on the frozen detector output.
"""

from pathlib import Path
import json
import numpy as np


ACTIONS = {
    0: "SUPPRESS",
    1: "ALERT",
    2: "UNVERIFIABLE",
}


class GapGuardRLPolicy:
    def __init__(
        self,
        alpha=0.10,
        gamma=0.90,
        epsilon=0.20,
        epsilon_decay=0.995,
        epsilon_min=0.02,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # severity(3) x evidence(3) x fatigue(3)
        self.q_table = np.zeros((3, 3, 3, 3), dtype=np.float64)

    @staticmethod
    def severity_bin(score, threshold):
        ratio = score / max(threshold, 1e-12)

        if ratio < 0.75:
            return 0
        elif ratio < 1.50:
            return 1
        return 2

    @staticmethod
    def evidence_bin(evidence):
        if evidence < 0.33:
            return 0
        elif evidence < 0.66:
            return 1
        return 2

    @staticmethod
    def fatigue_bin(recent_alerts):
        if recent_alerts == 0:
            return 0
        elif recent_alerts < 3:
            return 1
        return 2

    def state(self, score, threshold, evidence, recent_alerts):
        s = self.severity_bin(score, threshold)
        e = self.evidence_bin(evidence)
        f = self.fatigue_bin(recent_alerts)
        return s, e, f

    def choose_action(self, state, training=True):
        if training and np.random.random() < self.epsilon:
            return np.random.randint(0, 3)

        return int(np.argmax(self.q_table[state]))

    def update(self, state, action, reward, next_state):
        old_value = self.q_table[state + (action,)]

        next_best = np.max(self.q_table[next_state])

        target = reward + self.gamma * next_best

        self.q_table[state + (action,)] = (
            old_value + self.alpha * (target - old_value)
        )

    def decay_epsilon(self):
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay,
        )

    @staticmethod
    def reward(label, action, evidence):
        """
        label:
            0 = normal
            1 = attack

        evidence:
            normalized evidence quality [0, 1]
        """

        if label == 0:
            if action == 0:       # suppress normal
                return 1.0
            elif action == 1:     # false alert
                return -2.0
            else:                 # unverifiable
                return 0.0

        # Attack
        if evidence < 0.33:
            if action == 2:
                return 2.0
            elif action == 1:
                return -1.0
            else:
                return -2.0

        if action == 1:
            return 4.0
        elif action == 0:
            return -5.0
        else:
            return 1.0

    def train(self, samples, episodes=10):
        """
        samples:
            list of tuples:
            (score, threshold, evidence, label)

        label is binary:
            0 normal
            1 attack
        """

        for episode in range(episodes):

            recent_alerts = 0

            for score, threshold, evidence, label in samples:

                state = self.state(
                    score,
                    threshold,
                    evidence,
                    recent_alerts,
                )

                action = self.choose_action(
                    state,
                    training=True,
                )

                reward = self.reward(
                    label,
                    action,
                    evidence,
                )

                next_recent_alerts = (
                    recent_alerts + 1
                    if action == 1
                    else max(0, recent_alerts - 1)
                )

                next_state = self.state(
                    score,
                    threshold,
                    evidence,
                    next_recent_alerts,
                )

                self.update(
                    state,
                    action,
                    reward,
                    next_state,
                )

                if action == 1:
                    recent_alerts += 1
                else:
                    recent_alerts = max(
                        0,
                        recent_alerts - 1,
                    )

            self.decay_epsilon()

            if episode % 2 == 0:
                print(
                    f"Episode {episode + 1}/{episodes} "
                    f"epsilon={self.epsilon:.4f}"
                )

    def predict(
        self,
        score,
        threshold,
        evidence,
        recent_alerts,
    ):
        state = self.state(
            score,
            threshold,
            evidence,
            recent_alerts,
        )

        action = int(np.argmax(self.q_table[state]))

        return ACTIONS[action]

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        np.save(
            path,
            self.q_table,
        )

        metadata = {
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "actions": ACTIONS,
            "state_shape": list(self.q_table.shape),
        }

        with open(
            path.with_suffix(".json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                metadata,
                f,
                indent=2,
            )

    def load(self, path):
        self.q_table = np.load(path)


if __name__ == "__main__":
    policy = GapGuardRLPolicy()

    print("GapGuard RL policy created.")
    print("State space:", policy.q_table.shape)
    print("Actions:", ACTIONS)