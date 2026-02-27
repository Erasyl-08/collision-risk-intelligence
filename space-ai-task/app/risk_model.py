"""
AI Risk Assessment Module for AstroGuard.

Uses a two-layer approach:
  1. Physics-based: Chan's formula for collision probability (Pc)
  2. ML layer: Random Forest classifier trained on synthetic physics data
     to learn the non-linear risk surface and generalize to edge cases.

Additionally uses Isolation Forest for anomaly detection —
flagging unusual orbital conjunction patterns not covered by historical norms.
"""
import math
import logging
import numpy as np
from typing import Tuple, Optional

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

logger = logging.getLogger(__name__)

# Risk thresholds based on NASA CARA guidelines
PC_THRESHOLDS = {
    "CRITICAL": 1e-3,
    "HIGH": 1e-4,
    "MEDIUM": 1e-5,
    "LOW": 0.0,
}

RISK_LABELS = {0: "LOW", 1: "MEDIUM", 2: "HIGH", 3: "CRITICAL"}
RISK_COLORS = {0: "#00ff88", 1: "#ffcc00", 2: "#ff6b35", 3: "#ff3355"}

RECOMMENDATIONS = {
    0: "No action required. Continue routine monitoring.",
    1: "Increased monitoring recommended. Pre-alert satellite operator.",
    2: "Maneuver assessment required. Notify operator immediately.",
    3: "EMERGENCY: Execute avoidance maneuver within 24 hours.",
}


def chan_collision_probability(
    miss_distance_km: float,
    combined_hard_body_radius_m: float = 10.0,
    position_uncertainty_m: float = 200.0,
) -> float:
    """
    Simplified Chan formula for collision probability.
    Assumes Gaussian position uncertainty and circular hard-body model.

    Reference: Chan, F.K. (2008). Spacecraft Collision Probability.
    """
    d_m = miss_distance_km * 1000.0
    sigma = position_uncertainty_m
    r_hb = combined_hard_body_radius_m

    if sigma <= 0 or d_m < 0:
        return 0.0

    # Combined cross-sectional area
    A_comb = math.pi * r_hb ** 2

    # 2D Gaussian probability at miss distance
    pc = (A_comb / (2 * math.pi * sigma ** 2)) * math.exp(-0.5 * (d_m / sigma) ** 2)
    return min(pc, 1.0)


def pc_to_risk_level(pc: float) -> int:
    if pc >= PC_THRESHOLDS["CRITICAL"]:
        return 3
    elif pc >= PC_THRESHOLDS["HIGH"]:
        return 2
    elif pc >= PC_THRESHOLDS["MEDIUM"]:
        return 1
    return 0


def generate_training_data(n_samples: int = 5000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic training data using physics-based collision probability.
    The ML model learns the risk surface from this physically grounded dataset.
    Uses a stratified sampling approach to ensure all risk levels are well-represented.
    """
    rng = np.random.default_rng(42)

    # Stratified: 30% of samples designed to be HIGH/CRITICAL (rare but important)
    n_rare = n_samples // 3
    n_normal = n_samples - n_rare

    # Normal distribution (most conjunctions are low-risk)
    miss_normal = rng.exponential(scale=15.0, size=n_normal).clip(1.0, 200.0)
    miss_rare = rng.uniform(0.001, 1.0, size=n_rare)  # close approaches
    miss_distances = np.concatenate([miss_normal, miss_rare])

    rel_velocities = rng.uniform(0.1, 15.0, size=n_samples)
    altitudes = rng.uniform(200.0, 2000.0, size=n_samples)

    # Large objects (ISS-class) in rare set → larger cross-section
    obj1_rcs_normal = rng.exponential(scale=2.0, size=n_normal).clip(0.01, 50.0)
    obj1_rcs_rare = rng.uniform(1.0, 400.0, size=n_rare)
    obj1_rcs = np.concatenate([obj1_rcs_normal, obj1_rcs_rare])

    obj2_rcs_normal = rng.exponential(scale=2.0, size=n_normal).clip(0.01, 50.0)
    obj2_rcs_rare = rng.exponential(scale=5.0, size=n_rare).clip(0.01, 100.0)
    obj2_rcs = np.concatenate([obj2_rcs_normal, obj2_rcs_rare])

    sigma_pos = rng.uniform(50.0, 500.0, size=n_samples)

    X = []
    y = []

    for i in range(n_samples):
        d = miss_distances[i]
        v = rel_velocities[i]
        alt = altitudes[i]
        r1 = math.sqrt(obj1_rcs[i] / math.pi)
        r2 = math.sqrt(obj2_rcs[i] / math.pi)
        combined_r = r1 + r2
        sigma = sigma_pos[i]

        # Atmospheric drag degrades uncertainty at lower orbits
        effective_sigma = sigma * (1.0 + 500.0 / max(alt, 200.0))

        pc = chan_collision_probability(d, combined_r, effective_sigma)

        # Kinetic energy proxy (higher velocity = larger debris field if collision)
        ke_factor = min(v / 10.0, 1.0)
        adjusted_pc = pc * (1.0 + 0.5 * ke_factor)

        risk = pc_to_risk_level(adjusted_pc)
        X.append([d, v, alt, obj1_rcs[i], obj2_rcs[i], sigma, adjusted_pc, ke_factor])
        y.append(risk)

    return np.array(X), np.array(y)


class AstroGuardRiskModel:
    """
    Ensemble AI risk model combining:
    - Random Forest classifier for risk level prediction
    - Isolation Forest for anomaly detection of unusual conjunction patterns
    """

    def __init__(self):
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        self.anomaly_detector = IsolationForest(
            n_estimators=100,
            contamination=0.05,
            random_state=42,
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.train_accuracy = 0.0
        self.feature_names = [
            "miss_distance_km",
            "relative_velocity_km_s",
            "altitude_km",
            "object1_rcs_m2",
            "object2_rcs_m2",
            "position_sigma_m",
            "collision_probability",
            "kinetic_energy_factor",
        ]

    def train(self) -> float:
        """Train on synthetic physics data. Returns validation accuracy."""
        logger.info("Generating synthetic training data...")
        X, y = generate_training_data(n_samples=6000)

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        logger.info("Training Random Forest risk classifier...")
        self.classifier.fit(X_train_scaled, y_train)

        logger.info("Training Isolation Forest anomaly detector...")
        self.anomaly_detector.fit(X_train_scaled)

        y_pred = self.classifier.predict(X_val_scaled)
        self.train_accuracy = float(accuracy_score(y_val, y_pred))
        self.is_trained = True

        logger.info(f"Model trained. Validation accuracy: {self.train_accuracy:.3f}")
        return self.train_accuracy

    def _build_features(
        self,
        miss_distance_km: float,
        relative_velocity_km_s: float,
        altitude_km: float,
        object1_rcs_m2: float = 1.0,
        object2_rcs_m2: float = 1.0,
        position_sigma_m: float = 200.0,
    ) -> np.ndarray:
        r1 = math.sqrt(max(object1_rcs_m2, 0.01) / math.pi)
        r2 = math.sqrt(max(object2_rcs_m2, 0.01) / math.pi)
        combined_r = r1 + r2
        effective_sigma = position_sigma_m * (1.0 + 500.0 / max(altitude_km, 200.0))
        pc = chan_collision_probability(miss_distance_km, combined_r, effective_sigma)
        ke_factor = min(relative_velocity_km_s / 10.0, 1.0)
        return np.array([[
            miss_distance_km,
            relative_velocity_km_s,
            altitude_km,
            object1_rcs_m2,
            object2_rcs_m2,
            position_sigma_m,
            pc,
            ke_factor,
        ]])

    def predict(
        self,
        miss_distance_km: float,
        relative_velocity_km_s: float,
        altitude_km: float,
        object1_rcs_m2: float = 1.0,
        object2_rcs_m2: float = 1.0,
        position_sigma_m: float = 200.0,
    ) -> dict:
        """
        Predict risk level and detect anomalies for a conjunction event.
        Returns full assessment dict.
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")

        features = self._build_features(
            miss_distance_km, relative_velocity_km_s, altitude_km,
            object1_rcs_m2, object2_rcs_m2, position_sigma_m,
        )
        features_scaled = self.scaler.transform(features)

        ml_risk = int(self.classifier.predict(features_scaled)[0])
        probabilities = self.classifier.predict_proba(features_scaled)[0]
        confidence = float(max(probabilities))

        # Physics-based hard floor: never downgrade below what the Chan formula demands.
        # This ensures safety-critical events are never missed due to model uncertainty.
        pc = float(features[0, 6])
        physics_risk = pc_to_risk_level(pc)
        risk_level = max(ml_risk, physics_risk)

        anomaly_score = self.anomaly_detector.decision_function(features_scaled)[0]
        is_anomaly = bool(self.anomaly_detector.predict(features_scaled)[0] == -1)

        pc = float(features[0, 6])

        explanation = self._build_explanation(
            risk_level, miss_distance_km, relative_velocity_km_s,
            altitude_km, pc, is_anomaly, anomaly_score,
        )

        return {
            "risk_level": risk_level,
            "risk_label": RISK_LABELS[risk_level],
            "risk_color": RISK_COLORS[risk_level],
            "collision_probability": pc,
            "confidence": confidence,
            "is_anomaly": is_anomaly,
            "anomaly_score": float(anomaly_score),
            "recommendation": RECOMMENDATIONS[risk_level],
            "explanation": explanation,
            "features": {
                "miss_distance_km": round(miss_distance_km, 4),
                "relative_velocity_km_s": round(relative_velocity_km_s, 4),
                "altitude_km": round(altitude_km, 1),
                "collision_probability": f"{pc:.2e}",
            },
        }

    def _build_explanation(
        self,
        risk_level: int,
        miss_distance_km: float,
        rel_velocity_km_s: float,
        altitude_km: float,
        pc: float,
        is_anomaly: bool,
        anomaly_score: float,
    ) -> str:
        label = RISK_LABELS[risk_level]
        parts = [
            f"Risk assessed as {label} based on {miss_distance_km:.2f} km miss distance "
            f"at {rel_velocity_km_s:.1f} km/s relative velocity "
            f"(altitude: {altitude_km:.0f} km). "
            f"Collision probability: {pc:.2e}."
        ]
        if is_anomaly:
            parts.append(
                f" ⚠ ANOMALY DETECTED: This conjunction exhibits unusual orbital characteristics "
                f"(anomaly score: {anomaly_score:.3f}). Recommend independent verification."
            )
        if altitude_km < 400:
            parts.append(" Low orbit increases atmospheric drag uncertainty.")
        if rel_velocity_km_s > 10.0:
            parts.append(f" High relative velocity ({rel_velocity_km_s:.1f} km/s) indicates polar/retrograde crossing.")
        return "".join(parts)

    def get_feature_importances(self) -> dict:
        if not self.is_trained:
            return {}
        importances = self.classifier.feature_importances_
        return {name: round(float(imp), 4) for name, imp in zip(self.feature_names, importances)}
