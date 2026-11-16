from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize
from sklearn.cluster import KMeans, MiniBatchKMeans

from .he import decrypt_float, encrypt_float, generate_paillier_keypair


EPS = 1e-12


@dataclass
class Rule:
    centroid: np.ndarray
    scale: float
    beta: np.ndarray
    theta: float
    reliability: float
    support: float
    class_label: int | None


def _sigmoid(values: np.ndarray) -> np.ndarray:
    positive = values >= 0
    result = np.empty_like(values, dtype=float)
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.maximum(np.sum(exp_values, axis=1, keepdims=True), EPS)


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.maximum(np.asarray(values, dtype=float), 0.0)
    row_sum = np.sum(values, axis=1, keepdims=True)
    uniform = np.full_like(values, 1.0 / max(values.shape[1], 1))
    return np.divide(values, row_sum, out=uniform, where=row_sum > EPS)


def _activation_weights(match: np.ndarray, theta: np.ndarray) -> np.ndarray:
    activation = np.maximum(match, EPS) * np.maximum(theta[None, :], EPS)
    return activation / np.maximum(np.sum(activation, axis=1, keepdims=True), EPS)


def er_aggregate(weights: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Vectorized analytical ER aggregation from Eq. (3.5)."""
    weights = np.asarray(weights, dtype=float)
    beta = np.asarray(beta, dtype=float)
    beta_sum = np.sum(beta, axis=1)
    mu_factor = 1.0 - weights[:, :, None] * (beta_sum[None, :, None] - beta[None, :, :])
    eta_factor = 1.0 - weights * beta_sum[None, :]
    rho_factor = 1.0 - weights
    mu = np.prod(np.maximum(mu_factor, EPS), axis=1)
    eta = np.prod(np.maximum(eta_factor, EPS), axis=1)
    rho = np.prod(np.maximum(rho_factor, EPS), axis=1)
    denominator = np.sum(mu, axis=1) - (beta.shape[1] - 1) * eta - rho
    beliefs = (mu - eta[:, None]) / np.maximum(denominator[:, None], EPS)
    return np.clip(beliefs, 0.0, 1.0)


def _as_beliefs(y: np.ndarray, classes: np.ndarray) -> np.ndarray:
    values = np.asarray(y)
    if values.ndim == 2:
        if values.shape[1] != len(classes):
            raise ValueError("Soft labels must follow the shared conclusion-level order")
        beliefs = values.astype(float)
        if np.any(beliefs < 0.0) or np.any(np.sum(beliefs, axis=1) > 1.0 + 1e-9):
            raise ValueError("Label beliefs must be nonnegative and sum to at most one")
        return beliefs
    if values.ndim != 1:
        raise ValueError("Labels must be a class vector or a belief matrix")
    beliefs = np.zeros((len(values), len(classes)), dtype=float)
    for class_idx, class_label in enumerate(classes):
        beliefs[:, class_idx] = values == class_label
    if np.any(np.sum(beliefs, axis=1) == 0):
        raise ValueError("Every hard label must belong to the shared conclusion levels")
    return beliefs


class ClientBRB:
    """Feature-local BRB whose parameters are learned from local target beliefs."""

    def __init__(
        self,
        n_rules_per_class: int = 3,
        class_balance_power: float = 0.0,
        rule_strategy: str = "prototype",
        scale_multiplier: float = 1.0,
        random_state: int = 13,
        supervised: bool = True,
        n_rules: int | None = None,
        max_iter: int = 60,
        tolerance: float = 1e-6,
        optimize_rule_weights: bool = True,
        belief_mass_strength: float = 0.0,
        beta_initialization: str = "random",
        objective_loss: str = "squared_error",
    ) -> None:
        self.n_rules_per_class = int(n_rules_per_class)
        self.n_rules = None if n_rules is None else int(n_rules)
        if self.n_rules_per_class < 1 or (self.n_rules is not None and self.n_rules < 1):
            raise ValueError("Rule counts must be positive")
        self.class_balance_power = float(class_balance_power)
        if self.class_balance_power < 0.0:
            raise ValueError("class_balance_power must be nonnegative")
        if rule_strategy not in {
            "prototype",
            "classwise",
            "mixed",
            "supervised_classwise",
            "supervised_classwise_local_scale",
        }:
            raise ValueError(
                "rule_strategy must be 'prototype', 'classwise', 'mixed', "
                "'supervised_classwise', or 'supervised_classwise_local_scale'"
            )
        # Legacy strategy names are accepted for old runners, but antecedents are
        # always constructed by label-independent local clustering.
        self.rule_strategy = (
            rule_strategy
            if rule_strategy
            in {"supervised_classwise", "supervised_classwise_local_scale"}
            else "prototype"
        )
        self.scale_multiplier = float(scale_multiplier)
        self.random_state = int(random_state)
        self.supervised = bool(supervised)
        self.max_iter = int(max_iter)
        self.tolerance = float(tolerance)
        if self.max_iter < 1 or self.tolerance <= 0.0:
            raise ValueError("Optimizer settings must be positive")
        self.optimize_rule_weights = bool(optimize_rule_weights)
        self.belief_mass_strength = float(belief_mass_strength)
        if self.belief_mass_strength < 0.0:
            raise ValueError("belief_mass_strength must be nonnegative")
        if beta_initialization not in {"random", "local_label"}:
            raise ValueError("beta_initialization must be 'random' or 'local_label'")
        self.beta_initialization = beta_initialization
        if objective_loss not in {"squared_error", "cross_entropy"}:
            raise ValueError(
                "objective_loss must be 'squared_error' or 'cross_entropy'"
            )
        self.objective_loss = objective_loss
        self.classes_: np.ndarray | None = None
        self.rules_: List[Rule] = []
        self.prior_: np.ndarray | None = None
        self.training_loss_: float = float("nan")
        self.initial_loss_: float = float("nan")
        self.optimization_success_: bool = False
        self.optimization_iterations_: int = 0
        self.optimizer_converged_: bool = False
        self.hit_iteration_limit_: bool = False
        self.final_gradient_norm_: float = float("nan")

    def _make_centers(self, X: np.ndarray, n_clusters: int) -> Tuple[np.ndarray, np.ndarray]:
        unique_count = len(np.unique(X, axis=0)) if len(X) <= 20_000 else len(X)
        n_clusters = max(1, min(int(n_clusters), len(X), unique_count))
        if n_clusters == 1:
            return np.mean(X, axis=0, keepdims=True), np.zeros(len(X), dtype=int)
        if len(X) > 20_000:
            kmeans = MiniBatchKMeans(
                n_clusters=n_clusters,
                batch_size=min(4096, len(X)),
                n_init=3,
                max_iter=100,
                reassignment_ratio=0.01,
                random_state=self.random_state,
            )
        else:
            kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=self.random_state)
        labels = kmeans.fit_predict(X)
        return np.asarray(kmeans.cluster_centers_, dtype=float), labels

    def _rule_count(self, n_classes: int) -> int:
        if self.n_rules is not None:
            return max(1, self.n_rules)
        return max(1, self.n_rules_per_class * n_classes)

    def _build_feature_rules(self, X: np.ndarray, n_classes: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        centers, assignments = self._make_centers(X, self._rule_count(n_classes))
        return self._centers_to_rule_geometry(X, centers, assignments)

    def _centers_to_rule_geometry(
        self,
        X: np.ndarray,
        centers: np.ndarray,
        assignments: np.ndarray | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        centers = np.asarray(centers, dtype=float)
        if assignments is None:
            squared_distance = np.sum(
                (X[:, None, :] - centers[None, :, :]) ** 2,
                axis=2,
            )
            assignments = np.argmin(squared_distance, axis=1)
        scales = []
        supports = []
        global_scale = float(np.median(np.linalg.norm(X - np.mean(X, axis=0), axis=1)) + 1e-3)
        for rule_idx, center in enumerate(centers):
            members = X[assignments == rule_idx]
            if len(members) == 0:
                distances = np.asarray([global_scale])
            else:
                distances = np.linalg.norm(members - center, axis=1)
            scale = np.median(distances) + np.std(distances) + 1e-3
            scales.append(max(float(scale * self.scale_multiplier), 0.05))
            supports.append(float(len(members)))
        return centers, np.asarray(scales), np.asarray(supports)

    def _build_supervised_feature_rules(
        self,
        X: np.ndarray,
        targets: np.ndarray,
        mask: np.ndarray,
        n_classes: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Construct a balanced set of class-conditional rule antecedents.

        This option is intentionally separate from the default label-independent
        prototype strategy.  It is used by performance-ceiling experiments in
        which the distributed label records are first combined into an observed
        training target; no test labels enter antecedent construction.
        """
        total_rules = self._rule_count(n_classes)
        hard_targets = np.argmax(targets, axis=1)
        base, remainder = divmod(total_rules, n_classes)
        centers = []
        for class_idx in range(n_classes):
            class_rows = X[mask & (hard_targets == class_idx)]
            requested = base + int(class_idx < remainder)
            if requested <= 0 or len(class_rows) == 0:
                continue
            class_centers, _ = self._make_centers(
                class_rows,
                min(requested, len(class_rows)),
            )
            centers.append(class_centers)
        if not centers:
            return self._build_feature_rules(X, n_classes)
        combined = np.concatenate(centers, axis=0)
        if len(combined) < total_rules:
            extra, _ = self._make_centers(X, total_rules - len(combined))
            combined = np.concatenate([combined, extra], axis=0)
        return self._centers_to_rule_geometry(X, combined)

    def _build_supervised_local_scale_rules(
        self,
        X: np.ndarray,
        targets: np.ndarray,
        mask: np.ndarray,
        n_classes: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Use class-conditional samples for both centers and rule radii."""
        total_rules = self._rule_count(n_classes)
        hard_targets = np.argmax(targets, axis=1)
        base, remainder = divmod(total_rules, n_classes)
        global_scale = float(
            np.median(np.linalg.norm(X - np.mean(X, axis=0), axis=1)) + 1e-3
        )
        centers = []
        scales = []
        supports = []
        for class_idx in range(n_classes):
            class_rows = X[mask & (hard_targets == class_idx)]
            requested = base + int(class_idx < remainder)
            if requested <= 0 or len(class_rows) == 0:
                continue
            class_centers, assignments = self._make_centers(
                class_rows,
                min(requested, len(class_rows)),
            )
            for center_idx, center in enumerate(class_centers):
                members = class_rows[assignments == center_idx]
                distances = (
                    np.linalg.norm(members - center, axis=1)
                    if len(members)
                    else np.asarray([global_scale])
                )
                scale = np.median(distances) + np.std(distances) + 1e-3
                centers.append(np.asarray(center, dtype=float))
                scales.append(max(float(scale * self.scale_multiplier), 0.05))
                supports.append(float(len(members)))
        if not centers:
            return self._build_feature_rules(X, n_classes)
        return (
            np.stack(centers, axis=0),
            np.asarray(scales, dtype=float),
            np.asarray(supports, dtype=float),
        )

    @staticmethod
    def _match(X: np.ndarray, centers: np.ndarray, scales: np.ndarray) -> np.ndarray:
        squared_distance = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        return np.maximum(np.exp(-0.5 * squared_distance / np.maximum(scales[None, :] ** 2, EPS)), EPS)

    def _initial_parameters(
        self,
        n_rules: int,
        n_classes: int,
        supports: np.ndarray,
        complete_beta: np.ndarray | None = None,
    ) -> np.ndarray:
        rng = np.random.default_rng(self.random_state)
        if complete_beta is None:
            beta_logits = rng.normal(0.0, 0.03, size=(n_rules, n_classes + 1))
            beta_logits[:, -1] = np.log(0.05 / max(0.95 / n_classes, EPS))
        else:
            complete_beta = np.maximum(np.asarray(complete_beta, dtype=float), 1e-6)
            complete_beta /= np.maximum(np.sum(complete_beta, axis=1, keepdims=True), EPS)
            beta_logits = np.log(complete_beta)
        coverage = (supports + 1.0) / (np.max(supports) + 1.0)
        coverage = np.clip(coverage, 0.05, 0.95)
        theta_logits = np.log(coverage / (1.0 - coverage))
        return np.concatenate([beta_logits.ravel(), theta_logits])

    @staticmethod
    def _decode_parameters(parameters: np.ndarray, n_rules: int, n_classes: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        split = n_rules * (n_classes + 1)
        complete_beta = _softmax(parameters[:split].reshape(n_rules, n_classes + 1))
        beta = complete_beta[:, :n_classes]
        theta = _sigmoid(parameters[split:])
        return beta, theta, complete_beta

    def _objective_and_gradient(
        self,
        parameters: np.ndarray,
        match: np.ndarray,
        targets: np.ndarray,
        sample_weights: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        n_samples, n_rules = match.shape
        n_classes = targets.shape[1]
        beta, theta, complete_beta = self._decode_parameters(parameters, n_rules, n_classes)
        activation = np.maximum(match, EPS) * theta[None, :]
        activation_sum = np.maximum(np.sum(activation, axis=1, keepdims=True), EPS)
        weights = activation / activation_sum

        beta_sum = np.sum(beta, axis=1)
        mu_factor = np.maximum(
            1.0 - weights[:, :, None] * (beta_sum[None, :, None] - beta[None, :, :]),
            EPS,
        )
        eta_factor = np.maximum(1.0 - weights * beta_sum[None, :], EPS)
        rho_factor = np.maximum(1.0 - weights, EPS)
        mu = np.prod(mu_factor, axis=1)
        eta = np.prod(eta_factor, axis=1)
        rho = np.prod(rho_factor, axis=1)
        denominator = np.maximum(np.sum(mu, axis=1) - (n_classes - 1) * eta - rho, EPS)
        numerator = mu - eta[:, None]
        beliefs = numerator / denominator[:, None]

        weight_total = max(float(np.sum(sample_weights)), EPS)
        residual = beliefs - targets
        if self.objective_loss == "squared_error":
            loss = (
                0.5
                * float(np.sum(sample_weights[:, None] * residual * residual))
                / weight_total
            )
            belief_gradient = sample_weights[:, None] * residual / weight_total
        else:
            assigned = np.maximum(np.sum(beliefs, axis=1, keepdims=True), EPS)
            normalized_beliefs = np.maximum(beliefs / assigned, EPS)
            target_mass = np.sum(targets, axis=1, keepdims=True)
            loss = -float(
                np.sum(
                    sample_weights[:, None]
                    * targets
                    * np.log(normalized_beliefs)
                )
            ) / weight_total
            belief_gradient = (
                sample_weights[:, None]
                * (
                    -targets / np.maximum(beliefs, EPS)
                    + target_mass / assigned
                )
                / weight_total
            )
        if self.belief_mass_strength > 0.0:
            mass_residual = np.sum(residual, axis=1)
            loss += (
                0.5
                * self.belief_mass_strength
                * float(np.sum(sample_weights * mass_residual * mass_residual))
                / weight_total
            )
            belief_gradient += (
                self.belief_mass_strength
                * sample_weights[:, None]
                * mass_residual[:, None]
                / weight_total
            )

        shared = np.sum(belief_gradient * numerator, axis=1)
        mu_gradient = belief_gradient / denominator[:, None] - shared[:, None] / denominator[:, None] ** 2
        eta_gradient = (
            -np.sum(belief_gradient, axis=1) / denominator
            + (n_classes - 1) * shared / denominator**2
        )
        rho_gradient = shared / denominator**2

        mu_factor_gradient = mu_gradient[:, None, :] * mu[:, None, :] / mu_factor
        eta_factor_gradient = eta_gradient[:, None] * eta[:, None] / eta_factor
        rho_factor_gradient = rho_gradient[:, None] * rho[:, None] / rho_factor

        beta_gradient = np.zeros_like(beta)
        for class_idx in range(n_classes):
            factor_derivative = -weights[:, :, None] * np.ones((1, 1, n_classes))
            factor_derivative[:, :, class_idx] += weights
            beta_gradient[:, class_idx] = np.sum(mu_factor_gradient * factor_derivative, axis=(0, 2))
            beta_gradient[:, class_idx] += np.sum(eta_factor_gradient * (-weights), axis=0)

        weight_gradient = np.sum(
            mu_factor_gradient * (-(beta_sum[None, :, None] - beta[None, :, :])),
            axis=2,
        )
        weight_gradient += eta_factor_gradient * (-beta_sum[None, :])
        weight_gradient += -rho_factor_gradient

        activation_gradient = (
            weight_gradient - np.sum(weight_gradient * weights, axis=1, keepdims=True)
        ) / activation_sum
        theta_gradient = np.sum(activation_gradient * match, axis=0)

        full_beta_gradient = np.zeros_like(complete_beta)
        full_beta_gradient[:, :n_classes] = beta_gradient
        beta_logit_gradient = complete_beta * (
            full_beta_gradient - np.sum(full_beta_gradient * complete_beta, axis=1, keepdims=True)
        )
        theta_logit_gradient = theta_gradient * theta * (1.0 - theta)
        if not self.optimize_rule_weights:
            theta_logit_gradient[:] = 0.0
        gradient = np.concatenate([beta_logit_gradient.ravel(), theta_logit_gradient])
        return loss, gradient

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        label_mask: np.ndarray | None = None,
        classes: np.ndarray | None = None,
    ) -> "ClientBRB":
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or len(X) == 0:
            raise ValueError("X must be a nonempty two-dimensional local feature block")
        values = np.asarray(y)
        if len(values) != len(X):
            raise ValueError("X and local labels must contain the same aligned samples")
        if classes is None:
            classes = np.arange(values.shape[1]) if values.ndim == 2 else np.unique(values)
        self.classes_ = np.asarray(classes)
        targets = _as_beliefs(values, self.classes_)
        mask = np.ones(len(X), dtype=bool) if label_mask is None else np.asarray(label_mask, dtype=bool)
        if mask.shape != (len(X),):
            raise ValueError("label_mask must contain one availability flag per aligned sample")
        if not self.supervised:
            mask[:] = False

        if np.any(mask):
            self.prior_ = np.mean(targets[mask], axis=0)
        else:
            self.prior_ = np.full(len(self.classes_), 1.0 / len(self.classes_))

        if self.rule_strategy.startswith("supervised_classwise") and np.any(mask):
            builder = (
                self._build_supervised_local_scale_rules
                if self.rule_strategy == "supervised_classwise_local_scale"
                else self._build_supervised_feature_rules
            )
            centers, scales, supports = builder(X, targets, mask, len(self.classes_))
        else:
            centers, scales, supports = self._build_feature_rules(
                X,
                len(self.classes_),
            )
        match = self._match(X, centers, scales)
        initial_complete_beta = None
        if self.beta_initialization == "local_label" and np.any(mask):
            responsibilities = match[mask] / np.maximum(
                np.sum(match[mask], axis=1, keepdims=True),
                EPS,
            )
            assigned = responsibilities.T @ targets[mask]
            denominators = np.maximum(np.sum(responsibilities, axis=0)[:, None], EPS)
            assigned = assigned / denominators
            unassigned = np.maximum(1.0 - np.sum(assigned, axis=1, keepdims=True), 0.0)
            initial_complete_beta = np.concatenate([assigned, unassigned], axis=1)
            # A small symmetric smoothing term avoids saturated softmax logits
            # without changing the client-local label semantics.
            initial_complete_beta += 1e-3
            initial_complete_beta /= np.sum(initial_complete_beta, axis=1, keepdims=True)
        parameters = self._initial_parameters(
            len(centers),
            len(self.classes_),
            supports,
            complete_beta=initial_complete_beta,
        )

        if np.any(mask):
            local_targets = targets[mask]
            sample_weights = np.ones(np.sum(mask), dtype=float)
            if self.class_balance_power > 0.0:
                hard = np.argmax(local_targets, axis=1)
                counts = np.bincount(hard, minlength=len(self.classes_)).astype(float)
                class_weights = (
                    len(hard) / np.maximum(counts * len(self.classes_), EPS)
                ) ** self.class_balance_power
                sample_weights = class_weights[hard]
            objective = lambda vector: self._objective_and_gradient(
                vector,
                match[mask],
                local_targets,
                sample_weights,
            )
            self.initial_loss_ = float(objective(parameters)[0])
            result = minimize(
                objective,
                parameters,
                method="L-BFGS-B",
                jac=True,
                options={"maxiter": self.max_iter, "ftol": self.tolerance, "maxls": 30},
            )
            parameters = np.asarray(result.x, dtype=float)
            self.training_loss_ = float(result.fun)
            self.optimization_success_ = bool(result.success or result.fun <= self.initial_loss_)
            self.optimization_iterations_ = int(result.nit)
            self.optimizer_converged_ = bool(result.success)
            self.hit_iteration_limit_ = bool(result.nit >= self.max_iter)
            self.final_gradient_norm_ = float(np.linalg.norm(np.asarray(result.jac, dtype=float)))
        else:
            self.initial_loss_ = float("nan")
            self.training_loss_ = float("nan")
            self.optimization_success_ = False
            self.optimization_iterations_ = 0
            self.optimizer_converged_ = False
            self.hit_iteration_limit_ = False
            self.final_gradient_norm_ = float("nan")

        beta, theta, _ = self._decode_parameters(parameters, len(centers), len(self.classes_))
        self.rules_ = []
        for rule_idx, center in enumerate(centers):
            self.rules_.append(
                Rule(
                    centroid=np.asarray(center, dtype=float),
                    scale=float(scales[rule_idx]),
                    beta=np.asarray(beta[rule_idx], dtype=float),
                    theta=float(theta[rule_idx]),
                    reliability=1.0,
                    support=float(supports[rule_idx]),
                    class_label=int(self.classes_[int(np.argmax(beta[rule_idx]))]),
                )
            )
        return self

    def _rule_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.classes_ is None or not self.rules_:
            raise ValueError("ClientBRB has not been fitted")
        centers = np.stack([rule.centroid for rule in self.rules_], axis=0)
        scales = np.asarray([rule.scale for rule in self.rules_], dtype=float)
        beta = np.stack([rule.beta for rule in self.rules_], axis=0)
        theta = np.asarray([rule.theta * rule.reliability for rule in self.rules_], dtype=float)
        return centers, scales, beta, theta

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        centers, scales, beta, theta = self._rule_arrays()
        match = self._match(np.asarray(X, dtype=float), centers, scales)
        return er_aggregate(_activation_weights(match, theta), beta)

    def rule_count(self) -> int:
        return len(self.rules_)


def _normalized_source_weights(
    n_clients: int,
    source_weights: Sequence[float] | None,
    n_classes: int | None = None,
) -> np.ndarray:
    if source_weights is None:
        return np.full(n_clients, 1.0 / max(n_clients, 1), dtype=float)
    weights = np.asarray(source_weights, dtype=float)
    if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("source_weights must be finite and nonnegative")
    if weights.shape == (n_clients,):
        weights = np.maximum(weights, EPS)
        return weights / np.maximum(np.sum(weights), EPS)
    if n_classes is not None and weights.shape == (n_clients, n_classes):
        weights = np.maximum(weights, EPS)
        return weights / np.maximum(np.sum(weights, axis=0, keepdims=True), EPS)
    expected = f"({n_clients},)" if n_classes is None else f"({n_clients},) or ({n_clients}, {n_classes})"
    raise ValueError(f"source_weights must have shape {expected}")


def _moment_fusion_from_aggregates(
    first_moment: np.ndarray,
    second_weighted_moment: np.ndarray,
    source_weights: np.ndarray,
    scer_lambda: float,
) -> np.ndarray:
    if scer_lambda <= 0.0 or source_weights.shape[0] < 2:
        return np.clip(first_moment, 0.0, 1.0)
    if source_weights.ndim == 1:
        denominator = float(1.0 - np.sum(source_weights**2))
    else:
        denominator = 1.0 - np.sum(source_weights**2, axis=0)
    common = np.sqrt(
        np.maximum(
            (first_moment**2 - second_weighted_moment) / np.maximum(denominator, EPS),
            0.0,
        )
    )
    fused = (1.0 - scer_lambda) * first_moment + scer_lambda * common
    return np.clip(fused, 0.0, 1.0)


def _apply_scer_moment_fusion(
    client_probs: np.ndarray,
    scer_lambda: float,
    source_weights: Sequence[float] | None = None,
) -> np.ndarray:
    client_probs = np.asarray(client_probs, dtype=float)
    if client_probs.ndim != 3 or client_probs.shape[1] < 1 or client_probs.shape[2] < 1:
        raise ValueError("client_probs must have shape (samples, clients, conclusions)")
    if np.any(~np.isfinite(client_probs)) or np.any(client_probs < 0.0):
        raise ValueError("Client beliefs must be finite and nonnegative")
    if not 0.0 <= scer_lambda <= 1.0:
        raise ValueError("scer_lambda must lie in [0, 1]")
    weights = _normalized_source_weights(client_probs.shape[1], source_weights, client_probs.shape[2])
    if weights.ndim == 1:
        first = np.einsum("skc,k->sc", client_probs, weights)
        second = np.einsum("skc,k->sc", client_probs * client_probs, weights * weights)
    else:
        first = np.einsum("skc,kc->sc", client_probs, weights)
        second = np.einsum("skc,kc->sc", client_probs * client_probs, weights * weights)
    return _moment_fusion_from_aggregates(first, second, weights, scer_lambda)


def _he_scer_moment_fusion(
    client_probs: np.ndarray,
    scer_lambda: float,
    key_bits: int,
    scale: int,
    seed: int,
    source_weights: Sequence[float] | None = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    client_probs = np.asarray(client_probs, dtype=float)
    if client_probs.ndim != 3:
        raise ValueError("client_probs must have shape (samples, clients, conclusions)")
    if scale < 1:
        raise ValueError("The fixed-point scale must be positive")
    if not 0.0 <= scer_lambda <= 1.0:
        raise ValueError("scer_lambda must lie in [0, 1]")
    keygen_start = time.perf_counter()
    public_key, private_key = generate_paillier_keypair(bits=key_bits, seed=seed)
    keygen_elapsed = time.perf_counter() - keygen_start
    rng = random.Random(seed + 1009)
    n_samples, n_clients, n_classes = client_probs.shape
    weights = _normalized_source_weights(n_clients, source_weights, n_classes)
    aggregated_sum = np.zeros((n_samples, n_classes), dtype=float)
    aggregated_squares = np.zeros((n_samples, n_classes), dtype=float)
    encryptions = 0
    start = time.perf_counter()
    for sample_id in range(n_samples):
        for class_id in range(n_classes):
            coefficients = weights if weights.ndim == 1 else weights[:, class_id]
            ciphertexts = [
                encrypt_float(
                    float(coefficients[client_id] * client_probs[sample_id, client_id, class_id]),
                    public_key,
                    scale,
                    rng,
                )
                for client_id in range(n_clients)
            ]
            ciphertexts_sq = [
                encrypt_float(
                    float(
                        coefficients[client_id] ** 2
                        * client_probs[sample_id, client_id, class_id] ** 2
                    ),
                    public_key,
                    scale,
                    rng,
                )
                for client_id in range(n_clients)
            ]
            encryptions += 2 * n_clients
            aggregated_sum[sample_id, class_id] = decrypt_float(
                public_key.add(ciphertexts), private_key, scale
            )
            aggregated_squares[sample_id, class_id] = decrypt_float(
                public_key.add(ciphertexts_sq), private_key, scale
            )
    elapsed = time.perf_counter() - start
    fused = _moment_fusion_from_aggregates(
        aggregated_sum,
        aggregated_squares,
        weights,
        scer_lambda,
    )
    stats = {
        "he_keygen_seconds": keygen_elapsed,
        "he_seconds": elapsed,
        "he_end_to_end_seconds": keygen_elapsed + elapsed,
        "he_encryptions": float(encryptions),
        "he_ciphertext_bytes": float(public_key.ciphertext_bytes),
        "he_total_upload_bytes": float(encryptions * public_key.ciphertext_bytes),
        "he_scale": float(scale),
        "he_key_bits": float(key_bits),
    }
    return fused, stats


class DistributedLabelHEBRB:
    """Distributed local BRBs with optional Paillier-protected moment fusion."""

    def __init__(
        self,
        n_rules_per_class: int = 3,
        class_balance_power: float = 0.0,
        label_mode: str = "distributed",
        enable_scer: bool = True,
        scer_lambda: float = 0.25,
        use_he: bool = False,
        he_key_bits: int = 128,
        he_scale: int = 1_000_000,
        decision_rule: str = "argmax",
        rule_strategy: str = "prototype",
        scale_multiplier: float = 1.0,
        random_state: int = 13,
        source_weights: Sequence[float] | None = None,
        n_rules: int | None = None,
        max_iter: int = 60,
        tolerance: float = 1e-6,
        belief_mass_strength: float = 0.0,
        beta_initialization: str = "random",
    ) -> None:
        self.n_rules_per_class = int(n_rules_per_class)
        self.n_rules = None if n_rules is None else int(n_rules)
        self.class_balance_power = float(class_balance_power)
        self.label_mode = label_mode
        self.enable_scer = bool(enable_scer)
        self.scer_lambda = float(scer_lambda)
        if not 0.0 <= self.scer_lambda <= 1.0:
            raise ValueError("scer_lambda must lie in [0, 1]")
        self.use_he = bool(use_he)
        self.he_key_bits = int(he_key_bits)
        self.he_scale = int(he_scale)
        if self.he_key_bits < 64 or self.he_scale < 1:
            raise ValueError("Invalid Paillier experiment settings")
        if decision_rule not in {"threshold", "argmax"}:
            raise ValueError("decision_rule must be 'threshold' or 'argmax'")
        self.decision_rule = decision_rule
        self.rule_strategy = rule_strategy
        self.scale_multiplier = float(scale_multiplier)
        self.source_weights = None if source_weights is None else np.asarray(source_weights, dtype=float)
        self.random_state = int(random_state)
        self.max_iter = int(max_iter)
        self.tolerance = float(tolerance)
        if self.max_iter < 1 or self.tolerance <= 0.0:
            raise ValueError("Optimizer settings must be positive")
        self.belief_mass_strength = float(belief_mass_strength)
        if self.belief_mass_strength < 0.0:
            raise ValueError("belief_mass_strength must be nonnegative")
        if beta_initialization not in {"random", "local_label"}:
            raise ValueError("beta_initialization must be 'random' or 'local_label'")
        self.beta_initialization = beta_initialization
        self.clients_: Dict[int, ClientBRB] = {}
        self.classes_: np.ndarray | None = None
        self.minority_label_: int | None = None
        self.minority_threshold_: float = 0.5
        self.last_he_stats_: Dict[str, float] = {}

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        party_features: Mapping[int, Sequence[int]],
        client_labels: Mapping[int, np.ndarray] | None = None,
        label_masks: Mapping[int, np.ndarray] | None = None,
        classes: np.ndarray | None = None,
    ) -> "DistributedLabelHEBRB":
        X = np.asarray(X, dtype=float)
        global_values = np.asarray(y)
        if classes is None:
            classes = np.arange(global_values.shape[1]) if global_values.ndim == 2 else np.unique(global_values)
        self.classes_ = np.asarray(classes)
        hard_global = (
            self.classes_[np.argmax(global_values, axis=1)]
            if global_values.ndim == 2
            else global_values
        )
        counts = np.asarray([np.sum(hard_global == label) for label in self.classes_], dtype=float)
        self.minority_label_ = int(self.classes_[int(np.argmin(counts))])
        self.minority_threshold_ = 0.5
        self.clients_ = {}
        for client_id, features in party_features.items():
            client_id = int(client_id)
            local_y = global_values if client_labels is None else np.asarray(client_labels[client_id])
            local_mask = (
                np.ones(len(X), dtype=bool)
                if label_masks is None
                else np.asarray(label_masks[client_id], dtype=bool)
            )
            supervised = self.label_mode == "distributed" or (
                self.label_mode == "active_only" and client_id == 0
            )
            if self.label_mode == "no_label_calibration" or not supervised:
                local_mask = np.zeros(len(X), dtype=bool)
            client = ClientBRB(
                n_rules_per_class=self.n_rules_per_class,
                n_rules=self.n_rules,
                class_balance_power=self.class_balance_power if supervised else 0.0,
                rule_strategy=self.rule_strategy,
                scale_multiplier=self.scale_multiplier,
                random_state=self.random_state + client_id,
                supervised=supervised,
                max_iter=self.max_iter,
                tolerance=self.tolerance,
                belief_mass_strength=self.belief_mass_strength,
                beta_initialization=self.beta_initialization,
            )
            feature_idx = np.asarray(features, dtype=int)
            client.fit(X[:, feature_idx], local_y, label_mask=local_mask, classes=self.classes_)
            self.clients_[client_id] = client
        return self

    def _client_prob_stack(
        self,
        X: np.ndarray,
        party_features: Mapping[int, Sequence[int]],
    ) -> np.ndarray:
        if self.classes_ is None:
            raise ValueError("Model has not been fitted")
        probs = []
        for client_id in sorted(self.clients_):
            features = np.asarray(party_features[client_id], dtype=int)
            probs.append(self.clients_[client_id].predict_proba(np.asarray(X)[:, features]))
        return np.stack(probs, axis=1)

    def predict_proba(
        self,
        X: np.ndarray,
        party_features: Mapping[int, Sequence[int]],
    ) -> np.ndarray:
        client_probs = self._client_prob_stack(X, party_features)
        fusion_lambda = self.scer_lambda if self.enable_scer else 0.0
        if self.use_he:
            fused, stats = _he_scer_moment_fusion(
                client_probs,
                scer_lambda=fusion_lambda,
                key_bits=self.he_key_bits,
                scale=self.he_scale,
                seed=self.random_state,
                source_weights=self.source_weights,
            )
            plaintext = _apply_scer_moment_fusion(
                client_probs,
                scer_lambda=fusion_lambda,
                source_weights=self.source_weights,
            )
            stats["he_max_abs_score_diff"] = float(np.max(np.abs(fused - plaintext)))
            stats["plaintext_float_upload_bytes"] = float(client_probs.size * 8 * 2)
            stats["ciphertext_expansion"] = stats["he_total_upload_bytes"] / max(
                stats["plaintext_float_upload_bytes"], 1.0
            )
            self.last_he_stats_ = stats
            return fused
        self.last_he_stats_ = {}
        return _apply_scer_moment_fusion(
            client_probs,
            scer_lambda=fusion_lambda,
            source_weights=self.source_weights,
        )

    def predict(
        self,
        X: np.ndarray,
        party_features: Mapping[int, Sequence[int]],
    ) -> np.ndarray:
        if self.classes_ is None or self.minority_label_ is None:
            raise ValueError("Model has not been fitted")
        beliefs = self.predict_proba(X, party_features)
        pred_idx = np.argmax(beliefs, axis=1)
        if self.decision_rule == "threshold" and len(self.classes_) == 2:
            normalized = _normalize_rows(beliefs)
            minority_idx = int(np.flatnonzero(self.classes_ == self.minority_label_)[0])
            majority_idx = 1 - minority_idx
            pred_idx = np.where(
                normalized[:, minority_idx] >= self.minority_threshold_,
                minority_idx,
                majority_idx,
            )
        return self.classes_[pred_idx]

    def diagnostics(self) -> Dict[str, float]:
        losses = [client.training_loss_ for client in self.clients_.values()]
        initial = [client.initial_loss_ for client in self.clients_.values()]
        iterations = [client.optimization_iterations_ for client in self.clients_.values()]
        converged = [client.optimizer_converged_ for client in self.clients_.values()]
        limited = [client.hit_iteration_limit_ for client in self.clients_.values()]
        gradient_norms = [client.final_gradient_norm_ for client in self.clients_.values()]
        relative_reductions = [
            (client.initial_loss_ - client.training_loss_) / max(abs(client.initial_loss_), EPS)
            for client in self.clients_.values()
            if np.isfinite(client.initial_loss_) and np.isfinite(client.training_loss_)
        ]
        return {
            "rule_count": float(sum(client.rule_count() for client in self.clients_.values())),
            "client_count": float(len(self.clients_)),
            "minority_threshold": float(self.minority_threshold_),
            "decision_rule": self.decision_rule,
            "rule_strategy": "prototype",
            "scale_multiplier": float(self.scale_multiplier),
            "belief_mass_strength": float(self.belief_mass_strength),
            "beta_initialization": self.beta_initialization,
            "mean_initial_loss": float(np.nanmean(initial)) if initial else float("nan"),
            "mean_training_loss": float(np.nanmean(losses)) if losses else float("nan"),
            "mean_optimization_iterations": float(np.mean(iterations)) if iterations else 0.0,
            "optimizer_convergence_rate": float(np.mean(converged)) if converged else 0.0,
            "iteration_limit_rate": float(np.mean(limited)) if limited else 0.0,
            "mean_final_gradient_norm": (
                float(np.nanmean(gradient_norms)) if gradient_norms else float("nan")
            ),
            "mean_relative_loss_reduction": (
                float(np.mean(relative_reductions)) if relative_reductions else float("nan")
            ),
            **self.last_he_stats_,
        }
