"""Particle-based joint-antecedent BRB used by the revised JA-HE-BRB paper.

The implementation keeps every antecedent block party-local.  Only additive
local triangular mismatches and label sufficient statistics are combined.
Cryptographic execution is audited separately; the sums here are the exact
plaintext counterparts of the Paillier operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import minimize

from .brb import ClientBRB, EPS, _activation_weights, er_aggregate


@dataclass(frozen=True)
class ParticleJointBRBConfig:
    n_patterns: int = 12
    n_particles: int = 8
    n_iterations: int = 30
    tau: float = 1.0
    inertia: float = 0.55
    cognitive: float = 1.35
    social: float = 1.35
    class_balance_power: float = 0.5
    optimizer_max_iter: int = 80
    optimizer_tolerance: float = 1e-6
    belief_mass_strength: float = 0.05
    random_state: int = 13
    antecedent_sample_limit: int | None = None
    initialization: str = "aligned_samples"
    kmeans_max_iter: int = 50
    kmeans_n_init: int = 3
    kmeans_tolerance: float = 1e-5
    initialization_jitter: float = 0.025
    balance_strength: float = 0.0
    separation_strength: float = 0.0
    assignment_temperature: float = 0.1
    separation_margin: float = 0.60
    match_mapping: str = "linear_complement"
    supervised_refinement_strength: float = 0.0
    supervised_refinement_iterations: int = 10
    supervised_refinement_objective: str = "conditional_entropy"
    supervised_refinement_rounds: int = 1
    supervised_movement_rate: float = 1.0
    supervised_margin_strength: float = 0.0
    supervised_margin: float = 0.10
    consequent_loss: str = "squared_error"
    activation_power: float = 1.0
    candidate_selection_redundancy: float = 0.20
    adaptive_triangular_width: bool = False
    client_balanced_mismatch: bool = False


def aggregate_distributed_labels(
    beliefs: Mapping[int, np.ndarray], masks: Mapping[int, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return mean belief, observed mask, and contributing-party count."""
    party_ids = sorted(beliefs)
    if party_ids != sorted(masks):
        raise ValueError("Belief and mask parties do not match")
    arrays = [np.asarray(beliefs[k], dtype=float) for k in party_ids]
    flags = [np.asarray(masks[k], dtype=bool) for k in party_ids]
    if not arrays or len({x.shape for x in arrays}) != 1:
        raise ValueError("Aligned label-belief arrays must share one shape")
    counts = np.sum(np.stack(flags, axis=0), axis=0).astype(float)
    total = np.zeros_like(arrays[0], dtype=float)
    for value, flag in zip(arrays, flags):
        total += value * flag[:, None]
    observed = counts > 0
    targets = np.zeros_like(total)
    targets[observed] = total[observed] / counts[observed, None]
    return targets, observed, counts


class ParticleJointAntecedentBRB:
    """Label-independent particle antecedents with grouped BRB inference."""

    def __init__(
        self, config: ParticleJointBRBConfig,
        feature_weights: Mapping[int, np.ndarray] | None = None,
    ) -> None:
        self.config = config
        if self.config.activation_power <= 0:
            raise ValueError("activation_power must be positive")
        if self.config.supervised_refinement_rounds < 1:
            raise ValueError("supervised_refinement_rounds must be positive")
        if not 0 < self.config.supervised_movement_rate <= 1:
            raise ValueError("supervised_movement_rate must lie in (0, 1]")
        if self.config.supervised_margin_strength < 0 or self.config.supervised_margin < 0:
            raise ValueError("Supervised margin settings must be nonnegative")
        self.feature_weights = None if feature_weights is None else {
            int(k): np.asarray(v, dtype=float) for k, v in feature_weights.items()
        }

    def _weights(self, k: int, width: int) -> np.ndarray:
        if self.feature_weights is None:
            return np.ones(width, dtype=float)
        value = self.feature_weights[k]
        if value.shape != (width,) or np.any(value <= 0):
            raise ValueError("Feature weights must be positive and match local width")
        return value

    @staticmethod
    def _validate_parts(parts: Mapping[int, np.ndarray]) -> list[int]:
        party_ids = sorted(parts)
        if not party_ids:
            raise ValueError("At least one feature party is required")
        n = {np.asarray(parts[k]).shape[0] for k in party_ids}
        if len(n) != 1 or any(np.asarray(parts[k]).ndim != 2 for k in party_ids):
            raise ValueError("Feature blocks must be aligned two-dimensional arrays")
        return party_ids

    def _initial_particles(
        self, parts: Mapping[int, np.ndarray], rng: np.random.Generator
    ) -> tuple[list[dict[int, np.ndarray]], dict[int, tuple[np.ndarray, np.ndarray]]]:
        particles: list[dict[int, np.ndarray]] = []
        bounds: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        q = (np.arange(self.config.n_patterns, dtype=float) + 0.5) / self.config.n_patterns
        bases: dict[int, np.ndarray] = {}
        for k in self.party_ids_:
            x = np.asarray(parts[k], dtype=float)
            lo, hi = np.min(x, axis=0), np.max(x, axis=0)
            bounds[k] = (lo, hi)
            bases[k] = np.quantile(x, q, axis=0)
        kmeans_centers = None
        if self.config.initialization == "distributed_kmeans":
            kmeans_centers = self._distributed_kmeans_centers(parts, rng)
        for p in range(self.config.n_particles):
            candidate: dict[int, np.ndarray] = {}
            aligned_rows = rng.choice(
                len(parts[self.party_ids_[0]]), size=self.config.n_patterns,
                replace=len(parts[self.party_ids_[0]]) < self.config.n_patterns,
            )
            for k in self.party_ids_:
                lo, hi = bounds[k]
                span = np.maximum(hi - lo, EPS)
                jitter = 0.0 if p == 0 else float(self.config.initialization_jitter)
                noise = rng.normal(0.0, jitter, size=bases[k].shape) * span
                if self.config.initialization == "aligned_samples":
                    arranged = np.asarray(parts[k], dtype=float)[aligned_rows]
                elif self.config.initialization == "quantile_permutations":
                    arranged = np.empty_like(bases[k])
                    for m in range(arranged.shape[1]):
                        shift = (p * (m + 1) + 3 * k + m) % self.config.n_patterns
                        arranged[:, m] = bases[k][np.roll(np.arange(self.config.n_patterns), shift), m]
                elif self.config.initialization == "distributed_kmeans":
                    arranged = np.asarray(kmeans_centers[k], dtype=float)
                else:
                    raise ValueError("Unknown particle initialization strategy")
                candidate[k] = np.clip(arranged + noise, lo, hi)
            particles.append(candidate)
        return particles, bounds

    def _distributed_squared_distance(
        self, parts: Mapping[int, np.ndarray], centers: Mapping[int, np.ndarray]
    ) -> np.ndarray:
        """Add party-local squared distances without assembling full records."""
        n = len(np.asarray(parts[self.party_ids_[0]]))
        n_centers = len(np.asarray(centers[self.party_ids_[0]]))
        result = np.zeros((n, n_centers), dtype=float)
        for k in self.party_ids_:
            x = np.asarray(parts[k], dtype=float)
            weights = self._weights(k, x.shape[1])
            difference = x[:, None, :] - np.asarray(centers[k])[None, :, :]
            result += np.sum(difference * difference * weights[None, None, :], axis=2)
        return result

    def _kmeans_plus_plus_centers(
        self, parts: Mapping[int, np.ndarray], rng: np.random.Generator
    ) -> dict[int, np.ndarray]:
        n = len(np.asarray(parts[self.party_ids_[0]]))
        if self.config.n_patterns > n:
            raise ValueError("n_patterns cannot exceed the antecedent sample count")
        selected = [int(rng.integers(n))]
        while len(selected) < self.config.n_patterns:
            centers = {
                k: np.asarray(parts[k], dtype=float)[selected]
                for k in self.party_ids_
            }
            distance = self._distributed_squared_distance(parts, centers)
            closest = np.min(distance, axis=1)
            closest[np.asarray(selected, dtype=int)] = 0.0
            total = float(np.sum(closest))
            available = np.setdiff1d(np.arange(n), np.asarray(selected), assume_unique=False)
            if total <= EPS:
                selected.append(int(available[0]))
            else:
                selected.append(int(rng.choice(n, p=closest / total)))
        return {
            k: np.asarray(parts[k], dtype=float)[selected].copy()
            for k in self.party_ids_
        }

    def _distributed_kmeans_centers(
        self, parts: Mapping[int, np.ndarray], rng: np.random.Generator
    ) -> dict[int, np.ndarray]:
        """Run label-independent K-means using only additive local distances."""
        best_centers = None
        best_inertia = float("inf")
        total_iterations = 0
        for _ in range(max(1, int(self.config.kmeans_n_init))):
            centers = self._kmeans_plus_plus_centers(parts, rng)
            for _ in range(max(1, int(self.config.kmeans_max_iter))):
                distance = self._distributed_squared_distance(parts, centers)
                assignments = np.argmin(distance, axis=1)
                closest = distance[np.arange(len(assignments)), assignments]
                updated: dict[int, np.ndarray] = {}
                replacement_order = iter(np.argsort(closest)[::-1].tolist())
                replacements: dict[int, int] = {}
                used: set[int] = set()
                for r in range(self.config.n_patterns):
                    if not np.any(assignments == r):
                        row = next(x for x in replacement_order if int(x) not in used)
                        used.add(int(row))
                        replacements[r] = int(row)
                shift = 0.0
                for k in self.party_ids_:
                    x = np.asarray(parts[k], dtype=float)
                    current = np.asarray(centers[k], dtype=float)
                    value = np.empty_like(current)
                    for r in range(self.config.n_patterns):
                        member = assignments == r
                        value[r] = x[replacements[r]] if r in replacements else np.mean(x[member], axis=0)
                    shift += float(np.sum((value - current) ** 2))
                    updated[k] = value
                centers = updated
                total_iterations += 1
                if np.sqrt(shift) <= float(self.config.kmeans_tolerance):
                    break
            distance = self._distributed_squared_distance(parts, centers)
            inertia = float(np.sum(np.min(distance, axis=1)))
            if inertia < best_inertia:
                best_inertia = inertia
                best_centers = {k: value.copy() for k, value in centers.items()}
        if best_centers is None:
            raise RuntimeError("Distributed K-means initialization failed")
        self.kmeans_iterations_ = int(total_iterations)
        self.kmeans_inertia_ = float(best_inertia)
        return best_centers

    def _joint_mismatch(
        self, parts: Mapping[int, np.ndarray], centers: Mapping[int, np.ndarray],
        scales: Mapping[int, np.ndarray] | None = None,
    ) -> np.ndarray:
        n = len(np.asarray(parts[self.party_ids_[0]]))
        n_patterns = len(np.asarray(centers[self.party_ids_[0]]))
        result = np.zeros((n, n_patterns), dtype=float)
        total_width = sum(
            float(np.sum(self._weights(k, np.asarray(parts[k]).shape[1])))
            for k in self.party_ids_
        )
        for k in self.party_ids_:
            x = np.asarray(parts[k], dtype=float)
            denominator = self.config.tau
            if scales is not None:
                denominator = self.config.tau * np.maximum(
                    np.asarray(scales[k], dtype=float)[None, :, :], EPS
                )
            local = np.minimum(
                1.0,
                np.abs(x[:, None, :] - centers[k][None, :, :]) / denominator,
            )
            weights = self._weights(k, x.shape[1])
            local_sum = np.sum(local * weights[None, None, :], axis=2)
            if self.config.client_balanced_mismatch:
                result += local_sum / (
                    max(float(np.sum(weights)), EPS) * len(self.party_ids_)
                )
            else:
                result += local_sum / total_width
        return result

    def _adaptive_scales(
        self, parts: Mapping[int, np.ndarray], centers: Mapping[int, np.ndarray]
    ) -> dict[int, np.ndarray]:
        """Estimate client-local, rule-specific robust triangular widths."""
        fixed_mismatch = self._joint_mismatch(parts, centers)
        assignments = np.argmin(fixed_mismatch, axis=1)
        n_patterns = fixed_mismatch.shape[1]
        scales: dict[int, np.ndarray] = {}
        for k in self.party_ids_:
            x = np.asarray(parts[k], dtype=float)
            center = np.asarray(centers[k], dtype=float)
            global_width = np.quantile(
                np.abs(x - np.median(x, axis=0, keepdims=True)), .75, axis=0
            )
            fallback = np.maximum(global_width, 0.10)
            value = np.empty_like(center)
            for r in range(n_patterns):
                member = x[assignments == r]
                if len(member) >= 3:
                    local_width = np.quantile(
                        np.abs(member - center[r][None, :]), .75, axis=0
                    )
                    value[r] = np.maximum(local_width, 0.25 * fallback)
                else:
                    value[r] = fallback
            scales[k] = np.maximum(value, 0.05)
        return scales

    def _select_antecedent_candidates(
        self, parts: Mapping[int, np.ndarray], centers: Mapping[int, np.ndarray],
        targets: np.ndarray, observed: np.ndarray, selection_count: int,
    ) -> dict[int, np.ndarray]:
        """Select class-discriminative, nonredundant candidates without moving them."""
        candidate_count = len(np.asarray(centers[self.party_ids_[0]]))
        keep = int(selection_count)
        if keep < 1 or keep > candidate_count:
            raise ValueError("Antecedent selection count must lie within candidate count")
        if keep == candidate_count:
            self.candidate_selection_ = {
                "candidate_count": candidate_count,
                "selected_count": keep,
                "selected_indices": list(range(keep)),
                "class_counts": [],
            }
            return {k: np.asarray(v).copy() for k, v in centers.items()}

        active_targets = np.asarray(targets, dtype=float)[observed]
        match = self._pattern_match(self._joint_mismatch(parts, centers))[observed]
        responsibility = match / np.maximum(np.sum(match, axis=1, keepdims=True), EPS)
        class_support = responsibility.T @ active_targets
        total_support = np.sum(class_support, axis=1)
        conditional = class_support / np.maximum(total_support[:, None], EPS)
        class_prior = np.sum(active_targets, axis=0)
        class_prior /= np.maximum(np.sum(class_prior), EPS)
        lift = conditional / np.maximum(class_prior[None, :], EPS)
        purity = np.max(conditional, axis=1)
        coverage = total_support / np.maximum(np.sum(total_support), EPS)
        class_score = lift * np.sqrt(np.maximum(coverage[:, None], 0.0))

        norms = np.linalg.norm(match, axis=0)
        similarity = (match.T @ match) / np.maximum(norms[:, None] * norms[None, :], EPS)
        np.fill_diagonal(similarity, 0.0)
        n_classes = active_targets.shape[1]
        base_quota, remainder = divmod(keep, n_classes)
        quotas = [base_quota + int(c < remainder) for c in range(n_classes)]
        selected: list[int] = []
        selected_classes: list[int] = []

        def choose(class_idx: int, amount: int) -> None:
            pool = [r for r in range(candidate_count) if r not in selected]
            available = list(pool)
            for _ in range(min(amount, len(available))):
                best = max(
                    available,
                    key=lambda r: (
                        class_score[r, class_idx]
                        - self.config.candidate_selection_redundancy
                        * (max(similarity[r, selected]) if selected else 0.0),
                        class_score[r, class_idx], -r,
                    ),
                )
                selected.append(int(best))
                selected_classes.append(int(class_idx))
                available.remove(best)

        for class_idx, quota in enumerate(quotas):
            choose(class_idx, quota)
        if len(selected) < keep:
            choose(int(np.argmin(class_prior)), keep - len(selected))

        selected = selected[:keep]
        self.candidate_selection_ = {
            "candidate_count": candidate_count,
            "selected_count": keep,
            "selected_indices": selected,
            "class_counts": np.bincount(selected_classes, minlength=n_classes).tolist(),
            "class_prior": class_prior.tolist(),
            "mean_selected_lift": float(np.mean([
                lift[r, c] for r, c in zip(selected, selected_classes)
            ])),
            "mean_selected_purity": float(np.mean(purity[selected])),
            "mean_candidate_purity": float(np.mean(purity)),
            "mean_selected_coverage": float(np.mean(coverage[selected])),
        }
        return {k: np.asarray(v)[selected].copy() for k, v in centers.items()}

    def _antecedent_rows(self, n: int, rng: np.random.Generator) -> np.ndarray:
        limit = self.config.antecedent_sample_limit
        if limit is None or limit >= n:
            return np.arange(n)
        return np.sort(rng.choice(n, size=int(limit), replace=False))

    def _antecedent_objective(
        self, parts: Mapping[int, np.ndarray], centers: Mapping[int, np.ndarray]
    ) -> tuple[float, float, float, float]:
        mismatch = self._joint_mismatch(parts, centers)
        coverage = float(np.mean(np.min(mismatch, axis=1)))
        logits = -mismatch / max(self.config.assignment_temperature, EPS)
        logits -= np.max(logits, axis=1, keepdims=True)
        assignment = np.exp(logits)
        assignment /= np.maximum(np.sum(assignment, axis=1, keepdims=True), EPS)
        occupancy = np.mean(assignment, axis=0)
        # Scaled squared distance from uniform occupancy.  It is zero only when
        # all modes receive equal soft responsibility.
        balance = float(self.config.n_patterns * np.sum(
            (occupancy - 1.0 / self.config.n_patterns) ** 2
        ))
        if self.config.n_patterns < 2:
            separation = 0.0
        else:
            pairwise = np.zeros((self.config.n_patterns, self.config.n_patterns))
            total_width = sum(
                float(np.sum(self._weights(k, np.asarray(parts[k]).shape[1])))
                for k in self.party_ids_
            )
            for k in self.party_ids_:
                local = np.minimum(
                    1.0,
                    np.abs(centers[k][:, None, :] - centers[k][None, :, :])
                    / self.config.tau,
                )
                weights = self._weights(k, centers[k].shape[1])
                local_sum = np.sum(local * weights[None, None, :], axis=2)
                if self.config.client_balanced_mismatch:
                    pairwise += local_sum / (
                        max(float(np.sum(weights)), EPS) * len(self.party_ids_)
                    )
                else:
                    pairwise += local_sum / total_width
            upper = pairwise[np.triu_indices(self.config.n_patterns, 1)]
            separation = float(np.mean(
                np.maximum(0.0, self.config.separation_margin - upper) ** 2
            ))
        total = (
            coverage
            + self.config.balance_strength * balance
            + self.config.separation_strength * separation
        )
        return float(total), coverage, balance, separation

    def _search(self, parts: Mapping[int, np.ndarray]) -> dict[int, np.ndarray]:
        rng = np.random.default_rng(self.config.random_state)
        rows = self._antecedent_rows(len(parts[self.party_ids_[0]]), rng)
        sample = {k: np.asarray(parts[k])[rows] for k in self.party_ids_}
        particles, bounds = self._initial_particles(sample, rng)
        velocities = [{k: np.zeros_like(v[k]) for k in self.party_ids_} for v in particles]
        personal = [{k: v[k].copy() for k in self.party_ids_} for v in particles]
        losses = np.asarray([self._antecedent_objective(sample, v)[0] for v in particles])
        personal_losses = losses.copy()
        best_idx = int(np.argmin(losses))
        global_best = {k: particles[best_idx][k].copy() for k in self.party_ids_}
        self.antecedent_loss_curve_ = [float(losses[best_idx])]
        for _ in range(self.config.n_iterations):
            for p, candidate in enumerate(particles):
                for k in self.party_ids_:
                    r1 = float(rng.random())
                    r2 = float(rng.random())
                    velocities[p][k] = (
                        self.config.inertia * velocities[p][k]
                        + self.config.cognitive * r1 * (personal[p][k] - candidate[k])
                        + self.config.social * r2 * (global_best[k] - candidate[k])
                    )
                    lo, hi = bounds[k]
                    candidate[k] = np.clip(candidate[k] + velocities[p][k], lo, hi)
                loss = self._antecedent_objective(sample, candidate)[0]
                if loss < personal_losses[p]:
                    personal_losses[p] = loss
                    personal[p] = {k: candidate[k].copy() for k in self.party_ids_}
            best_idx = int(np.argmin(personal_losses))
            global_best = {k: personal[best_idx][k].copy() for k in self.party_ids_}
            self.antecedent_loss_curve_.append(float(personal_losses[best_idx]))
        self.antecedent_sample_count_ = int(len(rows))
        total, coverage, balance, separation = self._antecedent_objective(sample, global_best)
        self.antecedent_objective_components_ = {
            "total": total, "coverage": coverage,
            "balance_unweighted": balance,
            "separation_unweighted": separation,
        }
        return global_best

    def _conditional_entropy(
        self, parts: Mapping[int, np.ndarray], centers: Mapping[int, np.ndarray],
        targets: np.ndarray, observed: np.ndarray,
    ) -> float:
        """Normalized H(Y|R) from soft pattern assignments and aggregate targets."""
        mismatch = self._joint_mismatch(parts, centers)[observed]
        active_targets = np.asarray(targets, dtype=float)[observed]
        logits = -mismatch / max(self.config.assignment_temperature, EPS)
        logits -= np.max(logits, axis=1, keepdims=True)
        assignment = np.exp(logits)
        assignment /= np.maximum(np.sum(assignment, axis=1, keepdims=True), EPS)
        joint_mass = assignment.T @ active_targets
        pattern_mass = np.sum(joint_mass, axis=1)
        conditional = joint_mass / np.maximum(pattern_mass[:, None], EPS)
        entropy = -np.sum(conditional * np.log(np.maximum(conditional, EPS)), axis=1)
        normalizer = max(np.log(active_targets.shape[1]), EPS)
        return float(np.sum(pattern_mass * entropy) / np.maximum(np.sum(pattern_mass), EPS) / normalizer)

    def _supervised_refine(
        self, parts: Mapping[int, np.ndarray], targets: np.ndarray,
        observed: np.ndarray, initial: Mapping[int, np.ndarray],
    ) -> dict[int, np.ndarray]:
        """Locally refine label-independent antecedents using aggregate supervision."""
        strength = float(self.config.supervised_refinement_strength)
        if strength <= 0 or self.config.supervised_refinement_iterations <= 0:
            self.supervised_refinement_curve_ = []
            self.supervised_refinement_components_ = {}
            return {k: np.asarray(v).copy() for k, v in initial.items()}
        rng = np.random.default_rng(self.config.random_state + 7919)
        rows = self._antecedent_rows(len(parts[self.party_ids_[0]]), rng)
        sample = {k: np.asarray(parts[k])[rows] for k in self.party_ids_}
        sample_targets = np.asarray(targets)[rows]
        sample_observed = np.asarray(observed)[rows]
        if not np.any(sample_observed):
            raise ValueError("Supervised refinement requires observed targets")
        bounds = {
            k: (np.min(np.asarray(parts[k]), axis=0), np.max(np.asarray(parts[k]), axis=0))
            for k in self.party_ids_
        }

        refinement_objective = self.config.supervised_refinement_objective
        if refinement_objective not in {
            "conditional_entropy", "belief_output", "logistic_activation"
        }:
            raise ValueError("Unknown supervised antecedent-refinement objective")

        fixed_beta = fixed_theta = None

        def supervised_loss(candidate):
            if refinement_objective == "conditional_entropy":
                return self._conditional_entropy(
                    sample, candidate, sample_targets, sample_observed
                )
            if refinement_objective == "logistic_activation":
                match = self._effective_match(
                    self._pattern_match(self._joint_mismatch(sample, candidate))
                )[sample_observed]
                active_targets = np.asarray(sample_targets, dtype=float)[sample_observed]
                active_targets /= np.maximum(
                    np.sum(active_targets, axis=1, keepdims=True), EPS
                )
                design = np.column_stack([match, np.ones(len(match), dtype=float)])
                n_classes = active_targets.shape[1]

                def logistic_objective(vector):
                    coefficients = vector.reshape(design.shape[1], n_classes)
                    logits = design @ coefficients
                    logits -= np.max(logits, axis=1, keepdims=True)
                    probabilities = np.exp(np.clip(logits, -40.0, 40.0))
                    probabilities /= np.maximum(
                        np.sum(probabilities, axis=1, keepdims=True), EPS
                    )
                    loss = -np.mean(
                        np.sum(active_targets * np.log(np.maximum(probabilities, EPS)), axis=1)
                    )
                    # The intercept is not regularized.
                    loss += 5e-4 * float(np.sum(coefficients[:-1] ** 2))
                    gradient = design.T @ (probabilities - active_targets) / len(design)
                    gradient[:-1] += 1e-3 * coefficients[:-1]
                    return float(loss), gradient.ravel()

                fitted = minimize(
                    logistic_objective,
                    np.zeros(design.shape[1] * n_classes, dtype=float),
                    method="L-BFGS-B",
                    jac=True,
                    options={"maxiter": 20, "ftol": 1e-7, "maxls": 20},
                )
                return float(fitted.fun)
            match = self._effective_match(
                self._pattern_match(self._joint_mismatch(sample, candidate))
            )[sample_observed]
            prediction = er_aggregate(
                _activation_weights(match, fixed_theta), fixed_beta
            )
            active_targets = sample_targets[sample_observed]
            hard = np.argmax(active_targets, axis=1)
            counts = np.bincount(
                hard, minlength=active_targets.shape[1]
            ).astype(float)
            class_weights = (
                len(hard) / np.maximum(counts * active_targets.shape[1], EPS)
            ) ** self.config.class_balance_power
            errors = np.sum((active_targets - prediction) ** 2, axis=1)
            if self.config.supervised_margin_strength > 0:
                true_support = prediction[np.arange(len(hard)), hard]
                competing = prediction.copy()
                competing[np.arange(len(hard)), hard] = -np.inf
                margin = true_support - np.max(competing, axis=1)
                errors += self.config.supervised_margin_strength * np.maximum(
                    0.0, self.config.supervised_margin - margin
                ) ** 2
            return float(np.average(errors, weights=class_weights[hard]))

        def objective(candidate):
            coverage = self._antecedent_objective(sample, candidate)[1]
            supervised = supervised_loss(candidate)
            return float(coverage + strength * supervised), coverage, supervised

        current = {k: np.asarray(initial[k]).copy() for k in self.party_ids_}
        self.supervised_refinement_curve_ = []
        rounds = (
            self.config.supervised_refinement_rounds
            if refinement_objective == "belief_output" else 1
        )
        for _round in range(rounds):
            if refinement_objective == "belief_output":
                current_match = self._effective_match(
                    self._pattern_match(self._joint_mismatch(sample, current))
                )
                fixed_beta, fixed_theta, _ = self._fit_parameters(
                    current_match, sample_targets, sample_observed
                )
            particles = [{k: current[k].copy() for k in self.party_ids_}]
            for _ in range(1, self.config.n_particles):
                candidate = {}
                for k in self.party_ids_:
                    lo, hi = bounds[k]
                    span = np.maximum(hi - lo, EPS)
                    candidate[k] = np.clip(
                        current[k] + rng.normal(0., .015, current[k].shape) * span,
                        lo, hi,
                    )
                particles.append(candidate)
            velocities = [{k: np.zeros_like(v[k]) for k in self.party_ids_} for v in particles]
            personal = [{k: v[k].copy() for k in self.party_ids_} for v in particles]
            personal_losses = np.asarray([objective(v)[0] for v in particles])
            best_idx = int(np.argmin(personal_losses))
            global_best = {k: personal[best_idx][k].copy() for k in self.party_ids_}
            self.supervised_refinement_curve_.append(float(personal_losses[best_idx]))
            for _ in range(self.config.supervised_refinement_iterations):
                for p, candidate in enumerate(particles):
                    for k in self.party_ids_:
                        velocities[p][k] = (
                            self.config.inertia * velocities[p][k]
                            + self.config.cognitive * float(rng.random()) * (personal[p][k] - candidate[k])
                            + self.config.social * float(rng.random()) * (global_best[k] - candidate[k])
                        )
                        lo, hi = bounds[k]
                        candidate[k] = np.clip(candidate[k] + velocities[p][k], lo, hi)
                    loss = objective(candidate)[0]
                    if loss < personal_losses[p]:
                        personal_losses[p] = loss
                        personal[p] = {k: candidate[k].copy() for k in self.party_ids_}
                best_idx = int(np.argmin(personal_losses))
                global_best = {k: personal[best_idx][k].copy() for k in self.party_ids_}
                self.supervised_refinement_curve_.append(float(personal_losses[best_idx]))
            rate = float(self.config.supervised_movement_rate)
            current = {
                k: (1.0 - rate) * current[k] + rate * global_best[k]
                for k in self.party_ids_
            }
        total, coverage, supervised = objective(current)
        self.supervised_refinement_components_ = {
            "total": total, "coverage": coverage,
            "supervised_loss": supervised, "strength": strength,
            "objective": refinement_objective,
            "rounds": rounds,
            "movement_rate": float(self.config.supervised_movement_rate),
            "margin_strength": float(self.config.supervised_margin_strength),
        }
        if refinement_objective == "conditional_entropy":
            self.supervised_refinement_components_["conditional_entropy"] = supervised
        elif refinement_objective == "logistic_activation":
            self.supervised_refinement_components_["logistic_cross_entropy"] = supervised
        else:
            self.supervised_refinement_components_["belief_output_loss"] = supervised
        return current

    def _fit_parameters(
        self, match: np.ndarray, targets: np.ndarray, observed: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, dict[str, float | int | bool]]:
        helper = ClientBRB(
            n_rules=match.shape[1], random_state=self.config.random_state,
            max_iter=self.config.optimizer_max_iter,
            tolerance=self.config.optimizer_tolerance,
            class_balance_power=self.config.class_balance_power,
            belief_mass_strength=self.config.belief_mass_strength,
            beta_initialization="local_label",
            objective_loss=self.config.consequent_loss,
        )
        active_match, active_targets = match[observed], targets[observed]
        responsibilities = active_match / np.maximum(active_match.sum(axis=1, keepdims=True), EPS)
        assigned = responsibilities.T @ active_targets
        assigned /= np.maximum(responsibilities.sum(axis=0)[:, None], EPS)
        unassigned = np.maximum(1.0 - assigned.sum(axis=1, keepdims=True), 0.0)
        complete = np.concatenate([assigned, unassigned], axis=1) + 1e-3
        complete /= complete.sum(axis=1, keepdims=True)
        supports = responsibilities.sum(axis=0)
        vector = helper._initial_parameters(match.shape[1], targets.shape[1], supports, complete)
        hard = np.argmax(active_targets, axis=1)
        counts = np.bincount(hard, minlength=targets.shape[1]).astype(float)
        class_weights = (len(hard) / np.maximum(counts * targets.shape[1], EPS)) ** self.config.class_balance_power
        weights = class_weights[hard]
        objective = lambda z: helper._objective_and_gradient(z, active_match, active_targets, weights)
        initial_loss = float(objective(vector)[0])
        result = minimize(objective, vector, method="L-BFGS-B", jac=True,
                          options={"maxiter": self.config.optimizer_max_iter,
                                   "ftol": self.config.optimizer_tolerance, "maxls": 30})
        beta, theta, _ = helper._decode_parameters(result.x, match.shape[1], targets.shape[1])
        return beta, theta, {"initial_loss": initial_loss, "training_loss": float(result.fun),
                             "iterations": int(result.nit), "converged": bool(result.success),
                             "consequent_loss": self.config.consequent_loss}

    @staticmethod
    def _consequent_distance(a: np.ndarray, b: np.ndarray) -> float:
        aa = np.append(a, max(0.0, 1.0 - float(a.sum())))
        bb = np.append(b, max(0.0, 1.0 - float(b.sum())))
        return float(np.linalg.norm(aa - bb))

    @staticmethod
    def _group_match(pattern_match: np.ndarray, groups: Sequence[Sequence[int]]) -> np.ndarray:
        return np.column_stack([np.max(pattern_match[:, list(g)], axis=1) for g in groups])

    def _pattern_match(self, mismatch: np.ndarray) -> np.ndarray:
        if self.config.match_mapping == "linear_complement":
            return np.maximum(1.0 - mismatch, EPS)
        if self.config.match_mapping == "exponential":
            return np.maximum(np.exp(-mismatch), EPS)
        raise ValueError("Unknown mismatch-to-match mapping")

    def _effective_match(self, match: np.ndarray) -> np.ndarray:
        """Apply a monotone contrast transform before BRB activation."""
        return np.maximum(match, EPS) ** float(self.config.activation_power)

    def fit(
        self, parts: Mapping[int, np.ndarray], targets: np.ndarray, observed: np.ndarray,
        *, n_groups: int | None = None,
        antecedent_selection_count: int | None = None,
    ) -> "ParticleJointAntecedentBRB":
        chosen = self.config.n_patterns if n_groups is None else int(n_groups)
        return self.fit_group_candidates(
            parts, targets, observed, [chosen],
            antecedent_selection_count=antecedent_selection_count,
        )

    def fit_group_candidates(
        self, parts: Mapping[int, np.ndarray], targets: np.ndarray,
        observed: np.ndarray, group_counts: Sequence[int],
        initial_centers: Mapping[int, np.ndarray] | None = None,
        antecedent_selection_count: int | None = None,
    ) -> "ParticleJointAntecedentBRB":
        self.party_ids_ = self._validate_parts(parts)
        targets = np.asarray(targets, dtype=float)
        observed = np.asarray(observed, dtype=bool)
        if targets.shape[0] != len(parts[self.party_ids_[0]]) or not np.any(observed):
            raise ValueError("Targets must align with features and contain observed supervision")
        if initial_centers is None:
            initial = self._search(parts)
        else:
            initial = {k: np.asarray(initial_centers[k]).copy() for k in self.party_ids_}
            self.antecedent_loss_curve_ = []
            self.antecedent_objective_components_ = {}
            self.antecedent_sample_count_ = 0
        candidates = self._supervised_refine(parts, targets, observed, initial)
        selection_count = (
            len(np.asarray(candidates[self.party_ids_[0]]))
            if antecedent_selection_count is None else int(antecedent_selection_count)
        )
        self.centers_ = self._select_antecedent_candidates(
            parts, candidates, targets, observed, selection_count
        )
        self.scales_ = (
            self._adaptive_scales(parts, self.centers_)
            if self.config.adaptive_triangular_width else None
        )
        mismatch = self._joint_mismatch(parts, self.centers_, self.scales_)
        pattern_match = self._pattern_match(mismatch)
        requested = sorted({int(g) for g in group_counts}, reverse=True)
        if not requested or min(requested) < 1 or max(requested) > selection_count:
            raise ValueError("Every group count must lie between one and n_patterns")
        self.groups_ = [[i] for i in range(selection_count)]
        grouped_match = self._effective_match(pattern_match)
        self.beta_, self.theta_, self.fit_statistics_ = self._fit_parameters(
            grouped_match, targets, observed
        )
        self.group_models_ = {}
        if len(self.groups_) in requested:
            self.group_models_[len(self.groups_)] = self._group_snapshot()
        # Consequent distances are recomputed after every merge because the
        # shared consequent is a refitted parameter, not an arithmetic average.
        while len(self.groups_) > min(requested):
            pair = min(
                ((i, j) for i in range(len(self.groups_)) for j in range(i + 1, len(self.groups_))),
                key=lambda ij: (
                    self._consequent_distance(self.beta_[ij[0]], self.beta_[ij[1]]),
                    min(self.groups_[ij[0]]), min(self.groups_[ij[1]]),
                ),
            )
            i, j = pair
            self.groups_[i] += self.groups_[j]
            del self.groups_[j]
            grouped_match = self._effective_match(
                self._group_match(pattern_match, self.groups_)
            )
            self.beta_, self.theta_, self.fit_statistics_ = self._fit_parameters(
                grouped_match, targets, observed
            )
            if len(self.groups_) in requested:
                self.group_models_[len(self.groups_)] = self._group_snapshot()
        assignments = np.argmin(mismatch, axis=1)
        self.pattern_occupancy_ = np.bincount(assignments, minlength=selection_count)
        self.n_classes_ = targets.shape[1]
        self.select_group_count(requested[-1])
        return self

    def _group_snapshot(self) -> dict[str, object]:
        return {
            "groups": [list(g) for g in self.groups_],
            "beta": self.beta_.copy(), "theta": self.theta_.copy(),
            "fit_statistics": dict(self.fit_statistics_),
        }

    def select_group_count(self, n_groups: int) -> "ParticleJointAntecedentBRB":
        snapshot = self.group_models_[int(n_groups)]
        self.groups_ = [list(g) for g in snapshot["groups"]]
        self.beta_ = np.asarray(snapshot["beta"]).copy()
        self.theta_ = np.asarray(snapshot["theta"]).copy()
        self.fit_statistics_ = dict(snapshot["fit_statistics"])
        return self

    def predict_proba(
        self, parts: Mapping[int, np.ndarray], n_groups: int | None = None,
        *, retained_rules: Sequence[int] | None = None,
        reliability_power: float = 0.0,
    ) -> np.ndarray:
        if sorted(parts) != self.party_ids_:
            raise ValueError("Prediction parties differ from fitted parties")
        mismatch = self._joint_mismatch(parts, self.centers_, self.scales_)
        if n_groups is None:
            groups, beta, theta = self.groups_, self.beta_, self.theta_
        else:
            snapshot = self.group_models_[int(n_groups)]
            groups = snapshot["groups"]
            beta, theta = np.asarray(snapshot["beta"]), np.asarray(snapshot["theta"])
        grouped = self._effective_match(
            self._group_match(self._pattern_match(mismatch), groups)
        )
        if retained_rules is not None:
            keep = np.asarray(retained_rules, dtype=int)
            if keep.ndim != 1 or len(keep) == 0 or np.any(keep < 0) or np.any(keep >= len(groups)):
                raise ValueError("retained_rules must identify at least one valid grouped rule")
            grouped, beta, theta = grouped[:, keep], beta[keep], theta[keep]
        if reliability_power < 0:
            raise ValueError("reliability_power must be nonnegative")
        if reliability_power > 0:
            theta = theta * np.maximum(self._rule_reliability(beta), EPS) ** float(reliability_power)
        return er_aggregate(_activation_weights(grouped, theta), beta)

    @staticmethod
    def _rule_reliability(beta: np.ndarray) -> np.ndarray:
        """Return belief-mass-adjusted consequent certainty for each rule."""
        beta = np.clip(np.asarray(beta, dtype=float), 0.0, 1.0)
        mass = np.clip(np.sum(beta, axis=1), 0.0, 1.0)
        if beta.shape[1] <= 1:
            return mass
        normalized = beta / np.maximum(mass[:, None], EPS)
        entropy = -np.sum(normalized * np.log(np.maximum(normalized, EPS)), axis=1)
        certainty = 1.0 - entropy / np.log(beta.shape[1])
        return np.clip(mass * certainty, 0.0, 1.0)

    def select_rules_for_pruning(
        self,
        parts: Mapping[int, np.ndarray],
        targets: np.ndarray,
        observed: np.ndarray,
        retention_fraction: float,
        *,
        n_groups: int | None = None,
        reliability_power: float = 0.0,
    ) -> np.ndarray:
        """Select grouped rules using training-only ablation contribution.

        Validation may choose ``retention_fraction`` and ``reliability_power``, but
        the identities retained here depend only on the supplied training records.
        At least one rule dominated by every represented conclusion is protected.
        """
        if not 0 < retention_fraction <= 1:
            raise ValueError("retention_fraction must lie in (0, 1]")
        if n_groups is None:
            groups, beta, theta = self.groups_, self.beta_, self.theta_
        else:
            snapshot = self.group_models_[int(n_groups)]
            groups = snapshot["groups"]
            beta, theta = np.asarray(snapshot["beta"]), np.asarray(snapshot["theta"])
        n_rules = len(groups)
        n_keep = min(n_rules, max(1, int(np.ceil(retention_fraction * n_rules))))
        represented = np.unique(np.argmax(beta, axis=1))
        n_keep = min(n_rules, max(n_keep, len(represented)))
        if n_keep == n_rules:
            return np.arange(n_rules, dtype=int)

        mismatch = self._joint_mismatch(parts, self.centers_, self.scales_)
        grouped = self._effective_match(
            self._group_match(self._pattern_match(mismatch), groups)
        )
        adjusted_theta = np.asarray(theta, dtype=float).copy()
        if reliability_power > 0:
            adjusted_theta *= np.maximum(self._rule_reliability(beta), EPS) ** float(reliability_power)
        activation = _activation_weights(grouped, adjusted_theta)
        support = np.mean(activation, axis=0)
        flags = np.asarray(observed, dtype=bool)
        target = np.asarray(targets, dtype=float)[flags]

        def loss_for(indices: np.ndarray) -> float:
            prediction = er_aggregate(
                _activation_weights(grouped[flags][:, indices], adjusted_theta[indices]),
                beta[indices],
            )
            return float(np.mean(np.sum((prediction - target) ** 2, axis=1)))

        all_rules = np.arange(n_rules, dtype=int)
        base_loss = loss_for(all_rules)
        contribution = np.empty(n_rules, dtype=float)
        for rule_idx in range(n_rules):
            contribution[rule_idx] = loss_for(all_rules[all_rules != rule_idx]) - base_loss

        ranking = sorted(range(n_rules), key=lambda h: (contribution[h], support[h]), reverse=True)
        protected: set[int] = set()
        dominant = np.argmax(beta, axis=1)
        for class_idx in represented:
            members = np.flatnonzero(dominant == class_idx)
            protected.add(max(members, key=lambda h: (contribution[h], support[h])))
        selected = list(sorted(protected))
        selected.extend(h for h in ranking if h not in protected and len(selected) < n_keep)
        return np.asarray(sorted(selected[:n_keep]), dtype=int)

    def rule_diagnostics(
        self,
        parts: Mapping[int, np.ndarray],
        targets: np.ndarray | None = None,
        observed: np.ndarray | None = None,
        n_groups: int | None = None,
    ) -> dict[str, object]:
        """Summarize rule use and redundant activation without changing inference.

        Label-dependent quantities are reported only for observed samples.  They
        are intended for post-fit diagnosis, not for selecting hyperparameters.
        """
        if sorted(parts) != self.party_ids_:
            raise ValueError("Diagnostic parties differ from fitted parties")
        mismatch = self._joint_mismatch(parts, self.centers_, self.scales_)
        if n_groups is None:
            groups, beta, theta = self.groups_, self.beta_, self.theta_
        else:
            snapshot = self.group_models_[int(n_groups)]
            groups = snapshot["groups"]
            beta, theta = np.asarray(snapshot["beta"]), np.asarray(snapshot["theta"])
        grouped = self._effective_match(
            self._group_match(self._pattern_match(mismatch), groups)
        )
        activation = _activation_weights(grouped, theta)
        n_samples, n_rules = activation.shape
        top = np.argmax(activation, axis=1)
        usage = np.bincount(top, minlength=n_rules) / max(n_samples, 1)
        entropy = -np.sum(
            activation * np.log(np.maximum(activation, EPS)), axis=1
        )
        if n_rules > 1:
            normalized_entropy = entropy / np.log(n_rules)
            norms = np.linalg.norm(activation, axis=0)
            cosine = (activation.T @ activation) / np.maximum(
                norms[:, None] * norms[None, :], EPS
            )
            upper = cosine[np.triu_indices(n_rules, 1)]
            beta_distances = np.asarray([
                self._consequent_distance(beta[i], beta[j])
                for i in range(n_rules) for j in range(i + 1, n_rules)
            ])
        else:
            normalized_entropy = np.zeros(n_samples, dtype=float)
            upper = np.asarray([], dtype=float)
            beta_distances = np.asarray([], dtype=float)
        result: dict[str, object] = {
            "n_samples": int(n_samples),
            "n_rules": int(n_rules),
            "group_sizes": [len(g) for g in groups],
            "top_rule_usage": usage.tolist(),
            "unused_rule_fraction": float(np.mean(usage == 0.0)),
            "low_use_rule_fraction": float(np.mean(usage < 0.1 / n_rules)),
            "mean_normalized_activation_entropy": float(np.mean(normalized_entropy)),
            "mean_effective_rule_count": float(np.mean(np.exp(entropy))),
            "mean_pairwise_activation_cosine": (
                float(np.mean(upper)) if upper.size else 0.0
            ),
            "max_pairwise_activation_cosine": (
                float(np.max(upper)) if upper.size else 0.0
            ),
            "near_duplicate_activation_pair_fraction": (
                float(np.mean(upper >= 0.95)) if upper.size else 0.0
            ),
            "mean_pairwise_consequent_distance": (
                float(np.mean(beta_distances)) if beta_distances.size else 0.0
            ),
            "min_pairwise_consequent_distance": (
                float(np.min(beta_distances)) if beta_distances.size else 0.0
            ),
        }
        if targets is not None:
            target_array = np.asarray(targets, dtype=float)
            flags = (
                np.ones(n_samples, dtype=bool)
                if observed is None else np.asarray(observed, dtype=bool)
            )
            if target_array.shape[0] != n_samples or flags.shape != (n_samples,):
                raise ValueError("Diagnostic targets and masks must align with features")
            hard = np.argmax(target_array[flags], axis=1)
            class_usage = {}
            for class_idx in range(target_array.shape[1]):
                member = hard == class_idx
                counts = np.bincount(top[flags][member], minlength=n_rules)
                class_usage[str(class_idx)] = {
                    "n_samples": int(np.sum(member)),
                    "top_rule_usage": (
                        counts / max(int(np.sum(member)), 1)
                    ).tolist(),
                    "rules_reached": int(np.sum(counts > 0)),
                }
            result["class_conditional_usage"] = class_usage
        return result
