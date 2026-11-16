"""Synthetic held-out smoke example, not a paper benchmark."""
import json
import numpy as np
from dlhebrb.particle_joint_brb import ParticleJointAntecedentBRB, ParticleJointBRBConfig, aggregate_distributed_labels
rng = np.random.default_rng(7)
x = rng.uniform(0, 1, (100, 4))
y = (x[:, 0] + x[:, 2] > 1).astype(int)
idx = rng.permutation(len(y))
train, test = idx[:80], idx[80:]
parts = {0: x[train, :2], 1: x[train, 2:]}
test_parts = {0: x[test, :2], 1: x[test, 2:]}
beliefs = {0: np.eye(2)[y[train]], 1: np.eye(2)[y[train]]}
masks = {0: np.arange(80) % 2 == 0, 1: np.arange(80) % 2 == 1}
targets, observed, counts = aggregate_distributed_labels(beliefs, masks)
config = ParticleJointBRBConfig(n_patterns=6, n_particles=4, n_iterations=5, optimizer_max_iter=15, random_state=11)
model = ParticleJointAntecedentBRB(config).fit(parts, targets, observed, n_groups=3)
p = model.predict_proba(test_parts)
assert p.shape == (20, 2) and np.isfinite(p).all()
assert ((p >= 0) & (p <= 1)).all()
assert counts.sum() == 80
print(json.dumps({"kind": "synthetic held-out smoke check", "seed": 7, "train": 80, "test": 20, "accuracy": float(np.mean(p.argmax(1) == y[test])), "encrypted_execution": False}))
