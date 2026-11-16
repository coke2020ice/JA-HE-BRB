# JA-HE-BRB

Code companion for **Interpretable and Protected Joint-Antecedent Belief-Rule Learning for Distributed-Label Vertical Federated Learning**. The authors report that the paper has been published; final bibliographic details and DOI await confirmation.

## Run

Use Python 3.10 or newer.

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
PYTHONPATH=src python examples/run_reference.py
```

## Scope and evidence

This release contains the particle joint-antecedent BRB core and a fixed-seed synthetic reference example, tested in the local environment recorded in VALIDATION.json. A clean installation of pinned dependencies has not been independently verified. This is not a verified full-paper reproduction package.

The example executes plaintext single-process aggregation, not an actual multi-party encrypted deployment. Paillier helper source is included, but end-to-end cryptographic execution is not validated by this example. Synthetic accuracy is not a paper benchmark.

paper_results/tables and paper_results/figures preserve eight tables and seven figures from the final 2026-10-14 submission package, not independently regenerated results. They have not been compared with the final publisher version. Read the protocol disclosure: CDC tuning is exploratory after outer-fold inspection, and run counts and budgets differ.

SOURCE_MANIFEST.json identifies original working-tree core source. PUBLIC_MANIFEST.json identifies this release. Raw datasets, caches, per-record predictions, logs, reviewer correspondence, and previous Git history are excluded. Historical code and experiments remain in private repositories.

## License

No additional open-source license grant is made in this initial source release. Author and third-party licensing review remains pending. Public access alone does not grant an open-source license.
