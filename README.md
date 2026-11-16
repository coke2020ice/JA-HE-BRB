# JA-HE-BRB

Code companion for **Interpretable and privacy-preserving joint-antecedent belief rule learning in vertical federated learning with distributed labels**.

**Ziqian Lin, Longhao Yang, Ying-Ming Wang**  
*Knowledge-Based Systems*, article 117277. Scheduled issue date: 10 January 2027.

[Paper on ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0950705126020034) · [DOI: 10.1016/j.knosys.2026.117277](https://doi.org/10.1016/j.knosys.2026.117277)

## Abstract

Vertical federated learning (VFL) enables institutions holding different attributes of aligned entities to train a joint model without sharing raw records. Existing VFL methods exchange model-dependent intermediate information and often assume a single label-holding party, offering limited support for transparent and traceable rule-based reasoning. Conventional belief rule bases (BRBs) generally require centrally available antecedent attributes and supervision. This paper proposes a joint-antecedent belief rule base protected by homomorphic encryption (JA-HE-BRB) for distributed-label VFL. It decomposes each complete joint antecedent into aligned client-local referential-value blocks and learns the locally retained antecedent referential values through label-independent distributed particle search with bounded triangular mismatches. Paillier encryption protects the aggregation of local mismatches and supervision contributions. Complete joint antecedents with compatible consequent belief distributions are synthesized into belief rules with alternative antecedents; the best-matching antecedent alternative represents each rule before the activated belief rules are aggregated using the evidential reasoning (ER) approach. On six public medical datasets, JA-HE-BRB shows competitive accuracy, macro-F1, balanced accuracy and G-Mean. It achieves the highest mean macro-F1 on five datasets, the highest balanced accuracy and G-Mean on SPECT, and the highest balanced accuracy on New-Thyroid. Robustness and communication-cost analyses assess stability and overhead, while encrypted tests confirm numerical consistency. In the New-Thyroid case, removing the strongest supporting rule reduces the normal-class belief from 0.798 by 0.126, demonstrating inspectable decision evidence.

## Citation

```bibtex
@article{lin2027jahebrb,
  title = {Interpretable and privacy-preserving joint-antecedent belief rule learning in vertical federated learning with distributed labels},
  author = {Lin, Ziqian and Yang, Longhao and Wang, Ying-Ming},
  journal = {Knowledge-Based Systems},
  year = {2027},
  pages = {117277},
  doi = {10.1016/j.knosys.2026.117277},
  url = {https://doi.org/10.1016/j.knosys.2026.117277}
}
```

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
