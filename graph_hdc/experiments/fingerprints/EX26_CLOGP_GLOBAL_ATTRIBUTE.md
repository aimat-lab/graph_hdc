# ex_26: Crippen's logP as a further global attribute of HDF

Runs: 2026-10-12 on KCIST. Report: 2026-10-13.

**Status:** done, not used in the paper. The opening reply of the response letter (NCOMMS-26-058162) states that
"Crippen's logP, for example, could easily be added as a further global attribute for datasets that are known to
benefit from it, as is the case for the prediction of aqueous solubility". On 2026-10-13 the user decided to keep that
sentence as a claim without citing this experiment.

## Question

Does adding Crippen's logP as a further global attribute of HDF improve the predictions? In particular, does it
improve aqueous solubility, the example given in the letter?

## Setup

- **New attribute.** RDKit's `MolLogP` (Crippen's logP) of every molecule, with option `GRAPH_CLOGP = True` in
  `predict_molecules__hdc.py`.
  - It is encoded with the same fractional power encoding as the graph size (`make_value_graph_encoder` in
    `graph_hdc/special/molecules.py`), using unit-modulus base vectors and seed offset 50.
  - Bandwidth: one seventh of the logP range of the dataset, taken between the 1st and 99th percentile, and at least
    1. This gives AqSolDB 1.75, FreeSolv 1.19, HOPV15 2.36 and 1.0 for all other datasets.
  - The rule was fixed before any cluster run and not tuned. The first version used the plain min–max range, which
    gave AqSolDB a bandwidth of 6.2 because of a few extreme molecules (logP −17 to 26).
  - The logP hypervector is summed with the hypervectors of the graph size and diameter:
    `normalize(normalize(readout) + normalize(h_size + h_diameter + h_logP))`.
- **Everything else** is the HDF arm of ex_14 round 4, i.e. the HDF results of SI Section S9.
  - Encoder: D = 2048, L = 2, continuous encoding, unit-modulus codebooks, bidirectional message passing, total
    hydrogen counts.
  - Models: the ex_14 MLP (2 × 100, lr 1e-3, best weights on an internal 5% validation split) and KNN (k = 5,
    Euclidean).
  - Data: the nine ex_14 targets, seeds 0 to 9, random 81/9/10% splits.
- **Baseline.** The existing ex_14 round 4 runs (prefix `ex_14_gnn`), which were not re-run. Every logP run is paired
  with the run of the same target and seed.
  - All 90 pairs have identical train, validation and test indices.
  - Smoke-test control: re-running the ex_14 command (FreeSolv, seed 0) reproduced its archive. The KNN MAE was
    identical and the MLP MAE differed by 3e-8.
- **Statistics.** Per target and model:
  - mean ± std of the test MAE over the 10 seeds;
  - the mean relative change (logP − HDF) / HDF;
  - the number of seeds on which logP lowers the MAE;
  - a two-sided Wilcoxon signed-rank test, paired by seed, with Holm correction over the nine targets (separately per
    model). With 10 seeds, the smallest attainable p is 0.002 (Holm: 0.018).

## Results

Test MAE in the units of the datasets: FreeSolv kcal/mol, AqSolDB log mol/L, Lipophilicity logD, BACE pIC50, HOPV15
as in SI S9, COMPAS-3X gap eV, QM9 hartree. "Better" counts the seeds on which logP lowers the MAE. Bold marks a significant change
after Holm correction.

| Target | Model | HDF | HDF + logP | Change | Better | p (Holm) |
|---|---|---|---|---|---|---|
| FreeSolv | MLP | 0.800 ± 0.12 | 0.764 ± 0.10 | −4.0% | 7/10 | 0.96 |
| | KNN | 2.031 ± 0.26 | 1.261 ± 0.18 | **−37.7%** | 10/10 | 0.018 |
| AqSolDB | MLP | 0.600 ± 0.017 | 0.602 ± 0.021 | +0.3% | 6/10 | 1 |
| | KNN | 0.766 ± 0.011 | 0.680 ± 0.016 | **−11.2%** | 10/10 | 0.018 |
| Lipophilicity | MLP | 0.493 ± 0.015 | 0.502 ± 0.019 | +1.9% | 2/10 | 1 |
| | KNN | 0.667 ± 0.022 | 0.653 ± 0.033 | −2.0% | 8/10 | 0.58 |
| BACE | MLP | 0.549 ± 0.051 | 0.605 ± 0.052 | +11.1% | 2/10 | 0.12 |
| | KNN | 0.572 ± 0.041 | 0.698 ± 0.057 | **+22.3%** | 0/10 | 0.018 |
| HOPV15 | MLP | 0.210 ± 0.038 | 0.231 ± 0.032 | +12.2% | 5/10 | 1 |
| | KNN | 0.257 ± 0.030 | 0.242 ± 0.042 | −5.8% | 6/10 | 0.65 |
| COMPAS-3X gap | MLP | 0.1584 ± 0.0026 | 0.1605 ± 0.0026 | +1.3% | 3/10 | 0.52 |
| | KNN | 0.1527 ± 0.0014 | 0.1527 ± 0.0014 | 0.0% | 6/10 | 0.85 |
| QM9 gap | MLP | 0.00727 ± 0.00003 | 0.00734 ± 0.00010 | +0.9% | 2/10 | 0.52 |
| | KNN | 0.01057 ± 0.00011 | 0.01071 ± 0.00011 | **+1.3%** | 0/10 | 0.018 |
| QM9 U0 | MLP | 0.522 ± 0.030 | 0.549 ± 0.057 | +5.5% | 3/10 | 1 |
| | KNN | 3.559 ± 0.14 | 2.672 ± 0.11 | **−24.9%** | 10/10 | 0.018 |
| QM9 ZPVE | MLP | 0.000816 ± 0.000018 | 0.000826 ± 0.000025 | +1.4% | 4/10 | 1 |
| | KNN | 0.00425 ± 0.00010 | 0.00372 ± 0.00008 | **−12.6%** | 10/10 | 0.018 |

## Findings

- **KNN.** logP clearly lowers the error on the targets related to partitioning between phases: hydration free energy
  (FreeSolv, −38%) and aqueous solubility (AqSolDB, −11%), on all 10 seeds each. It also lowers the error on the
  extensive QM9 energies (U0 −25%, ZPVE −13%). It increases the error on BACE (+22%) and slightly on the QM9 gap
  (+1.3%), on all seeds each. Lipophilicity, HOPV15 and COMPAS-3X do not change significantly.
- **MLP.** No significant change on any target. On AqSolDB, the MAE is unchanged (+0.3%). Most other targets tend to get
  worse, BACE the most (+11%, uncorrected p = 0.014).
- **Interpretation (hypothesis, not tested here).**
  - The MLP can probably already derive the information of logP from HDF itself. Crippen's logP is a sum of atom-type
    contributions, and HDF is the best of all representations at predicting ClogP itself in Table 1. The explicit
    attribute therefore adds little.
  - KNN cannot re-weight the features, so the attribute changes which molecules count as neighbours. That helps where
    the target follows logP (solubility, hydration free energy, and the QM9 energies through the atom composition) and
    hurts where it does not (BACE).
- **Consequence for the letter.** The solubility example holds for distance-based use of HDF (nearest neighbours), not
  for the MLP of the paper's main comparisons.

## Caveats

- **Encoding.** One design only: the bandwidth rule was fixed beforehand, and the weight of the global part in the
  embedding is unchanged.
- **Statistics.** 10 seeds per target.
- **Scope.** No comparison with Morgan fingerprints augmented with logP; baselines stay unchanged by design.

## Reproduction

- **Code.** graph_hdc branch `revision/ex26-clogp`: commits f3d047e and f86cfd5 on top of 89f3843, plus this report.
  - `predict_molecules__hdc.py` (`GRAPH_CLOGP`) and `graph_hdc/special/molecules.py` (`make_value_graph_encoder`).
  - `_slurm_ex_26.py` (command files; modes smoke, with a control line, and missing).
  - `analyze_ex_26.py` (pairing with ex_14, split check, statistics).
  - `tests/test_graph_value_encoder.py`.
- **KCIST.**
  - Pinned worktree `~/Programming/graph_hdc_ex26`, whose `.cache` links only to the load__/stats_ caches of the main
    checkout; those caches fix the ex_14 splits.
  - Runner `run_ex14_kcist.sbatch`, stage hdf, with `HDF_FILE=_ex26/hdf.txt`.
  - Jobs: smoke 51174 and production 51176 (90/90 runs ok, 1 h 56 min on one small node).
  - Archives are in that worktree's `results/predict_molecules__hdc/ex_26_clogp__*`; the ex_14 baseline is in
    `~/Programming/graph_hdc_ex14/.../results`.
  - Helper scripts are in `~/ctun_scripts/ex26/`. Analysis: `bash ~/ctun_scripts/ex26/analyze.sh`.
- **Result files.** `_ex26/kcist/summary_ex_26_clogp.md` and `_ex26/kcist/results_ex_26_clogp.csv` (one row per arm,
  target, seed and model).
- **Caches.** The `__clogp` encoding caches (about 30 GB) were deleted on 2026-10-13. A re-run has to encode again, which
  takes about 2 h on one node.
