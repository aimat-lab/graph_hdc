# Fingerprint Comparison

This folder contains all the experiment scripts related to the investigation of the hypervectors 
as viable alternatives for molecular fingerprints on molecular graphs.

- ``predict_molecules.py``: Base experiment loading molecular datasets and training ML models (neural net, random forest, SVM etc.) on encoded vector representations with test set evaluation.
- ``predict_molecules__hdc.py``: Encodes molecules using HDC encoder from this package.
- ``predict_molecules__fp.py``: Encodes molecules using RDKit fingerprints as baseline for HDC comparison.


# Experiment Records

## Experiment 0

Experiment 0 is a pre-requisite for the other experiments and performs a simple hyperparameter optimization separately for each combination of method and dataset... to 
determine the optimal set of hyperparameters for each method and dataset combination, which can then later be used in the other experiments.

- ``ex_00_b``: Full hyperparameter optimization grid search for all encodings (GNN, FP, HDC), ML methods (NN, RF, etc.), and 6 core datasets (qm9 gap, aqsoldb, clogp, conjugated, bace, bbbp).
- ``ex_00_c``: Hyperparameter optimization for 4 regression datasets (qm9 gap, aqsoldb, clogp, qm9 u0) with RDKit fingerprints using larger embedding sizes.

## Experiment 1

Experiment 1 is the main experiment which compares the different encoding methods and different machine learning methods on the 6 core datasets. The results are summarized in a table and a figure.

- ``ex_01_o``: Results using hyperparameters from ``ex_00_b`` for all encodings (GNN, FP, HDC) and ML methods (NN, RF, SVM etc.) on 6 core datasets with average ranks.
- ``ex_01_p``: Results using hyperparameters from ``ex_00_c`` only for FP (HDC/GNN use standard params) on 4 regression datasets (qm9 gap, aqsoldb, clogp, qm9 u0).

## Experiment 2

Experiment 2 chooses a smaller number of machine learning methods but a larger number of datasets to compare the performance of the different encoding methods on a wider range of datasets. The results are summarized in a table where the datasets are the rows and the methods are the columns. Together with an average rank across all datasets for each method.

- ``ex_02_a``: ---

## Experiment 3 

Experiment 3 performs an ablation study on the size and depth of the vector representations of both the HDC vectors and the molecular fingerprints for a selected dataset and a selection of machine learning methods.

- ``ex_03_a``: AqSolDB solubility, GNN baseline, FP and HDC, NN and RF, embedding size and depth sweep, max training size.
  - Results show that smaller depth is better always (kind of). Results also show that the HDC performs much better at smaller embedding sizes than the fingerprints but towards larger embedding sizes, the fingerprints converge towards almost the same performance.
- ``ex_03_aa``: AqSolDB solubility, GNN baseline, Morgan FP and HDC, NN and RF, embedding size and depth sweep, max training size.
- ``ex_03_ab``: QM9 U0, Morgan FP and HDC, NN and RF, embedding size and depth sweep, max training size.
- ``ex_03_ac``: HOPV15, FP and HDC, NN and KNN, embedding size and depth sweep.
- ``ex_03_ad``: LIPOP, FP and HDC, NN and KNN, embedding size and dephth sweep.
- ``ex_03_ae``: BACE IC50, FP and HDC, NN and KNN, embedding size and depth sweep.

## Experiment 4

Experiment 4 performs a sweep over different dataset sizes to compute a learning curve (log-log plot 
of dataset size vs. error residuals).

- ``ex_04_a``: QM9 GAP, GNN baseline, FP and HDC, NN and RF, training size sweep, embedding depth 2 and size 2048.
  - result shows GNN performs best across the board. HDC performs better for smaller dataset sizes but for larger sizes, all converge toward similar performance.
- ``ex_04_b``: QM9 GAP, GNN baseline, FP and HDC, NN and RF, training size sweep, embedding depth 2 and size 256.
  - result shows that GNN performs best across the board. HDC performs better for all dataset sizes. Unlike before, they do not converge toward similar performance but the slope stays the same.
- ``ex_04_c``: AqSolDB solubility, GNN baseline, FP and HDC, NN and RF, training size sweep, embedding depth 2 and size 2048.
  - result shows GNN performs best across the board. HDC performs better for all dataset sizes. Interestingly, even though this is a noisy dataset, the performance plateaus only at the very end even for the HDC. Neural net approaches much better than random forest.
- ``ex_04_d``: Zinc250k QED, GNN baseline, FP and HDC, NN and RF, training size sweep, embedding depth 2 and size 2048.
  - Here, fingerprints and HDC perform about the same, while GNN clearly performs better.

- ``ex_04_ab``: QM9 U0, GNN baseline, HDC and FP, NN and RF, training size sweep, embedding size 8192 and depth 2.
  - Result shows that FP performs really badly and that FP is much better - at the same level as GNNs and even better for larger dataset sizes at some point.
- ``ex_04_ac``: AqSolDB solubility, GNN baseline, HDC and FP, NN and RF, training size sweep, embedding size 8192 and depth 2.
- ``ex_04_ae``: Ames mutagenicity, GNN baseline, HDC and FP, NN and RF, training size sweep, embedding size 2048 and depth 2.
  - Result shows GNNs are much worse (as expected due to to no balancing mechanism) and that HDC and FP are essentially exactly the same.
- ``ex_04_ag``: BBBP blood-brain barrier, GNN baseline, HDC and FP, NN and RF, training size sweep, embedding size 8192 and depth 2.
- ``ex_04_ag``: Zinc250k logP, GNN baseline, HDC and FP, NN and RF, training size sweep, embedding size 8192 and depth 2.
  - Result shows that GNNs are much better than both HDC and FP, which are essentially exactly the same.
- ``ex_04_ah``: QM9 GAP, GNN baseline, HDC and FP, NN and RF, training size sweep, embedding size 2048 and depth 2.
- ``ex_04_ai``: QM9 Energy, GNN baseline, HDC and FP, KNN, embedding size 2048 and depth sweep.
- ``ex_04_aj``: QM9 Dipole Moment, GNN baseline, HDC and FP, KNN, embedding size 2048 and depth sweep.
- ``ex_04_ak``: QM9 GAP, GNN baseline, HDC and FP, KNN, embedding size 2048 and depth sweep.
- ``ex_04_al``: QM9 Heat capacity, GNN baseline, HDC and FP, KNN, embedding size 2048 and depth sweep.


## Experiment 5

Experiment 5 deals with the reconstruction of molecules from the hypervectors. More specifically, this experiment is concerned with the reconstruction of the molecular composition from the hypervectors. This experiment sweeps different embedding sizes to get the accuracy of the composition reconstruction.

- ``ex_05_a``: AqSolDB ~10k molecules, embedding size sweep for molecular composition reconstruction accuracy.
  - results show that larger embedding size leads to better reconstruction accuracy.

## Experiment 6

Experiment 6 focuses on molecular generation using normalizing flows to learn distributions over the hypervector space, enabling sampling of novel molecular structures.

## Experiment 7

Experiment 7 performs a systematic comparison of molecular representation methods for Bayesian Optimization-based molecular search. Instead of gradient-based optimization in continuous representation space, this experiment treats molecular search as a discrete black-box optimization problem where Bayesian Optimization with Gaussian Processes guides the search toward target property values.

**Key Methodology:**
- **Bayesian Optimization Setup**: Uses BotTorch with Gaussian Process surrogate models and Expected Improvement acquisition function
- **Comparison Metrics**: Area Under Curve (AUC) as primary metric for convergence speed, plus simple regret and rounds-to-threshold
- **Statistical Robustness**: Each experiment runs 25 independent BO trials (NUM_TRIALS=25) for averaging, no seed loop needed
- **Representation Methods**: Random baseline (Gaussian N(0,1)), Morgan fingerprints (ECFP4, radius=2), and HDC (2-layer continuous encoding)

**Research Questions:**
1. How does representation dimensionality affect BO convergence speed?
2. Which representation method achieves best performance at minimal embedding size?
3. How much better are structured representations (FP, HDC) compared to random baseline?
4. At what embedding size do we see diminishing returns for each method?

- ``ex_07_a``: AqSolDB with CLogP target property, embedding size sweep [8, 16, 32, 64, 128], 3 representation methods (random, FP, HDC), 20 initial samples, 25 BO rounds, 3 samples per round, 25 trials per experiment.
  - Total: 15 SLURM jobs (5 sizes × 3 methods), each running 25 independent BO trials
  - Goal: Quantify how representation quality and dimensionality affect molecular search efficiency using standardized BO comparison metrics

## Hyperparameter Optimization

- ``hyperopt_a``: Hyperparameter optimization for Ames dataset.
  - gnn: batch_size=128 learning_rate=0.0001
  - fp: fingerprint_size=1024 fingerprint_radius=1
  - hdc: embedding_size=4096 num_layers=2
- ``hyperopt_b``: Hyperparameter optimization for Conjugated dataset.
  - gnn: batch_size=16 learning_rate=0.001
  - fp: fingerprint_size=8192 fingerprint_radius=1
  - hdc: embedding_size=8192 num_layers=1
