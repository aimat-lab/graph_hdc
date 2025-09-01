# Fingerprint Comparison

This folder contains all the experiment scripts related to the investigation of the hypervectors 
as viable alternatives for molecular fingerprints on molecular graphs.

- ``predict_molecules.py``: The base implementation of the experiment which loads the dataset of 
  molecular graphs and trains the various vanilla machine learning models (simple neural net, random 
  forest, support vector machines etc.) on the encoded vector representation of the molecular graphs.
  The trained models are subsequently evaluated on a test set and compared.
  The implementation of the vector representation is left up to the sub-experiments which may either 
  be moleculer fingerprints, for example.
- ``predict_molecules__hdc.py``: Encodes the molecules using the hyperdimensional computing (HDC) 
  encoder implemented in this package.
- ``predict_molecules__fp.py``: Encodes the molecules using a molecular fingerprint from RDKit.
  Acts as a baseline to compare the HDC encoder against.


# Experiment Records

## Experiment 0

Experiment 0 is a pre-requisite for the other experiments and performs a simple hyperparameter optimization separately for each combination of method and dataset... to 
determine the optimal set of hyperparameters for each method and dataset combination, which can then later be used in the other experiments.

- ``ex_00_b``: The full hyperparameter optimization for the Experiment 1 table. This means that it 
  was a grid search over all the hyperparameters (encoding method + model parameters) for each 
  combination of encoding (gnn, fp, hdc), the machine learning method (neural net, random forest etc)
  and the 6 core datasets (qm9 gap, aqsoldb, clogp, conjugated, bace, bbbp).
- ``ex_00_c``: hopt only for the 4 regression datasets (qm9 gap, aqsoldb, clogp, qm9 u0) but also for the rdkit fingerprints in addition and now also using larger embedding size for the sweep to be more fair towards the fingerprints hopefully.

## Experiment 1

Experiment 1 is the main experiment which compares the different encoding methods and different machine learning methods on the 6 core datasets. The results are summarized in a table and a figure.

- ``ex_01_o``: Results using the results of the full hyperparameter optimization from ``ex_00_b``.
  The table contains the results of the different encoding methods (gnn, fp, hdc) and the different machine learning methods (neural net, random forest, support vector machines etc.) on the 6 core datasets (qm9 gap, aqsoldb, clogp, conjugated, bace, bbbp).
  The table also contains the average rank across all datasets for each method.
- ``ex_01_p``: Results using the results of the hyperparameter optimization from ``ex_00_c``. Except that the optimized hyperparameters are only used for the fingerprint methods but not for the HDC and GNN methods - these use the standard parameters. And also only results for the 4 regression datasets (qm9 gap, aqsoldb, clogp, qm9 u0) are included.

## Experiment 2

Experiment 2 chooses a smaller number of machine learning methods but a larger number of datasets to compare the performance of the different encoding methods on a wider range of datasets. The results are summarized in a table where the datasets are the rows and the methods are the columns. Together with an average rank across all datasets for each method.

- ``ex_02_a``: ---

## Experiment 3 

Experiment 3 performs an ablation study on the size and depth of the vector representations of both the HDC vectors and the molecular fingerprints for a selected dataset and a selection of machine learning methods.

- ``ex_03_a``: On the AqSolDB dataset using the solubility as the target property. Comparison between GNN as the baseline, fingerprint and HDC (neural_net and random_forest each). Sweep over different embedding sizes and depths. Max training size.
  - Results show that smaller depth is better always (kind of). Results also show that the HDC performs much better at smaller embedding sizes than the fingerprints but towards larger embedding sizes, the fingerprints converge towards almost the same performance.
- ``ex_03_aa``: On the AqSolDB dataset using the solubility as the target property. Comparisong between GNN as the baseline, morgan fingerprints and HDC vectors on neural net and random forest. Sweep over different embedding sizes and depths. Max training size.
  

## Experiment 4

Experiment 4 performs a sweep over different dataset sizes to compute a learning curve (log-log plot 
of dataset size vs. error residuals).

- ``ex_04_a``: On the QM9 dataset using the GAP as the target property. Comparison between GNN as the baseline, fingerprint and HDC (neural_net and random_forest each). Sweep over different training dataset sizes. Fixes configuration for embedding depth 2 and embedding size 2048.
  - result shows GNN performs best across the board. HDC performs better for smaller dataset sizes but for larger sizes, all converge toward similar performance.
- ``ex_04_b``: On the QM9 dataset using the GAP as the target property. Comparison between GNN as the baseline, fingerprint and HDC (neural_net and random_forest each). Sweep over different training dataset sizes. Fixes configuration for embedding depth 2 and embedding size 256. This experiment was to see if there is a larger difference for smaller embeddings sizes or not.
  - result shows that GNN performs best across the board. HDC performs better for all dataset sizes. Unlike before, they do not converge toward similar performance but the slope stays the same.
- ``ex_04_c``: On the AqSolDB dataset using the solubility as the target property. Comparison between GNN as the baseline, fingerprint and HDC (neural_net and random_forest each). Sweep over different training dataset sizes. Fixes configuration for embedding depth 2 and embedding size 2048.
  - result shows GNN performs best across the board. HDC performs better for all dataset sizes. Interestingly, even though this is a noisy dataset, the performance plateaus only at the very end even for the HDC. Neural net approaches much better than random forest.
- ``ex_04_d``: On the Zinc250k dataset using the QED as the target property. Comparison between GNN as the baseline, fingerpring and HDC (neural_net and random_forest each). Sweep over different training dataset sizes. Fixex configuration for embedding depth 2 and embedding size 2048.
  - Here, fingerprints and HDC perform about the same, while GNN clearly performs better.

## Experiment 5

Experiment 5 deals with the reconstruction of molecules from the hypervectors. More specifically, this experiment is concerned with the reconstruction of the molecular composition from the hypervectors. This experiment sweeps different embedding sizes to get the accuracy of the composition reconstruction.

- ``ex_05_a``: On the AqSolDB dataset using all of the ~10k molecules. A sweep over a large number of different embedding sizes to determine the accuracy of the reconstruction of the molecular composition from the hypervectors.
- results show that larger embedding size leads to better reconstruction accuracy.

## Hyperparameter Optimization

- ``hyperopt_a``: Hyperparameter optimization for the "ames" dataset
  - gnn: batch_size=128 learning_rate=0.0001
  - fp: fingerprint_size=1024 fingerprint_radius=1
  - hdc: embedding_size=4096 num_layers=2
- ``hyperopt_b``: Hyperparameter optimization for the "conjugated" dataset
  - gnn: batch_size=16 learning_rate=0.001
  - fp: fingerprint_size=8192 fingerprint_radius=1
  - hdc: embedding_size=8192 num_layers=1
