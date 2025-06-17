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

## Experiment 4

Experiment 4 performs a sweep over different dataset sizes to compute a learning curve (log-log plot 
of dataset size vs. error residuals).

- ``ex_04_b``: hdc and gnn for "clogp" deterministic classification dataset
- ``ex_04_c``: hdc, gnn and fp for "bbbp" experimental classification dataset
- ``ex_04_d``: hdc, gnn and fp for "conjugated" deterministic classification dataset
- ``ex_04_e``: hdc ,gnn and fp for "ames" experimental classification dataset (hyperparameter optimized)
- ``ex_04_f``: hdc, gnn and fp for "conjugated" experimental classification dataset (hyperparameter optimized)

## Hyperparameter Optimization

- ``hyperopt_a``: Hyperparameter optimization for the "ames" dataset
  - gnn: batch_size=128 learning_rate=0.0001
  - fp: fingerprint_size=1024 fingerprint_radius=1
  - hdc: embedding_size=4096 num_layers=2
- ``hyperopt_b``: Hyperparameter optimization for the "conjugated" dataset
  - gnn: batch_size=16 learning_rate=0.001
  - fp: fingerprint_size=8192 fingerprint_radius=1
  - hdc: embedding_size=8192 num_layers=1
