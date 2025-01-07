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