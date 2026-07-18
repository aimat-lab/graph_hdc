"""
Sherlock-fingerprint baseline for the molecular property prediction experiments.

This is a thin variant of :mod:`predict_molecules` that replaces the per-molecule
feature vector with the *Sherlock Fingerprint* (entropy-ranked, collision-free
Morgan; Xu et al., SPECTRE, JCIM 2026). Everything downstream -- model training,
evaluation and result aggregation -- is inherited unchanged from the base
experiment, so the Sherlock fingerprint slots into the main comparison table as
just another feature column next to ``fp`` (Morgan), ``gnn`` and ``hdc``.

The descriptor dictionary is *task independent*: it is fit once on a large external
reference corpus (COCONUT + LOTUS + DeepSAT training, matching the paper) via
``graph_hdc/baselines/fit_sherlock.py`` and then applied here as a stateless
featurizer. Point :param:`SHERLOCK_DICTIONARY_PATH` at that artifact.

To obtain the paper's native fingerprint set ``FINGERPRINT_SIZE = 16384`` and
``SHERLOCK_RADIUS = 6``. For a *size-matched* comparison against the other columns,
lower ``FINGERPRINT_SIZE`` (e.g. to 2048 to match the Morgan baseline); this keeps
the top-N highest-entropy descriptors of the same fitted dictionary.
"""

import os

import numpy as np
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

from graph_hdc.baselines.sherlock import SherlockFingerprint

# == SHERLOCK FINGERPRINT PARAMETERS ==

# :param FINGERPRINT_SIZE:
#       Output dimensionality of the Sherlock fingerprint, i.e. the number of retained
#       (highest-entropy) descriptors. The native fingerprint from the SPECTRE paper
#       uses 16384. Setting a smaller value yields a *size-matched* variant -- the
#       top-N highest-entropy descriptors of the same dictionary -- which requires that
#       the loaded dictionary was fit with at least this many descriptors.
FINGERPRINT_SIZE: int = 16384
# :param SHERLOCK_RADIUS:
#       Circular radius of the fingerprint (the paper uses 6). Must be <= the radius the
#       dictionary was fit with.
SHERLOCK_RADIUS: int = 6
# :param SHERLOCK_DICTIONARY_PATH:
#       Path to a pickled ``SherlockFingerprint`` fit artifact produced by
#       ``graph_hdc/baselines/fit_sherlock.py`` on a reference corpus. This is the
#       recommended, paper-faithful, task-independent way to use the baseline. If the
#       path is ``None`` or missing, the experiment falls back to fitting the dictionary
#       on the TRAINING SPLIT ONLY of the current dataset (leakage-safe, but NOT the
#       paper corpus) -- intended only for quick local tests.
SHERLOCK_DICTIONARY_PATH: str = None
# :param SHERLOCK_FIT_JOBS:
#       Number of worker processes used for the fallback train-split fit (ignored when a
#       pre-fit dictionary is loaded).
SHERLOCK_FIT_JOBS: int = 4

# == EXPERIMENT PARAMETERS ==

experiment = Experiment.extend(
    'predict_molecules.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals()
)


@experiment.hook('process_dataset', replace=True, default=False)
def process_dataset(e: Experiment,
                    index_data_map: dict
                    ) -> None:

    # RDKit is noisy about the (expected) invalid SMILES it encounters; silence it.
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')

    path = e.SHERLOCK_DICTIONARY_PATH

    if path and os.path.exists(path):
        # The Sherlock transform depends only on the molecule set, the output size and the radius --
        # not on the model, the seed, or the prediction target. So one transform is reused across all
        # HPO configs, all table seeds, AND all targets of a dataset (every QM9 target shares a single
        # qm9_smiles transform). Cache it. Full-data caches are seed-independent; subsampled runs (whose
        # molecule set depends on the seed) add the seed to the key.
        if e.NUM_DATA is None or e.NUM_DATA == 1.0:
            cache_name = f'sherlock_{e.DATASET_NAME}__numdata_{e.NUM_DATA}__size_{e.FINGERPRINT_SIZE}__radius_{e.SHERLOCK_RADIUS}'
        else:
            cache_name = f'sherlock_{e.DATASET_NAME}__numdata_{e.NUM_DATA}__seed_{e.SEED}__size_{e.FINGERPRINT_SIZE}__radius_{e.SHERLOCK_RADIUS}'

        @experiment.cache.cached(name=cache_name)
        def _sherlock_features():
            e.log(f'loading pre-fit Sherlock dictionary from: {path}')
            sherlock = SherlockFingerprint.load(path)
            e.log(f' * dictionary fit on {sherlock.corpus_size_} corpus molecules, '
                  f'{sherlock.n_descriptors_} distinct descriptors, fit_radius={sherlock.fit_radius_}')
            sherlock.set_output(size=e.FINGERPRINT_SIZE, max_radius=e.SHERLOCK_RADIUS)
            e.log(f'transforming {len(index_data_map)} molecules into {sherlock.size}-dim '
                  f'Sherlock fingerprints (radius {sherlock.radius})...')
            feats = {}
            for c, (index, graph) in enumerate(index_data_map.items()):
                feats[index] = sherlock.transform_smiles(graph['graph_repr']).astype(float)
                if c % 1000 == 0:
                    e.log(f' * {c} molecules done')
            return feats

        feats = _sherlock_features()
        for index in index_data_map:
            index_data_map[index]['graph_features'] = feats[index]
    else:
        e.log('WARNING: no valid SHERLOCK_DICTIONARY_PATH given -- falling back to '
              'fitting the descriptor dictionary on the TRAINING SPLIT of this dataset. '
              'This is leakage-safe but does NOT reproduce the paper corpus; provide a '
              'pre-fit dictionary via SHERLOCK_DICTIONARY_PATH for paper-faithful results.')
        train_indices = e['indices/train']
        train_smiles = [index_data_map[i]['graph_repr'] for i in train_indices]
        e.log(f' * fitting Sherlock on {len(train_smiles)} training molecules '
              f'(radius {e.SHERLOCK_RADIUS}, size {e.FINGERPRINT_SIZE}, jobs {e.SHERLOCK_FIT_JOBS})')
        sherlock = SherlockFingerprint(radius=e.SHERLOCK_RADIUS, size=e.FINGERPRINT_SIZE)
        sherlock.fit(train_smiles, n_jobs=e.SHERLOCK_FIT_JOBS, log=e.log)
        for index, graph in index_data_map.items():
            graph['graph_features'] = sherlock.transform_smiles(graph['graph_repr']).astype(float)


experiment.run_if_main()
