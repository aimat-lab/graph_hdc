"""
Training-free recovery of the atom composition from HDF vectors (ex_23, SI section on partial invertibility).

For random molecules of each molecule set (MOLECULE_SETS), the experiment encodes the molecules with HDF and
decodes their atom composition from three vectors, using only the known codebooks (graph_hdc.special.composition):

- 'final': the HDF vector used throughout the paper, normalize(normalize(structure) + normalize(size + diameter)).
  The number of atoms is decoded jointly with the diameter from its size/diameter component, and the atom types
  are decoded with this number.
- 'structure': the same vector without the size and diameter encodings, normalize(structure). It contains no
  size encoding, so the true number of atoms is used to scale the decoded counts.
- 'layer0': the layer-0 readout, i.e. the sum of the bound atom vectors before message passing (computed by HDF
  as part of the encoding). Every atom contributes one unit-norm vector, so the decoded coefficients are rounded
  directly.

Atom types are all combinations of the encoder's elements with a heavy-atom degree and a hydrogen count in
DEGREES and HYDROGENS. Per molecule, we record whether the number of atoms ('final' only), the set of atom types
and the counts of all atom types are recovered exactly, and the share of atoms that are assigned the correct type.
The experiment covers every size in EMBEDDING_SIZES at depth NUM_LAYERS and every depth in DEPTHS at size
DEPTH_EMBEDDING_SIZE, each with the codebook seeds SEEDS.

Outputs in the archive: molecules.csv (one row per molecule, configuration, seed and vector) and results.csv
(the averages per configuration, seed and vector).
"""
import os
import random
import time
from typing import List, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

from graph_hdc.models import HyperNet
from graph_hdc.special.composition import (atom_composition, atom_type_codebook, composition_metrics,
                                           decode_composition, decode_size_and_diameter)
from graph_hdc.special.molecules import (graph_dict_from_mol, make_molecule_graph_encoder_map_cont,
                                         make_molecule_node_encoder_map_cont)

RDLogger.DisableLog('rdApp.*')

# == INPUT PARAMETERS ==

# :param MOLECULE_SETS:
#       The molecule sets as (DATASET_NAME, MAX_GRAPH_SIZE, MAX_GRAPH_DIAMETER). As in the other HDF experiments,
#       the size and diameter encodings are scaled by the maximum size and diameter over all valid, connected
#       molecules of the full dataset (QM9: 9 and 8, ZINC250k: 38 and 23, as in ged_exact_correlation.py).
MOLECULE_SETS: list = [('qm9_smiles', 9, 8), ('zinc250k', 38, 23)]

# :param NUM_MOLECULES:
#       The number of random molecules per molecule set.
NUM_MOLECULES: int = 1000

# :param SAMPLE_SEED:
#       The seed of the random selection of the molecules.
SAMPLE_SEED: int = 1

# == HDF PARAMETERS ==

# :param EMBEDDING_SIZES:
#       The HDF dimensions D of the dimension sweep (at depth NUM_LAYERS).
EMBEDDING_SIZES: list = [256, 512, 1024, 2048, 4096, 8192]

# :param NUM_LAYERS:
#       The message-passing depth L of the dimension sweep.
NUM_LAYERS: int = 2

# :param DEPTHS:
#       The depths L of the depth sweep (at dimension DEPTH_EMBEDDING_SIZE).
DEPTHS: list = [1, 2, 3]

# :param DEPTH_EMBEDDING_SIZE:
#       The dimension D of the depth sweep.
DEPTH_EMBEDDING_SIZE: int = 2048

# :param SEEDS:
#       The seeds of the HDF codebooks. Seed 1 is the default of the other fingerprint experiments.
SEEDS: list = [1, 2, 3]

# :param SPECTRUM:
#       Fourier magnitudes of the random codebook vectors (see predict_molecules__hdc.py): "unit" (default) or
#       "gaussian" (the original encoder).
SPECTRUM: str = 'unit'

# :param BIDIRECTIONAL:
#       Whether HDF passes messages along both directions of every bond (see predict_molecules__hdc.py).
BIDIRECTIONAL: bool = True

# :param HYDROGEN_COUNT:
#       How the hydrogen count of each atom is determined (see predict_molecules__hdc.py).
HYDROGEN_COUNT: str = 'total'

# == DECODER PARAMETERS ==

# :param DEGREES:
#       The heavy-atom degrees of the atom-type codebook of the decoder.
DEGREES: list = [0, 1, 2, 3, 4]

# :param HYDROGENS:
#       The hydrogen counts of the atom-type codebook of the decoder.
HYDROGENS: list = [0, 1, 2, 3, 4]

# :param SHORTLIST:
#       The number of candidate atom types (most similar to the vector) in the least-squares fit.
SHORTLIST: int = 64

# :param BATCH_SIZE:
#       The number of molecules that HyperNet encodes at once.
BATCH_SIZE: int = 250

__DEBUG__ = False


def sample_molecules(dataset_name: str, num: int, seed: int, codebook_types: set) -> Tuple[List[str], int]:
    """Random valid, connected molecules whose atom types are all part of the decoder's codebook."""
    from chem_mat_data import load_smiles_dataset
    df = load_smiles_dataset(dataset_name)
    column = 'smiles' if 'smiles' in df.columns else df.columns[0]
    smiles = [str(s).strip() for s in df[column]]
    order = list(range(len(smiles)))
    random.Random(seed).shuffle(order)
    selected, outside = [], 0
    for index in order:
        mol = Chem.MolFromSmiles(smiles[index])
        if mol is None or mol.GetNumAtoms() < 2 or mol.GetNumBonds() < 1 or '.' in smiles[index]:
            continue
        graph = graph_dict_from_mol(mol, hydrogens='total')
        if any((int(z), int(d), int(h)) not in codebook_types
               for z, d, h in zip(graph['node_atoms'], graph['node_degrees'], graph['node_valences'])):
            outside += 1
            continue
        selected.append(smiles[index])
        if len(selected) == num:
            break
    return selected, outside


def encode(graphs: list, dim: int, depth: int, seed: int, e: Experiment, max_size: int, max_diameter: int):
    """The three vectors of every molecule and the encoders: (final, structure, layer0, node_map, graph_map)."""
    unit = (e.SPECTRUM == 'unit')
    node_map = make_molecule_node_encoder_map_cont(dim=dim, seed=seed, unit_modulus=unit)
    graph_map = make_molecule_graph_encoder_map_cont(dim=dim, seed=seed, max_graph_size=max_size,
                                                     max_graph_diameter=max_diameter, unit_modulus=unit)
    common = dict(hidden_dim=dim, depth=depth, device='cpu', node_encoder_map=node_map, seed=seed,
                  normalize_all=True, bidirectional=e.BIDIRECTIONAL)
    full = HyperNet(graph_encoder_map=graph_map, **common)
    structure_only = HyperNet(graph_encoder_map={}, **common)
    final = np.stack([r['graph_embedding'] for r in full.forward_graphs(graphs, batch_size=e.BATCH_SIZE)])
    layer0 = np.stack([np.asarray(r['graph_hv_stack'])[0] for r in full.forward_graphs(graphs, batch_size=e.BATCH_SIZE)])
    structure = np.stack([r['graph_embedding'] for r in structure_only.forward_graphs(graphs, batch_size=e.BATCH_SIZE)])
    structure = structure / np.linalg.norm(structure, axis=1, keepdims=True)   # HyperNet does not normalize without graph encoders
    return final.astype(np.float64), structure.astype(np.float64), layer0.astype(np.float64), node_map, graph_map


@Experiment(
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)
def experiment(e: Experiment):

    @e.testing
    def testing(e: Experiment):
        e.NUM_MOLECULES = 20
        e.EMBEDDING_SIZES = [256, 1024]
        e.DEPTHS = [1, 2]
        e.DEPTH_EMBEDDING_SIZE = 1024
        e.SEEDS = [1]

    e.log_parameters()
    if e.SPECTRUM not in ('gaussian', 'unit'):
        raise ValueError(f'SPECTRUM must be "gaussian" or "unit", got {e.SPECTRUM!r}')
    if e.HYDROGEN_COUNT != 'total':
        raise ValueError('the decoder codebook counts all bonded hydrogens, so HYDROGEN_COUNT must be "total"')

    configs = [(dim, e.NUM_LAYERS) for dim in e.EMBEDDING_SIZES]
    configs += [(e.DEPTH_EMBEDDING_SIZE, depth) for depth in e.DEPTHS if (e.DEPTH_EMBEDDING_SIZE, depth) not in configs]
    e.log(f'configurations (D, L): {configs}, seeds {e.SEEDS}')

    rows = []
    for dataset_name, max_size, max_diameter in e.MOLECULE_SETS:
        # the codebook types do not depend on D or the seed, so a small codebook suffices for the selection
        types, _ = atom_type_codebook(make_molecule_node_encoder_map_cont(dim=16, seed=0), e.DEGREES, e.HYDROGENS)
        smiles, outside = sample_molecules(dataset_name, e.NUM_MOLECULES, e.SAMPLE_SEED, set(types))
        graphs = [graph_dict_from_mol(Chem.MolFromSmiles(s), hydrogens=e.HYDROGEN_COUNT) for s in smiles]
        sizes = np.array([len(g['node_atoms']) for g in graphs])
        e.log(f'{dataset_name}: {len(graphs)} molecules ({sizes.min()}-{sizes.max()} heavy atoms, mean '
              f'{sizes.mean():.1f}); skipped {outside} molecules with atom types outside the codebook')
        e[f'molecules/{dataset_name}/smiles'] = smiles
        e[f'molecules/{dataset_name}/num_outside_codebook'] = outside

        for dim, depth in configs:
            for seed in e.SEEDS:
                start = time.time()
                final, structure, layer0, node_map, graph_map = encode(graphs, dim, depth, seed, e,
                                                                       max_size, max_diameter)
                types, codebook = atom_type_codebook(node_map, e.DEGREES, e.HYDROGENS)
                num_hat, _ = decode_size_and_diameter(final, graph_map['graph_size'], graph_map['graph_diameter'],
                                                      max_size, max_diameter)
                for i, graph in enumerate(graphs):
                    true = atom_composition(graph, types)
                    decoded = {
                        'final': decode_composition(final[i], codebook, int(num_hat[i]), e.SHORTLIST),
                        'structure': decode_composition(structure[i], codebook, int(sizes[i]), e.SHORTLIST),
                        'layer0': decode_composition(layer0[i], codebook, None, e.SHORTLIST),
                    }
                    for vector, counts in decoded.items():
                        row = {'dataset': dataset_name, 'dim': dim, 'depth': depth, 'seed': seed, 'vector': vector,
                               'molecule': i, 'num_atoms': int(sizes[i]),
                               'atom_count': float(num_hat[i] == sizes[i]) if vector == 'final' else np.nan}
                        row.update(composition_metrics(counts, true))
                        rows.append(row)
                e.log(f'  {dataset_name} D={dim} L={depth} seed={seed}: atom count {np.mean(num_hat == sizes):.3f} '
                      f'({time.time() - start:.0f} s)')

    molecules = pd.DataFrame(rows)
    molecules.to_csv(os.path.join(e.path, 'molecules.csv'), index=False)
    metrics = ['atom_count', 'exact_set', 'exact_counts', 'atoms_correct']
    results = molecules.groupby(['dataset', 'dim', 'depth', 'seed', 'vector'], as_index=False)[metrics].mean()
    results.to_csv(os.path.join(e.path, 'results.csv'), index=False)
    e.log('saved molecules.csv and results.csv')

    summary = results.groupby(['dataset', 'dim', 'depth', 'vector'])[metrics].agg(['mean', 'min', 'max'])
    with pd.option_context('display.width', 200, 'display.max_columns', 20):
        e.log('\n' + summary.round(3).to_string())


experiment.run_if_main()
