import networkx as nx
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, rdmolops
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

from graph_hdc.baselines.minhash_fps import secfp_fingerprint, map4_fingerprint
from graph_hdc.baselines.sort_and_slice import SortAndSliceFingerprint, fit_transform_split

# == FINGERPRINT PARAMETERS ==

# :param FINGERPRINT_SIZE:
#       The size of the fingerprint to be generated. This will be the number of elements in the 
#       fingerprint vector representation of each molecule.
FINGERPRINT_SIZE: int = 2048
# :param FINGERPRINT_RADIUS:
#       The radius of the fingerprint to be generated. This parameter determines the number of
#       bonds to be considered when generating the fingerprint. For 'secfp' it is the maximum radius of
#       the circular substructures (3 = SECFP6, the MHFP default), for 'map4' the maximum radius of the
#       atom environments in the atom pairs (2 = MAP4), for 'sort_slice' the ECFP radius (2 = ECFP4).
FINGERPRINT_RADIUS: int = 2
# :param FINGERPRINT_TYPE:
#       The type of fingerprint to be generated: 'morgan', 'count_morgan', 'rdkit', 'atom' (atom pair),
#       'torsion' (topological torsion), 'secfp' (folded MHFP, Probst & Reymond 2018), 'map4' (folded
#       MinHashed atom-pair fingerprint, Capecchi et al. 2020) or 'sort_slice' (binary ECFP vectorised via
#       Sort & Slice instead of folding, Dablander et al. 2024). See graph_hdc.baselines.minhash_fps and
#       graph_hdc.baselines.sort_and_slice. 'sort_slice' keeps the FINGERPRINT_SIZE substructures that occur
#       in the most training molecules (no labels involved; validation and test molecules are not used).
FINGERPRINT_TYPE: str = 'morgan'
# :param FINGERPRINT_MAX_DISTANCE:
#       'atom' only: the largest topological distance of the atom pairs (RDKit maxDistance, default 30).
FINGERPRINT_MAX_DISTANCE: int = 30
# :param FINGERPRINT_TORSION_ATOMS:
#       'torsion' only: the number of atoms of a topological torsion (RDKit torsionAtomCount, default 4).
FINGERPRINT_TORSION_ATOMS: int = 4
# :param GRAPH_DESCRIPTORS:
#       Whether the two global descriptors that HDF encodes, the graph size (number of heavy atoms) and the
#       graph diameter (longest shortest path), are appended to the fingerprint as two additional features
#       (ex_25, reviewer comment R3.3: baseline fingerprints augmented with the same descriptors). Both are
#       computed from the heavy-atom graph exactly as in HDF (graph_dict_from_mol, predict_molecules__hdc) and
#       divided by their maximum over the dataset, from which HDF derives the bandwidths of their encodings, so
#       that they lie in [0, 1] like the fingerprint bits. Default False: the fingerprint alone.
GRAPH_DESCRIPTORS: bool = False
# :param FINGERPRINT_CHIRALITY:
#       Whether the Morgan fingerprints ('morgan', 'count_morgan') are computed with RDKit's ``includeChirality``,
#       which takes the CIP descriptors (R/S) of the stereocenters into account. RDKit does not distinguish every
#       pair of stereoisomers this way (in ex_21, some enantiomers keep identical fingerprints at radius 2). The
#       default (False) ignores stereochemistry, as in all fingerprints of the paper; True is the stereo-aware
#       reference for the optional stereo attribute of HDF (ex_21).
FINGERPRINT_CHIRALITY: bool = False

# == EXPERIMENT PARAMETERS ==

experiment = Experiment.extend(
    'predict_molecules.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals()
)

def graph_descriptors(mol: Chem.Mol) -> tuple:
    """
    Graph size (number of heavy atoms) and graph diameter of a molecule, computed on the heavy-atom graph as for
    the global descriptors of HDF (``graph_dict_from_mol`` and the dataset statistics of predict_molecules__hdc).
    """
    graph = nx.from_numpy_array(rdmolops.GetAdjacencyMatrix(mol))
    return graph.number_of_nodes(), nx.diameter(graph)


def append_graph_descriptors(e: Experiment, index_data_map: dict) -> None:
    """
    Append the graph size and the graph diameter, each divided by its maximum over the dataset, to the fingerprint
    of every molecule (GRAPH_DESCRIPTORS). The maxima are those from which HDF derives its encoding bandwidths.
    """
    descriptors = {index: graph_descriptors(Chem.MolFromSmiles(graph['graph_repr']))
                   for index, graph in index_data_map.items()}
    max_size = max(size for size, _ in descriptors.values())
    max_diameter = max(diameter for _, diameter in descriptors.values())
    e.log(f'appending graph size (max {max_size}) and graph diameter (max {max_diameter}) as two features...')
    for index, graph in index_data_map.items():
        size, diameter = descriptors[index]
        graph['graph_features'] = np.concatenate([graph['graph_features'], [size / max_size, diameter / max_diameter]])
    e['graph_descriptors/max_size'] = max_size
    e['graph_descriptors/max_diameter'] = max_diameter


@experiment.hook('process_dataset', replace=True, default=False)
def process_dataset(e: Experiment,
                    index_data_map: dict
                    ) -> None:

    encode_fingerprints(e, index_data_map)
    if e.GRAPH_DESCRIPTORS:
        append_graph_descriptors(e, index_data_map)


def encode_fingerprints(e: Experiment, index_data_map: dict) -> None:
    """Store the fingerprint of FINGERPRINT_TYPE of every molecule as its "graph_features"."""

    if e.FINGERPRINT_CHIRALITY and e.FINGERPRINT_TYPE not in ('morgan', 'count_morgan'):
        raise ValueError(f'FINGERPRINT_CHIRALITY is only implemented for Morgan fingerprints, '
                         f'not for "{e.FINGERPRINT_TYPE}"')

    if e.FINGERPRINT_TYPE == 'sort_slice':
        # The vocabulary is fitted on the training molecules of this split, then all molecules are encoded.
        fingerprint = SortAndSliceFingerprint(size=e.FINGERPRINT_SIZE, radius=e.FINGERPRINT_RADIUS)
        mols = {index: Chem.MolFromSmiles(graph['graph_repr']) for index, graph in index_data_map.items()}
        e.log(f'fitting the Sort & Slice vocabulary on the {len(e["indices/train"])} training molecules...')
        features = fit_transform_split(fingerprint, mols, e['indices/train'])
        e.log(f' * {fingerprint.num_reference_substructures} distinct substructures in the training molecules, '
              f'{len(fingerprint.vocabulary)} kept')
        for index, graph in index_data_map.items():
            graph['graph_features'] = features[index]
        return

    if e.FINGERPRINT_TYPE == 'morgan':
        gen = rdFingerprintGenerator.GetMorganGenerator(
            radius=e.FINGERPRINT_RADIUS,
            fpSize=e.FINGERPRINT_SIZE,
            includeChirality=e.FINGERPRINT_CHIRALITY,
        )
        
    elif e.FINGERPRINT_TYPE == 'rdkit':
        gen = rdFingerprintGenerator.GetRDKitFPGenerator(
            fpSize=e.FINGERPRINT_SIZE,
            maxPath=2*e.FINGERPRINT_RADIUS,
        )
        
    elif e.FINGERPRINT_TYPE == 'atom':
        gen = rdFingerprintGenerator.GetAtomPairGenerator(
            fpSize=e.FINGERPRINT_SIZE,
            maxDistance=e.FINGERPRINT_MAX_DISTANCE,
        )

    elif e.FINGERPRINT_TYPE == 'torsion':
        gen = rdFingerprintGenerator.GetTopologicalTorsionGenerator(
            fpSize=e.FINGERPRINT_SIZE,
            torsionAtomCount=e.FINGERPRINT_TORSION_ATOMS,
        )

    elif e.FINGERPRINT_TYPE == 'count_morgan':
        # Count-based Morgan: identical substructure hashing to 'morgan', but the output vector holds
        # the number of times each bit is set (integer counts) instead of a 0/1 presence indicator.
        gen = rdFingerprintGenerator.GetMorganGenerator(
            radius=e.FINGERPRINT_RADIUS,
            fpSize=e.FINGERPRINT_SIZE,
            includeChirality=e.FINGERPRINT_CHIRALITY,
        )

    elif e.FINGERPRINT_TYPE == 'secfp':
        gen = None
        encode = lambda mol: secfp_fingerprint(mol, length=e.FINGERPRINT_SIZE, radius=e.FINGERPRINT_RADIUS)

    elif e.FINGERPRINT_TYPE == 'map4':
        gen = None
        encode = lambda mol: map4_fingerprint(mol, length=e.FINGERPRINT_SIZE, radius=e.FINGERPRINT_RADIUS)

    else:
        raise ValueError(f'unknown FINGERPRINT_TYPE "{e.FINGERPRINT_TYPE}"')

    count = (e.FINGERPRINT_TYPE == 'count_morgan')
    e.log(f'processing molecules into {"count " if count else ""}{e.FINGERPRINT_TYPE} fingerprints...')

    for c, (index, graph) in enumerate(index_data_map.items()):
        mol = Chem.MolFromSmiles(graph['graph_repr'])
        if gen is None:
            graph['graph_features'] = encode(mol).astype(float)
        elif count:
            graph['graph_features'] = np.array(gen.GetCountFingerprint(mol).ToList()).astype(float)
        else:
            graph['graph_features'] = np.array(gen.GetFingerprint(mol)).astype(float)

        if c % 1000 == 0:
            e.log(f' * {c} molecules done')

experiment.run_if_main()