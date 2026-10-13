"""
Experiment 24 (ex_24): Bayesian optimization (BO) benchmark of the HDF revision (Reviewer 2, comment R2.4).

The original BO experiment (ex_07) used a Matern-5/2 GP on standardized inputs for every representation,
including the binary Morgan fingerprints, for which a Tanimoto kernel is the standard choice. This benchmark
repeats the size sweep of ex_07 with a kernel that suits each representation and with more objectives:

    objectives    clogp          ClogP (Crippen) closest to 15 on AqSolDB      (target, as in ex_07)
                  qed            QED closest to 0.2 on ZINC250k                (target, as in ex_07)
                  solubility     maximal AqSolDB solubility (logS)
                  lipophilicity  maximal lipophilicity (logD)
                  freesolv       minimal FreeSolv hydration free energy
                  beet           maximal honey-bee LD50 (BEET, exact values only, log10 scale: the raw values
                                 span 0.0015-3980 ug/bee, and on that scale 124 of 128 molecules lie at
                                 normalized regret >= 0.91 behind four outliers)
                  qm9_gap        minimal HOMO-LUMO gap on QM9 (DFT)
                  qm9_dipole     maximal dipole moment on QM9 (DFT)
                                 Both QM9 objectives never propose zwitterions (user decisions 2026-10-12/13, see
                                 zwitterion_mask): molecules with a formal charge that no opposite charge on a
                                 bonded atom compensates, and ylides with a charged carbon. In the gas-phase DFT of
                                 QM9 they hold all 100 largest dipoles (up to 29.6 D) and most of the smallest
                                 gaps. They stay in the library (input space, HDF statistics). Nitro compounds are
                                 candidates (the first version excluded every molecule with a formal charge).
                  melting_point  maximal melting point (open melting point dataset)
    methods       hdf_matern       HDF (standard configuration) + Matern-5/2
                  morgan_tanimoto  Morgan radius 2, binary, folded to D bits + Tanimoto
                  morgan_pca       Morgan-2048 -> first D PCA components + Matern-5/2
                  random           random search
                  hdf_linear       HDF + linear (cosine) kernel (SI; on qed since 2026-10-13)
    sizes         D = 8, 16, 32, 64, 128, 256, 512, 1024 (the PCA needs more molecules than components: no
                  morgan_pca cell above num_molecules - 1 components)
    campaigns     100 per (objective, method, size) (25 until 2026-10-13); campaign k shares its initial
                  molecules across methods
                  and uses its own HDF codebook (seed 1000 + 100 k)
    budget        20 initial molecules + 150 rounds of one molecule; freesolv 20 + 80; beet 10 + 40. The
                  acquisition scores the whole library in every round, also on ZINC250k.

The BO procedure itself lives in ``graph_hdc.bo``. All data and results are written to ``_ex24/`` next to this
file (or to $EX24_ROOT): ``data/<library>/`` holds the cleaned library and its features, ``data/objectives/``
the objective values and initial designs, ``runs/`` one result file per campaign. Stages:

    python bo_benchmark.py featurize [--libraries ...] [--workers N]
    python bo_benchmark.py morgan [--libraries ...] [--sizes ...] [--workers N]
    python bo_benchmark.py objectives --objectives ...
    python bo_benchmark.py encode [--libraries ...] [--sizes ...] [--campaigns ...] [--workers N]
    python bo_benchmark.py tasks [--campaigns ...] [--output FILE] [--open-only]
    python bo_benchmark.py run --workers W --threads T [--gpus G] [--shard i --num-shards n] [--tasks-file FILE]
    python bo_benchmark.py cleanup --campaigns ... [--dry-run]
    python bo_benchmark.py calibrate --workers-list 16 32 64 [--gpus G]
    python bo_benchmark.py status
    python bo_benchmark.py collect

On KCIST the stages run as whole-node SLURM jobs through ``run_ex24_kcist.sbatch``.
"""
import argparse
import json
import multiprocessing as mp
import os
import pickle
import socket
import subprocess
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional

import numpy as np

PATH = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get('EX24_ROOT', os.path.join(PATH, '_ex24'))
DATA = os.path.join(ROOT, 'data')
RUNS = os.path.join(ROOT, 'runs')

SIZES = [8, 16, 32, 64, 128, 256, 512, 1024]
# 25 campaigns until 2026-10-13, then 100 (user). Campaign k keeps its initial design and codebook seed, so the
# first 25 campaigns stay valid.
NUM_CAMPAIGNS = 100
HDF_DEPTH = 2
MORGAN_RADIUS = 2
PCA_SOURCE_BITS = 2048
PCA_COMPONENTS = max(SIZES)
# Libraries with more molecules than this run their acquisition on a GPU (if the worker has one). The KCIST
# calibration (2026-10-12) measured about 0.01 s per acquisition on an RTX 4090 for AqSolDB (8,574 molecules)
# at every size, against 0.04-0.38 s on one CPU core; FreeSolv and BEET are too small to gain anything.
GPU_MIN_MOLECULES = 2000

# name -> chem_mat_data dataset, SMILES column and the property columns used as objectives
LIBRARIES = {
    'aqsoldb': {'smiles': 'SMILES', 'properties': {'solubility': 'Solubility'}},
    'zinc250k': {'smiles': 'smiles', 'properties': {'qed': 'qed'}},
    'freesolv': {'smiles': 'smiles', 'properties': {'freesolv': 'expt'}},
    'lipophilicity': {'smiles': 'smiles', 'properties': {'lipophilicity': 'exp'}},
    'beet': {'smiles': 'SMILES', 'properties': {'beet': 'Experimental value'}},
    'qm9_smiles': {'smiles': 'smiles', 'properties': {'qm9_gap': 'gap', 'qm9_dipole': 'mu'}},
    'open_melting_point': {'smiles': 'smiles', 'properties': {'melting_point': 'target'}},
}

OBJECTIVES = {
    'clogp': dict(library='aqsoldb', property='clogp', mode='target', target=15.0, num_initial=20, num_rounds=150),
    'qed': dict(library='zinc250k', property='qed', mode='target', target=0.2, num_initial=20, num_rounds=150),
    'solubility': dict(library='aqsoldb', property='solubility', mode='maximize', target=None,
                       num_initial=20, num_rounds=150),
    'lipophilicity': dict(library='lipophilicity', property='lipophilicity', mode='maximize', target=None,
                          num_initial=20, num_rounds=150),
    'freesolv': dict(library='freesolv', property='freesolv', mode='minimize', target=None,
                     num_initial=20, num_rounds=80),
    'beet': dict(library='beet', property='beet', mode='maximize', target=None, num_initial=10, num_rounds=40,
                 transform='log10'),
    'qm9_gap': dict(library='qm9_smiles', property='qm9_gap', mode='minimize', target=None,
                    num_initial=20, num_rounds=150, exclude_zwitterions=True),
    'qm9_dipole': dict(library='qm9_smiles', property='qm9_dipole', mode='maximize', target=None,
                       num_initial=20, num_rounds=150, exclude_zwitterions=True),
    'melting_point': dict(library='open_melting_point', property='melting_point', mode='maximize', target=None,
                          num_initial=20, num_rounds=150),
}

GP_METHODS = ['hdf_matern', 'morgan_tanimoto', 'morgan_pca', 'hdf_linear']


def methods_for(objective: str) -> List[str]:
    """The methods of an objective: all of them (the linear-kernel HDF skipped ZINC250k until 2026-10-13)."""
    return ['hdf_matern', 'morgan_tanimoto', 'morgan_pca', 'random', 'hdf_linear']


def hdf_seed(campaign: int) -> int:
    """
    HDF codebook seed of a campaign. The encoder maps add offsets of up to +30 to their seed, so seeds 100
    apart never share a random stream between campaigns.
    """
    return 1000 + 100 * campaign


def campaign_seed(campaign: int) -> int:
    """Seed of the random tie-breaking and of the random search of a campaign (the same for all methods)."""
    return 24000 + campaign


def git_commit() -> str:
    try:
        return subprocess.run(['git', '-C', PATH, 'rev-parse', '--short', 'HEAD'], capture_output=True,
                              text=True, timeout=10).stdout.strip()
    except Exception:
        return 'unknown'


def set_thread_env(threads: int) -> None:
    """
    Limits the thread pools of every numerical library for the worker processes spawned after this call (they
    inherit the environment and read it on import). Without it, numpy's OpenBLAS starts one thread per core in
    every worker, and dozens of workers per node spend their time in the kernel.
    """
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = str(threads)


def log(message: str) -> None:
    print(f'[{time.strftime("%H:%M:%S")}] {message}', flush=True)


def save_npy(path: str, array: np.ndarray) -> None:
    """Writes ``array`` atomically (temporary file + rename), so concurrent readers never see a partial file."""
    tmp = f'{path}.tmp{os.getpid()}.npy'
    np.save(tmp, array)
    os.replace(tmp, path)


# == FEATURIZATION ==

def _parse_row(args):
    """Worker of ``featurize``: one dataset row -> (stereo-free canonical SMILES, values) or None."""
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.*')
    smiles, values = args
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    canonical = Chem.MolToSmiles(mol, isomericSmiles=False)
    # The same filter as the original BO experiment: valid, at least two atoms and one bond, one component.
    # HDF needs the last two (single atoms have no messages, the graph diameter needs a connected graph). The
    # filter applies to the stereo-free canonical SMILES as it is featurized later: without isotopes, the
    # deuterium atoms of e.g. heavy water ([2H]O[2H] -> [H]O[H]) become plain hydrogens, which RDKit drops.
    heavy = Chem.MolFromSmiles(canonical)
    if heavy is None or heavy.GetNumAtoms() < 2 or heavy.GetNumBonds() < 1 or len(Chem.GetMolFrags(heavy)) > 1:
        return None
    return canonical, values


_GENERATORS = {}


def _featurize_molecule(smiles: str):
    """Worker of ``featurize``: graph dict, Morgan bits of every size and ClogP of one canonical SMILES."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Crippen, rdFingerprintGenerator
    from graph_hdc.special.molecules import graph_dict_from_mol
    RDLogger.DisableLog('rdApp.*')
    if not _GENERATORS:
        # binary Morgan fingerprints without chirality, folded to ``size`` bits (as in ex_07)
        for size in SIZES + [PCA_SOURCE_BITS]:
            _GENERATORS[size] = rdFingerprintGenerator.GetMorganGenerator(radius=MORGAN_RADIUS, fpSize=size)
    mol = Chem.MolFromSmiles(smiles)
    graph = graph_dict_from_mol(mol, hydrogens='total')
    graph.pop('graph_labels', None)
    bits = {size: generator.GetFingerprintAsNumPy(mol).astype(np.uint8) for size, generator in _GENERATORS.items()}
    return graph, bits, Crippen.MolLogP(mol)


_MORGAN_GENERATORS = {}


def _morgan_bits(smiles: str, sizes: List[int]) -> Dict[int, np.ndarray]:
    """Worker of ``add_morgan``: the Morgan bits of ``sizes`` with the generator settings of ``featurize``."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog('rdApp.*')
    for size in sizes:
        if size not in _MORGAN_GENERATORS:
            _MORGAN_GENERATORS[size] = rdFingerprintGenerator.GetMorganGenerator(radius=MORGAN_RADIUS, fpSize=size)
    mol = Chem.MolFromSmiles(smiles)
    return {size: _MORGAN_GENERATORS[size].GetFingerprintAsNumPy(mol).astype(np.uint8) for size in sizes}


def add_morgan(library: str, sizes: List[int], workers: int) -> None:
    """
    Adds the Morgan fingerprints of the ``sizes`` that a featurized library does not have yet (D=256 since
    2026-10-13), from the stereo-free canonical SMILES in library.tsv, the strings that ``featurize`` computed its
    bits from. Graphs, PCA and objectives stay untouched, so finished campaigns stay valid. The explained variance
    of the PCA at the new sizes goes into meta.json.
    """
    folder = os.path.join(DATA, library)
    missing = [s for s in sizes if not os.path.exists(os.path.join(folder, f'morgan_{s}.npy'))]
    if not missing:
        log(f'{library}: Morgan bits of sizes {sizes} exist')
        return
    with open(os.path.join(folder, 'library.tsv')) as file:
        smiles_list = [line.split('\t')[1] for line in file.read().splitlines()[1:]]
    set_thread_env(1)
    with mp.get_context('spawn').Pool(workers) as pool:
        bits = pool.starmap(_morgan_bits, [(smiles, missing) for smiles in smiles_list], chunksize=500)
    for size in missing:
        save_npy(os.path.join(folder, f'morgan_{size}.npy'), np.stack([b[size] for b in bits]))
    meta = library_meta(library)
    explained = np.load(os.path.join(folder, 'pca_explained.npy'))
    for size in missing:
        if size <= meta['pca_components']:
            meta['pca_explained_by_size'][str(size)] = float(explained[:size].sum())
    meta['pca_explained_by_size'] = dict(sorted(meta['pca_explained_by_size'].items(), key=lambda kv: int(kv[0])))
    # atomic, as other stages read meta.json
    tmp = os.path.join(folder, f'meta.json.tmp{os.getpid()}')
    with open(tmp, 'w') as file:
        json.dump(meta, file, indent=2)
    os.replace(tmp, os.path.join(folder, 'meta.json'))
    log(f'{library}: Morgan bits of sizes {missing} for {len(smiles_list)} molecules')


def _parse_value(value) -> float:
    """Property value as float; NaN for missing, censored (">...") or non-numeric entries."""
    try:
        value = float(str(value).strip())
    except ValueError:
        return float('nan')
    return value if np.isfinite(value) else float('nan')


def featurize(library: str, workers: int) -> None:
    """
    Cleans one library and computes everything the BO campaigns need from it: graph dicts for HDF, Morgan bits
    of every size, the PCA of the 2048-bit Morgan fingerprints, property values, and for every objective on the
    library its objective values and the initial designs of all campaigns.
    """
    from chem_mat_data import load_smiles_dataset
    from sklearn.decomposition import PCA

    spec = LIBRARIES[library]
    folder = os.path.join(DATA, library)
    os.makedirs(folder, exist_ok=True)
    os.makedirs(os.path.join(DATA, 'objectives'), exist_ok=True)

    set_thread_env(1)
    df = load_smiles_dataset(library)
    log(f'{library}: {len(df)} rows')
    names = list(spec['properties'])
    rows = [(smiles, [_parse_value(df[column].iloc[i]) for column in spec['properties'].values()])
            for i, smiles in enumerate(df[spec['smiles']])]
    with mp.get_context('spawn').Pool(workers) as pool:
        parsed = pool.map(_parse_row, rows, chunksize=500)

    # Molecules that only differ in stereochemistry are identical for both HDF and Morgan without chirality:
    # one candidate per stereo-free canonical SMILES, with the mean of its (finite) values.
    groups: Dict[str, List[List[float]]] = {}
    for item in parsed:
        if item is not None:
            groups.setdefault(item[0], []).append(item[1])
    smiles_list = sorted(groups)
    values = np.full((len(smiles_list), len(names)), np.nan)
    for i, smiles in enumerate(smiles_list):
        group = np.array(groups[smiles], dtype=np.float64)  # (num_duplicates, num_properties)
        for j in range(len(names)):
            finite = group[:, j][np.isfinite(group[:, j])]
            if len(finite) > 0:
                values[i, j] = finite.mean()
    # every candidate needs a value for every objective of the library (BEET: the censored rows have none)
    keep = np.isfinite(values).all(axis=1)
    num_invalid = sum(item is None for item in parsed)
    log(f'{library}: {len(parsed) - num_invalid} valid rows -> {len(smiles_list)} structures without stereo -> '
        f'{int(keep.sum())} candidates with values ({num_invalid} rows filtered as invalid, multi-component or '
        f'smaller than two atoms)')
    smiles_list = [s for s, k in zip(smiles_list, keep) if k]
    values = values[keep]

    with mp.get_context('spawn').Pool(workers) as pool:
        featurized = pool.map(_featurize_molecule, smiles_list, chunksize=200)
    graphs = [f[0] for f in featurized]
    clogp = np.array([f[2] for f in featurized], dtype=np.float64)
    for size in SIZES + [PCA_SOURCE_BITS]:
        save_npy(os.path.join(folder, f'morgan_{size}.npy'), np.stack([f[1][size] for f in featurized]))
    with open(os.path.join(folder, 'graphs.pkl'), 'wb') as file:
        pickle.dump(graphs, file, protocol=pickle.HIGHEST_PROTOCOL)

    # PCA of the 2048-bit fingerprints, fitted once on the whole library (unsupervised). The components are
    # nested, so the first D columns are the D-dimensional projection for every size of the sweep. A library
    # with fewer molecules has fewer usable components (rank <= num_molecules - 1).
    bits = np.load(os.path.join(folder, f'morgan_{PCA_SOURCE_BITS}.npy')).astype(np.float32)
    num_components = min(PCA_COMPONENTS, len(smiles_list) - 1, PCA_SOURCE_BITS)
    pca = PCA(n_components=num_components, svd_solver='covariance_eigh' if len(bits) > PCA_SOURCE_BITS else 'full')
    scores = pca.fit_transform(bits).astype(np.float32)
    save_npy(os.path.join(folder, 'pca_scores.npy'), scores)
    save_npy(os.path.join(folder, 'pca_explained.npy'), pca.explained_variance_ratio_.astype(np.float64))

    with open(os.path.join(folder, 'library.tsv'), 'w') as file:
        file.write('\t'.join(['index', 'smiles', 'clogp'] + names) + '\n')
        for i, smiles in enumerate(smiles_list):
            file.write('\t'.join([str(i), smiles, f'{clogp[i]:.6g}'] + [f'{v:.6g}' for v in values[i]]) + '\n')

    meta = {
        'library': library,
        'num_rows': len(df),
        'num_candidates': len(smiles_list),
        'num_filtered': num_invalid,
        'max_graph_size': int(max(g['graph_size'] for g in graphs)),
        'num_atoms': int(sum(g['graph_size'] for g in graphs)),
        'max_graph_diameter': int(max(g['graph_diameter'] for g in graphs)),
        'pca_components': int(num_components),
        'pca_explained_by_size': {str(s): float(pca.explained_variance_ratio_[:s].sum())
                                  for s in SIZES if s <= num_components},
        'git': git_commit(),
        'time': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(os.path.join(folder, 'meta.json'), 'w') as file:
        json.dump(meta, file, indent=2)

    properties = dict(zip(names, values.T))
    properties['clogp'] = clogp
    write_objectives(library, smiles_list, properties)
    log(f'{library}: done, {json.dumps({k: meta[k] for k in ("max_graph_size", "max_graph_diameter")})}')


def zwitterion_mask(smiles_list: List[str]) -> np.ndarray:
    """
    True for every zwitterion or ion: a molecule with a formal charge that no opposite charge on a bonded atom
    compensates (e.g. an ammonium and a carboxylate group), or with a charged carbon atom. The latter are ylides
    such as [NH3+][c-] or [C-]-[NH+]=, 28 molecules of QM9, of which 16 have larger gas-phase dipoles than any
    neutral molecule (up to 12.2 D against 9.4 D). Charges that cancel within a bond between nitrogen and oxygen
    or nitrogen, as in nitro groups, are not masked: 139 nitro compounds and 7 aromatic N+-N- rings of QM9, none of
    them beyond the extremes of the neutral molecules (QM9: 1,698 masked of 133,795). The bonded pairs come from a
    maximum matching between the positively and the negatively charged atoms (Kuhn's augmenting paths), so the order of
    the bonds does not matter. Atoms with a charge other than +1 or -1 count as uncompensated.
    """
    from rdkit import Chem
    mask = np.zeros(len(smiles_list), dtype=bool)
    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        charges = {atom.GetIdx(): atom.GetFormalCharge() for atom in mol.GetAtoms() if atom.GetFormalCharge() != 0}
        if not charges:
            continue
        if any(abs(charge) != 1 or mol.GetAtomWithIdx(idx).GetSymbol() == 'C' for idx, charge in charges.items()):
            mask[i] = True
            continue
        negatives = {idx for idx, charge in charges.items() if charge < 0}
        neighbours = {idx: [n.GetIdx() for n in mol.GetAtomWithIdx(idx).GetNeighbors() if n.GetIdx() in negatives]
                      for idx, charge in charges.items() if charge > 0}
        partner = {}

        def augment(positive: int, seen: set) -> bool:
            for negative in neighbours[positive]:
                if negative not in seen:
                    seen.add(negative)
                    if negative not in partner or augment(partner[negative], seen):
                        partner[negative] = positive
                        return True
            return False

        matched = sum(augment(positive, set()) for positive in neighbours)
        mask[i] = 2 * matched < len(charges)
    return mask


def write_objectives(library: str, smiles_list: List[str], properties: Dict[str, np.ndarray],
                     only: Optional[List[str]] = None) -> None:
    """
    Writes ``data/objectives/<objective>.npz`` for the objectives of ``library`` (or the subset ``only``): the raw
    property values, the objective values (to be maximized), the mask of molecules that may be proposed, the
    initial designs of all campaigns (inside the mask) and the objective range over the mask (for the
    normalized regret).
    """
    from graph_hdc.bo import objective_values
    zwitterions = None
    for objective, ospec in OBJECTIVES.items():
        if ospec['library'] != library or (only is not None and objective not in only):
            continue
        prop = properties[ospec['property']]
        if ospec.get('transform') == 'log10':
            if (prop <= 0).any():
                raise ValueError(f'{objective}: log10 transform of non-positive values')
            prop = np.log10(prop)
        objective_array = objective_values(prop, ospec['mode'], ospec['target'])
        if ospec.get('exclude_zwitterions'):
            if zwitterions is None:
                zwitterions = zwitterion_mask(smiles_list)
            allowed = ~zwitterions
            population = np.flatnonzero(allowed)
        else:
            allowed = np.ones(len(smiles_list), dtype=bool)
            # an integer population (not the index array): keeps the initial designs of the first round exactly
            population = len(smiles_list)
        initial = np.stack([
            np.random.default_rng([24, zlib.crc32(objective.encode()), k]).choice(
                population, size=ospec['num_initial'], replace=False)
            for k in range(NUM_CAMPAIGNS)
        ])
        allowed_objective = objective_array[allowed]
        np.savez(os.path.join(DATA, 'objectives', f'{objective}.npz'),
                 values=properties[ospec['property']], transform=ospec.get('transform') or 'none',
                 objective=objective_array, initial=initial, allowed=allowed,
                 objective_max=allowed_objective.max(), objective_min=allowed_objective.min())
        best = int(np.flatnonzero(allowed)[allowed_objective.argmax()])
        log(f'{objective}: {int(allowed.sum())} of {len(smiles_list)} molecules are candidates, objective range '
            f'[{allowed_objective.min():.4g}, {allowed_objective.max():.4g}], best molecule {smiles_list[best]}')


def verify_objectives(objectives: List[str], backup: str) -> None:
    """
    Compares the rewritten objective files with the copies in ``backup`` (taken before the rewrite): values,
    objective values, mask, objective range and the initial designs of the campaigns that the backup holds must
    be identical, so that finished campaigns stay valid. Objectives whose candidate mask changed on purpose (the
    QM9 objectives, 2026-10-13) are only reported. Raises on any other difference.
    """
    problems = []
    for objective in objectives:
        old_path = os.path.join(backup, f'{objective}.npz')
        if not os.path.exists(old_path):
            continue
        old = np.load(old_path)
        new = np.load(os.path.join(DATA, 'objectives', f'{objective}.npz'))
        n = len(old['initial'])
        # files written before the QM9 mask (2026-10-13) have no mask: every molecule is a candidate (as in run_task)
        old_allowed = old['allowed'] if 'allowed' in old.files else np.ones(len(old['objective']), dtype=bool)
        same = {
            'values': np.array_equal(old['values'], new['values'], equal_nan=True),
            'objective': np.array_equal(old['objective'], new['objective']),
            'allowed': np.array_equal(old_allowed, new['allowed']),
            'range': float(old['objective_max']) == float(new['objective_max'])
            and float(old['objective_min']) == float(new['objective_min']),
            f'initial[:{n}]': np.array_equal(old['initial'], new['initial'][:n]),
        }
        differ = [name for name, ok in same.items() if not ok]
        if OBJECTIVES[objective].get('exclude_zwitterions'):
            log(f'{objective}: new candidate mask, differs from the backup in {differ or "nothing"}')
        elif differ:
            problems.append(f'{objective}: {differ}')
        else:
            log(f'{objective}: identical to the backup for the first {n} campaigns')
    if problems:
        raise RuntimeError('rewritten objective files differ from the backup: ' + '; '.join(problems))


def rewrite_objectives(objectives: List[str]) -> None:
    """
    Rewrites the objective files of ``objectives`` from the stored library (SMILES in library.tsv) and the raw
    property values in the existing objective files, without touching graphs or features, e.g. after a change
    of the candidate mask while features are in use.
    """
    for library in sorted({OBJECTIVES[o]['library'] for o in objectives}):
        with open(os.path.join(DATA, library, 'library.tsv')) as file:
            smiles_list = [line.split('\t')[1] for line in file.read().splitlines()[1:]]
        properties = {}
        for objective, ospec in OBJECTIVES.items():
            path = os.path.join(DATA, 'objectives', f'{objective}.npz')
            if ospec['library'] == library and os.path.exists(path):
                properties[ospec['property']] = np.load(path)['values']
        write_objectives(library, smiles_list, properties, only=objectives)


# == HDF ENCODING ==

_ENCODE_STATE = {}


def _encode_init(threads: int):
    import torch
    torch.set_num_threads(threads)


def _encode_load(library: str) -> None:
    """Loads the PyG graphs of ``library`` into the worker (replacing the previously loaded library)."""
    from graph_hdc.graph import data_list_from_graph_dicts
    if _ENCODE_STATE.get('library') == library:
        return
    _ENCODE_STATE.clear()
    with open(os.path.join(DATA, library, 'graphs.pkl'), 'rb') as file:
        graphs = pickle.load(file)
    with open(os.path.join(DATA, library, 'meta.json')) as file:
        meta = json.load(file)
    _ENCODE_STATE.update(library=library, data=data_list_from_graph_dicts(graphs), meta=meta)


def encode_hdf(data_list, size: int, seed: int, max_size: float, max_diameter: float,
               batch_size: int = 1024) -> np.ndarray:
    """
    HDF embeddings in the standard configuration (bidirectional message passing, unit-modulus codebooks, total
    hydrogen counts in the graph dicts, size and diameter encodings, depth 2) of the PyG ``data_list``.
    """
    import torch
    from torch_geometric.loader import DataLoader
    from graph_hdc.models import HyperNet
    from graph_hdc.special.molecules import make_molecule_graph_encoder_map_cont, make_molecule_node_encoder_map_cont

    net = HyperNet(
        hidden_dim=size,
        depth=HDF_DEPTH,
        node_encoder_map=make_molecule_node_encoder_map_cont(dim=size, seed=seed),
        graph_encoder_map=make_molecule_graph_encoder_map_cont(dim=size, seed=seed, max_graph_size=max_size,
                                                               max_graph_diameter=max_diameter),
        seed=seed,
        normalize_all=True,
        bidirectional=True,
    )
    embeddings = []
    with torch.no_grad():
        for batch in DataLoader(data_list, batch_size=batch_size, shuffle=False):
            embeddings.append(net.forward(batch)['graph_embedding'].to(torch.float32).cpu().numpy())
    return np.concatenate(embeddings)


def _encode_one(job):
    library, size, campaign = job
    _encode_load(library)
    meta = _ENCODE_STATE['meta']
    path = os.path.join(DATA, library, f'hdf_{size}_c{campaign}.npy')
    if os.path.exists(path):
        return path, 0.0
    start = time.perf_counter()
    embeddings = encode_hdf(_ENCODE_STATE['data'], size, hdf_seed(campaign), meta['max_graph_size'],
                            meta['max_graph_diameter'])
    if not np.isfinite(embeddings).all():
        raise RuntimeError(f'non-finite HDF embeddings for {library} D={size} campaign {campaign}')
    save_npy(path, embeddings)
    return path, time.perf_counter() - start


def encode(libraries: List[str], sizes: List[int], campaigns: List[int], workers: int, threads: int) -> None:
    """
    HDF embeddings of every (library, size, campaign) that is still missing, in one worker pool over all
    libraries, the most expensive encodings first (cost ~ number of atoms x size), so that the long ZINC250k
    encodings run alongside the many short ones instead of leaving workers idle.
    """
    jobs = []
    for library in libraries:
        meta = library_meta(library)
        num_atoms = meta.get('num_atoms', meta['num_candidates'] * 20)
        for size in sizes:
            for campaign in campaigns:
                if not os.path.exists(os.path.join(DATA, library, f'hdf_{size}_c{campaign}.npy')):
                    jobs.append((num_atoms * size, (library, size, campaign)))
    jobs = [job for _, job in sorted(jobs, key=lambda item: -item[0])]
    log(f'{len(jobs)} HDF encodings to compute with {workers} workers x {threads} threads')
    if not jobs:
        return
    set_thread_env(threads)
    # ProcessPoolExecutor (unlike multiprocessing.Pool) fails with BrokenProcessPool when a worker dies, e.g.
    # killed for memory, instead of waiting forever. A worker holds the PyG graphs of one library at a time
    # (about 5 GB for ZINC250k).
    with ProcessPoolExecutor(min(workers, len(jobs)), mp_context=mp.get_context('spawn'),
                             initializer=_encode_init, initargs=(threads, )) as executor:
        futures = {executor.submit(_encode_one, job): job for job in jobs}
        try:
            for i, future in enumerate(as_completed(futures), start=1):
                path, seconds = future.result()
                log(f'  [{i}/{len(jobs)}] {futures[future][0]} {os.path.basename(path)} in {seconds:.1f} s')
        except BaseException:
            # do not compute the remaining encodings when one failed (e.g. a full disk)
            executor.shutdown(wait=False, cancel_futures=True)
            raise


# == TASKS ==

def result_path(task: dict) -> str:
    size = 'na' if task['size'] is None else task['size']
    return os.path.join(RUNS, task['objective'], task['method'], f'D{size}', f'c{task["campaign"]}.npz')


def library_meta(library: str) -> dict:
    with open(os.path.join(DATA, library, 'meta.json')) as file:
        return json.load(file)


def estimated_cost(task: dict, num_molecules: int) -> float:
    """Rough relative cost of a campaign (for longest-first scheduling): GP fits plus library scans."""
    if task['method'] == 'random':
        return 0.001
    rounds = OBJECTIVES[task['objective']]['num_rounds']
    scan = num_molecules * task['size'] / 2e7
    if num_molecules > GPU_MIN_MOLECULES:
        scan /= 20.0  # GPU acquisition (only the order of the task list depends on this)
    return rounds * (0.5 + scan)


def make_tasks(objectives: Optional[List[str]] = None, sizes: Optional[List[int]] = None,
               campaigns: Optional[List[int]] = None) -> List[dict]:
    """All campaigns of the benchmark (or of a subset), the most expensive first."""
    tasks = []
    for objective, spec in OBJECTIVES.items():
        if objectives is not None and objective not in objectives:
            continue
        meta = library_meta(spec['library'])
        for method in methods_for(objective):
            for size in [None] if method == 'random' else (sizes or SIZES):
                # a PCA cannot have more components than the library allows (BEET: 127)
                if method == 'morgan_pca' and size > meta['pca_components']:
                    continue
                for campaign in (campaigns if campaigns is not None else range(NUM_CAMPAIGNS)):
                    task = dict(objective=objective, method=method, size=size, campaign=campaign)
                    task['cost'] = estimated_cost(task, meta['num_candidates'])
                    tasks.append(task)
    tasks.sort(key=lambda t: -t['cost'])
    return tasks


# == RUNNING CAMPAIGNS ==

_WORKER = {}


def _worker_init(threads: int, gpus: int, gpu_min_molecules: int = GPU_MIN_MOLECULES):
    import torch
    os.environ['OMP_NUM_THREADS'] = str(threads)
    torch.set_num_threads(threads)
    # worker processes are numbered 1, 2, ... in the order of their creation: spread them over the GPUs
    identity = mp.current_process()._identity
    _WORKER['gpu'] = (identity[0] - 1) % gpus if gpus > 0 and identity else None
    _WORKER['threads'] = threads
    _WORKER['gpu_min_molecules'] = gpu_min_molecules
    _WORKER['features'] = {}


def _warmup(_) -> int:
    """Imports the BO stack and fits one tiny GP, so that the workers are ready before a timed measurement."""
    import torch
    from graph_hdc.bo import fit_gp, make_gp
    x = torch.rand(10, 4, dtype=torch.float64)
    fit_gp(make_gp('hdf_matern', x, x[:, :1]))
    return os.getpid()


def resolve_gpus(gpus: int) -> int:
    """The number of GPUs to use: ``gpus``, or all visible ones if negative."""
    if gpus >= 0:
        return gpus
    import torch
    return torch.cuda.device_count()


def load_features(objective: str, method: str, size: Optional[int], campaign: int) -> Optional[np.ndarray]:
    """GP inputs of a campaign, in the form the method's kernel expects (see graph_hdc.bo)."""
    from graph_hdc.bo import center_scale
    library = OBJECTIVES[objective]['library']
    folder = os.path.join(DATA, library)
    if method == 'random':
        return None
    key = (library, method, size, campaign if method.startswith('hdf') else None)
    cache = _WORKER.setdefault('features', {})
    if key in cache:
        return cache[key]
    if method == 'hdf_matern':
        features = center_scale(np.load(os.path.join(folder, f'hdf_{size}_c{campaign}.npy')))
    elif method == 'hdf_linear':
        features = np.load(os.path.join(folder, f'hdf_{size}_c{campaign}.npy'))
        features = (features / np.linalg.norm(features, axis=1, keepdims=True)).astype(np.float32)
    elif method == 'morgan_tanimoto':
        features = np.load(os.path.join(folder, f'morgan_{size}.npy')).astype(np.float32)
    elif method == 'morgan_pca':
        features = center_scale(np.load(os.path.join(folder, 'pca_scores.npy'), mmap_mode='r')[:, :size])
    else:
        raise ValueError(method)
    # keep only the deterministic (campaign-independent) features of the current library; HDF features change
    # with every campaign and are never reused
    if not method.startswith('hdf'):
        cache.clear()
        cache[key] = features
    return features


def run_task(task: dict) -> dict:
    import botorch
    import gpytorch
    import torch
    from graph_hdc.bo import run_campaign
    path = result_path(task)
    if os.path.exists(path):
        return {**task, 'status': 'exists'}
    spec = OBJECTIVES[task['objective']]
    data = np.load(os.path.join(DATA, 'objectives', f'{task["objective"]}.npz'))
    num_molecules = len(data['objective'])
    gpu = _WORKER.get('gpu')
    use_gpu = gpu is not None and num_molecules > _WORKER.get('gpu_min_molecules', GPU_MIN_MOLECULES)
    device = f'cuda:{gpu}' if use_gpu else 'cpu'
    start = time.time()
    features = load_features(task['objective'], task['method'], task['size'], task['campaign'])
    time_features = time.time() - start
    allowed = data['allowed'] if 'allowed' in data.files else None
    oom_retries = 0
    while True:
        try:
            result = run_campaign(
                method=task['method'],
                objective=data['objective'],
                initial=data['initial'][task['campaign']],
                num_rounds=spec['num_rounds'],
                seed=campaign_seed(task['campaign']),
                features=features,
                device=device,
                allowed=allowed,
            )
            break
        except torch.OutOfMemoryError:
            # Several workers share each GPU, and the largest libraries at the largest sizes (about 1 GB of
            # candidates each) can exhaust its memory when they run at the same time. Free the cached memory,
            # wait for the others and try again; the last attempt runs on the CPU. The campaign is seeded, so a
            # retry repeats it exactly (up to the device).
            oom_retries += 1
            torch.cuda.empty_cache()
            if oom_retries > 3:
                raise
            if oom_retries == 3:
                device = 'cpu'
            time.sleep(30 * oom_retries)
    meta = {
        **{k: task[k] for k in ('objective', 'method', 'size', 'campaign')},
        'hdf_seed': hdf_seed(task['campaign']) if task['method'].startswith('hdf') else None,
        'campaign_seed': campaign_seed(task['campaign']),
        'num_initial': spec['num_initial'], 'num_rounds': spec['num_rounds'],
        'num_molecules': num_molecules,
        'num_candidates': int(num_molecules if allowed is None else allowed.sum()),
        'objective_max': float(data['objective_max']), 'objective_min': float(data['objective_min']),
        'device': device, 'oom_retries': oom_retries, 'threads': _WORKER.get('threads'),
        'host': socket.gethostname(),
        'git': _WORKER.setdefault('git', git_commit()), 'torch': torch.__version__,
        'botorch': botorch.__version__, 'gpytorch': gpytorch.__version__,
        'seconds_features': time_features, 'seconds_total': time.time() - start,
        'finished': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f'{path}.tmp{os.getpid()}.npz'
    np.savez(tmp, meta=json.dumps(meta), **result)
    os.replace(tmp, path)
    return {**task, 'status': 'done', 'seconds': meta['seconds_total'], 'device': device}


def run(workers: int, threads: int, gpus: int, shard: int, num_shards: int, limit: Optional[int] = None,
        gpu_min_molecules: int = GPU_MIN_MOLECULES, tasks_file: Optional[str] = None) -> int:
    """
    Runs the open campaigns of one shard of the task list (``tasks_file``, default ``tasks.jsonl``); returns the
    number of failed campaigns.
    """
    gpus = resolve_gpus(gpus)
    with open(tasks_file or os.path.join(ROOT, 'tasks.jsonl')) as file:
        tasks = [json.loads(line) for line in file]
    # round-robin over the cost-sorted list gives every shard the same mix of long and short campaigns
    tasks = [t for i, t in enumerate(tasks) if i % num_shards == shard]
    tasks = [t for t in tasks if not os.path.exists(result_path(t))]
    if limit is not None:
        tasks = tasks[:limit]
    # Alternate expensive and cheap campaigns (the list is sorted by cost). Starting with only the largest
    # libraries at the largest sizes would put about 1 GB of candidates per worker on every GPU at once; the mix
    # keeps the GPU memory and the load of cores and GPUs even.
    tasks = [tasks[i // 2] if i % 2 == 0 else tasks[len(tasks) - 1 - i // 2] for i in range(len(tasks))]
    log(f'shard {shard}/{num_shards}: {len(tasks)} open campaigns, {workers} workers x {threads} threads, '
        f'{gpus} GPUs, host {socket.gethostname()}, code {git_commit()}')
    if not tasks:
        return 0
    set_thread_env(threads)
    failed = 0
    start = time.time()
    # The executor hands out the tasks in the submitted (longest-first) order. A worker that dies (e.g. killed
    # for memory) breaks the executor with BrokenProcessPool and ends the job instead of hanging it; finished
    # campaigns are kept, and resubmitting the stage resumes the rest.
    with ProcessPoolExecutor(workers, mp_context=mp.get_context('spawn'), initializer=_worker_init,
                             initargs=(threads, gpus, gpu_min_molecules)) as executor:
        futures = [executor.submit(_run_task_safe, task) for task in tasks]
        for i, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            if result['status'] == 'error':
                failed += 1
                log(f'  !!! {result["objective"]} {result["method"]} D={result["size"]} c{result["campaign"]}: '
                    f'{result["error"]}')
            if i % 50 == 0 or i == len(tasks):
                log(f'  {i}/{len(tasks)} campaigns, {failed} failed, {(time.time() - start) / 60:.1f} min')
    log(f'shard {shard}: finished, {failed} failed')
    return failed


def _run_task_safe(task: dict) -> dict:
    import gc
    import traceback
    import torch
    try:
        return run_task(task)
    except Exception as error:
        return {**task, 'status': 'error', 'error': f'{error!r}\n{traceback.format_exc()}'}
    finally:
        # Give the GPU memory of the finished campaign back to the device. Several workers share each GPU, and
        # memory that PyTorch keeps cached for finished campaigns would otherwise fill it up, so that the next
        # large campaign of any worker fails with CUDA out-of-memory.
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            gc.collect()
            torch.cuda.empty_cache()


# == CALIBRATION ==

# (objective, size, device) of the single-worker timings: ZINC250k at every size on the GPU and at one size on the
# CPU, AqSolDB on the CPU and on the GPU (device "cuda" falls back to the CPU without GPUs)
CALIBRATION_SINGLE = [('solubility', 8, 'cpu'), ('solubility', 64, 'cpu'), ('solubility', 512, 'cpu'),
                      ('qed', 64, 'cpu'), ('solubility', 64, 'cuda'), ('solubility', 512, 'cuda'),
                      ('qed', 8, 'cuda'), ('qed', 64, 'cuda'), ('qed', 512, 'cuda')]
# (objective, method, size, device) of the throughput mix: small libraries on the CPU, ZINC250k on the GPU
CALIBRATION_MIX = [('solubility', 'hdf_matern', 64, 'cpu'), ('lipophilicity', 'morgan_tanimoto', 512, 'cpu'),
                   ('freesolv', 'morgan_pca', 128, 'cpu'), ('qed', 'morgan_tanimoto', 64, 'cuda')]

def _calibrate_task(job):
    """Times ``rounds`` BO rounds of one configuration at the full-budget number of observations."""
    import torch
    from graph_hdc.bo import run_campaign
    objective, method, size, rounds, device = job
    data = np.load(os.path.join(DATA, 'objectives', f'{objective}.npz'))
    spec = OBJECTIVES[objective]
    # start with the number of observations of the last rounds of a real campaign
    num_observed = spec['num_initial'] + spec['num_rounds'] - rounds
    initial = np.random.default_rng(0).choice(len(data['objective']), size=num_observed, replace=False)
    features = load_features(objective, method, size, 0)
    if device != 'cpu' and _WORKER.get('gpu') is not None:
        device = f'cuda:{_WORKER["gpu"]}'
    elif device != 'cpu':
        device = 'cpu'
    start = time.perf_counter()
    result = run_campaign(method, data['objective'], initial, rounds, seed=0, features=features, device=device)
    return {'objective': objective, 'method': method, 'size': size, 'device': device,
            'threads': _WORKER['threads'], 'seconds': time.perf_counter() - start,
            'fit': float(np.median(result['fit_seconds'])), 'acq': float(np.median(result['acq_seconds']))}


def calibrate(workers_list: List[int], threads_list: List[int], gpus: int, rounds: int,
              throughput_rounds: int) -> None:
    """
    Two measurements for planning the full run on a node: (1) single-worker seconds per round of every GP
    method on the largest and a mid-sized library at several sizes and thread counts (CPU and GPU acquisition),
    and (2) the throughput of the node (BO rounds per second) with W concurrent workers on a representative mix
    of campaigns, measured after all workers have started and imported the BO stack.
    """
    gpus = resolve_gpus(gpus)
    cores = len(os.sched_getaffinity(0))
    report = {'host': socket.gethostname(), 'code': git_commit(), 'cores': cores, 'gpus': gpus,
              'single': [], 'throughput': []}
    context = mp.get_context('spawn')

    # (1) single-worker timings
    configs = [c for c in CALIBRATION_SINGLE if c[2] == 'cpu' or gpus > 0]
    for threads in threads_list:
        jobs = [(objective, method, size, rounds, device)
                for method in ('hdf_matern', 'morgan_tanimoto', 'morgan_pca') for objective, size, device in configs]
        set_thread_env(threads)
        with ProcessPoolExecutor(1, mp_context=context, initializer=_worker_init,
                                 initargs=(threads, gpus, 0)) as executor:
            executor.submit(_warmup, 0).result()
            for job in jobs:
                timing = executor.submit(_calibrate_task, job).result()
                report['single'].append(timing)
                log(f'  single: {json.dumps(timing)}')

    # (2) throughput with W concurrent workers: every configuration runs the same mix of small-library campaigns
    # (CPU) and ZINC campaigns (GPU acquisition), two per worker
    mix = [(objective, method, size, throughput_rounds, device) for objective, method, size, device in CALIBRATION_MIX]
    for workers in workers_list:
        for threads in threads_list:
            if workers * threads > cores:
                continue
            jobs = [mix[i % len(mix)] for i in range(2 * workers)]
            set_thread_env(threads)
            with ProcessPoolExecutor(workers, mp_context=context, initializer=_worker_init,
                                     initargs=(threads, gpus, 0)) as executor:
                list(executor.map(_warmup, range(workers)))
                start = time.perf_counter()
                timings = list(executor.map(_calibrate_task, jobs))
                seconds = time.perf_counter() - start
            entry = {'workers': workers, 'threads': threads, 'jobs': len(jobs), 'seconds': seconds,
                     'rounds_per_second': len(jobs) * throughput_rounds / seconds,
                     'mean_job_seconds': float(np.mean([t['seconds'] for t in timings])),
                     'mean_job_seconds_by_objective': {o: float(np.mean([t['seconds'] for t in timings
                                                                          if t['objective'] == o]))
                                                       for o in sorted({t['objective'] for t in timings})}}
            report['throughput'].append(entry)
            log(f'  throughput: {json.dumps(entry)}')

    os.makedirs(ROOT, exist_ok=True)
    path = os.path.join(ROOT, f'calibration_{socket.gethostname()}_{time.strftime("%Y%m%d_%H%M%S")}.json')
    with open(path, 'w') as file:
        json.dump(report, file, indent=2)
    log(f'calibration report: {path}')


# == STATUS & COLLECTION ==

def status() -> None:
    with open(os.path.join(ROOT, 'tasks.jsonl')) as file:
        tasks = [json.loads(line) for line in file]
    counts: Dict[tuple, List[int]] = {}
    for task in tasks:
        key = (task['objective'], task['method'])
        done, total = counts.get(key, [0, 0])
        counts[key] = [done + os.path.exists(result_path(task)), total + 1]
    for (objective, method), (done, total) in sorted(counts.items()):
        print(f'{objective:15s} {method:16s} {done:5d}/{total}')
    print(f'total: {sum(c[0] for c in counts.values())}/{len(tasks)}')


def collect() -> None:
    """
    One row per finished campaign with its metrics and timings: the normalized-regret AUC, the final regret,
    whether the regret reached 0.05 ("hit") and the number of BO rounds after the initial design until it did
    ("first_hit", 0 = already in the initial design, -1 = never).
    """
    from graph_hdc.bo import normalized_regret, regret_auc
    with open(os.path.join(ROOT, 'tasks.jsonl')) as file:
        tasks = [json.loads(line) for line in file]
    objective_files = {}
    rows = []
    for task in tasks:
        path = result_path(task)
        if not os.path.exists(path):
            continue
        result = np.load(path)
        meta = json.loads(str(result['meta']))
        # a result must belong to the current objective file: same initial design and candidate count (catches
        # results that survived a change of the candidate mask, e.g. the QM9 mask of 2026-10-13)
        if task['objective'] not in objective_files:
            objective_files[task['objective']] = np.load(
                os.path.join(DATA, 'objectives', f'{task["objective"]}.npz'))
        current = objective_files[task['objective']]
        num_candidates = int(current['allowed'].sum()) if 'allowed' in current.files else len(current['objective'])
        initial = current['initial'][task['campaign']]
        if (not np.array_equal(result['observed'][:len(initial)], initial)
                or meta.get('num_candidates', meta['num_molecules']) != num_candidates):
            raise RuntimeError(f'{path} does not match the current objective file (stale result?)')
        regret = normalized_regret(result['best'], meta['objective_max'], meta['objective_min'])
        rows.append({
            **{k: meta[k] for k in ('objective', 'method', 'size', 'campaign')},
            'auc': regret_auc(regret), 'final_regret': float(regret[-1]), 'hit': bool(regret.min() <= 0.05),
            'first_hit': int(np.argmax(regret <= 0.05)) if (regret <= 0.05).any() else -1,
            'fit_seconds': float(result['fit_seconds'].sum()), 'acq_seconds': float(result['acq_seconds'].sum()),
            'fit_failures': int(result['fit_failures']), 'device': meta['device'],
            'seconds_total': meta['seconds_total'], 'git': meta['git'],
        })
    if not rows:
        log(f'no finished campaigns among {len(tasks)}')
        return
    path = os.path.join(ROOT, 'campaigns.tsv')
    with open(path, 'w') as file:
        file.write('\t'.join(rows[0]) + '\n')
        for row in rows:
            file.write('\t'.join(str(v) for v in row.values()) + '\n')
    log(f'{len(rows)}/{len(tasks)} campaigns -> {path}')


# == CLEANUP ==

def cleanup(campaigns: List[int], dry_run: bool = False) -> int:
    """
    Deletes the HDF encodings of ``campaigns`` (every library and size) whose campaigns all have their result
    files. The encodings take most of the disk space of the benchmark (about 3.4 GB per campaign over all
    libraries and sizes, mostly ZINC250k and QM9) and are reproducible from their seeds, so the rerun of
    2026-10-13 encodes and runs the campaigns in waves and removes the encodings of each wave afterwards (user).
    An encoding that an unfinished campaign still needs is kept. Returns the number of kept encodings.
    """
    import glob
    libraries = [library for library in LIBRARIES if os.path.exists(os.path.join(DATA, library, 'meta.json'))]
    objectives = [objective for objective, spec in OBJECTIVES.items() if spec['library'] in libraries]
    needed: Dict[tuple, List[str]] = {}
    for task in make_tasks(objectives=objectives, campaigns=campaigns):
        if task['method'] in ('hdf_matern', 'hdf_linear'):
            key = (OBJECTIVES[task['objective']]['library'], task['size'], task['campaign'])
            needed.setdefault(key, []).append(result_path(task))
    kept, removed, freed = 0, 0, 0
    for library in libraries:
        for campaign in campaigns:
            for path in sorted(glob.glob(os.path.join(DATA, library, f'hdf_*_c{campaign}.npy'))):
                size = int(os.path.basename(path).split('_')[1])
                missing = [p for p in needed.get((library, size, campaign), []) if not os.path.exists(p)]
                if missing:
                    kept += 1
                    log(f'  keep {library}/{os.path.basename(path)}: {len(missing)} campaigns without result, '
                        f'e.g. {missing[0]}')
                    continue
                removed += 1
                freed += os.path.getsize(path)
                if not dry_run:
                    os.remove(path)
    log(f'{"would remove" if dry_run else "removed"} {removed} encodings ({freed / 1e9:.1f} GB), kept {kept}')
    return kept


# == COMMAND LINE ==

def parse_campaigns(text: str) -> List[int]:
    if '-' in text:
        first, last = text.split('-')
        return list(range(int(first), int(last) + 1))
    return [int(c) for c in text.split(',')]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    sub = parser.add_subparsers(dest='stage', required=True)
    p = sub.add_parser('featurize')
    p.add_argument('--libraries', nargs='+', default=list(LIBRARIES))
    p.add_argument('--workers', type=int, default=8)
    p = sub.add_parser('morgan')
    p.add_argument('--libraries', nargs='+', default=list(LIBRARIES))
    p.add_argument('--sizes', nargs='+', type=int, default=SIZES)
    p.add_argument('--workers', type=int, default=8)
    p = sub.add_parser('encode')
    p.add_argument('--libraries', nargs='+', default=list(LIBRARIES))
    p.add_argument('--sizes', nargs='+', type=int, default=SIZES)
    p.add_argument('--campaigns', default=f'0-{NUM_CAMPAIGNS - 1}')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--threads', type=int, default=1)
    p = sub.add_parser('objectives')
    p.add_argument('--objectives', nargs='+', required=True)
    p.add_argument('--verify-against', default=None,
                   help='folder with copies of the objective files from before the rewrite (see verify_objectives)')
    p = sub.add_parser('tasks')
    p.add_argument('--objectives', nargs='+', default=None)
    p.add_argument('--sizes', nargs='+', type=int, default=None)
    p.add_argument('--campaigns', default=None)
    p.add_argument('--output', default=None, help='task file, default _ex24/tasks.jsonl (status and collect read it)')
    p.add_argument('--open-only', action='store_true', help='only campaigns without a result file')
    p = sub.add_parser('run')
    p.add_argument('--workers', type=int, required=True)
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--gpus', type=int, default=-1, help='number of GPUs, -1 = all visible')
    p.add_argument('--gpu-min-molecules', type=int, default=GPU_MIN_MOLECULES,
                   help='libraries larger than this run their acquisition on a GPU')
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--num-shards', type=int, default=1)
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--tasks-file', default=None, help='task file, default _ex24/tasks.jsonl')
    p = sub.add_parser('calibrate')
    p.add_argument('--workers-list', nargs='+', type=int, default=[16, 32, 64])
    p.add_argument('--threads-list', nargs='+', type=int, default=[1, 2])
    p.add_argument('--gpus', type=int, default=-1, help='number of GPUs, -1 = all visible')
    p.add_argument('--rounds', type=int, default=5)
    p.add_argument('--throughput-rounds', type=int, default=20)
    sub.add_parser('status')
    sub.add_parser('collect')
    p = sub.add_parser('cleanup')
    p.add_argument('--campaigns', required=True)
    p.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)

    if args.stage == 'featurize':
        for library in args.libraries:
            featurize(library, args.workers)
    elif args.stage == 'morgan':
        for library in args.libraries:
            add_morgan(library, args.sizes, args.workers)
    elif args.stage == 'encode':
        encode(args.libraries, args.sizes, parse_campaigns(args.campaigns), args.workers, args.threads)
    elif args.stage == 'objectives':
        rewrite_objectives(args.objectives)
        if args.verify_against:
            verify_objectives(args.objectives, args.verify_against)
    elif args.stage == 'tasks':
        tasks = make_tasks(args.objectives, args.sizes,
                           parse_campaigns(args.campaigns) if args.campaigns else None)
        if args.open_only:
            tasks = [task for task in tasks if not os.path.exists(result_path(task))]
        os.makedirs(ROOT, exist_ok=True)
        output = args.output or os.path.join(ROOT, 'tasks.jsonl')
        with open(output, 'w') as file:
            for task in tasks:
                file.write(json.dumps(task) + '\n')
        log(f'{len(tasks)} campaigns -> {output}')
    elif args.stage == 'run':
        failed = run(args.workers, args.threads, args.gpus, args.shard, args.num_shards, args.limit,
                     args.gpu_min_molecules, args.tasks_file)
        return 1 if failed else 0
    elif args.stage == 'calibrate':
        calibrate(args.workers_list, args.threads_list, args.gpus, args.rounds, args.throughput_rounds)
    elif args.stage == 'status':
        status()
    elif args.stage == 'collect':
        collect()
    elif args.stage == 'cleanup':
        return 1 if cleanup(parse_campaigns(args.campaigns), args.dry_run) else 0
    return 0


if __name__ == '__main__':
    sys.exit(main())
