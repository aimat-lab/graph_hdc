"""
Prepares the enantiomer docking benchmark of Adams et al. (ChIRo, ICLR 2022) for experiment ex_21.

The benchmark consists of pairs of small enantiomers with one stereocenter, each docked three times into the chiral
pocket of PDB 4JBV with AutoDock Vina. A pair was kept if the best docking scores of its enantiomers differ by at least
0.3 kcal/mol. The authors provide train/validation/test splits (70/15/15, the two enantiomers of a pair always in the
same split) with one row per conformer; every conformer of an enantiomer carries the best ("top") docking score.

This script downloads the three split files from the authors' figshare item (the same links as in the ChiENN code),
checks with pickletools, without unpickling, that they only reference pandas, numpy and RDKit objects, loads them with
an unpickler restricted to exactly these classes, keeps one row per enantiomer and writes

    data/chiro_docking/chiro_docking.csv    columns: smiles (isomeric SMILES), smiles_nostereo, top_score, split

    python prepare_chiro_docking.py [raw_folder]    # default raw folder: data/chiro_docking/raw
"""
import os
import sys
import pickle
import pickletools
import urllib.request
import warnings

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')

PATH = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(PATH, '..', '..', '..'))
OUT = os.path.join(REPO, 'data', 'chiro_docking')

URLS = {
    'train': 'https://ndownloader.figshare.com/files/30975697?private_link=e23be65a884ce7fc8543',
    'val': 'https://ndownloader.figshare.com/files/30975706?private_link=e23be65a884ce7fc8543',
    'test': 'https://ndownloader.figshare.com/files/30975682?private_link=e23be65a884ce7fc8543',
}

# The globals a pandas DataFrame pickle with an RDKit Mol column references.
ALLOWED = {
    ('pandas.core.frame', 'DataFrame'), ('pandas.core.indexes.base', '_new_Index'),
    ('pandas.core.indexes.base', 'Index'), ('pandas.core.indexes.range', 'RangeIndex'),
    ('pandas.core.internals.managers', 'BlockManager'), ('numpy', 'ndarray'), ('numpy', 'dtype'),
    ('numpy.core.multiarray', '_reconstruct'), ('numpy.core.numeric', '_frombuffer'),
    ('builtins', 'slice'), ('rdkit.Chem.rdchem', 'Mol'),
}
ALLOWED_MODULES = {module for module, _ in ALLOWED}


class RestrictedUnpickler(pickle.Unpickler):
    """Unpickler that refuses every global outside ALLOWED."""

    def find_class(self, module, name):
        if (module, name) not in ALLOWED:
            raise pickle.UnpicklingError(f'refusing to load {module}.{name}')
        return super().find_class(module, name)


def referenced_modules(path: str) -> set:
    """All string arguments of the pickle that look like module paths of a GLOBAL / STACK_GLOBAL, found without
    unpickling. Every module must be one of ALLOWED_MODULES."""
    modules = set()
    with open(path, 'rb') as f:
        for opcode, arg, _ in pickletools.genops(f):
            if opcode.name == 'GLOBAL':
                modules.add(arg.split(' ', 1)[0])
            elif opcode.name in ('SHORT_BINUNICODE', 'BINUNICODE', 'BINUNICODE8', 'UNICODE') and \
                    isinstance(arg, str) and arg.count('.') >= 1 and ' ' not in arg and \
                    arg.split('.')[0] in ('pandas', 'numpy', 'rdkit', 'builtins', 'os', 'subprocess', 'posix',
                                          'nt', 'sys', 'shutil', 'socket', 'importlib', 'runpy', 'pickle'):
                modules.add(arg)
    return modules


def download(url: str, path: str):
    request = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(request, timeout=600) as response, open(path, 'wb') as f:
        f.write(response.read())


def main(raw: str):
    os.makedirs(raw, exist_ok=True)
    frames = []
    for split, url in URLS.items():
        path = os.path.join(raw, f'{split}.pickle')
        if not os.path.exists(path):
            print(f'downloading {split} split...')
            download(url, path)
        unexpected = {m for m in referenced_modules(path) if m not in ALLOWED_MODULES}
        if unexpected:
            sys.exit(f'{path} references unexpected modules {sorted(unexpected)}, not loading it')
        with open(path, 'rb') as f, warnings.catch_warnings():
            warnings.simplefilter('ignore')
            df = RestrictedUnpickler(f).load()
        print(f'{split}: {len(df)} conformers, {df["ID"].nunique()} enantiomers, '
              f'{df["SMILES_nostereo"].nunique()} pairs')
        # one row per enantiomer; all conformers of an enantiomer carry the same top score
        scores = df.groupby('ID')['top_score'].nunique()
        if (scores > 1).any():
            sys.exit(f'{split}: {int((scores > 1).sum())} enantiomers with differing top scores')
        rows = df.drop_duplicates(subset='ID')[['ID', 'SMILES_nostereo', 'top_score']].copy()
        rows.columns = ['smiles', 'smiles_nostereo', 'top_score']
        rows['split'] = split
        frames.append(rows)
    table = pd.concat(frames, ignore_index=True)

    # checks: complete pairs within one split, score margins, parsable SMILES with one specified stereocenter
    members = table.groupby('smiles_nostereo')
    if not (members.size() == 2).all() or not (members['split'].nunique() == 1).all():
        sys.exit('not every pair consists of exactly two enantiomers in the same split')
    margin = members['top_score'].agg(lambda s: abs(s.iloc[0] - s.iloc[1]))
    centers = []
    for smiles in table['smiles']:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            sys.exit(f'unparsable SMILES {smiles}')
        centers.append(len(Chem.FindMolChiralCenters(mol, includeUnassigned=False, useLegacyImplementation=False)))
    table['num_specified_centers'] = centers
    print(f'{len(table)} enantiomers in {table["smiles_nostereo"].nunique()} pairs; score margin within pairs: '
          f'min {margin.min():.2f}, median {margin.median():.2f} kcal/mol; specified stereocenters per molecule: '
          f'{table["num_specified_centers"].value_counts().to_dict()}')

    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, 'chiro_docking.csv')
    table.drop(columns='num_specified_centers').to_csv(out, index=False)
    print(f'wrote {out}')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(OUT, 'raw'))
