"""
Prepares the elution order benchmark of Baimacheva et al. (J Cheminform 17:137, 2025) for experiment ex_21.

The data are the enantiomer pairs of the CMRT dataset (Xu et al., Nat Commun 14:3095, 2023) that were separated on
the chiral HPLC column Chiralpak AD-H, as curated by Baimacheva et al.: 3,858 molecules in 1,929 enantiomer pairs, each
labeled as eluted first (F) or last (L), with the authors' training/test split by pair (9:1). Their file
RF_models/class_all.csv (repository jairesdesousa/chiraldlsv) is downloaded, checked (two enantiomers per pair, one F
and one L, both in the same split, parsable SMILES) and written as

    data/cmrt_elution/cmrt_adh.csv    columns: smiles, smiles_nostereo, first_eluted (1/0), split (train/test), r_s

where smiles_nostereo identifies the pair: the SMILES without stereochemistry, followed by a running number if one
constitution forms several enantiomer pairs.

    python prepare_cmrt_elution.py [class_all.csv]    # default: download into data/cmrt_elution/raw
"""
import os
import sys
import urllib.request

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')

PATH = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(PATH, '..', '..', '..'))
OUT = os.path.join(REPO, 'data', 'cmrt_elution')
URL = 'https://raw.githubusercontent.com/jairesdesousa/chiraldlsv/main/RF_models/class_all.csv'


def main(path: str):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f'downloading {URL}...')
        request = urllib.request.Request(URL, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(request, timeout=300) as response, open(path, 'wb') as f:
            f.write(response.read())
    df = pd.read_csv(path, index_col=0)
    for smiles in list(df['SMILES']) + list(df['SMILES_opp']):
        if Chem.MolFromSmiles(smiles) is None:
            sys.exit(f'unparsable SMILES {smiles}')

    # A pair consists of a molecule and its mirror image (SMILES_opp). One constitution can form several pairs (here:
    # the E and the Z isomer of a molecule with one stereocenter), so the stereo-free SMILES receives a running
    # number in that case, such that it identifies the pair.
    df['pair'] = [' | '.join(sorted((Chem.CanonSmiles(a), Chem.CanonSmiles(b))))
                  for a, b in zip(df['SMILES'], df['SMILES_opp'])]
    pairs_of = df.groupby('SMILES_ns')['pair'].unique()
    number = {pair: (ns if len(pairs) == 1 else f'{ns}#{k}') for ns, pairs in pairs_of.items()
              for k, pair in enumerate(sorted(pairs), start=1)}
    df['pair_id'] = df['pair'].map(number)
    print(f'{len(df)} molecules, {df["pair_id"].nunique()} pairs, {df["SMILES_ns"].nunique()} constitutions')

    # checks: two enantiomers per pair, one eluted first and one last, both in the same split
    members = df.groupby('pair_id')
    if not (members.size() == 2).all():
        sys.exit('not every pair consists of exactly two molecules')
    if not (members['F-L_class'].agg(lambda s: sorted(s) == ['F', 'L'])).all():
        sys.exit('not every pair has one first and one last eluted enantiomer')
    if not (members['TR/TE'].nunique() == 1).all():
        sys.exit('the two enantiomers of a pair are in different splits')

    table = pd.DataFrame({
        'smiles': df['SMILES'],
        'smiles_nostereo': df['pair_id'],
        'first_eluted': (df['F-L_class'] == 'F').astype(int),
        'split': df['TR/TE'].map({'TR': 'train', 'TE': 'test'}),
        'r_s': df['R-S_class'],
    })
    pairs = table.groupby('split')['smiles_nostereo'].nunique().to_dict()
    r_first = (table.loc[table['first_eluted'] == 1, 'r_s'] == 'R').mean()
    print(f'pairs per split {pairs}; the R enantiomer elutes first in {r_first:.1%} of the pairs')
    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, 'cmrt_adh.csv')
    table.to_csv(out, index=False)
    print(f'wrote {out}')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(OUT, 'raw', 'class_all.csv'))
