"""
Unit tests for the GNN baselines (:mod:`graph_hdc.baselines.gnn`): the HDF-matched node features (in particular
the hydrogen counts, which must match what the HDF encoder sees) and the optional cosine learning-rate schedule.
"""
import numpy as np
import pytest
import torch
import pytorch_lightning as pl
from rdkit import Chem
from torch_geometric.loader import DataLoader

from graph_hdc.baselines.gnn import (
    GNN_CLASSES, HDF_ATOMS, HDF_MAX_DEGREE, HDF_MAX_HYDROGENS, build_pyg_list, hdf_matched_graph,
)
from graph_hdc.special.molecules import graph_dict_from_mol


def hydrogen_counts(smiles: str, hydrogens: str) -> list:
    """Per-atom hydrogen count encoded in the last one-hot block of the HDF-matched node features."""
    graph = hdf_matched_graph({'graph_repr': smiles}, hydrogens=hydrogens)
    block = graph['node_attributes'][:, len(HDF_ATOMS) + 1 + HDF_MAX_DEGREE:]
    assert block.shape[1] == HDF_MAX_HYDROGENS
    return block.argmax(axis=1).tolist()


@pytest.mark.parametrize('smiles,implicit,total', [
    ('CCO', [3, 2, 1], [3, 2, 1]),                     # no bracket atoms: both modes agree
    ('c1cc[nH]c1', [1, 1, 1, 0, 1], [1, 1, 1, 1, 1]),  # pyrrole N-H written in brackets
    ('C[NH3+]', [3, 0], [3, 3]),                        # charged bracket atom
    ('C[C@@H](O)Cl', [3, 0, 1, 0], [3, 1, 1, 0]),       # stereo centre written in brackets
])
def test_hydrogen_counts(smiles, implicit, total):
    assert hydrogen_counts(smiles, 'implicit') == implicit
    assert hydrogen_counts(smiles, 'total') == total


@pytest.mark.parametrize('smiles', ['c1cc[nH]c1', 'C[NH3+]', 'C[C@@H](O)Cl', 'O=C([O-])c1ccc[nH]1', 'CCO'])
@pytest.mark.parametrize('hydrogens', ['implicit', 'total'])
def test_hydrogen_counts_match_the_hdf_encoder(smiles, hydrogens):
    """The GNNs must see exactly the hydrogen counts that the HDF encoder encodes (node_valences)."""
    mol = Chem.MolFromSmiles(smiles)
    expected = graph_dict_from_mol(mol, hydrogens=hydrogens)['node_valences']
    assert hydrogen_counts(smiles, hydrogens) == [int(v) for v in expected]


def test_default_and_invalid_hydrogen_mode():
    assert hydrogen_counts('C[NH3+]', 'implicit') == hdf_matched_graph({'graph_repr': 'C[NH3+]'})['node_attributes'][
        :, len(HDF_ATOMS) + 1 + HDF_MAX_DEGREE:].argmax(axis=1).tolist()
    with pytest.raises(ValueError):
        hdf_matched_graph({'graph_repr': 'CCO'}, hydrogens='explicit')


def make_model(**kwargs):
    graph = hdf_matched_graph({'graph_repr': 'CCO'})
    return GNN_CLASSES['gin'](input_dim=graph['node_attributes'].shape[1], output_dim=1, output_type='regression',
                              conv_units=[8, 8], dense_units=[4], learning_rate=1e-4, **kwargs)


def test_constant_learning_rate_by_default():
    assert isinstance(make_model().configure_optimizers(), torch.optim.Adam)


def test_cosine_schedule_values():
    config = make_model(lr_schedule='cosine', lr_min=1e-6, epochs=10).configure_optimizers()
    optimizer, scheduler = config['optimizer'], config['lr_scheduler']['scheduler']
    assert config['lr_scheduler']['interval'] == 'epoch'
    lrs = []
    for _ in range(10):
        lrs.append(optimizer.param_groups[0]['lr'])
        optimizer.step()
        scheduler.step()
    assert lrs[0] == pytest.approx(1e-4)
    assert lrs[5] == pytest.approx((1e-4 + 1e-6) / 2)          # half way: the cosine midpoint
    assert all(a > b for a, b in zip(lrs, lrs[1:]))          # strictly decreasing
    assert optimizer.param_groups[0]['lr'] == pytest.approx(1e-6)


def test_cosine_schedule_needs_epochs_and_known_name():
    with pytest.raises(ValueError):
        make_model(lr_schedule='cosine')
    with pytest.raises(ValueError):
        make_model(lr_schedule='step', epochs=10)


def test_cosine_schedule_runs_in_lightning():
    """A short fit: Lightning accepts the scheduler config and the rate ends at lr_min."""
    pl.seed_everything(0)
    graphs = {i: {'graph_repr': s, 'graph_labels': np.array([float(i)])}
              for i, s in enumerate(['CCO', 'c1cc[nH]c1', 'C[NH3+]', 'CC(=O)O', 'CCN', 'c1ccccc1'])}
    for graph in graphs.values():
        hdf_matched_graph(graph, hydrogens='total')
    loader = DataLoader(build_pyg_list(graphs, list(graphs)), batch_size=3, shuffle=True)
    model = make_model(lr_schedule='cosine', lr_min=1e-6, epochs=3)
    trainer = pl.Trainer(max_epochs=3, logger=False, enable_checkpointing=False, enable_progress_bar=False,
                         enable_model_summary=False, accelerator='cpu')
    trainer.fit(model, loader, loader)
    assert trainer.optimizers[0].param_groups[0]['lr'] == pytest.approx(1e-6)
    assert model.model_restorer.best_epoch is not None
