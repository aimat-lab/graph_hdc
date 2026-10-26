"""
Pool-based Bayesian optimization of the BO benchmark (graph_hdc.bo): the Tanimoto kernel, the
geometry-preserving input scaling, the objective conventions and the campaign loop.
"""
import numpy as np
import pytest
import torch

from graph_hdc.bo import (
    METHODS,
    TanimotoKernel,
    center_scale,
    normalized_regret,
    objective_values,
    pick_best,
    regret_auc,
    run_campaign,
)


def gauche_tanimoto(x1: np.ndarray, x2: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """GAUCHE's batch_tanimoto_sim, written out in numpy as an independent reference."""
    dot = x1 @ x2.T
    return (dot + eps) / (eps + (x1 ** 2).sum(-1)[:, None] + (x2 ** 2).sum(-1)[None, :] - dot)


def test_tanimoto_kernel_matches_reference():
    rng = np.random.default_rng(0)
    x1 = (rng.random((7, 32)) < 0.3).astype(np.float64)
    x2 = (rng.random((5, 32)) < 0.3).astype(np.float64)
    kernel = TanimotoKernel()
    value = kernel(torch.tensor(x1), torch.tensor(x2)).to_dense().numpy()
    assert np.allclose(value, gauche_tanimoto(x1, x2), atol=1e-12)
    # plain Jaccard on a hand-made pair: 1 shared bit out of 3 set bits
    pair = torch.tensor([[1., 1., 0.], [1., 0., 1.]], dtype=torch.float64)
    assert kernel(pair[:1], pair[1:]).to_dense().item() == pytest.approx(1 / 3, abs=1e-6)


def test_tanimoto_kernel_diag_and_identity():
    rng = np.random.default_rng(1)
    x = torch.tensor((rng.random((6, 16)) < 0.4).astype(np.float64))
    kernel = TanimotoKernel()
    full = kernel(x, x).to_dense()
    assert torch.allclose(kernel(x, x, diag=True), torch.diagonal(full))
    assert torch.allclose(torch.diagonal(full), torch.ones(6, dtype=torch.float64))


def test_center_scale_keeps_geometry():
    rng = np.random.default_rng(2)
    features = rng.normal(size=(50, 8)) * np.arange(1, 9) + 3.0
    scaled = center_scale(features)
    assert np.allclose(scaled.mean(axis=0), 0, atol=1e-6)
    assert (scaled.astype(np.float64) ** 2).sum(axis=1).mean() == pytest.approx(1.0, rel=1e-5)
    # all pairwise distances shrink by the same factor (no per-coordinate rescaling)
    ratio = np.linalg.norm(scaled[1:] - scaled[:-1], axis=1) / np.linalg.norm(features[1:] - features[:-1], axis=1)
    assert np.allclose(ratio, ratio[0], rtol=1e-5)


def test_objective_values():
    values = np.array([1.0, 5.0, -2.0])
    assert np.array_equal(objective_values(values, 'maximize'), values)
    assert np.array_equal(objective_values(values, 'minimize'), -values)
    assert np.array_equal(objective_values(values, 'target', target=4.0), [-3.0, -1.0, -6.0])
    with pytest.raises(ValueError):
        objective_values(values, 'target')


def test_regret_helpers():
    regret = normalized_regret(np.array([0.0, 2.0, 4.0]), objective_max=4.0, objective_min=0.0)
    assert np.allclose(regret, [1.0, 0.5, 0.0])
    # trapezoid over rounds 0..2, divided by the 2 rounds
    assert regret_auc(regret) == pytest.approx((0.75 + 0.25) / 2)


@pytest.mark.parametrize('method', ['hdf_matern', 'morgan_tanimoto', 'hdf_linear'])
def test_log_ei_scores_match_botorch(method):
    # the chunked joint-posterior LogEI equals BoTorch's LogExpectedImprovement on a batch of q = 1 candidates
    from botorch.acquisition.analytic import LogExpectedImprovement
    from graph_hdc.bo import fit_gp, log_ei_scores, make_gp
    rng = np.random.default_rng(5)
    if method == 'morgan_tanimoto':
        x = (rng.random((300, 32)) < 0.2).astype(np.float64)
    else:
        x = rng.normal(size=(300, 8))
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    y = np.sin(3 * x[:, 0]) + x[:, 1]
    gp = make_gp(method, torch.tensor(x[:40]), torch.tensor(y[:40])[:, None])
    fit_gp(gp)
    gp.eval()
    best_f = float(y[:40].max())
    reference = LogExpectedImprovement(gp, best_f=best_f)(torch.tensor(x).unsqueeze(-2)).detach()
    scores = log_ei_scores(gp, torch.tensor(x), best_f=best_f, chunk_size=64)
    # Far in the tail (LogEI around -350) the two ways of computing a tiny posterior variance differ by float64
    # rounding that LogEI amplifies (relative differences up to ~1e-7); the ranking of the candidates is the same.
    assert torch.allclose(scores, reference, rtol=1e-5, atol=1e-6)
    assert torch.equal(torch.topk(scores, 5).indices, torch.topk(reference, 5).indices)
    # the chunking does not change the scores
    assert torch.allclose(scores, log_ei_scores(gp, torch.tensor(x), best_f=best_f, chunk_size=1000), atol=1e-12)


def test_log_ei_scores_match_botorch_lazy_path():
    # more than 4096 candidates in one chunk make gpytorch evaluate the posterior lazily
    from botorch.acquisition.analytic import LogExpectedImprovement
    from graph_hdc.bo import fit_gp, log_ei_scores, make_gp
    rng = np.random.default_rng(6)
    x = rng.normal(size=(6000, 8))
    y = np.sin(3 * x[:, 0]) + x[:, 1]
    gp = make_gp('hdf_matern', torch.tensor(x[:50]), torch.tensor(y[:50])[:, None])
    fit_gp(gp)
    gp.eval()
    reference = LogExpectedImprovement(gp, best_f=float(y[:50].max()))(torch.tensor(x).unsqueeze(-2)).detach()
    scores = log_ei_scores(gp, torch.tensor(x), best_f=float(y[:50].max()), chunk_size=6000)
    assert torch.allclose(scores, reference, rtol=1e-5, atol=1e-6)
    assert torch.equal(torch.topk(scores, 10).indices, torch.topk(reference, 10).indices)


def test_pick_best_breaks_ties_among_maxima():
    scores = torch.tensor([0.1, 0.9, 0.3, 0.9, 0.9 - 1e-12], dtype=torch.float64)
    picks = {pick_best(scores, np.random.default_rng(seed)) for seed in range(30)}
    assert picks == {1, 3, 4}


def synthetic_library(num: int = 120, dim: int = 6, seed: int = 0):
    rng = np.random.default_rng(seed)
    bits = (rng.random((num, 64)) < 0.2).astype(np.float32)
    continuous = rng.normal(size=(num, dim)).astype(np.float32)
    objective = -np.linalg.norm(continuous - continuous[0], axis=1)  # smooth, optimum at molecule 0
    return bits, continuous, objective


@pytest.mark.parametrize('method', METHODS)
def test_run_campaign_bookkeeping(method):
    bits, continuous, objective = synthetic_library()
    features = {
        'hdf_matern': center_scale(continuous),
        'morgan_pca': center_scale(continuous),
        'hdf_linear': continuous / np.linalg.norm(continuous, axis=1, keepdims=True),
        'morgan_tanimoto': bits,
        'random': None,
    }[method]
    initial = [5, 17, 33, 48, 90]
    result = run_campaign(method, objective, initial, num_rounds=8, seed=3, features=features)

    observed = result['observed']
    assert list(observed[:5]) == initial
    assert len(observed) == 13 and len(set(observed.tolist())) == 13
    assert result['best'].shape == (9, )
    assert np.all(np.diff(result['best']) >= 0)
    assert np.allclose(result['best'], np.maximum.accumulate(objective[observed])[4:])
    if method != 'random':
        assert result['fit_seconds'].shape == (8, ) and result['acq_seconds'].shape == (8, )
        assert any(key.startswith('hp_') for key in result)

    # the same seed gives the same campaign
    again = run_campaign(method, objective, initial, num_rounds=8, seed=3, features=features)
    assert np.array_equal(again['observed'], observed)


@pytest.mark.parametrize('method', ['hdf_matern', 'random'])
def test_run_campaign_respects_allowed_candidates(method):
    _, continuous, objective = synthetic_library(num=80, seed=7)
    allowed = np.zeros(80, dtype=bool)
    allowed[::2] = True  # only the even molecules may be proposed (the optimum, molecule 0, is among them)
    features = center_scale(continuous) if method != 'random' else None
    result = run_campaign(method, objective, [2, 4, 6], num_rounds=50, seed=1, features=features, allowed=allowed)
    assert allowed[result['observed']].all()
    assert len(result['observed']) == 40  # 37 rounds until the 40 allowed molecules are used up
    with pytest.raises(ValueError):
        run_campaign(method, objective, [1, 2], num_rounds=5, seed=1, features=features, allowed=allowed)


def test_run_campaign_stops_when_library_is_exhausted():
    bits, _, objective = synthetic_library(num=12)
    result = run_campaign('morgan_tanimoto', objective, [0, 1, 2], num_rounds=50, seed=0, features=bits)
    assert len(result['observed']) == 12


def test_matern_bo_beats_random_on_smooth_objective():
    # 10 paired campaigns on a smooth objective: the GP should find the optimum far earlier than random search
    _, continuous, objective = synthetic_library(num=300, dim=3, seed=4)
    features = center_scale(continuous)
    aucs = {'hdf_matern': [], 'random': []}
    for campaign in range(10):
        initial = np.random.default_rng(100 + campaign).choice(300, size=5, replace=False)
        for method in aucs:
            result = run_campaign(method, objective, initial, num_rounds=20, seed=campaign,
                                  features=features if method != 'random' else None)
            regret = normalized_regret(result['best'], objective.max(), objective.min())
            aucs[method].append(regret_auc(regret))
    assert np.mean(aucs['hdf_matern']) < 0.5 * np.mean(aucs['random'])
