"""
Pool-based Bayesian optimization (BO) over a fixed molecular library.

This module holds the BO procedure of the HDF revision benchmark (ex_24, ``bo_benchmark.py``). A *campaign*
starts from a set of initial molecules whose objective values are known. In every round it fits a Gaussian
process (GP) surrogate to all observed values, scores every unobserved molecule of the library with the log
expected improvement (LogEI) and observes the molecule with the highest score. The objective is always
maximized; minimization and target-value tasks are expressed through the sign of the objective values (see
``objective_values``).

The surrogate model depends on the representation of the molecules:

    method            representation                                 GP kernel
    hdf_matern        HDF, centred and globally scaled               Matern-5/2 (one lengthscale)
    hdf_linear        HDF as computed (unit-length vectors)          linear, i.e. cosine similarity
    morgan_tanimoto   binary Morgan fingerprint folded to D bits     Tanimoto
    morgan_pca        Morgan-2048 projected on D PCA components,     Matern-5/2 (one lengthscale)
                      globally scaled
    random            none: random search

The inputs of the Matern kernels are centred and divided by *one* global factor, the root-mean-square norm
of the centred library vectors (``center_scale``). This keeps the geometry of the representation: all distances
between molecules change by the same factor. A per-coordinate standardization would instead inflate weak
coordinates, e.g. the many low-variance PCA components at large sizes.
"""
import time
import warnings
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import gpytorch
from botorch.acquisition.analytic import _log_ei_helper
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.kernels import LinearKernel, MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood


METHODS = ('hdf_matern', 'hdf_linear', 'morgan_tanimoto', 'morgan_pca', 'random')
# methods whose GP inputs go through ``center_scale``
MATERN_METHODS = ('hdf_matern', 'morgan_pca')


class TanimotoKernel(gpytorch.kernels.Kernel):
    """
    Tanimoto (Jaccard) kernel for fingerprint vectors,

        k(x, x') = <x, x'> / (|x|^2 + |x'|^2 - <x, x'>),

    in the form of GAUCHE's ``batch_tanimoto_sim`` (Griffiths et al., 2023), including its small ``eps`` in the
    numerator and denominator, which keeps the kernel finite for all-zero vectors. The kernel has no
    hyperparameters; wrap it in a ScaleKernel for an output scale.

    :param eps: The small constant added to numerator and denominator.
    """
    has_lengthscale = False

    def __init__(self, eps: float = 1e-6, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, diag: bool = False,
                last_dim_is_batch: bool = False, **params) -> torch.Tensor:
        if last_dim_is_batch:
            raise NotImplementedError('TanimotoKernel does not support last_dim_is_batch')
        if diag:
            dot = (x1 * x2).sum(dim=-1)
            return (dot + self.eps) / (self.eps + (x1 * x1).sum(dim=-1) + (x2 * x2).sum(dim=-1) - dot)

        dot = x1 @ x2.transpose(-2, -1)
        norm1 = (x1 * x1).sum(dim=-1, keepdim=True)
        norm2 = (x2 * x2).sum(dim=-1, keepdim=True)
        return (dot + self.eps) / (self.eps + norm1 + norm2.transpose(-2, -1) - dot)


def center_scale(features: np.ndarray) -> np.ndarray:
    """
    Centres the rows of ``features`` on their mean and divides them by the root-mean-square norm of the
    centred rows, so that the average squared norm is one. Distances between rows change only by one common
    factor, i.e. the geometry of the representation is kept.

    :param features: Array of shape (num_molecules, dim).

    :returns: Float32 array of the same shape.
    """
    features = np.asarray(features, dtype=np.float64)
    centred = features - features.mean(axis=0, keepdims=True)
    scale = np.sqrt((centred ** 2).sum(axis=1).mean())
    if scale <= 0:
        raise ValueError('cannot scale features that are identical for every molecule')
    return (centred / scale).astype(np.float32)


def objective_values(values: np.ndarray, mode: str, target: Optional[float] = None) -> np.ndarray:
    """
    Converts property ``values`` into objective values that BO maximizes.

    :param values: The property values of all library molecules.
    :param mode: "maximize", "minimize" or "target" (minimize the distance to ``target``).
    :param target: The target value of mode "target".

    :returns: Float64 array, larger is better.
    """
    values = np.asarray(values, dtype=np.float64)
    if mode == 'maximize':
        return values.copy()
    if mode == 'minimize':
        return -values
    if mode == 'target':
        if target is None:
            raise ValueError('mode "target" needs a target value')
        return -np.abs(values - target)
    raise ValueError(f'unknown objective mode "{mode}"')


def make_gp(method: str, train_x: torch.Tensor, train_y: torch.Tensor) -> SingleTaskGP:
    """
    Creates the (unfitted) GP surrogate of ``method`` for the observed inputs ``train_x`` (n, d) and objective
    values ``train_y`` (n, 1), both float64. All models use BoTorch's SingleTaskGP with a constant mean, its
    default Gaussian likelihood (LogNormal noise prior, noise >= 1e-4) and standardized outcomes.
    """
    if method in MATERN_METHODS:
        covar_module = ScaleKernel(MaternKernel(nu=2.5))
    elif method == 'hdf_linear':
        # LinearKernel carries its own variance, so it needs no ScaleKernel
        covar_module = LinearKernel()
    elif method == 'morgan_tanimoto':
        covar_module = ScaleKernel(TanimotoKernel())
    else:
        raise ValueError(f'method "{method}" has no GP surrogate')

    return SingleTaskGP(train_x, train_y, covar_module=covar_module, outcome_transform=Standardize(m=1))


def fit_gp(gp: SingleTaskGP) -> bool:
    """
    Fits the hyperparameters of ``gp`` by maximizing the exact marginal log likelihood (BoTorch's
    ``fit_gpytorch_mll``, L-BFGS-B with its retry logic; the retries draw from the global torch RNG, which
    ``run_campaign`` seeds). BoTorch's warnings about input scaling are expected here (the inputs are
    deliberately not min-max scaled) and are silenced.

    :returns: False if the fit failed (BoTorch gave up, or any other error such as a failed Cholesky
        decomposition), True otherwise. After a failure the caller should not trust the parameters of ``gp``.
    """
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            fit_gpytorch_mll(mll)
            return True
        except Exception:
            return False


def gp_hyperparameters(gp: SingleTaskGP) -> Dict[str, float]:
    """
    The fitted hyperparameters of ``gp`` as plain floats (in the standardized outcome space), for logging.
    """
    params = {
        'noise': float(gp.likelihood.noise.detach().reshape(-1)[0]),
        'mean': float(gp.mean_module.constant.detach().reshape(-1)[0]),
    }
    covar = gp.covar_module
    if isinstance(covar, ScaleKernel):
        params['outputscale'] = float(covar.outputscale.detach().reshape(-1)[0])
        if covar.base_kernel.has_lengthscale:
            params['lengthscale'] = float(covar.base_kernel.lengthscale.detach().reshape(-1)[0])
    elif isinstance(covar, LinearKernel):
        params['variance'] = float(covar.variance.detach().reshape(-1)[0])
    return params


@torch.no_grad()
def log_ei_scores(gp: SingleTaskGP,
                  candidates: torch.Tensor,
                  best_f: float,
                  chunk_size: int = 8192,
                  ) -> torch.Tensor:
    """
    The log expected improvement (LogEI, q = 1) of every row of ``candidates`` over ``best_f`` under the
    posterior of ``gp``. The candidates are processed in chunks (cast to float64 chunk by chunk, so that a
    float32 candidate matrix can stay on the GPU), and ``gp`` must live on the same device as ``candidates``.

    This computes exactly what BoTorch's ``LogExpectedImprovement`` computes for a (b, 1, d) batch of
    candidates, ``_log_ei_helper((mean - best_f) / sigma) + log(sigma)`` with the marginal posterior mean and
    standard deviation (variance clamped at 1e-12), but takes the marginals from one joint posterior per chunk,
    whose covariance gpytorch keeps lazy, so that only its diagonal is computed. BoTorch's batch of b separate
    q = 1 posteriors repeats the training-data part of the computation per candidate and is 7-30 times slower
    on a library of 20,000 molecules (identical scores up to 1e-8, see tests/test_bo.py).

    :param gp: The fitted GP.
    :param candidates: Tensor (num_candidates, dim) on the device of ``gp``.
    :param best_f: The best objective value observed so far (in the original, untransformed units).
    :param chunk_size: Number of candidates per chunk.

    :returns: Float64 tensor (num_candidates, ) on the device of ``candidates``.
    """
    gp.eval()
    scores = torch.empty(candidates.shape[0], dtype=torch.float64, device=candidates.device)
    for start in range(0, candidates.shape[0], chunk_size):
        posterior = gp.posterior(candidates[start:start + chunk_size].to(torch.float64))
        mean = posterior.mean.squeeze(-1)
        sigma = posterior.variance.squeeze(-1).clamp_min(1e-12).sqrt()
        scores[start:start + chunk_size] = _log_ei_helper((mean - best_f) / sigma) + sigma.log()
    return scores


def pick_best(scores: torch.Tensor, rng: np.random.Generator, tolerance: float = 1e-9) -> int:
    """
    The index of the highest entry of ``scores``. Ties are broken uniformly at random with ``rng`` instead of by
    library order. Molecules with identical feature vectors (e.g. Morgan collisions at small sizes) are true
    ties, but their scores can differ in the last bits when they are computed in different chunks, so every
    score within ``tolerance`` (relative) of the maximum counts as tied.
    """
    best = scores.max()
    ties = torch.nonzero(scores >= best - tolerance * (1 + best.abs())).flatten().cpu().numpy()
    if len(ties) == 1:
        return int(ties[0])
    return int(ties[rng.integers(len(ties))])


def run_campaign(method: str,
                 objective: np.ndarray,
                 initial: Sequence[int],
                 num_rounds: int,
                 seed: int,
                 features: Optional[np.ndarray] = None,
                 device: str = 'cpu',
                 chunk_size: int = 8192,
                 allowed: Optional[np.ndarray] = None,
                 ) -> Dict[str, np.ndarray]:
    """
    Runs one BO campaign of ``method`` on a fixed library.

    :param method: One of METHODS.
    :param objective: Objective values (to be maximized) of all library molecules, shape (num_molecules, ).
    :param initial: Library indices of the initial molecules (shared by all methods of a campaign).
    :param num_rounds: Number of BO rounds, one new molecule per round.
    :param seed: Seed of the random tie-breaking (and of the random search).
    :param features: The GP input vectors of all library molecules, shape (num_molecules, dim), already in the
        form the kernel expects (``center_scale`` for the Matern methods, raw for the others). Not used by
        "random".
    :param device: Device of the acquisition step ("cpu" or "cuda:<i>"). The GP is always fitted on the CPU,
        because the fit of a few hundred points is latency-bound.
    :param chunk_size: Candidates per chunk of the acquisition step.
    :param allowed: Optional boolean mask over the library: only these molecules can be observed (the initial
        design must lie inside it). The others still define the input space but are never proposed.

    :returns: Dict with the observed library indices in order ("observed", initial ones first), the best
        objective value after the initial design and after every round ("best", length num_rounds + 1), the
        seconds of GP fit and acquisition per round ("fit_seconds", "acq_seconds"), the number of failed fits
        ("fit_failures") and the hyperparameters per round ("hp_<name>").
    """
    if method not in METHODS:
        raise ValueError(f'unknown method "{method}"')
    objective = np.asarray(objective, dtype=np.float64)
    num_molecules = len(objective)
    rng = np.random.default_rng(seed)
    # BoTorch's fit retries sample from the global torch RNG: seed it, so a campaign does not depend on what
    # its worker process ran before
    torch.manual_seed(seed)
    observed: List[int] = [int(i) for i in initial]
    if len(set(observed)) != len(observed):
        raise ValueError('the initial design contains duplicates')
    allowed = np.ones(num_molecules, dtype=bool) if allowed is None else np.asarray(allowed, dtype=bool)
    if not allowed[observed].all():
        raise ValueError('the initial design contains molecules outside the allowed candidates')
    num_rounds = min(num_rounds, int(allowed.sum()) - len(observed))

    best = [float(objective[observed].max())]
    fit_seconds, acq_seconds = [], []
    hyperparameters: Dict[str, List[float]] = {}
    fit_failures = 0

    if method == 'random':
        remaining = np.setdiff1d(np.flatnonzero(allowed), observed)
        for index in rng.permutation(remaining)[:num_rounds]:
            observed.append(int(index))
            best.append(max(best[-1], float(objective[index])))
    else:
        if features is None or len(features) != num_molecules:
            raise ValueError('method needs one feature vector per library molecule')
        device = torch.device(device)
        # The candidate matrix stays float32 on the acquisition device; the training data are float64 on the
        # CPU (BoTorch's recommendation for a stable GP fit).
        candidates = torch.as_tensor(np.asarray(features), dtype=torch.float32, device=device)
        # molecules that cannot be proposed: the observed ones and those outside the allowed candidates
        is_blocked = torch.as_tensor(~allowed, device=device)
        is_blocked[observed] = True
        train_x = torch.as_tensor(np.asarray(features)[observed], dtype=torch.float64)
        train_y = torch.as_tensor(objective[observed], dtype=torch.float64).unsqueeze(-1)

        for _ in range(num_rounds):
            time_start = time.perf_counter()
            gp = make_gp(method, train_x, train_y)
            if not fit_gp(gp):
                # continue with the default hyperparameters of a fresh model (what BoTorch itself restores
                # after giving up)
                fit_failures += 1
                gp = make_gp(method, train_x, train_y)
            for name, value in gp_hyperparameters(gp).items():
                hyperparameters.setdefault(name, []).append(value)
            time_fit = time.perf_counter()

            gp = gp.to(device)
            scores = log_ei_scores(gp, candidates, best_f=float(train_y.max()), chunk_size=chunk_size)
            scores[is_blocked] = -float('inf')
            index = pick_best(scores, rng)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            time_acq = time.perf_counter()

            observed.append(index)
            is_blocked[index] = True
            train_x = torch.cat([train_x, torch.as_tensor(np.asarray(features[index:index + 1]),
                                                          dtype=torch.float64)])
            train_y = torch.cat([train_y, torch.tensor([[objective[index]]], dtype=torch.float64)])
            best.append(max(best[-1], float(objective[index])))
            fit_seconds.append(time_fit - time_start)
            acq_seconds.append(time_acq - time_fit)
            del gp, scores

    result = {
        'observed': np.array(observed, dtype=np.int64),
        'best': np.array(best, dtype=np.float64),
        'fit_seconds': np.array(fit_seconds, dtype=np.float64),
        'acq_seconds': np.array(acq_seconds, dtype=np.float64),
        'fit_failures': np.array(fit_failures, dtype=np.int64),
    }
    for name, values in hyperparameters.items():
        result[f'hp_{name}'] = np.array(values, dtype=np.float64)
    return result


def normalized_regret(best: np.ndarray, objective_max: float, objective_min: float) -> np.ndarray:
    """
    The normalized regret of a best-so-far curve: 0 when the best molecule of the library has been found, 1 at
    the worst molecule of the library.
    """
    return (objective_max - np.asarray(best, dtype=np.float64)) / (objective_max - objective_min)


def regret_auc(regret: np.ndarray) -> float:
    """
    The area under a normalized-regret curve over the BO rounds (trapezoid rule, from the initial design at
    round 0 to the last round), divided by the number of rounds, i.e. the average regret of the campaign.
    """
    regret = np.asarray(regret, dtype=np.float64)
    if len(regret) < 2:
        return float(regret[0])
    return float((regret[:-1] + regret[1:]).sum() / 2 / (len(regret) - 1))
