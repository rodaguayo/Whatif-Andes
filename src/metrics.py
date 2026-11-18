"""
Make neuralhydrology's ``KGE`` metric the modified KGE (KGE', Kling et al., 2012).

neuralhydrology 1.13 computes ``KGE`` as the 2009 version (Gupta et al., 2009), whose variability
term is ``alpha = sigma_sim / sigma_obs``. Since alpha also scales with the mean, it is
cross-correlated with the bias term ``beta``. KGE' replaces alpha with ``gamma``, the ratio of the
coefficients of variation, ``(sigma_sim / mu_sim) / (sigma_obs / mu_obs)``.

Importing this module, without modifying the installed package:

- makes the ``KGE`` metric return KGE' wherever neuralhydrology computes it (validation during
  training, the tester, ``validation_metrics.csv``, and therefore ``find_best_epoch``);
- adds the metric ``Gamma-KGE`` (the gamma component). ``Beta-KGE`` (mean ratio) is the same in
  both versions, and ``Alpha-NSE`` (std ratio) is still available.

It is imported by ``src.conceptual_models``, so every process that registers the glacier SHM
also gets KGE'. Import it before ``start_run`` / ``get_tester``::

    import src.metrics  # noqa: F401
"""

from typing import Dict, List

import numpy as np
from scipy import stats
from xarray import DataArray

import neuralhydrology.evaluation.metrics as nh_metrics
import neuralhydrology.evaluation.tester as nh_tester
import neuralhydrology.utils.nh_results_ensemble as nh_ensemble


def gamma_kge(obs: DataArray, sim: DataArray) -> float:
    """Ratio of the coefficients of variation, simulated over observed (KGE' gamma)."""
    nh_metrics._validate_inputs(obs, sim)
    obs, sim = nh_metrics._mask_valid(obs, sim)
    if len(obs) < 2:
        return np.nan
    return float((sim.std() / sim.mean()) / (obs.std() / obs.mean()))


def kge(obs: DataArray, sim: DataArray, weights: List[float] = [1.0, 1.0, 1.0]) -> float:
    """Modified Kling-Gupta efficiency, KGE' (Kling et al., 2012).

    ``KGE' = 1 - sqrt([s_r (r - 1)]^2 + [s_g (gamma - 1)]^2 + [s_b (beta - 1)]^2)``, with ``r`` the
    Pearson correlation, ``gamma`` the ratio of the coefficients of variation and ``beta`` the
    ratio of the means. Same signature as the neuralhydrology 2009 version it replaces.

    Kling, H., Fuchs, M., & Paulin, M. (2012). Runoff conditions in the upper Danube basin under
    an ensemble of climate change scenarios. Journal of Hydrology, 424-425, 264-277.
    """
    if len(weights) != 3:
        raise ValueError("Weights of the KGE must be a list of three values")
    nh_metrics._validate_inputs(obs, sim)
    obs, sim = nh_metrics._mask_valid(obs, sim)
    if len(obs) < 2:
        return np.nan

    r, _ = stats.pearsonr(obs.values, sim.values)
    beta = sim.mean() / obs.mean()
    gamma = (sim.std() / sim.mean()) / (obs.std() / obs.mean())
    value = weights[0] * (r - 1) ** 2 + weights[1] * (gamma - 1) ** 2 + weights[2] * (beta - 1) ** 2
    return 1 - np.sqrt(float(value))


EXTRA_METRICS = {"gamma-kge": ("Gamma-KGE", gamma_kge)}

# guarded against double import: always wrap the upstream functions
_get_available = getattr(nh_metrics, "_upstream_get_available_metrics", nh_metrics.get_available_metrics)
_calculate = getattr(nh_metrics, "_upstream_calculate_metrics", nh_metrics.calculate_metrics)
_calculate_all = getattr(nh_metrics, "_upstream_calculate_all_metrics", nh_metrics.calculate_all_metrics)


def get_available_metrics() -> List[str]:
    return _get_available() + [name for name, _ in EXTRA_METRICS.values()]


def calculate_metrics(obs: DataArray, sim: DataArray, metrics: List[str], resolution: str = "1D",
                      datetime_coord: str = None) -> Dict[str, float]:
    if "all" in metrics:
        return calculate_all_metrics(obs, sim, resolution=resolution, datetime_coord=datetime_coord)
    extra = [m for m in metrics if m.lower() in EXTRA_METRICS]
    upstream = [m for m in metrics if m.lower() not in EXTRA_METRICS]
    values = _calculate(obs, sim, metrics=upstream, resolution=resolution, datetime_coord=datetime_coord)
    for m in extra:
        name, func = EXTRA_METRICS[m.lower()]
        values[name] = func(obs, sim)
    return values


def calculate_all_metrics(obs: DataArray, sim: DataArray, resolution: str = "1D",
                          datetime_coord: str = None) -> Dict[str, float]:
    values = _calculate_all(obs, sim, resolution=resolution, datetime_coord=datetime_coord)
    for name, func in EXTRA_METRICS.values():
        values[name] = func(obs, sim)
    return values


nh_metrics._upstream_get_available_metrics = _get_available
nh_metrics._upstream_calculate_metrics = _calculate
nh_metrics._upstream_calculate_all_metrics = _calculate_all
nh_metrics.kge = kge  # calculate_metrics / calculate_all_metrics look kge up in the module globals
nh_metrics.get_available_metrics = get_available_metrics
nh_metrics.calculate_metrics = calculate_metrics
nh_metrics.calculate_all_metrics = calculate_all_metrics
for _module in (nh_tester, nh_ensemble):  # these imported the functions by name
    _module.get_available_metrics = get_available_metrics
    _module.calculate_metrics = calculate_metrics
