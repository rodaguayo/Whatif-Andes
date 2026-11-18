"""
Glacier-aware SHM conceptual model for the neuralhydrology ``hybrid_model``.

neuralhydrology 1.13 only knows ``conceptual_model: shm``. Importing this package
extends ``HybridModel._get_conceptual_model`` so that the names in ``CONCEPTUAL_MODELS``
resolve to the classes kept here, leaving the installed neuralhydrology untouched.

Import it before ``start_run`` or ``get_tester``, in the same process::

    import src.conceptual_models  # noqa: F401
"""

from neuralhydrology.modelzoo.hybridmodel import HybridModel

import src.metrics  # noqa: F401  the KGE metric becomes KGE' (2012) wherever these models are run
from src.conceptual_models.shm_glacier_snow_routing import SHMGlacierSnowRouting

CONCEPTUAL_MODELS = {
    "shm_glacier_snow_routing": SHMGlacierSnowRouting,
}

# keep the upstream resolver for the built-in models (guarded against double import)
_upstream = getattr(HybridModel, "_upstream_get_conceptual_model", HybridModel._get_conceptual_model)


def _get_conceptual_model(cfg):
    model = CONCEPTUAL_MODELS.get(cfg.conceptual_model.lower())
    return model(cfg=cfg) if model is not None else _upstream(cfg=cfg)


HybridModel._upstream_get_conceptual_model = staticmethod(_upstream)
HybridModel._get_conceptual_model = staticmethod(_get_conceptual_model)
