"""tpacmon: Temporal Bayesian Latent Factor Model with GP priors."""

from tpacmon.core.models import TemporalPACMON
from tpacmon.tools.enrichment import enrich_factors

__all__ = ["TemporalPACMON", "enrich_factors"]
__version__ = "0.1.0"
