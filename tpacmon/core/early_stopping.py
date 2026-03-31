"""Early stopping callback for SVI training."""

import logging
import numpy as np

logger = logging.getLogger(__name__)


class EarlyStoppingCallback:
    """Stop training when smoothed ELBO improvement stalls.

    Uses an exponential moving average of the ELBO to smooth out the
    stochastic noise inherent in SVI, preventing premature stopping.
    """

    def __init__(
        self,
        min_epochs: int = 100,
        tolerance: float = 1e-5,
        patience: int = 10,
        smoothing: int = 50,
    ):
        self.min_epochs = min_epochs
        self.tolerance = tolerance
        self.patience = patience
        self.smoothing = smoothing
        self._counter = 0
        self._best_loss = np.inf

    def __call__(self, history) -> bool:
        epoch = len(history)
        if epoch < self.min_epochs:
            return False

        # Use moving average to smooth SVI noise
        window = min(self.smoothing, len(history))
        current = np.mean(history[-window:])

        if current < self._best_loss - self.tolerance * abs(self._best_loss):
            self._best_loss = current
            self._counter = 0
        else:
            self._counter += 1

        if self._counter >= self.patience:
            logger.info(f"Early stopping at epoch {epoch} (patience={self.patience})")
            return True
        return False
