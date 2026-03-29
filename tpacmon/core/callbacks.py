"""Training callbacks for tpacmon."""

import logging
import os
import torch

logger = logging.getLogger(__name__)


class CheckpointCallback:
    """Save model checkpoints during training."""

    def __init__(self, save_dir: str = "checkpoints", every_n_epochs: int = 100):
        self.save_dir = save_dir
        self.every_n_epochs = every_n_epochs
        os.makedirs(save_dir, exist_ok=True)

    def __call__(self, epoch: int, model, guide, loss: float):
        if (epoch + 1) % self.every_n_epochs == 0:
            path = os.path.join(self.save_dir, f"checkpoint_epoch{epoch+1}.pt")
            torch.save({
                "epoch": epoch + 1,
                "loss": loss,
                "model_state": model.state_dict(),
                "guide_state": guide.state_dict(),
            }, path)
            logger.info(f"Checkpoint saved: {path}")
