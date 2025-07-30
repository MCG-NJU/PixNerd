from typing import Any, Optional

import torch
import lightning.pytorch as pl
from lightning.pytorch import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT
from lightning_utilities.core.rank_zero import rank_zero_info

from src.models.metrics import UnifiedMetric


class UnifiedMetricHook(UnifiedMetric, Callback):
    def __init__(self, *args, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def on_predict_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.to(pl_module.device)
        self.reset()
        if self.precompute_data_path:
            self.load_precompute_data(self.precompute_data_path, trainer.global_rank, trainer.world_size)

    def on_predict_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        with torch.no_grad():
            self.update(outputs, False)

    def on_predict_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> dict[str, Any]:
        trainer.strategy.barrier()
        metric_dict = self.compute()
        for k, v in metric_dict.items():
            rank_zero_info(f"{k}:{v.item()}")
        self.cpu()
        return metric_dict

    def on_validation_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        return self.on_predict_epoch_start(trainer, pl_module)

    def on_validation_batch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            outputs: STEP_OUTPUT,
            batch: Any,
            batch_idx: int,
            dataloader_idx: int = 0,
    ) -> None:
        self.on_predict_batch_end(trainer, pl_module, outputs, batch_idx, dataloader_idx)

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        metric_dict =  self.on_predict_epoch_end(trainer, pl_module)
        pl_module.log_dict(metric_dict, prog_bar=True, sync_dist=True, on_epoch=True)

    def state_dict(  # type: ignore[override]  # todo
        self,
        destination: Optional[dict[str, Any]] = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> dict[str, Any]:
        return dict()






