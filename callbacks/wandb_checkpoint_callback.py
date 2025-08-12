import os
from typing import Dict, Any
import torch
import wandb

class WandbCheckpointCallback:

    def __init__(self, save_dir: str = "checkpoints"):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.last_model_path = os.path.join(self.save_dir, "model-last.pt")
        self.best_model_path = os.path.join(self.save_dir, "model-best.pt")

    def _save_and_log(self, model, path: str, artifact_name: str, epoch: int, eval_metric: Any, is_best: bool):
        torch.save(model.state_dict(), path)
        artifact = wandb.Artifact(artifact_name, type="model")
        artifact.add_file(path)
        artifact.metadata.update({
            "epoch": epoch,
            "eval_metric": eval_metric,
            "is_best": is_best,
        })
        wandb.log_artifact(artifact)
        print(f"[W&B] Logged {artifact_name.upper()} checkpoint (epoch {epoch})")

    def __call__(self, metrics: Dict[str, Any]) -> None:
        epoch = metrics["epoch"]
        model = metrics["model"]
        is_best = metrics["is_best"]
        eval_metric = metrics.get("eval_metric", None)
        self._save_and_log(model, self.last_model_path, "model-last", epoch, eval_metric, is_best)
        if is_best:
            self._save_and_log(model, self.best_model_path, "model-best", epoch, eval_metric, is_best)