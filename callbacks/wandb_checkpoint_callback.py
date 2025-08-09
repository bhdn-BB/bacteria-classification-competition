import wandb
import torch
import os
from typing import Dict, Any

class WandbCheckpointCallback:

    def __init__(self, save_dir: str = "checkpoints"):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.last_model_path = os.path.join(self.save_dir, "model-last.pt")

    def __call__(self, metrics: Dict[str, Any]) -> None:
        epoch = metrics["epoch"]
        model = metrics["model"]
        is_best = metrics["is_best"]
        eval_metric = metrics.get("eval_metric", None)

        torch.save(model.state_dict(), self.last_model_path)
        last_artifact = wandb.Artifact("model-last", type="model")
        last_artifact.add_file(self.last_model_path)
        last_artifact.metadata.update({
            "epoch": epoch,
            "eval_metric": eval_metric,
            "is_best": is_best,
        })
        wandb.log_artifact(last_artifact)
        print(f"[W&B] Logged LAST checkpoint (epoch {epoch})")

        if is_best:
            best_model_path = os.path.join(self.save_dir, "model-best.pt")
            torch.save(model.state_dict(), best_model_path)
            best_artifact = wandb.Artifact("model-best", type="model")
            best_artifact.add_file(best_model_path)
            best_artifact.metadata.update({
                "epoch": epoch,
                "eval_metric": eval_metric,
                "is_best": is_best,
            })
            wandb.log_artifact(best_artifact)
            print(f"[W&B] Logged BEST checkpoint (epoch {epoch})")
