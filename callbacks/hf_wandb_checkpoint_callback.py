import wandb
import torch
import os
from transformers import TrainerCallback


class WandbCheckpointCallback(TrainerCallback):
    def __init__(
            self,
            save_dir: str = "checkpoints",
            metric_name="eval_f1_macro",
            greater_is_better=True
    ):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.metric_name = metric_name
        self.greater_is_better = greater_is_better
        self.best_metric = None
        self.best_model_path = os.path.join(self.save_dir, "model-best.pt")
        self.last_model_path = os.path.join(self.save_dir, "model-last.pt")

    def on_epoch_end(self, args, state, control, **kwargs):
        trainer = kwargs.get("trainer")
        model = trainer.model
        metrics = state.log_history[-1]
        epoch = int(metrics.get("epoch", state.epoch))
        current_metric = metrics.get(self.metric_name)
        torch.save(model.state_dict(), self.last_model_path)
        last_artifact = wandb.Artifact(f"model-last", type="model")
        last_artifact.add_file(self.last_model_path)
        last_artifact.metadata.update({
            "epoch": epoch,
            self.metric_name: current_metric,
            "is_best": False,
        })
        wandb.log_artifact(last_artifact)
        print(f"[W&B] Logged LAST checkpoint (epoch {epoch}, {self.metric_name}={current_metric})")
        if (self.best_metric is None) or \
           (self.greater_is_better and current_metric > self.best_metric) or \
           (not self.greater_is_better and current_metric < self.best_metric):
            self.best_metric = current_metric
            torch.save(model.state_dict(), self.best_model_path)
            best_artifact = wandb.Artifact(f"model-best", type="model")
            best_artifact.add_file(self.best_model_path)
            best_artifact.metadata.update({
                "epoch": epoch,
                self.metric_name: current_metric,
                "is_best": True,
            })
            wandb.log_artifact(best_artifact)
            print(f"[W&B] Logged BEST checkpoint (epoch {epoch}, {self.metric_name}={current_metric})")

    def on_init_end(self, args, state, control, **kwargs):
        pass