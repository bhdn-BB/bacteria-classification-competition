import wandb
import torch
import os
from transformers import TrainerCallback


class HfWandbCheckpointCallback(TrainerCallback):
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

    def _save_and_log(self, model, path, artifact_name, epoch, metric_value, is_best):
        torch.save(model.state_dict(), path)
        artifact = wandb.Artifact(artifact_name, type="model")
        artifact.add_file(path)
        artifact.metadata.update({
            "epoch": epoch,
            self.metric_name: metric_value,
            "is_best": is_best,
        })
        wandb.log_artifact(artifact)
        print(f"[W&B] Logged {artifact_name.upper()} checkpoint (epoch {epoch}, {self.metric_name}={metric_value})")

    def on_epoch_end(self, args, state, control, **kwargs):
        trainer = kwargs.get("trainer")
        model = trainer.model
        metrics = state.log_history[-1]
        epoch = int(metrics.get("epoch", state.epoch))
        current_metric = metrics.get(self.metric_name)

        self._save_and_log(model, self.last_model_path, "model-last", epoch, current_metric, is_best=False)

        if (self.best_metric is None) or \
           (self.greater_is_better and current_metric > self.best_metric) or \
           (not self.greater_is_better and current_metric < self.best_metric):
            self.best_metric = current_metric
            self._save_and_log(model, self.best_model_path, "model-best", epoch, current_metric, is_best=True)

    def on_init_end(self, args, state, control, **kwargs):
        pass