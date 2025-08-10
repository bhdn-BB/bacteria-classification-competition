import numpy as np
import wandb
from sklearn.metrics import f1_score
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    Trainer,
    TrainingArguments
)

from callbacks.wandb_checkpoint_callback import WandbCheckpointCallback
from configs.best_dino_v2 import MODEL, EPOCHS, BATCH_SIZE, LR
from global_config import NUM_CLASSES, ROOT


class DinoV2Classifier:
    def __init__(
        self,
        model_name=MODEL,
        num_labels=NUM_CLASSES,
        num_freeze_backbone=0
    ):
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForImageClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )

        if num_freeze_backbone > 0:
            try:
                blocks = list(self.model.base_model.encoder.layer)
            except AttributeError:
                raise ValueError("encoder layer not found")
            for block in blocks[:num_freeze_backbone]:
                for param in block.parameters():
                    param.requires_grad = False

        self.trainer = None

    @staticmethod
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        f1_macro = f1_score(labels, preds, average="macro")
        return {"f1_macro": f1_macro}

    def train(
        self,
        train_dataset,
        val_dataset,
        output_dir=f"{ROOT}/dino2-finetuned",
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR
    ):
        # wandb.login(key="...")
        wandb.init(project="dino-v2-classifier")

        args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=lr,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=0.01,
            logging_dir=f"{ROOT}/logs",
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            greater_is_better=True
        )

        self.trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[WandbCheckpointCallback()]
        )
        self.trainer.train()
        wandb.finish()

    def predict(self, dataset):
        predictions = self.trainer.predict(dataset)
        preds = np.argmax(predictions.predictions, axis=-1)
        return preds.tolist()