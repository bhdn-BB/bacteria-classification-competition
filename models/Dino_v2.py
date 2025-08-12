import numpy as np
import torch
import wandb
from sklearn.metrics import f1_score
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    Trainer,
    TrainingArguments
)
from callbacks.hf_wandb_checkpoint_callback import HfWandbCheckpointCallback
from global_config import NUM_CLASSES
from configs.best_dino_v2 import *


class DinoV2Classifier:

    def __init__(self, model_name=MODEL, num_labels=NUM_CLASSES):

        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForImageClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
        )
        self.model.config.num_channels = num_labels
        self.model.dinov2.embeddings.patch_embeddings.num_channels = num_labels
        self.model.config.use_cache = False
        self.model.gradient_checkpointing_enable()
        old_conv = self.model.dinov2.embeddings.patch_embeddings.projection
        new_conv = torch.nn.Conv2d(
            in_channels=num_labels,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )
        with torch.no_grad():
            new_conv.weight[:, :3, :, :] = old_conv.weight
            new_conv.weight[:, 3:4, :, :] = old_conv.weight.mean(dim=1, keepdim=True)
            if old_conv.bias is not None:
                new_conv.bias = old_conv.bias
        self.model.dinov2.embeddings.patch_embeddings.projection = new_conv
        self.trainer = None

    @staticmethod
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        f1_macro = f1_score(labels, preds, average="macro")
        return {"f1_macro": f1_macro}

    def my_data_collator(self, features):
        pixel_values = torch.stack([f[0] for f in features])
        labels = torch.stack([f[1] for f in features])
        return {
            "pixel_values": pixel_values,
            "labels": labels
        }

    def train(
            self,
            train_dataset,
            val_dataset,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            lr=LR
    ):
        # wandb.login(key="...")
        wandb.init(project="dino-v2-classifier")

        args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            save_strategy="epoch",
            # evaluation_strategy="epoch",
            learning_rate=lr,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=WEIGHT_DECAY,
            logging_dir= OUTPUT_DIR,
            logging_steps=LOG_INTERVAL,
            metric_for_best_model=metrics,
            greater_is_better=True,
            fp16=True,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        )
        self.trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
            data_collator=self.my_data_collator,

            callbacks=[HfWandbCheckpointCallback()]
        )
        self.trainer.train()
        wandb.finish()

    def predict(self, dataset):
        predictions = self.trainer.predict(dataset)
        preds = np.argmax(predictions.predictions, axis=-1)
        return preds.tolist()