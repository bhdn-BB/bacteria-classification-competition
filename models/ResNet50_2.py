import wandb
import torch
import torch.nn as nn
from torch import optim
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
from tqdm import tqdm
from callbacks.wandb_checkpoint_callback import WandbCheckpointCallback

class WideResNet50_2:
    def __init__(
            self,
            num_classes,
            channels_image,
            device,
            num_freeze_backbone=0,
    ):
        self.num_classes = num_classes
        self.channels_image = channels_image
        self.device = device
        self.num_freeze_backbone = num_freeze_backbone
        weights = Wide_ResNet50_2_Weights.IMAGENET1K_V1
        model = wide_resnet50_2(weights=weights)
        old_conv = model.conv1
        new_conv = nn.Conv2d(
            in_channels=channels_image,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None
        )
        with torch.no_grad():
            new_conv.weight[:, :3, :, :] = old_conv.weight
            if channels_image > 3:
                new_conv.weight[:, 3:4, :, :] = old_conv.weight.mean(dim=1, keepdim=True)
        model.conv1 = new_conv
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        if num_freeze_backbone > 0:
            backbone_blocks = [
                model.conv1,
                model.bn1,
                model.layer1,
                model.layer2,
                model.layer3,
                model.layer4
            ]
            freeze_blocks = backbone_blocks[:num_freeze_backbone]
            for block in freeze_blocks:
                for param in block.parameters():
                    param.requires_grad = False
        self.model = model.to(device)
    def fit(
            self,
            train_loader,
            val_loader,
            epochs,
            lr,
            weight_decay,
            use_wandb_checkpoint=False
    ):
        if use_wandb_checkpoint:
            # wandb.login(key="...")
            wandb.init(project="WideResNet50_2", name="model_training")
            checkpoint_callback = WandbCheckpointCallback()
        else:
            checkpoint_callback = None

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_val_loss = float('inf')

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0

            for images, targets in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} Train", leave=False):
                images, targets = images.to(self.device), targets.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)

            epoch_train_loss = running_loss / len(train_loader.dataset)
            scheduler.step()

            self.model.eval()
            val_running_loss = 0.0
            with torch.no_grad():
                for images, targets in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} Val", leave=False):
                    images, targets = images.to(self.device), targets.to(self.device)
                    outputs = self.model(images)
                    loss = criterion(outputs, targets)
                    val_running_loss += loss.item() * images.size(0)

            epoch_val_loss = val_running_loss / len(val_loader.dataset)

            print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")
            if checkpoint_callback:
                checkpoint_callback({
                    "epoch": epoch + 1,
                    "model": self.model,
                    "is_best": epoch_val_loss < best_val_loss,
                    "eval_metric": epoch_val_loss,
                })

            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                torch.save(self.model.state_dict(), "best_model.pt")
        if use_wandb_checkpoint:
            wandb.finish()