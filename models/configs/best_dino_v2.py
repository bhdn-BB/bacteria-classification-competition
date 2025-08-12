from global_config import WORKING_DIR

MODEL = 'facebook/dinov2-base-imagenet1k-1-layer'
EPOCHS = 15
BATCH_SIZE = 128
LR = 3E-4
LOG_INTERVAL = 10
WEIGHT_DECAY = 3E-3
OUTPUT_DIR = WORKING_DIR / "dino2-finetuned"
metrics = "f1_macro"
GRADIENT_ACCUMULATION_STEPS = 16
