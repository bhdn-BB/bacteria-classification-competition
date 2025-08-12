from pathlib import Path
import torch

DICT_LABELS = {
    0:  "Nucleoplasm",
    1:  "Nuclear membrane",
    2:  "Nucleoli",
    3:  "Nucleoli fibrillar center",
    4:  "Nuclear speckles",
    5:  "Nuclear bodies",
    6:  "Endoplasmic reticulum",
    7:  "Golgi apparatus",
    8:  "Peroxisomes",
    9:  "Endosomes",
    10: "Lysosomes",
    11: "Intermediate filaments",
    12: "Actin filaments",
    13: "Focal adhesion sites",
    14: "Microtubules",
    15: "Microtubule ends",
    16: "Cytokinetic bridge",
    17: "Mitotic spindle",
    18: "Microtubule organizing center",
    19: "Centrosome",
    20: "Lipid droplets",
    21: "Plasma membrane",
    22: "Cell junctions",
    23: "Mitochondria",
    24: "Aggresome",
    25: "Cytosol",
    26: "Cytoplasmic bodies",
    27: "Rods & rings"
}

ROOT = Path('/kaggle/input/human-protein-atlas-image-classification')
WORKING_DIR = Path('/kaggle/working')

TRAIN_IMAGE_DIR = ROOT / 'train'
TEST_IMAGE_DIR = ROOT / 'test'
TRAIN_LABEL_PATH = ROOT / 'train.csv'
TEST_LABEL_PATH = ROOT / 'sample_submission.csv'
TRAIN_TARGET = "Target"
TEST_TARGET = 'Predicted'


NUM_CLASSES = len(DICT_LABELS)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MEAN_IMAGENET = [0.485, 0.456, 0.406, 0.45]
STD_IMAGENET = [0.229, 0.224, 0.225, 0.2]

CALCULATE_MEAN = [
    0.08044196078431372,
    0.05262980392156863,
    0.05474705882352941,
    0.08270901960784315
]

CALCULATE_STD = [
    0.13000745098039215,
    0.08796627450980392,
    0.1386321568627451,
    0.12718039215686275
]

COLORS_CHANNEL = ['red', 'green', 'blue', 'yellow']

SIZE_IMAGENET = [224, 224]

CHANNELS_IMAGE = 4

LR = 1e-3
EPOCHS = 100

MAX_SAMPLES = 4000