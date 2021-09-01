NUM_LABELS = 8
NEPOCHSHP = 20
# GLOBAL VARIABLES for GRID (ResNet101)
IMG_HEIGHT, IMG_WIDTH = 224, 224
top_dropout_rate_grid = 0.5
# MM
HFixed = 768
# TEXT
MAX_SEQ = 50
TEXTMODEL = "bert-base-uncased"
# Training
BATCH_SIZE = 8
hparams = {
    "MM-Resnet-CONCAT": {"AvgAll": (0.25, 1e-5, 2), "SampleWImg": (0.25, 1e-5, 3)},
    "MM-Resnet-XATT": {"AvgAll": (0.15, 1e-5, 3), "SampleWImg": (0.05, 1e-5, 12)},
    "MM-EfficientNet-XATT":{"AvgAll": (0.25, 1e-5, 4), "SampleWImg": (0.05, 1e-5, 8)},
    "MM-EfficientNet-CONCAT": {"AvgAll": (0.25, 1e-5, 4), "SampleWImg": (0.05, 1e-5, 3)},
    "MM-Xception-ADD": {"AvgAll": (0.15, 1e-5, 2), "SampleWImg": (0.05, 1e-5, 4)},
    "MM-Xception-CONCAT": {"AvgAll": (0.25, 1e-5, 2), "SampleWImg": (0.05, 1e-5, 4)},
    "MM-Xception-XATT": {"AvgAll": (0.15, 1e-5, 3), "SampleWImg": (0.15, 1e-5, 7)},
    "MM-Xception-GLU": {"AvgAll": (0.05, 1e-5, 2), "SampleWImg": (0.25, 1e-5, 4)},
    "MM-Xception-GLUATT": {"AvgAll": (0.05, 1e-5, 4), "SampleWImg": (0.05, 1e-5, 4)},
    "MM-Xception-ATTM": {"AvgAll": (0.25, 1e-5, 3), "SampleWImg": (0.15, 1e-5, 3)}
    }

