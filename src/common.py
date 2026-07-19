import numpy as np
import torch

from torch.utils.data import DataLoader

from src.dataset import GaitDataset

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load processed data
X_train = np.load("../processed_data/X_train.npy")
X_val = np.load("../processed_data/X_val.npy")
X_test = np.load("../processed_data/X_test.npy")

y_train = np.load("../processed_data/y_train.npy")
y_val = np.load("../processed_data/y_val.npy")
y_test = np.load("../processed_data/y_test.npy")

# Datasets
train_dataset = GaitDataset(X_train, y_train)
val_dataset = GaitDataset(X_val, y_val)
test_dataset = GaitDataset(X_test, y_test)

# DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=128,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False
)
import random

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)