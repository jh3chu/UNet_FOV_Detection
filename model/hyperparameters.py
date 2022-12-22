import torch

LEARNING_RATE = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 1
NUM_EPOCH = 10
NUM_WORKERS = 4