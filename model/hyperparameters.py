import torch

LEARNING_RATE = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 1
NUM_EPOCH = 50
NUM_WORKERS = 0