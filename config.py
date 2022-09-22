import torch


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR_VIS = "data/vis/train"
VAL_DIR_VIS = "data/vis/val"
TRAIN_DIR_IR = "data/ir/train"
VAL_DIR_IR = "data/ir/val"
LEARNING_RATE = 2e-4
BATCH_SIZE = 32
NUM_WORKERS = 2
IMAGE_SIZE = 512
CHANNELS_IMG = 3
L1_LAMBDA = 100
# LAMBDA_GP = 10
NUM_EPOCHS = 500
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_DISC = "disc_KL_10.pth.tar"
CHECKPOINT_GEN = "gen_KL_10.pth.tar"



