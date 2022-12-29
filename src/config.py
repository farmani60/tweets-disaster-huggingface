import os

from transformers import AutoTokenizer, DistilBertTokenizer

# data
DATA_DIR = "../data"
MODEL_DIR = "../model"
TRAIN_SET_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_SET_PATH = os.path.join(DATA_DIR, "test.csv")
SUBMISSION_PATH = os.path.join(DATA_DIR, "sample_submission.csv")

# features
TEXT = "text"
TARGET = "target"

# validation
VALIDATION_SIZE = 0.1

# model & tokenizer
MODEL_CHECKPOINT = "distilbert-base-uncased"
NUM_LABELS = 2

# TRAINING
TRAIN_BATCH_SIZE = 64
VALID_BATCH_SIZE = 64
LEARNING_RATE = 0.00002
NUM_EPOCHS = 10
WEIGHT_DECAY = 0.01
EVALUATION_STRATEGY = "epoch"

# tokenizer
# TOKENIZER = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
TOKENIZER = DistilBertTokenizer.from_pretrained(MODEL_CHECKPOINT)
