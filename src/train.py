import pandas as pd
import torch
from transformers import (
    AutoModelForAudioFrameClassification,
    Trainer,
    TrainingArguments,
)

from src import config
from src.dataset import make_data, plot_avg_tweets_length, tokenize
from src.metrics import compute_metrics


def train(dataset):
    # define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # encode tweets
    tweets_encoded = dataset.map(tokenize, batched=True, batch_size=None)

    # make model
    model = AutoModelForAudioFrameClassification.from_pretrained(
        config.MODEL_CHECKPOINT, num_labels=config.NUM_LABELS
    ).to(device)

    logging_steps = len(tweets_encoded["train"]) // config.BATCH_SIZE

    training_args = TrainingArguments(
        output_dir=config.MODEL_DIR,
        num_train_epochs=config.NUM_EPOCHS,
        learning_rate=config.LEARNING_RATE,
        per_device_train_batch_size=config.TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=config.VALID_BATCH_SIZE,
        weight_decay=config.WEIGHT_DECAY,
        evaluation_strategy=config.EVALUATION_STRATEGY,
        disable_tqdm=False,
        logging_steps=logging_steps,
        push_to_hub=False,
        logging_dir="error",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=tweets_encoded["train"],
        eval_dataset=tweets_encoded["valid"],
        tokenizer=config.TOKENIZER,
    )

    trainer.train()


if __name__ == "__main__":
    df = pd.read_csv(config.TRAIN_SET_PATH, index_col=False)
    # plot_avg_tweets_length(df)
    df_test = pd.read_csv(config.TEST_SET_PATH)

    dataset = make_data(df)
    train_ds = dataset["train"]
    valid_ds = dataset["validation"]

    train(train_ds, valid_ds, device="cpu")
