import matplotlib
from datasets import Dataset, DatasetDict
from sklearn import model_selection

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from src import config


def make_data(df_train, df_test):
    df_train["label"] = df_train["target"]
    # split the data to train and validation sets
    df_train, df_valid = model_selection.train_test_split(
        df_train, test_size=config.VALIDATION_SIZE, random_state=42, stratify=df_train.target.values
    )
    # make Huggingface dataset
    train_ds = Dataset.from_pandas(df_train)
    valid_ds = Dataset.from_pandas(df_valid)
    test_ds = Dataset.from_pandas(df_test)

    dataset = DatasetDict()

    dataset["train"] = train_ds
    dataset["validation"] = valid_ds
    dataset["test"] = test_ds

    return dataset


def plot_avg_tweets_length(df):
    df["words_per_tweet"] = df["text"].str.split().apply(len)
    df.boxplot("words_per_tweet", by="target", grid=False, showfliers=False, color="black")
    plt.suptitle("")
    plt.xlabel("")
    plt.show()


def tokenize(batch):
    return config.TOKENIZER(batch[config.TEXT], padding=True, truncation=True)
