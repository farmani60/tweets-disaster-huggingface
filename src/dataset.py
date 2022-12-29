import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn import model_selection

from src import config


def make_data(df):
    # split the data to train and validation sets
    df_train, df_valid = model_selection.train_test_split(
        df, test_size=config.VALIDATION_SIZE, random_state=42, stratify=df.target.values
    )
    # make Huggingface dataset
    train_ds = Dataset.from_pandas(df_train)
    valid_ds = Dataset.from_pandas(df_valid)

    dataset = DatasetDict()

    dataset["train"] = train_ds
    dataset["validation"] = valid_ds

    return dataset


if __name__ == "__main__":
    df = pd.read_csv(config.TRAIN_SET_PATH)
    df_test = pd.read_csv(config.TEST_SET_PATH)

    dataset = make_data(df)
