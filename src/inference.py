import os

import numpy as np
import pandas as pd

from src import config
from src.dataset import tokenize


def predict(tweets_encoded, trainer):
    preds = trainer.predict(tweets_encoded["test"])
    y_pres = np.argmax(preds.predictions, axis=1)
    sample_submission = pd.read_csv(config.SUBMISSION_PATH)
    sample_submission["target"] = y_pres
    print("Saving predictions...")
    sample_submission.to_csv(os.path.join(config.DATA_DIR, "submission.csv"), index=False)
    return y_pres
