import pandas as pd
import numpy as np
from logger import logging

from sklearn.model_selection import KFold


def encode(df):
    # Map: A, C = not defaulted (0), B, D = defaulted (1)
    df['default'] = df['status'].map({
        'A': 0,  # Paid on time
        'C': 0,  # Paid late, but eventually paid
        'B': 1,  # Unpaid at term
        'D': 1   # Defaulted
    })
    df.drop(labels ='status',axis = 1 ,  inplace = True)
    logging.info("Step 1 of encoding is done.")
    return df


def target_encode(train_series , target , n_splits = 5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    encoded = pd.Series(index=train_series.index , dtype=float)
    for train_idx , val_idx in kf.split(train_series):
        train_fold = train_series.iloc[train_idx]
        val_fold = train_series.iloc[val_idx]
        target_fold = target.iloc[train_idx]
        
        means = target_fold.groupby(train_fold).mean()
        encoded.iloc[val_idx] = val_fold.map(means)
    logging.info("Step 2 of encoding is done.")
    return encoded

