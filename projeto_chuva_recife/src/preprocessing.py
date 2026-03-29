import pandas as pd

def preprocess(df):
    df = df.fillna(0)
    df['month'] = df['date'].dt.month
    df['dayofyear'] = df['date'].dt.dayofyear
    return df
