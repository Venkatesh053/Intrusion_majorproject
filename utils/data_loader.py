import pandas as pd

def load_nsl_dataset():
    """Load NSL-KDD processed dataset"""
    df = pd.read_csv("./datasets/nslkdd/nslkdd_processed.csv")
    return df
