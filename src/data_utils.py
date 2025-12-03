import pandas as pd

def load_processed_data(path):
    df = pd.read_csv(path, parse_dates=['date'])
    return df
