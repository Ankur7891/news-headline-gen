import pandas as pd


def load_news(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=["article", "highlights"])
    art = df["article"].astype(str).tolist()
    summ = df["highlights"].astype(str).tolist()
    return art, summ
