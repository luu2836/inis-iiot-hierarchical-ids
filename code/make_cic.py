import glob, os
import numpy as np
import pandas as pd

D = r"E:\EI_data\CIC-IDS2017"
OUT = os.path.join(D, "cic_capped.csv")
CAP = 50000
SEED = 42

ID_COLS = ["Flow ID", "Source IP", "Source Port", "Destination IP", "Timestamp"]

frames = []
for f in sorted(glob.glob(os.path.join(D, "*.csv"))):
    if os.path.basename(f) == "cic_capped.csv":
        continue
    df = pd.read_csv(f, low_memory=False, encoding="latin-1")
    df.columns = [c.strip() for c in df.columns]
    frames.append(df)
    print("loaded", os.path.basename(f), df.shape)

df = pd.concat(frames, ignore_index=True)
print("combined:", df.shape)
df = df.drop(columns=[c for c in ID_COLS if c in df.columns])
df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
print("after clean:", df.shape)

parts = []
for lab, g in df.groupby("Label"):
    if len(g) > CAP:
        g = g.sample(n=CAP, random_state=SEED)
    parts.append(g)
df = pd.concat(parts, ignore_index=True)
print("after cap:", df.shape)
print(df["Label"].value_counts().to_string())

df.to_csv(OUT, index=False, encoding="utf-8")
print("saved ->", OUT)
