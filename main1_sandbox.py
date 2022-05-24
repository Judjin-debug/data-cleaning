import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display
from matplotlib.colors import ListedColormap

# pd.set_option('display.max_columns', None)

df = pd.read_csv('data/sber_data.csv')

n = df.shape[0]
thresh = n * 0.5
df = df.dropna(how='any', thresh=thresh, axis=1)

m = df.shape[1]
df = df.dropna(how='any', thresh=m-2, axis=0)

for col in df.columns:
    if df[col].isnull().mean() > 0:
        if df[col].dtype.name == 'category' or df[col].dtype.name == 'object':
            df[col] = df[col].fillna(value=df[col].mode()[0])
        else:
            df[col] = df[col].fillna(value=df[col].mean())
