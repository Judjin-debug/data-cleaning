import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display
from matplotlib.colors import ListedColormap

# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

sber_data = pd.read_csv('data/sber_data.csv')

print(sber_data['id'].nunique == sber_data.shape[0])

dupl_columns = list(sber_data.columns)
dupl_columns.remove('id')

mask = sber_data.duplicated(subset=dupl_columns)
sber_duplicates = sber_data[mask]
print(f'Number of found duplicates is {sber_duplicates.shape[0]}')

sber_dedupped = sber_data.drop_duplicates(subset=dupl_columns)
print(f'Resulting number of entries is {sber_dedupped.shape[0]}')


def low_info_cols(dataframe):
    low_information_cols = []
    for col in dataframe.columns:
        top_freq = dataframe[col].value_counts(normalize=True).max()
        nunique_ratio = dataframe[col].nunique() / dataframe[col].count()
        if top_freq > 0.95:
            low_information_cols.append(col)
            print(f'{col}: {round(top_freq * 100, 2)}% of same values')
        if nunique_ratio > 0.95:
            low_information_cols.append(col)
            print(f'{col}: {round(nunique_ratio * 100, 2)}% of unique values')
    return low_information_cols


low_information_cols = low_info_cols(sber_data)
information_sber_data = sber_data.drop(low_information_cols, axis=1)
print(f'Resulting number of columns is {information_sber_data.shape[1]}')

