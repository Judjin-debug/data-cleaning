import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display
from matplotlib.colors import ListedColormap

# pd.set_option('display.max_columns', None)

sber_data = pd.read_csv('data/sber_data.csv')
sber_data.head()

drop_data = sber_data.copy()
print(drop_data.shape)
thresh = drop_data.shape[0]*0.7
drop_data = drop_data.dropna(how='any', thresh=thresh, axis=1)
drop_data = drop_data.dropna(how='any', axis=0)
display(drop_data.isnull().mean())
print(drop_data.shape)

cols_null_percent = sber_data.isnull().mean() * 100
cols_with_null = cols_null_percent[cols_null_percent > 0].sort_values(ascending=False)

cols = cols_with_null.index

sber_data[cols].hist(figsize=(20, 8))
# plt.show()

fill_data = sber_data.copy()

values = {
    'life_sq': fill_data['full_sq'],
    'metro_min_walk': fill_data['metro_min_walk'].median(),
    'metro_km_walk': fill_data['metro_km_walk'].median(),
    'railroad_station_walk_km': fill_data['railroad_station_walk_km'].median(),
    'railroad_station_walk_min': fill_data['railroad_station_walk_min'].median(),
    'hospital_beds_raion': fill_data['hospital_beds_raion'].mode()[0],
    'preschool_quota': fill_data['preschool_quota'].mode()[0],
    'school_quota': fill_data['school_quota'].mode()[0],
    'floor': fill_data['floor'].mode()[0]
}

fill_data = fill_data.fillna(values)
display(fill_data.isnull().mean())

cols = cols_with_null.index
fill_data[cols].hist(figsize=(20, 8))
# plt.show()

indicator_data = sber_data.copy()

for col in cols_with_null.index:

    indicator_data[col + '_was_null'] = indicator_data[col].isnull()

values = {
    'life_sq': indicator_data['full_sq'],
    'metro_min_walk': indicator_data['metro_min_walk'].median(),
    'metro_km_walk': indicator_data['metro_km_walk'].median(),
    'railroad_station_walk_km': indicator_data['railroad_station_walk_km'].median(),
    'railroad_station_walk_min': indicator_data['railroad_station_walk_min'].median(),
    'hospital_beds_raion': indicator_data['hospital_beds_raion'].mode()[0],
    'preschool_quota': indicator_data['preschool_quota'].mode()[0],
    'school_quota': indicator_data['school_quota'].mode()[0],
    'floor': indicator_data['floor'].mode()[0]
}

indicator_data = indicator_data.fillna(values)

display(indicator_data.isnull().mean())

combine_data = sber_data.copy()

n = combine_data.shape[0]
thresh = n*0.7
combine_data = combine_data.dropna(how='any', thresh=thresh, axis=1)

m = combine_data.shape[1]
combine_data = combine_data.dropna(how='any', thresh=m-2, axis=0)

values = {
    'life_sq': indicator_data['full_sq'],
    'metro_min_walk': indicator_data['metro_min_walk'].median(),
    'metro_km_walk': indicator_data['metro_km_walk'].median(),
    'railroad_station_walk_km': indicator_data['railroad_station_walk_km'].median(),
    'railroad_station_walk_min': indicator_data['railroad_station_walk_min'].median(),
    'hospital_beds_raion': indicator_data['hospital_beds_raion'].mode()[0],
    'preschool_quota': indicator_data['preschool_quota'].mode()[0],
    'school_quota': indicator_data['school_quota'].mode()[0],
    'floor': indicator_data['floor'].mode()[0]
}

combine_data = combine_data.fillna(values)

display(combine_data.isnull().mean())

print(combine_data.shape)
