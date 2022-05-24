import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display
from matplotlib.colors import ListedColormap

# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

diabetes_data = pd.read_csv('data/diabetes_data.csv')

diabetes_data.info()


def find_and_del_dupl(dataframe, cols_to_exclude=None):
    subset_columns = list(dataframe.columns)
    if cols_to_exclude is not None:
        for col in cols_to_exclude:
            subset_columns.remove(col)
    mask = dataframe.duplicated(subset=subset_columns)
    print(f'The total amount of duplicates is {dataframe[mask].shape[0]}')
    dataframe_without_duplicates = dataframe.drop_duplicates(subset=subset_columns)
    print(f'The resulting number of entries is {dataframe_without_duplicates.shape[0]}')
    return dataframe_without_duplicates


diabetes_data = find_and_del_dupl(diabetes_data)


def low_info_cols(dataframe, threshold=0.95):
    low_information_cols = []
    for col in dataframe.columns:
        top_freq = dataframe[col].value_counts(normalize=True).max()
        nunique_ratio = dataframe[col].nunique() / dataframe[col].count()
        if top_freq > threshold:
            low_information_cols.append(col)
            print(f'{col}: {round(top_freq * 100, 2)}% of same values')
        if nunique_ratio > threshold:
            low_information_cols.append(col)
            print(f'{col}: {round(nunique_ratio * 100, 2)}% of unique values')
    return dataframe.drop(columns=low_information_cols)


diabetes_data = low_info_cols(diabetes_data, threshold=0.99)

# print(diabetes_data['Glucose'].describe())

cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
diabetes_data[cols] = diabetes_data[cols].replace({'0': np.nan, 0: np.nan})

# print(diabetes_data.isnull().mean())

n = diabetes_data.shape[0]
thresh = n * 0.7

diabetes_data = diabetes_data.dropna(how='any', thresh=thresh, axis=1)

m = diabetes_data.shape[1]
diabetes_data = diabetes_data.dropna(how='any', thresh=m-2, axis=0)

# print(diabetes_data.shape[0])


def fill_na_in_df(df):
    for col in df.columns:
        if df[col].isnull().mean() > 0:
            if df[col].dtype.name == 'category' or df[col].dtype.name == 'object':
                df[col] = df[col].fillna(value=df[col].mode()[0])
            else:
                df[col] = df[col].fillna(value=df[col].median())
    return df


diabetes_data = fill_na_in_df(diabetes_data)
# print(diabetes_data.isnull().mean())
# print(diabetes_data['SkinThickness'].mean())


def outliers_iqr_mod(data, feature, log_scale=False, left=1.5, right=1.5):
    if log_scale:
        x = np.log(data[feature])
    else:
        x = data[feature]
    quantile_1, quantile_3 = x.quantile(.25), x.quantile(.75)
    iqr = quantile_3 - quantile_1
    lower_bound = quantile_1 - (iqr * left)
    upper_bound = quantile_3 + (iqr * right)
    outliers = data[(x < lower_bound) | (x > upper_bound)]
    cleaned = data[(x > lower_bound) & (x < upper_bound)]
    return outliers, cleaned


def format_iqr_mod(data, feature, log_scale=False, left=1.5, right=1.5):
    outliers, cleaned = outliers_iqr_mod(data, feature, log_scale, left, right)
    print(f'Number of outliers by Tukey\'s method for {feature} is {outliers.shape[0]}')
    print(f'Number of clean entries by Tukey\'s method {feature} is {cleaned.shape[0]}')


def outliers_z_score_mod(data, feature, log_scale=False, left=3.0, right=3.0):
    if log_scale:
        x = np.log(data[feature] + 1)
    else:
        x = data[feature]
    mu = x.mean()
    sigma = x.std()
    lower_bound = mu - left * sigma
    upper_bound = mu + right * sigma
    outliers = data[(x < lower_bound) | (x > upper_bound)]
    cleaned = data[(x > lower_bound) & (x < upper_bound)]
    return outliers, cleaned


def format_z_score_mod(data, feature, log_scale=False, left=3.0, right=3.0):
    outliers, cleaned = outliers_z_score_mod(data, feature, log_scale, left, right)
    print(f'Number of outliers for SkinThickness by Z-score method  for {feature} is {outliers.shape[0]}')
    print(f'Number of clean entries for SkinThickness by Z-score method  for {feature} is {cleaned.shape[0]}')


format_iqr_mod(diabetes_data, 'DiabetesPedigreeFunction')
format_iqr_mod(diabetes_data, 'DiabetesPedigreeFunction', log_scale=True)
