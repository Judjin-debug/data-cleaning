import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display
from matplotlib.colors import ListedColormap

# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

sber_data = pd.read_csv('data/sber_data.csv')
sber_data.head()

# print(sber_data['life_sq'].describe())

# print(sber_data[sber_data['life_sq'] == 0].shape[0])

# print(sber_data[sber_data['life_sq'] > 7000])

# outliers = sber_data[sber_data['life_sq'] > sber_data['full_sq']]
# print(outliers.shape[0])

# cleaned = sber_data.drop(outliers.index, axis=0)

# print(sber_data[sber_data['floor'] > 50]['floor'])


def show_hist_and_box(data, feature):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 4))
    histplot = sns.histplot(data=data, x=feature, ax=axes[0])
    histplot.set_title(feature + ' distribution')
    boxplot = sns.boxplot(data=data, x=feature, ax=axes[1])
    boxplot.set_title(feature + ' boxplot')
    plt.show()


# show_hist_and_box(sber_data, 'full_sq')

###

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


# outliers, cleaned = outliers_iqr_mod(sber_data, 'full_sq', left=1, right=6)
# print(f'Number of outliers by Tukey\'s method {outliers.shape[0]}')
# print(f'Number of clean entries by Tukey\'s method {cleaned.shape[0]}')
#
# show_hist_and_box(cleaned, 'full_sq')

###

# fig, axes = plt.subplots(1, 2, figsize=(15, 4))
#
# histplot = sns.histplot(sber_data['mkad_km'], bins=30, ax=axes[0])
# histplot.set_title('MKAD km distribution')
#
# log_mkad_km = np.log(sber_data['mkad_km'] + 1)
# histplot = sns.histplot(log_mkad_km, bins=30, ax=axes[1])
# histplot.set_title('MKAD km distribution')
#
# plt.show()
#
# print(log_mkad_km.skew())

###

# def outliers_z_score(data, feature, log_scale=False):
#     if log_scale:
#         x = np.log(data[feature] + 1)
#     else:
#         x = data[feature]
#     mu = x.mean()
#     sigma = x.std()
#     lower_bound = mu - 3 * sigma
#     upper_bound = mu + 3 * sigma
#     outliers = data[(x < lower_bound) | (x > upper_bound)]
#     cleaned = data[(x > lower_bound) & (x < upper_bound)]
#     return outliers, cleaned
#
#
# outliers, cleaned = outliers_z_score(sber_data, 'mkad_km', log_scale=True)
# print(f'Number of outliers by Z-score method {outliers.shape[0]}')
# print(f'Number of clean entries by Z-score method {cleaned.shape[0]}')
#
# # print(outliers['sub_area'].unique())
#
# fig, ax = plt.subplots(1, 1, figsize=(8, 4))
# log_mkad_km = np.log(sber_data['mkad_km'] + 1)
# histplot = sns.histplot(log_mkad_km, bins=30, ax=ax)
# histplot.axvline(log_mkad_km.mean(), color='k', lw=2)
# histplot.axvline(log_mkad_km.mean() + 3 * log_mkad_km.std(), color='k', ls='--', lw=2)
# histplot.axvline(log_mkad_km.mean() - 3 * log_mkad_km.std(), color='k', ls='--', lw=2)
# histplot.set_title('Log MKAD km distribution')
# plt.show()

###


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


# outliers, cleaned = outliers_z_score_mod(sber_data, 'mkad_km', log_scale=True, right=3.5)
# print(f'Number of outliers by Z-score method {outliers.shape[0]}')
# print(f'Number of clean entries by Z-score method {cleaned.shape[0]}')

###

# fig, ax = plt.subplots(1, 1, figsize=(8, 4))
# log_data = np.log(sber_data['price_doc'] + 1)
# histplot = sns.histplot(log_data, bins=30, ax=ax)
# histplot.axvline(log_data.mean(), color='k', lw=2)
# histplot.axvline(log_data.mean() + 3 * log_data.std(), color='k', ls='--', lw=2)
# histplot.axvline(log_data.mean() - 3 * log_data.std(), color='k', ls='--', lw=2)
# histplot.set_title('Price doc distribution')
# plt.show()

###

# outliers, cleaned = outliers_z_score_mod(sber_data, 'price_doc', log_scale=True, left=3.7, right=3.7)
#
# print(f'Number of outliers by Z-score method {outliers.shape[0]}')
# print(f'Number of clean entries by Z-score method {cleaned.shape[0]}')

###

outliers, cleaned = outliers_iqr_mod(sber_data, 'price_doc', log_scale=True, left=3, right=3)

print(f'Number of outliers by Tukey\'s method {outliers.shape[0]}')
print(f'Number of clean entries by Tukey\'s method {cleaned.shape[0]}')
