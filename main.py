import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display
from matplotlib.colors import ListedColormap

# pd.set_option('display.max_columns', None)

sber_data = pd.read_csv('data/sber_data.csv')
sber_data.head()

sber_data.info()
display(sber_data['ecology'].value_counts())

ax1 = sns.boxplot(data=sber_data, y='price_doc', x='ecology')
plt.show()

ax2 = sns.scatterplot(data=sber_data, x='kremlin_km', y='price_doc')
plt.show()

cols_null_percent = sber_data.isnull().mean() * 100
cols_with_null = cols_null_percent[cols_null_percent > 0].sort_values(ascending=False)
display(cols_with_null)
display(type(cols_with_null))

ax3 = sns.barplot(x=cols_with_null.index, y=cols_with_null.values)
ax3.set_xticklabels(ax3.get_xticklabels(), rotation=30)
plt.show()

cmap = sns.color_palette(['blue', 'yellow'], n_colors=2)
cols = cols_with_null.index
grid_kws = {'width_ratios': (0.9, 0.03), 'wspace': 0.18}
fig4, (ax4, cbar_ax4) = plt.subplots(1, 2, gridspec_kw=grid_kws)
ax4 = sns.heatmap(sber_data[cols].isnull(), ax=ax4, cbar_ax=cbar_ax4,
                  cmap=ListedColormap(cmap))
ax4.set_xticklabels(ax3.get_xticklabels(), rotation=20)
cbar_ax4.yaxis.set_ticks([0.25, 0.75])
cbar_ax4.set_yticklabels(['isnotnull', 'isnull'])
ax4.set_title('Null values in sber_data table')
ax4.set_xlabel('Column names')
ax4.set_ylabel('Row numbers')
plt.show()
