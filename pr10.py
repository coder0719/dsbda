# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.datasets import load_iris
import warnings
warnings.filterwarnings("ignore")


# %%
data = load_iris()


# %%
df = pd.DataFrame()
df[data['feature_names']] = data['data']
df['label'] = data['target']

# %%
df.head()

# %%
df.shape

# %%
df.info()

# %%
df.describe()

# %%
plt.figure(figsize=(10, 8))
for i, feature in enumerate(df.columns[:-1]):
    plt.subplot(2, 2, i + 1)
    sns.histplot(df[feature], kde=True)
    plt.title(f'Histogram of {feature}')
plt.tight_layout()
plt.show()

# %%
plt.figure(figsize=(10, 8))
for i, feature in enumerate(df.columns[:-1]):
    plt.subplot(2, 2, i + 1)
    sns.boxplot(y=df[feature])
    plt.title(f'Boxplot of {feature}')
plt.tight_layout()
plt.show()


