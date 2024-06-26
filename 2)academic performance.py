# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# %%
df = pd.DataFrame( {'Name': ['Adi', 'Deeksha', 'Jincy', 'Keerthi', 'Harish', 'Anu', 'Ram'],
        'Age': [17, 17, 18, 17, 18, 17, 17],
        'Gender': ['M', 'F', 'F', 'F', 'M', 'F', 'M'],
        'Marks': [90, 76, 'NULL', 74, 65, 'NULL', 71]})

# %%
df.describe()

# %%
df.info()

# %%
df.isnull()
df.isnull().sum()

# %%
df

# %%
df.replace(['NULL'],[0],inplace=True)

# %%
df

# %%
df['Gender'].replace(['M','F'],[0,1], inplace = True)

# %%
df

# %%
df['Transformed_Marks'] = np.log1p(df['Marks']) #log transformation
df

# %%
sns.histplot(df['Marks'], label = 'Original Marks')
sns.histplot(df['Transformed_Marks'],  color = 'Pink', label = 'Transformed marks')
plt.legend()
plt.title("Distribution of marks and transformed marks")
plt.show()


