import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_squared_error

data = pd.read_csv('gpu_1986-2026.csv')

# Plot GPUs releases per year

release_cols = [
    "Graphics Card__Release Date",
]

data["Release_raw"] = np.nan
for col in release_cols:
    if col in data.columns:
        data["Release_raw"] = data["Release_raw"].combine_first(data[col])


data["Release_clean"] = (
    data["Release_raw"]
    .astype(str)
    .replace("nan", "", regex=False)
    .str.replace(r"(\d+)(st|nd|rd|th)", r"\1", regex=True)
    .str.replace(",", "", regex=False)
    .str.strip()
)


data["Release_year"] = (
    data["Release_clean"]
    .str.extract(r"\b((?:19|20)\d{2})\b")[0]
    .astype("Int64")
)


plt.figure(figsize=(12, 6))
ax = sns.countplot(data=data, x='Release_year',)
plt.title('Distribution of Graphics Card Release Years')
plt.xlabel('Release Year')
plt.ylabel('Number of Models')
plt.xticks(rotation=45)


for p in ax.patches:
    height = p.get_height()
    if height > 0:
        ax.annotate(
            str(int(height)),
            (p.get_x() + p.get_width() / 2., height),
            ha='center', va='bottom', fontsize=10, fontweight='bold'
        )

plt.tight_layout()
plt.show()

# Theoretical Performance Pixel Rate of Nvidia GPUs over time (GPU with highest Pixel Rate each year)

nvidia_data = data[data['Brand'] == 'NVIDIA'].copy()
nvidia_data = nvidia_data.dropna(subset=['Release_year', 'Theoretical Performance__Pixel Rate'])
nvidia_data['Pixel Rate'] = (
    nvidia_data['Theoretical Performance__Pixel Rate']
    .str.replace(r'[^\d.]', '', regex=True)
    .astype(float)
)
    
nvidia_data['Biggest_TPixel'] = nvidia_data.groupby('Release_year')['Pixel Rate'].transform('max')
nvidia_data = nvidia_data.drop_duplicates(subset=['Release_year', 'Biggest_TPixel'])

plt.figure(figsize=(12, 6))
plt.plot(nvidia_data['Release_year'], nvidia_data['Biggest_TPixel'], color='orange', label='Data Points')
X = nvidia_data[['Release_year']]
y = nvidia_data['Biggest_TPixel']
model = lm.LinearRegression()
model.fit(X, y)
plt.xlabel('Release Year')
plt.ylabel('Theoretical Pixel Rate (MPixels/s)')
plt.legend()
plt.tight_layout()
plt.show()

# AMD GPUs number of transistor per DIE size over time

def clean_numeric(series):
    s = series.astype(str).str.replace(",", "", regex=False)
    s = s.str.replace(r'[^\d.]', '', regex=True)
    return pd.to_numeric(s, errors='coerce')

amd_data = data[data['Brand'] == 'AMD'].copy()


amd_data['Transistors_Million'] = clean_numeric(amd_data['Graphics Processor__Transistors'])
amd_data['Die_Size_mm2'] = clean_numeric(amd_data['Graphics Processor__Die Size'])

amd_data['Transistor_Density'] = amd_data['Transistors_Million'] / amd_data['Die_Size_mm2']

# Inspect how many NaNs / infs we have
print("Rows before drop:", len(amd_data))
print("NaNs in Transistors_Million:", amd_data['Transistors_Million'].isna().sum())
print("NaNs in Die_Size_mm2:", amd_data['Die_Size_mm2'].isna().sum())
print("NaNs in Transistor_Density:", amd_data['Transistor_Density'].isna().sum())
print("Infinite densities (die size == 0):", (~np.isfinite(amd_data['Transistor_Density'])).sum())

amd_clean = amd_data.loc[np.isfinite(amd_data['Transistor_Density'])].dropna(
    subset=['Release_year', 'Transistor_Density', 'Transistors_Million', 'Die_Size_mm2']
).copy()

print("Rows after clean:", len(amd_clean))

# Fit model with the cleaned data
X_amd = amd_clean[['Release_year']]
y_amd = amd_clean['Transistor_Density']

model_amd = lm.LinearRegression()
model_amd.fit(X_amd, y_amd)

plt.figure(figsize=(12, 6))
plt.bar(amd_clean['Release_year'], amd_clean['Transistor_Density'], label='Data Points', color='red')
plt.xlabel('Release Year')
plt.ylabel('Transistor Density (Million/mm²)')
plt.legend()
plt.tight_layout()
plt.show()