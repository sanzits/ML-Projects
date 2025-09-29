import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv('/Users/sanchitsuman/Satellite Data/train_data_indexed.csv')
#print(df)

print(df.describe())
df.describe().to_csv('/Users/sanchitsuman/Satellite Data/train_data_indexed_description.csv')   

yearly = df.groupby('year').agg(
        avg_gdp = ("GDP", "mean"),
        avg_rad = ("avg_rad", "mean")
).reset_index()

print(yearly)

#county-level average GDP against average night-time light intensity (avg_rad) on a log-log scale.
df['urban_quartile'] = pd.qcut(df['urban_pct'], 4,
                                    labels=['Q1 (Least urban)', 'Q2', 'Q3', 'Q4 (Most urban)'])
sns.scatterplot(data=df, x='avg_rad', y='GDP',  hue='urban_quartile',alpha=0.3)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Log Average Night-time Light Intensity (avg_rad)')
plt.ylabel('Log County GDP')
plt.title('County-level Average GDP vs. Average Night-time Light Intensity')
plt.show()

#Divide urban_pct into 4 bins (lowest 25%, 25–50%, 50–75%, highest 25%).
#This way, you group counties from most rural → most urban.
#Step 2: Compute average GDP growth per quartile per year
#For each quartile and each year, calculate mean GDP growth.
#Step 3: Plot
#X-axis: Year
#Y-axis: Average GDP growth
#Lines: One for each urbanization quartile (Q1 = least urban, Q4 = most urban).

