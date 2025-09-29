import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv('/Users/sanchitsuman/Satellite Data/train_data_indexed.csv')
#print(df)

#print(df.columns)
#df.describe().to_csv('/Users/sanchitsuman/Satellite Data/train_data_indexed_description.csv')   

df['urban_group'] = pd.qcut(df['urban_pct'], q=4, labels=['Low', 'Med-Low', 'Med-High', 'High'])

# Step 2: Focus on extreme groups (Low vs High)


df = df.sort_values(["county_id", "year"])  
df["gdp_growth"] = df.groupby("county_id")["GDP"].pct_change() * 100
df_filtered = df[df['urban_group'].isin(['Low', 'High'])]
# 5. Compute average GDP growth per quartile per year
#avg_growth = df_plot.groupby(['year', 'urban_quartile'])['gdp_growth'].mean().reset_index()

plot_df = (
    df_filtered.groupby(['year', 'urban_group'])['gdp_growth']
    .mean()
    .reset_index()
)

# Step 4: Plot
plt.figure(figsize=(8, 5))
sns.lineplot(
    data=plot_df,
    x='year', y='gdp_growth', hue='urban_group',
    marker='o'
)

plt.title("GDP Growth over Time: Low vs. High Urbanization Counties")
plt.ylabel("Average GDP Growth (%)")
plt.xlabel("Year")
plt.legend(title="Urbanization Level")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()