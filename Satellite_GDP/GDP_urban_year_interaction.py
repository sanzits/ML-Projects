import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
df = pd.read_csv('/Users/sanchitsuman/Satellite Data/train_data_indexed.csv')
# 1. Sort by county and year
df = df.sort_values(['county_id', 'year'])

# 2. Calculate GDP growth per county (year-over-year % change)
df['gdp_growth'] = df.groupby('county_id')['GDP'].pct_change() * 100  # percent

# 3. Drop rows where gdp_growth is NaN (first year per county)
df_plot = df.dropna(subset=['gdp_growth'])

# 4. Create urbanization quartiles
df_plot['urban_quartile'] = pd.qcut(df_plot['urban_pct'], 4,
                                    labels=['Q1 (Least urban)', 'Q2', 'Q3', 'Q4 (Most urban)'])

# 5. Compute average GDP growth per quartile per year
avg_growth = df_plot.groupby(['year', 'urban_quartile'])['gdp_growth'].mean().reset_index()

# 6. Optional: pivot for plotting (quartiles as columns)
avg_growth_pivot = avg_growth.pivot(index='year', columns='urban_quartile', values='gdp_growth')

print(avg_growth_pivot.head())
#avg_growth_pivot= avg_growth_pivot[avg_growth_pivot['year']!=2020]
avg_growth_pivot.plot(marker='o', figsize=(8,6))
plt.title("Average County GDP Growth by Urbanization Quartile")
plt.xlabel("Year")
plt.ylabel("Average GDP Growth (%)")
plt.legend(title="Urbanization Quartile")
plt.xticks(avg_growth_pivot.index)  # Ensure all years appear on x-axis
plt.grid(True)
plt.tight_layout()
plt.show()