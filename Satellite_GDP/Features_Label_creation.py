import pandas as pd
import csv
from scipy.stats import linregress

def read_labels_file():
    """
    Reads the labels CSV file and returns a DataFrame.
    """
    file_path = '/Users/sanchitsuman/Satellite Data/CountyGDP.csv'
    df = pd.read_csv(file_path)
    return df


def read_features_file(years):
    """
    Reads multiple CSV files and concatenates them into a single DataFrame.

    Args:
        file_list (list): List of file paths to CSV files.

    Returns:
        pd.DataFrame: Concatenated DataFrame containing data from all files.
    """
    i =0
    for year in years:
        file_path = f'/Users/sanchitsuman/Satellite Data/county_data_{year}.csv'
        df = pd.read_csv(file_path)
        if i==0:
             combined_df = df
        else:
             combined_df = pd.concat([combined_df, df], ignore_index=True)
        i += 1
    return combined_df

def feature_creation(df, value_columns):
    """
    For each rolling 5-year window, group by state_name and county_name,
    and calculate average, range, pct_change, std_dev, and regression stats
    for each specified value column.
    Returns a DataFrame with new feature rows for each group, window, and column.
    """
    years = sorted(df['year'].unique())
    results = []

    for i in range(len(years) - 4):
        years_trimmed = years[i:i+5]
        df_window = df[df['year'].isin(years_trimmed)]

        grouped = df_window.groupby(['state_name', 'county_name'])

        for (state, county), group in grouped:
            group_sorted = group.sort_values('year')
            years_vals = group_sorted['year'].values

            for col in value_columns:
                values = group_sorted[col].values

                if len(values) < 5:
                    continue  # Skip incomplete windows

                avg = values.mean()
                rng = values.max() - values.min()
                pct_chg = (values[-1] - values[0]) / values[0] if values[0] != 0 else None
                std_dev = values.std()
                slope, intercept, r_value, p_value, std_err = linregress(years_vals, values)

                results.append({
                    'state_name': state,
                    'county_name': county,
                    'start_year': years_trimmed[0],
                    'end_year': years_trimmed[-1],
                    'feature_column': col,
                    'average': avg,
                    'range': rng,
                    'pct_change': pct_chg, 
                    'std_dev': std_dev,
                    'slope': slope,
                    'intercept': intercept,
                    'r_value': r_value,
                    'p_value': p_value,
                    'std_err': std_err
                })

    return pd.DataFrame(results)

# Usage in main():
# Replace with your actual column names

#std_Dev
#slope, intercept, r_value, p_value, std_err



def main():
    df_features = read_features_file([2016, 2017, 2018, 2019,2020,2021,2022,2023])
    df_features.to_csv('/Users/sanchitsuman/Satellite Data/features.csv', index=False)
    #print(df_features.count())
    #print(df_features["state_name"]["county_name"].unique())
    df_labels = read_labels_file()
    #print(df_labels["State"]["County"].unique())
    # print(df_labels.head())
    # ---------------------------------------------------------------
    # The following lines convert the unique state/county pairs from 
    # each DataFrame into sets of tuples:
    #
    # - features_unique.values gives a NumPy array of unique rows.
    # - map(tuple, ...) converts each row (array) into a tuple, e.g., ('California', 'Los Angeles').
    # - set(...) collects all tuples into a set, removing duplicates and enabling set operations.
    #
    # This allows for efficient comparison (e.g., intersection) between the two sets.
    # ---------------------------------------------------------------
   
    features_unique = df_features[['state_name', 'county_name']].drop_duplicates()
    print(f"Unique state/county pairs in features: {len(features_unique)}")
    #print("features_unique",features_unique.values)

    # Unique (State, County) in labels
    labels_unique = df_labels[['State', 'County']].drop_duplicates()
    print(f"Unique state/county pairs in labels: {len(labels_unique)}")

    features_set = set(map(tuple,features_unique.values))
    labels_set = set(map(tuple,labels_unique.values))
    print(f"Features set sample: {list(features_set)[:5]}")
    print(f"Labels set sample: {list(labels_set)[:5]}")

    # Find common (State, County) pairs
    common_features_labels = features_set.intersection(labels_set)
    print(f"Common state/county pairs: {len(common_features_labels)}")

    print("Years :", df_features['year'].unique())
    df_features_extended =feature_creation(df_features, ['ppt_total_mm', 'road_length_km', 'tmean', 'urban_pct'])
    df_features_extended.to_csv('/Users/sanchitsuman/Satellite Data/extended_features.csv', index=False)
    df_wide = df_features_extended.pivot_table(
        index=["state_name", "county_name","start_year", "end_year"], 
        columns="feature_column", 
        values=["range", "pct_change", "std_dev", "slope", "intercept", "r_value", "p_value", "std_err"]
    ).reset_index()
    df_wide.columns = [
        f"{i}_{j}" if j else f"{i}"
        for i, j in df_wide.columns.to_flat_index()
    ]
    df_wide.to_csv('/Users/sanchitsuman/Satellite Data/wide_features.csv', index=False)

# ---------------------------
# Entry Point
# ---------------------------
if __name__ == "__main__":
    main()