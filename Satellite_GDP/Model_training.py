from linearmodels.panel import PanelOLS
import pandas as pd
import numpy as np

def training_data(df_features, df_labels, df_census):
    df = pd.merge(df_features, df_labels, how='inner', left_on=['state_name', 'county_name','year'], right_on=['State', 'County', 'Year'])
    print('Checking merge for GEOID:', df.head())
    df = pd.merge(
        df,
        df_census,
        how='left',
        left_on=['GEOID', 'year'],
        right_on=['GEOID', 'year']
    )
    print('Checking merge for GEOID:', df.head())
    return df

def mod_growth(df_train):
    df_train["urban_year_interaction"] = df_train["urban_pct"] * df_train["year"]
    df_train['log_gdp'] = df_train['GDP'].apply(np.log)
    df_train = df_train.set_index(['county_id', 'year'])
    df_train['gdp_growth'] = df_train.groupby(level='county_id')['log_gdp'].diff()
    # Remove missing growth rows
    df_train = df_train.dropna(subset=['gdp_growth'])

    df_train.to_csv('/Users/sanchitsuman/Satellite Data/train_data_indexed_growth.csv', index=True)
    df_train = df_train[~df_train.index.duplicated(keep="first")]
    
    

    mod_growth = PanelOLS.from_formula(
    'gdp_growth ~ avg_rad + tmean + ppt_total_mm + road_year_interaction + urban_year_interaction + EntityEffects + TimeEffects',
    data=df_train
    )
    res_growth = mod_growth.fit(cov_type='clustered', cluster_entity=True)
    return res_growth

def mod_simple(df_train):

   
    df_train = df_train.set_index(['county_id', 'year'])

    
    
    df_train.to_csv('/Users/sanchitsuman/Satellite Data/train_data_indexed.csv', index=True)
    df_train = df_train[~df_train.index.duplicated(keep="first")]
    #print("duplicates2 : ",df_train.index.duplicated().sum())
    #print(df_train.dtypes)
    #print("df_train :",len(df_train))
    print("unique :",df_train.nunique())
    mod = PanelOLS.from_formula(
        "GDP ~  road_length_km + avg_rad + tmean + urban_pct + ppt_total_mm + road_year_interaction + EntityEffects + TimeEffects",
        data=df_train,
        drop_absorbed=True
    )
    res = mod.fit(cov_type="clustered", cluster_entity=True)
    return res

def main():
    df_features = pd.read_csv('/Users/sanchitsuman/Satellite Data/features.csv')
    df_labels = pd.read_csv('/Users/sanchitsuman/Satellite Data/CountyGDP.csv')
    df_census = pd.read_csv('/Users/sanchitsuman/Satellite Data/county_population_2015_2023.csv')

    
    #df_train = training_data(df_features, df_labels,df_census)

    #model = PanelOLS.from_formula('GDP ~ 1 + road_length_km + tmean + urban_pct + EntityEffects', data=df)
    #results = model.fit()
    #df = df.merge(df_long, how='inner', left_on=['state_name', 'county_name', 'year'], right_on=['State', 'County', 'Year'])
    df_long = df_labels[["State", "County", "2020", "2021", "2022","2023"]].melt(
        id_vars=["State", "County"], 
        value_vars=["2020", "2021", "2022","2023"],
        var_name="Year", 
        value_name="GDP"
    )

    # Step 2: Convert Year to integer
    df_long["Year"] = df_long["Year"].astype(int)
    df_long['GDP'] = df_long["GDP"].str.replace(",", "").astype(float)

    # Step 3: Sort (important for creating lags later)
    df_long = df_long.sort_values(by=["State", "County", "Year"]).reset_index(drop=True)

    print(df_long.dtypes)
    df_long.to_csv('/Users/sanchitsuman/Satellite Data/long_labels.csv', index=False)
    df_train = training_data(df_features, df_long,df_census)
     
    
    print("df_features :",len(df_features))
    print("df_long :",len(df_long))

    print("df_train :",len(df_train))
    df_train["per_capita"] = df_train["GDP"] / df_train["population"]
    df_train["county_id"] = df_train["state_name"] + "_" + df_train["county_name"]
    df_train["road_year_interaction"] = df_train["road_length_km"] * df_train["year"]
    
    
    df_train =df_train[["county_id", "year", "GDP","road_length_km", "tmean", "urban_pct", "avg_rad","ppt_total_mm","road_year_interaction"]]
    #.to_csv('/Users/sanchitsuman/Satellite Data/train_data.csv', index=False)



    res =  mod_simple(df_train)
    print(res.summary)

    res_growth = mod_growth(df_train)
    print(res_growth.summary)

    with open('/Users/sanchitsuman/Satellite Data/res_growth_summary.txt', 'w') as f:
        f.write(str(res_growth.summary))

    '''
   
    df_train['log_gdp'] = df_train['GDP'].apply(np.log)
    df_train['gdp_growth'] = df_train.groupby(level='county_id')['log_gdp'].diff()
    # Remove missing growth rows
    df_growth = df_train.dropna(subset=['gdp_growth'])


    df_growth = df_growth.set_index(['county_id', 'year'])

    mod_growth = PanelOLS.from_formula(
    'gdp_growth ~ avg_rad + tmean + ppt_total_mm + road_year_interaction + urban_year_interaction + EntityEffects + TimeEffects',
    data=df_growth
    )
    res_growth = mod_growth.fit(cov_type='clustered', cluster_entity=True)
    print(res_growth.summary)
    '''
    
    #print(df_train.isnull().sum())
    #print(df_train.isnull().mean() * 100)
    #print(df_train.index.get_level_values('county_id').isnull().sum())
    #print(df_train.index.get_level_values('year').isnull().sum())
if __name__ == "__main__":
    main()