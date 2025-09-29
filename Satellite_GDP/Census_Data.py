import requests
import pandas as pd
import time

API_KEY = "065ef4d6178fd0c5aa0083c533a759fd3a47c362"
START_YEAR = 2015
END_YEAR = 2023
MAX_RETRIES = 3
SLEEP_BETWEEN = 1  # seconds

def get_county_population(year):
    """
    Fetches ACS 5-year total population for all U.S. counties for a given year.
    Handles API errors and retries.
    """
    endpoint = f"https://api.census.gov/data/{year}/acs/acs5"
    params = {
        "get": "NAME,B01003_001E",
        "for": "county:*",
        "in": "state:*",
        "key": API_KEY
    }
    
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(endpoint, params=params, timeout=10)
            r.raise_for_status()
            data = r.json()  # May raise ValueError if response is invalid
            df = pd.DataFrame(data[1:], columns=data[0])
            df["year"] = year
            df.rename(columns={
                "B01003_001E": "population",
                "state": "STATEFP",
                "county": "COUNTYFP"
            }, inplace=True)
            df["GEOID"] = df["STATEFP"] + df["COUNTYFP"]
            df["population"] = df["population"].astype(int)
            return df
        except requests.exceptions.RequestException as e:
            print(f"Request error for year {year}: {e}")
        except ValueError:
            print(f"Invalid JSON response for year {year}: {r.text[:200]}...")
        
        print(f"Retrying ({attempt}/{MAX_RETRIES})...")
        time.sleep(SLEEP_BETWEEN)
    
    print(f"Failed to fetch data for year {year}. Skipping.")
    return pd.DataFrame()  # Empty DF if all retries fail

# --- Fetch for all years and concatenate ---
all_years = []
for y in range(START_YEAR, END_YEAR + 1):
    print(f"Fetching population for year {y}...")
    df_year = get_county_population(y)
    if not df_year.empty:
        all_years.append(df_year)

all_county_pop = pd.concat(all_years, ignore_index=True)
print(all_county_pop.head())
print(all_county_pop.shape)

# Save to CSV
all_county_pop.to_csv("/Users/sanchitsuman/Satellite Data/county_population_2015_2023.csv", index=False)