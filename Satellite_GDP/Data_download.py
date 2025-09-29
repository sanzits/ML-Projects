import ee
import csv

# ---------------------------
# Authentication & Initialization
# ---------------------------
def initialize_earth_engine(project_name="causal1"):
    ee.Authenticate()
    ee.Initialize(project=project_name)
    print(ee.String('Hello from the Earth Engine servers!').getInfo())

# ---------------------------
# Data Retrieval Functions
# ---------------------------

'''def get_viirs_mean_light(geometry, year):
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"

    viirs = ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG") \
                .filterDate(start_date, end_date) \
                .select("avg_rad")

    mean_img = viirs.mean()
    mean_dict = mean_img.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geometry,
        scale=500,
        maxPixels=1e9
    )
    return [mean_dict.get("avg_rad").getInfo()]

def get_all_infrastructure_metrics(geometry, year):
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"

    # ROAD LENGTH
    try:
        roads = ee.FeatureCollection("TIGER/2016/Roads")
        clipped_roads = roads.filterBounds(geometry)
        road_length_km = clipped_roads.geometry().length().getInfo() / 1000
    except Exception:
        road_length_km = None

    return {
        "year": year,
        "road_length_km": road_length_km
    }
'''
def get_county_year_metrics(county_feature, year):
    """
    Returns a dictionary of infrastructure and environmental metrics for a county and year.

    Args:
        county_feature (ee.Feature): Single county feature
        year (int): Year to process

    Returns:
        dict: Metrics including VIIRS, road length, TerraClimate, and urban %
    """
    geom = county_feature.geometry()
    
    # --- VIIRS ---
    viirs_col = ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG") \
        .filterDate(f"{year}-01-01", f"{year}-12-31")
    viirs_mean = viirs_col.mean().reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geom,
        scale=500
    ).get('avg_rad')  # adjust band name if needed
    
    # --- Road length (fixed) ---
    roads = ee.FeatureCollection("TIGER/2016/Roads").filterBounds(geom)
    roads = roads.map(lambda f: f.set('length_m', f.geometry().length()))
    road_length = roads.aggregate_sum('length_m')
    
    # --- TerraClimate ---
    start = ee.Date.fromYMD(year, 1, 1)
    end = start.advance(1, 'year')
    terraclimate = ee.ImageCollection('IDAHO_EPSCOR/TERRACLIMATE').filterDate(start, end)
    
    # Mean annual temp
    tmean_monthly = terraclimate.map(lambda im: im.expression(
        '((tmmx + tmmn)/2)', {'tmmx': im.select('tmmx'), 'tmmn': im.select('tmmn')}
    ).rename('tmean'))
    tmean_annual = tmean_monthly.mean().reduceRegion(
        reducer=ee.Reducer.mean(), geometry=geom, scale=500
    ).get('tmean')
    
    # Total precipitation
    ppt_total = terraclimate.select('pr').sum().reduceRegion(
    reducer=ee.Reducer.sum(), geometry=geom, scale=500
    ).get('pr')
    
    # --- MODIS urban ---
    modis_lc = ee.ImageCollection('MODIS/061/MCD12Q1') \
        .filter(ee.Filter.calendarRange(year, year, 'year')).first().select('LC_Type1')
    urban_mask = modis_lc.eq(13)
    urban_pct = urban_mask.reduceRegion(
        reducer=ee.Reducer.mean(), geometry=geom, scale=500
    ).get('LC_Type1')  # fraction 0-1
    
    return {
        "avg_rad": viirs_mean.getInfo() if viirs_mean else None,
        "road_length_km": (road_length.getInfo()/1000) if road_length else 0,
        "tmean": tmean_annual.getInfo() if tmean_annual else None,
        "ppt_total_mm": ppt_total.getInfo() if ppt_total else None,
        "urban_pct": urban_pct.getInfo()*100 if urban_pct else 0
    }

# ---------------------------
# County Dictionary Creation
# ---------------------------
def get_county_dict():
    counties = ee.FeatureCollection("TIGER/2018/Counties")

    names = counties.aggregate_array("NAME").getInfo()
    statefps = counties.aggregate_array("STATEFP").getInfo()
    countyfps = counties.aggregate_array("COUNTYFP").getInfo() 

    statefp_to_name = {
        "01": "Alabama", "02": "Alaska", "04": "Arizona", "05": "Arkansas", "06": "California",
        "08": "Colorado", "09": "Connecticut", "10": "Delaware", "11": "District of Columbia",
        "12": "Florida", "13": "Georgia", "15": "Hawaii", "16": "Idaho", "17": "Illinois",
        "18": "Indiana", "19": "Iowa", "20": "Kansas", "21": "Kentucky", "22": "Louisiana",
        "23": "Maine", "24": "Maryland", "25": "Massachusetts", "26": "Michigan", "27": "Minnesota",
        "28": "Mississippi", "29": "Missouri", "30": "Montana", "31": "Nebraska", "32": "Nevada",
        "33": "New Hampshire", "34": "New Jersey", "35": "New Mexico", "36": "New York",
        "37": "North Carolina", "38": "North Dakota", "39": "Ohio", "40": "Oklahoma", "41": "Oregon",
        "42": "Pennsylvania", "44": "Rhode Island", "45": "South Carolina", "46": "South Dakota",
        "47": "Tennessee", "48": "Texas", "49": "Utah", "50": "Vermont", "51": "Virginia",
        "53": "Washington", "54": "West Virginia", "55": "Wisconsin", "56": "Wyoming"
    }

    county_dict = {}
    for name, statefp, countyfp in zip(names, statefps, countyfps):
        statefp = statefp.zfill(2)
        countyfp = countyfp.zfill(3)
        geoid = statefp + countyfp  # create GEOID
        state_name = statefp_to_name.get(statefp, "Unknown")
        if state_name not in county_dict:
            county_dict[state_name] = {
                "STATEFP": statefp,
                "counties": []
            }
        county_dict[state_name]["counties"].append({
            "name": name,
            "COUNTYFP": countyfp,
            "GEOID": geoid
        })

    return county_dict


# ---------------------------
# Main Execution
# ---------------------------
def main():
    initialize_earth_engine()

    years = [2021,2022,2023]
    
    county_dict = get_county_dict()

    for year in years:
        infra_over_years = []
        for state_name, info in county_dict.items():
            statefp = info["STATEFP"]
            for county_dict_item in info["counties"]:
                county_name = county_dict_item["name"]
                geoid = county_dict_item["GEOID"]
                try:
                    county = ee.FeatureCollection("TIGER/2018/Counties") \
                        .filter(ee.Filter.eq("STATEFP", statefp)) \
                        .filter(ee.Filter.eq("NAME", county_name)) \
                        .first()
                    
                    metrics = get_county_year_metrics(county, year)
                    
                    infra_over_years.append({
                        "state_name": state_name,
                        "county_name": county_name,
                        "year": year,
                        "STATEFP": statefp,
                        "GEOID": geoid,
                        **metrics
                    })
                    
                    print(f"{state_name} - {county_name} ({year}): {metrics}")
                except Exception as e:
                    print(f"Error processing {county_name}, {state_name}: {str(e)}")      
        keys = sorted({k for d in infra_over_years for k in d.keys()})
        file_path = f"/Users/sanchitsuman/Satellite Data/county_data_{year}.csv"
        with open(file_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(infra_over_years)
        print(f"Data saved to {year}.csv")
    # Save to CSV
    

    

# ---------------------------
# Entry Point
# ---------------------------
if __name__ == "__main__":
    main()