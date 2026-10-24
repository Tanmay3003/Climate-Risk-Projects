#Section - 1 Imports
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Section - 2 Loading the dataset
ds = xr.open_dataset(r"C:\Users\nandk\Downloads\P_2015_2100.nc")

# Define assets and their data
assets_climate = [
    {"asset_id": "A001", "name": "Mumbai Port",
     "lat": 19.0, "lon": 72.8,
     "baseline_mm_day": 6.5, "flood_threshold_mm_day": 11.0},

    {"asset_id": "A002", "name": "Chennai Warehouse",
     "lat": 13.0, "lon": 80.2,
     "baseline_mm_day": 3.8, "flood_threshold_mm_day": 7.0},

    {"asset_id": "A003", "name": "Delhi Office",
     "lat": 28.6, "lon": 77.2,
     "baseline_mm_day": 2.1, "flood_threshold_mm_day": 5.0},

    {"asset_id": "A004", "name": "Kolkata Factory",
     "lat": 22.5, "lon": 88.3,
     "baseline_mm_day": 5.2, "flood_threshold_mm_day": 9.0},

    {"asset_id": "A005", "name": "Pune Data Center",
     "lat": 18.5, "lon": 73.8,
     "baseline_mm_day": 4.1, "flood_threshold_mm_day": 8.0},
]

df = pd.DataFrame(assets_climate)

#Section - 3 Extraction Function
def extract_asset_precip(ds,lat,lon,baseline_mm_day):
    anomaly = ds['pr'].sel(
        lat = lat,
        lon = lon,
        method = 'nearest'
    ).values
    
    projected = baseline_mm_day * (1 + anomaly / 100)
    
    return np.array(projected).flatten()

#Section - 4 Compute Flood Probabilities
def compute_flood_probability(precip_array,flood_threshold):
    
    flood_years = np.sum(precip_array > flood_threshold)
    
    total_years = len(precip_array)
    
    return float(flood_years / total_years)

#Section - 5 Extract precipitation for all assets
asset_precip = df.apply(
    lambda row: extract_asset_precip(
        ds,
        row['lat'],
        row['lon'],
        row['baseline_mm_day']
    ),
    axis = 1
)

asset_precip = np.array(asset_precip.to_list())

for i, row in df.iterrows():
    
    df.at[i, 'derived_flood_prob'] = compute_flood_probability(
        asset_precip[i],
        row["flood_threshold_mm_day"]
    )

#Section - 5 Print Result
print("\nCLIMATE-DERIVED FLOOD PROBABILITIES\n")
print(df.to_string(index = False))

#Plot the result
plt.figure(figsize=(10,6))

plt.bar(
    df["name"],
    df["derived_flood_prob"]
)

plt.xlabel("Asset Name")
plt.ylabel("Annual Flood Probability")

plt.title(
    "Climate-Derived Annual Flood Probability by Asset (SSP1-1.9)"
)

plt.xticks(rotation=25, ha= "right")
plt.tight_layout()

plt.savefig("climate_flood_probs.png")
plt.show()

#Section - 7 Comparison
original_flood_probs = {
    "Mumbai Port": 0.08,
    "Chennai Warehouse": 0.04,
    "Delhi Office": 0.01,
    "Kolkata Factory": 0.06,
    "Pune Data Center": 0.02,
}

comparison = df[["name","derived_flood_prob"]].copy()

comparison["original_flood_probs"] = (
    comparison["name"].map(original_flood_probs)
)

comparison["difference"] = (
    comparison["derived_flood_prob"] - comparison["original_flood_probs"]
)

print("\nPORTFOLIO COMPARISON\n")
print(comparison.to_string(index=False))

print("\nProjected precip ranges per asset:\n")
for i, row in df.iterrows():
    precip = asset_precip[i]
    print(f"{row['name']}")
    print(f"  Min: {precip.min():.2f}  Max: {precip.max():.2f}  "
          f"Mean: {precip.mean():.2f}  Threshold: {row['flood_threshold_mm_day']}")
    print(f"  Years above threshold: {int(np.sum(precip > row['flood_threshold_mm_day']))}/86\n")

ds.close()
