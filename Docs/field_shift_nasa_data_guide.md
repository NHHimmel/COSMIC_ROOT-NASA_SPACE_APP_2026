# Field Shift NASA Earth Observation Data Pipeline & Developer Guide

## Four Unified API Backends Cover Every Field Shift Agricultural Dataset

The `field_shift_nasa_data.py` module provides a single, extensible interface for pulling, scaling, and merging all seven core satellite categories and five supplemental training features specified in the **Field Shift: Adapting Farms with NASA Data** project architecture. Instead of writing custom parsers for disparate archive centers, the library routes requests through four specialized backends matched to each sensor's resolution and delivery format:

* **[NASA POWER Agroclimatology API](https://www.google.com/search?q=https%3A%2F%2Fpower.larc.nasa.gov%2Fdocs%2Fservices%2Fapi%2F) (`NASAPowerClient`):** Delivers zero-authentication daily time series of temperature, relative/specific humidity, solar irradiance, photosynthetically active radiation (PAR), wind speed, bias-corrected precipitation, and three-layer soil wetness derived from [MERRA-2](https://www.google.com/search?q=https%3A%2F%2Fgmao.gsfc.nasa.gov%2Fgmao-products%2Fmerra-2%2F) and satellite solar observations ([POWER Dataset Catalog](https://www.google.com/search?q=https%3A%2F%2Fdata.nasa.gov%2Fdataset%2Fprediction-of-worldwide-energy-resources-power)).
* **[NASA AppEEARS REST API](https://www.google.com/search?q=https%3A%2F%2Fappeears.earthdatacloud.nasa.gov%2Fapi%2F) (`AppEEARSClient`):** Executes server-side point and polygon extractions across [MODIS](https://www.google.com/search?q=https%3A%2F%2Fwww.earthdata.nasa.gov%2Fdata%2Finstruments%2Fmodis), [Harmonized Landsat Sentinel-2 (HLS)](https://www.google.com/search?q=https%3A%2F%2Fwww.earthdata.nasa.gov%2Fdata%2Fprojects%2Fhls), [ECOSTRESS](https://www.google.com/search?q=https%3A%2F%2Fwww.earthdata.nasa.gov%2Fdata%2Finstruments%2Fecostress), [SMAP](https://www.google.com/search?q=https%3A%2F%2Fwww.earthdata.nasa.gov%2Fdata%2Fplatforms%2Fspace-based-platforms%2Fsmap), and [SRTM](https://www.google.com/search?q=https%3A%2F%2Fwww.earthdata.nasa.gov%2Fdata%2Ftools%2Fappeears), returning analysis-ready CSVs without downloading multi-gigabyte raw tiles.
* **[NASA earthaccess + xarray](https://www.google.com/search?q=https%3A%2F%2Fwww.earthdata.nasa.gov%2Fdata%2Ftools%2Fearthaccess) (`EarthAccessClient`):** Authenticates once with your [NASA Earthdata Login](https://www.google.com/search?q=https%3A%2F%2Furs.earthdata.nasa.gov%2F) to stream or download raw HDF5/NetCDF granules directly from NASA Earthdata Cloud, including [GPM IMERG Daily Rainfall](https://www.google.com/search?q=https%3A%2F%2Fnasa-openscapes.github.io%2F2023-Cloud-Workshop-AGU%2Ftutorials%2FEarthdata_Subset_and_Plot.html), [SMAP Level-4 Root-Zone Soil Moisture](https://www.google.com/search?q=https%3A%2F%2Fwww.earthdata.nasa.gov%2Fdata%2Fcatalog%2Fnsidc-cprd-spl4smgp-008), and [GRACE/GRACE-FO Drought Percentiles](https://www.google.com/search?q=https%3A%2F%2Fwww.earthdata.nasa.gov%2Fdata%2Fcatalog%2Fges-disc-gracedadm-clsm025gl-7d-3.0).
* **[Google Earth Engine Python API](https://www.google.com/search?q=https%3A%2F%2Fdevelopers.google.com%2Fearth-engine%2Ftutorials%2Fcommunity%2Fintro-to-python-api) (`EarthEngineClient`):** Performs cloud-side spatial reductions (`ee.Reducer.mean()` and `ee.Reducer.mode()`) over buffered farm points or exact field boundary polygons, aligning multi-resolution rasters into a clean `pandas.DataFrame`.

## Complete Data Dictionary Maps Every PDF Input to Verified NASA Collections

Every variable required for predicting crop-rotation sequences, soil-health improvements, water-smart selections, and climate-risk trade-offs is registered in `DATASET_REGISTRY` with its native scale factors and physical units.

| Field Shift PDF Input | NASA Mission / Sensor | Collection / Product ID | Extracted Variables / Bands | Spatial & Temporal Resolution | Physical Units & Scaling |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Rainfall** | [GPM IMERG](https://www.google.com/search?q=https%3A%2F%2Fdevelopers.google.com%2Fearth-engine%2Fdatasets%2Fcatalog%2FNASA_GPM_L3_IMERG_V07) & [POWER](https://www.google.com/search?q=https%3A%2F%2Fpower.larc.nasa.gov%2Fdocs%2Fservices%2Fapi%2F) | `GPM_3IMERGDF` (v07) / `NASA/GPM_L3/IMERG_V07` / `PRECTOTCORR` | `precipitation`, `probabilityLiquidPrecipitation`, `PRECTOTCORR` | $0.1^\circ$ (~11 km), half-hourly & daily | $\text{mm/day}$ or $\text{mm/hr}$ (scale `1.0`) |
| **Soil moisture** | [SMAP L4](https://www.google.com/search?q=https%3A%2F%2Fwww.earthdata.nasa.gov%2Fdata%2Fcatalog%2Fnsidc-cprd-spl4smgp-008), [SMAP L3](https://www.google.com/search?q=https%3A%2F%2Fwww.earthdata.nasa.gov%2Fdata%2Fcatalog%2Fnsidc-cprd-spl3smp-e-006), [POWER](https://www.google.com/search?q=https%3A%2F%2Fwww.mdpi.com%2F2624-7402%2F7%2F1%2F6) | `SPL4SMGP` (v008) / `SPL3SMP_E` (v006) / `NASA/SMAP/SPL4SMGP/008` | `sm_surface` (0–5 cm), `sm_rootzone` (0–100 cm), `sm_profile`, `GWETTOP`, `GWETROOT`, `GWETPROF` | 9 km EASE-Grid 2.0, 3-hourly & daily | $\text{m}^3/\text{m}^3$ volumetric or $0.0\text{–}1.0$ relative wetness |
| **Temperature, humidity, radiation** | [POWER](https://www.google.com/search?q=https%3A%2F%2Fdata.nasa.gov%2Fdataset%2Fprediction-of-worldwide-energy-resources-power) / [MERRA-2](https://www.google.com/search?q=https%3A%2F%2Fgmao.gsfc.nasa.gov%2Fgmao-products%2Fmerra-2%2F) | `community=AG` / `M2SDNXSLV` (v5.12.4) | `T2M`, `T2M_MAX`, `T2M_MIN`, `RH2M`, `QV2M`, `ALLSKY_SFC_SW_DWN`, `ALLSKY_SFC_PAR_TOT`, `WS2M` | $0.5^\circ \times 0.625^\circ$, daily | $^\circ\text{C}$, $\%$, $\text{MJ/m}^2/\text{day}$, $\text{W/m}^2$, $\text{m/s}$ |
| **Vegetation health (NDVI/EVI)** | [MODIS](https://www.google.com/search?q=https%3A%2F%2Fdevelopers.google.com%2Fearth-engine%2Fdatasets%2Fcatalog%2Fmodis), [Landsat 8](https://www.google.com/search?q=https%3A%2F%2Fwww.earthdata.nasa.gov%2Fdata%2Fplatforms%2Fspace-based-platforms%2Flandsat-8), [HLS](https://www.google.com/search?q=https%3A%2F%2Fwww.earthdata.nasa.gov%2Fdata%2Fprojects%2Fhls) | `MOD13Q1.061` / [HLSL30 v2.0](https://www.google.com/search?q=https%3A%2F%2Fwww.earthdata.nasa.gov%2Fdata%2Fcatalog%2Flpcloud-hlss30-2.0) / [GEE HLSS30](https://www.google.com/search?q=https%3A%2F%2Fdevelopers.google.com%2Fearth-engine%2Fdatasets%2Fcatalog%2FNASA_HLS_HLSS30_v002) | `NDVI`, `EVI`, `B05` (NIR), `B04` (Red), `B02` (Blue), `Fmask` | 250 m (16-day MODIS) & 30 m (2–3 day HLS) | Dimensionless $[-1, 1]$ (`MOD13Q1` scale `0.0001`) |
| **Evapotranspiration** | [MODIS](https://www.google.com/search?q=https%3A%2F%2Fdevelopers.google.com%2Fearth-engine%2Fdatasets%2Fcatalog%2Fmodis) / [ECOSTRESS](https://www.google.com/search?q=https%3A%2F%2Fwww.earthdata.nasa.gov%2Fdata%2Fcatalog%2Flpcloud-eco-l3t-jet-003) | `MOD16A2GF.061` / [ECO\_L3T\_JET.002](https://www.google.com/search?q=https%3A%2F%2Fwww.earthdata.nasa.gov%2Fdata%2Fcatalog%2Flpcloud-eco-l3t-jet-002) | `ET_500m`, `PET_500m`, `ETdaily`, `PTJPLSMinst` | 500 m (8-day MODIS) & 70 m (ECOSTRESS) | $\text{kg/m}^2/\text{8day}$ (`MOD16` scale `0.1`), $\text{mm/day}$ |
| **Plant water stress** | [ECOSTRESS](https://www.google.com/search?q=https%3A%2F%2Fwww.earthdata.nasa.gov%2Fdata%2Finstruments%2Fecostress) | [ECO\_L4T\_ESI.003](https://www.google.com/search?q=https%3A%2F%2Fwww.earthdata.nasa.gov%2Fdata%2Fcatalog%2Flpcloud-eco-l4t-esi-003) / [ECO\_L4T\_WUE.003](https://www.google.com/search?q=https%3A%2F%2Fwww.earthdata.nasa.gov%2Fdata%2Fcatalog%2Flpcloud-eco-l4t-wue-003) | `ESIavg` ($ET/PET$ ratio), `WUEavg` (water use efficiency) | 70 m, variable ISS overpass | Ratio $[0, 1]$ and $\text{g C / kg H}_2\text{O}$ |
| **Land cover / crop history** | [MODIS](https://www.google.com/search?q=https%3A%2F%2Fdevelopers.google.com%2Fearth-engine%2Fdatasets%2Fcatalog%2Fmodis), [Landsat 8](https://www.google.com/search?q=https%3A%2F%2Fwww.earthdata.nasa.gov%2Fdata%2Fplatforms%2Fspace-based-platforms%2Flandsat-8) | `MCD12Q1.061` (`MODIS/061/MCD12Q1`) | `LC_Type1` (IGBP class 12=Croplands, 14=Crop/Natural Mosaic), `LC_Prop2` (FAO LCCS) | 500 m, annual | Categorical integer class |
| **Flood / drought indicators** | [GRACE / GRACE-FO](https://www.google.com/search?q=https%3A%2F%2Fcmr.earthdata.nasa.gov%2Fsearch%2Fconcepts%2FC2283200566-GES_DISC.html) & [GPM](https://www.google.com/search?q=https%3A%2F%2Fdevelopers.google.com%2Fearth-engine%2Fdatasets%2Fcatalog%2FNASA_GPM_L3_IMERG_V07) | `GRACEDADM_CLSM025GL_7D` (v3.0) + derived GPM metrics | `gws_inst` (groundwater), `rtzsm_inst` (root-zone), `sfsm_inst` (surface), `consecutive_dry_days`, `precip_7d_sum_mm` | $0.25^\circ$ (~28 km), weekly (GRACE) & daily | Wetness percentiles $[0, 100]\%$, days, $\text{mm}$ |
| **Land-surface temperature** | [MODIS](https://www.google.com/search?q=https%3A%2F%2Fdevelopers.google.com%2Fearth-engine%2Fdatasets%2Fcatalog%2Fmodis) / [ECOSTRESS](https://www.google.com/search?q=https%3A%2F%2Fwww.earthdata.nasa.gov%2Fdata%2Fcatalog%2Flpcloud-eco-l2t-lste-003) | `MOD11A2.061` / `ECO_L2T_LSTE.003` | `LST_Day_1km`, `LST_Night_1km`, `LST` | 1 km (8-day MODIS) & 70 m (ECOSTRESS) | $^\circ\text{C} = \text{DN} \times 0.02 - 273.15$ |
| **Elevation & terrain** | [SRTM](https://www.google.com/search?q=https%3A%2F%2Fappeears.earthdatacloud.nasa.gov%2Fapi%2F) / NASADEM | `SRTMGL1_NC.003` / `USGS/SRTMGL1_003` / `NASADEM_HGT` | `elevation`, `slope_deg`, `aspect_deg` | 30 m, static | Meters ($\text{m}$) and degrees ($^\circ$) |

## Environment Setup and Authentication Take Under Five Minutes

### 1. Install Required Python Libraries

```bash
pip install requests pandas numpy earthaccess xarray h5netcdf earthengine-api
```

### 2. Configure NASA Earthdata Login Credentials (`earthaccess` & `AppEEARS`)

Register for a free account at [NASA Earthdata Login](https://www.google.com/search?q=https%3A%2F%2Furs.earthdata.nasa.gov%2F) and export your credentials as environment variables (or store them in `~/.netrc`):

```bash
export EARTHDATA_USERNAME="your_earthdata_username"
export EARTHDATA_PASSWORD="your_earthdata_password"
```

When using [earthaccess](https://www.google.com/search?q=https%3A%2F%2Fearthaccess.readthedocs.io%2F), calling `earthaccess.login(strategy="all", persist=True)` automatically detects `EARTHDATA_USERNAME` and `EARTHDATA_PASSWORD` or prompts once and writes `~/.netrc` for future runs.

### 3. Authenticate Google Earth Engine (Optional for Cloud Polygon Reductions)

If you use `EarthEngineClient` to aggregate 30 m [HLS](https://www.google.com/search?q=https%3A%2F%2Fdevelopers.google.com%2Fearth-engine%2Fdatasets%2Fcatalog%2FNASA_HLS_HLSS30_v002) or 250 m [MODIS](https://www.google.com/search?q=https%3A%2F%2Fdevelopers.google.com%2Fearth-engine%2Fdatasets%2Fcatalog%2Fmodis) rasters over farm polygons in the cloud, authenticate once via the CLI:

```bash
earthengine authenticate
```

## End-to-End Code Examples Show How to Fetch and Harmonize Farm Features

### Workflow A: Instant Single-Farm Daily Weather, Soil Moisture & Flood/Drought Baseline (No Auth)

Use `NASAPowerClient` or `FieldShiftFeatureBuilder` to pull a complete daily meteorological and soil-moisture feature matrix for any farm coordinate immediately:

```python
from field_shift_nasa_data import FarmLocation, FieldShiftFeatureBuilder

# 1. Define target farm location and farmer metadata
farm = FarmLocation(
    farm_id="rajshahi_plot_12",
    latitude=24.3745,
    longitude=88.6042,
    buffer_meters=250.0,
    irrigation_access=True,
    farmer_priority="water_conservation",
)

# 2. Build daily training table (POWER weather, radiation, rainfall, soil wetness + stress flags)
builder = FieldShiftFeatureBuilder()
daily_df = builder.build_farm_training_table(
    location=farm,
    start_date="2024-01-01",
    end_date="2024-12-31",
    include_gee_features=False,
    include_earthaccess_features=False,
)

# 3. Compute seasonal aggregates for crop-rotation model input
seasonal_features = builder.summarize_growing_season(daily_df)
print(daily_df[["T2M", "RH2M", "ALLSKY_SFC_SW_DWN", "PRECTOTCORR", "GWETROOT", "vpd_kpa"]].head())
print("Seasonal Summary:", seasonal_features)
```

### Workflow B: Full Multi-Sensor Fusion with Google Earth Engine or `earthaccess`

To merge 30 m [HLS](https://www.google.com/search?q=https%3A%2F%2Fwww.earthdata.nasa.gov%2Fdata%2Fprojects%2Fhls) and 250 m [MODIS](https://www.google.com/search?q=https%3A%2F%2Fwww.earthdata.nasa.gov%2Fdata%2Finstruments%2Fmodis) NDVI/EVI, 500 m [MODIS Evapotranspiration](https://www.google.com/search?q=https%3A%2F%2Fdevelopers.google.com%2Fearth-engine%2Fdatasets%2Fcatalog%2Fmodis), 1 km Land Surface Temperature, 30 m SRTM terrain slope/elevation, 9 km [SMAP L4 Root-Zone Soil Moisture](https://www.google.com/search?q=https%3A%2F%2Fwww.earthdata.nasa.gov%2Fdata%2Fcatalog%2Fnsidc-cprd-spl4smgp-008), and weekly [GRACE Groundwater Drought Percentiles](https://www.google.com/search?q=https%3A%2F%2Fwww.earthdata.nasa.gov%2Fdata%2Fcatalog%2Fges-disc-gracedadm-clsm025gl-7d-3.0) onto the same daily index:

```python
from field_shift_nasa_data import (
    EarthAccessClient,
    EarthEngineClient,
    FarmLocation,
    FieldShiftFeatureBuilder,
)

farm = FarmLocation(
    farm_id="bogura_field_04",
    latitude=24.8481,
    longitude=89.3730,
    buffer_meters=300.0,
    irrigation_access=False,
    farmer_priority="soil_health",
)

builder = FieldShiftFeatureBuilder(
    gee_client=EarthEngineClient(project_id="your-gcp-project-id"),
    earthaccess_client=EarthAccessClient(),
)

full_training_df = builder.build_farm_training_table(
    location=farm,
    start_date="2024-06-01",
    end_date="2024-11-30",
    include_gee_features=True,          # Adds MODIS/HLS NDVI & EVI, MOD16 ET/PET, LST, SRTM, Land Cover
    include_earthaccess_features=True,  # Adds GPM IMERG daily precip, SMAP L4, and GRACE drought percentiles
    interpolate_remote_sensing=True,    # Linearly interpolates 8-day/16-day composites onto daily grid
)

full_training_df.to_csv("bogura_field_04_training_features.csv")
```

### Workflow C: Batch Multi-Farm Extraction via NASA AppEEARS REST API

When building a historical training dataset across dozens or hundreds of farm coordinates including 70 m [ECOSTRESS Evapotranspiration (ECO\_L3T\_JET)](https://www.google.com/search?q=https%3A%2F%2Fwww.earthdata.nasa.gov%2Fdata%2Fcatalog%2Flpcloud-eco-l3t-jet-002) and [Evaporative Stress Index (ECO\_L4T\_ESI)](https://www.google.com/search?q=https%3A%2F%2Fwww.earthdata.nasa.gov%2Fdata%2Fcatalog%2Flpcloud-eco-l4t-esi-003), submit a single batch point request to [AppEEARS](https://www.google.com/search?q=https%3A%2F%2Fappeears.earthdatacloud.nasa.gov%2Fapi%2F):

```python
from field_shift_nasa_data import AppEEARSClient, FarmLocation

farms = [
    FarmLocation("farm_north", latitude=25.7439, longitude=89.2752, farmer_priority="balanced"),
    FarmLocation("farm_delta", latitude=22.8456, longitude=89.5403, farmer_priority="water_conservation"),
]

appeears = AppEEARSClient()  # Reads EARTHDATA_USERNAME and EARTHDATA_PASSWORD
task_id = appeears.submit_point_task(
    locations=farms,
    start_date="2024-01-01",
    end_date="2024-12-31",
    task_name="FieldShift_Regional_Training_Batch",
)

# Polls until NASA servers finish processing and downloads all CSV results
csv_files = appeears.wait_and_download_csvs(task_id=task_id, output_dir="./appeears_outputs")
```

## Derived Agronomic Indicators Map Directly to Field Shift System Outputs

To support the five recommendation goals on Page 2 of your PDF, `NASAPowerClient.compute_agronomic_indicators()` and `FieldShiftFeatureBuilder.summarize_growing_season()` automatically engineer domain-specific features from the raw satellite bands:

* **Water-smart crop selection (`vpd_kpa`, `evaporative_stress_ratio`, `GWETROOT`, `sm_rootzone`):** Combines Tetens vapor pressure deficit ($\text{kPa}$) with the ratio of actual to potential evapotranspiration ($ET / PET$) and 0–100 cm root-zone soil moisture to quantify seasonal crop water demand versus available soil water storage.
* **Climate-risk-aware planning (`consecutive_dry_days`, `heavy_rain_event_flag`, `heat_stress_day_flag`, `grace_groundwater_pctl`):** Tracks the longest dry-spell streak ($<1\text{ mm/day}$), extreme flood-inducing rain days ($\ge 50\text{ mm/day}$), reproductive heat-stress days ($T_{\max} \ge 35^\circ\text{C}$), and multi-year [GRACE groundwater percentiles](https://www.google.com/search?q=https%3A%2F%2Fcmr.earthdata.nasa.gov%2Fsearch%2Fconcepts%2FC2283200566-GES_DISC.html) to score candidate rotations against drought and waterlogging hazards.
* **Soil-health improvement & crop-rotation sequence (`modis_ndvi`, `hls_ndvi_30m`, `modis_igbp_landcover_class`):** Tracks multi-season vegetation vigor trajectories and fallow-period residue cover at 30 m–250 m resolution alongside farmer priority flags (`balanced`, `water_conservation`, `soil_health`).
