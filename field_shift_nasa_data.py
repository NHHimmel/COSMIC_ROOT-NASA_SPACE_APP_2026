"""
Field Shift: Generalizable NASA Earth Observation Data Fetching Library
=======================================================================

A modular, multi-backend Python library for fetching, scaling, and harmonizing
all NASA Earth science datasets required by the "Field Shift: Adapting Farms
with NASA Data" agricultural decision-support system.

Covered PDF Data Categories & Missions
--------------------------------------
1. Rainfall:
   - GPM / IMERG (`GPM_3IMERGDF` v07, `NASA/GPM_L3/IMERG_V07`, NASA POWER `PRECTOTCORR`)
2. Soil Moisture:
   - SMAP (`SPL4SMGP` v008 3-hourly 9 km surface & root-zone, `SPL3SMP_E` v006 daily 9 km,
     NASA POWER `GWETTOP`, `GWETROOT`, `GWETPROF`)
3. Temperature, Humidity, Radiation (& Wind):
   - NASA POWER / MERRA-2 (`T2M`, `T2M_MAX`, `T2M_MIN`, `RH2M`, `QV2M`,
     `ALLSKY_SFC_SW_DWN`, `ALLSKY_SFC_PAR_TOT`, `WS2M`, MERRA-2 `M2SDNXSLV`)
4. Vegetation Health (NDVI / EVI):
   - MODIS (`MOD13Q1` v061 250 m 16-day), Harmonized Landsat Sentinel-2 (`HLSL30` /
     `HLSS30` v2.0 30 m), Landsat 8/9 (`LANDSAT/LC08/C02/T1_L2`)
5. Evapotranspiration & Plant Water Stress:
   - MODIS (`MOD16A2GF` v061 500 m 8-day gap-filled ET & PET)
   - ECOSTRESS (`ECO_L3T_JET` 70 m ET, `ECO_L4T_ESI` 70 m Evaporative Stress Index,
     `ECO_L4T_WUE` 70 m Water Use Efficiency)
6. Land Cover / Crop History:
   - MODIS (`MCD12Q1` v061 500 m yearly IGBP & FAO LCCS land cover), Landsat/HLS
7. Flood / Drought Indicators:
   - GRACE / GRACE-FO Data Assimilation (`GRACEDADM_CLSM025GL_7D` v3.0 weekly
     groundwater, root-zone, and surface soil moisture percentiles)
   - GPM / IMERG rolling cumulative rainfall, extreme rainfall quantiles, and
     consecutive dry days (CDD)
8. Additional Training Features (PDF Page 1):
   - Land-Surface Temperature (LST): MODIS `MOD11A2` v061 (1 km) & ECOSTRESS `ECO_L2T_LSTE` (70 m)
   - Elevation & Terrain (Slope/Aspect): NASA SRTM 30 m (`SRTMGL1_NC.003` / `USGS/SRTMGL1_003`)

Installation
------------
    pip install requests pandas numpy earthaccess xarray h5netcdf earthengine-api
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger("field_shift_nasa")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", "%H:%M:%S")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# =============================================================================
# 1. DATA STRUCTURES & UNIFIED NASA DATASET REGISTRY
# =============================================================================


@dataclass
class FarmLocation:
    """
    Represents a target farm location (point coordinate, buffered field, or polygon)
    along with optional farmer/agronomic context from the Field Shift specification.

    Attributes
    ----------
    farm_id : str
        Unique identifier for the farm or field parcel.
    latitude : float
        Farm centroid latitude in decimal degrees (WGS84, [-90, 90]).
    longitude : float
        Farm centroid longitude in decimal degrees (WGS84, [-180, 180]).
    buffer_meters : float, default 250.0
        Radius around the centroid (in meters) used when reducing high-resolution
        rasters (e.g., 30 m HLS/Landsat/SRTM or 70 m ECOSTRESS).
    polygon_coords : list of (lon, lat) tuples, optional
        Optional closed ring of (longitude, latitude) vertices defining an exact
        field boundary polygon.
    irrigation_access : bool, optional
        Whether the farm has access to supplemental irrigation (PDF training input).
    farmer_priority : str, default 'balanced'
        Target optimization mode ('balanced', 'water_conservation', 'soil_health').
    """

    farm_id: str
    latitude: float
    longitude: float
    buffer_meters: float = 250.0
    polygon_coords: Optional[List[Tuple[float, float]]] = None
    irrigation_access: Optional[bool] = None
    farmer_priority: str = "balanced"

    def bounding_box(self, pad_degrees: float = 0.05) -> Tuple[float, float, float, float]:
        """
        Return (min_lon, min_lat, max_lon, max_lat) bounding box for granule searches.
        """
        if self.polygon_coords:
            lons = [pt[0] for pt in self.polygon_coords]
            lats = [pt[1] for pt in self.polygon_coords]
            return (min(lons), min(lats), max(lons), max(lats))
        return (
            self.longitude - pad_degrees,
            self.latitude - pad_degrees,
            self.longitude + pad_degrees,
            self.latitude + pad_degrees,
        )


# Comprehensive registry mapping every Field Shift PDF category to its official
# NASA collection identifiers, variables, scale factors, and units.
DATASET_REGISTRY: Dict[str, Dict[str, Any]] = {
    "weather_radiation": {
        "description": "Daily temperature, relative humidity, solar radiation, PAR, and wind speed",
        "mission": "POWER / MERRA-2",
        "power_parameters": [
            "T2M",                # 2m Mean Air Temperature (C)
            "T2M_MAX",            # 2m Max Air Temperature (C)
            "T2M_MIN",            # 2m Min Air Temperature (C)
            "RH2M",               # 2m Relative Humidity (%)
            "QV2M",               # 2m Specific Humidity (g/kg)
            "ALLSKY_SFC_SW_DWN",  # All-Sky Surface Shortwave Downward Irradiance (MJ/m^2/day)
            "ALLSKY_SFC_PAR_TOT", # All-Sky Surface Total PAR (W/m^2)
            "WS2M",               # 2m Wind Speed (m/s)
        ],
        "earthaccess": {
            "short_name": "M2SDNXSLV",
            "version": "5.12.4",
            "variables": ["T2MMEAN", "T2MMAX", "T2MMIN", "HOURNORAIN"],
        },
    },
    "rainfall": {
        "description": "Half-hourly and daily satellite-gauge merged precipitation estimates",
        "mission": "GPM / IMERG",
        "power_parameters": ["PRECTOTCORR"],  # Bias-corrected daily rainfall (mm/day)
        "earthaccess": {
            "short_name": "GPM_3IMERGDF",
            "version": "07",
            "group": None,
            "variables": ["precipitation", "probabilityLiquidPrecipitation"],
            "units": "mm/day",
        },
        "gee": {
            "collection": "NASA/GPM_L3/IMERG_V07",
            "bands": ["precipitation"],
            "scale_meters": 11132,
            "units": "mm/hr",
        },
    },
    "soil_moisture": {
        "description": "Surface (0-5 cm), root-zone (0-100 cm), and profile soil moisture",
        "mission": "SMAP",
        "power_parameters": [
            "GWETTOP",   # Surface Soil Wetness (0-5 cm, fraction 0-1)
            "GWETROOT",  # Root Zone Soil Wetness (0-100 cm, fraction 0-1)
            "GWETPROF",  # Total Profile Soil Moisture (surface to bedrock, fraction 0-1)
        ],
        "earthaccess_l4": {
            "short_name": "SPL4SMGP",
            "version": "008",
            "group": "Geophysical_Data",
            "variables": ["sm_surface", "sm_rootzone", "sm_profile", "sm_surface_wetness"],
            "units": "m^3/m^3",
        },
        "earthaccess_l3": {
            "short_name": "SPL3SMP_E",
            "version": "006",
            "group": "Soil_Moisture_Retrieval_Data_AM",
            "variables": ["soil_moisture", "retrieval_qual_flag"],
            "units": "cm^3/cm^3",
        },
        "appeears": [
            {"product": "SPL3SMP_E.006", "layer": "Soil_Moisture_Retrieval_Data_AM_soil_moisture"},
            {"product": "SPL3SMP_E.006", "layer": "Soil_Moisture_Retrieval_Data_PM_soil_moisture_pm"},
        ],
        "gee": {
            "collection": "NASA/SMAP/SPL4SMGP/008",
            "bands": ["sm_surface", "sm_rootzone", "sm_profile", "sm_surface_wetness"],
            "scale_meters": 9000,
            "units": "m^3/m^3",
        },
    },
    "vegetation_health": {
        "description": "Normalized Difference Vegetation Index (NDVI) & Enhanced Vegetation Index (EVI)",
        "mission": "MODIS, Landsat, HLS",
        "appeears": [
            {"product": "MOD13Q1.061", "layer": "_250m_16_days_NDVI", "scale_factor": 0.0001},
            {"product": "MOD13Q1.061", "layer": "_250m_16_days_EVI", "scale_factor": 0.0001},
            {"product": "MOD13Q1.061", "layer": "_250m_16_days_VI_Quality", "scale_factor": 1.0},
            {"product": "HLSL30.020", "layer": "B05", "scale_factor": 0.0001},  # NIR (Landsat 8/9 OLI)
            {"product": "HLSL30.020", "layer": "B04", "scale_factor": 0.0001},  # Red
            {"product": "HLSL30.020", "layer": "B02", "scale_factor": 0.0001},  # Blue
            {"product": "HLSL30.020", "layer": "Fmask", "scale_factor": 1.0},
        ],
        "earthaccess": {
            "short_name": "MOD13Q1",
            "version": "061",
            "hls_short_names": ["HLSL30", "HLSS30"],
            "hls_version": "2.0",
        },
        "gee_modis": {
            "collection": "MODIS/061/MOD13Q1",
            "bands": ["NDVI", "EVI", "SummaryQA"],
            "scale_factor": 0.0001,
            "scale_meters": 250,
        },
        "gee_hls": {
            "collection": "NASA/HLS/HLSL30/v002",
            "bands": ["B5", "B4", "B2", "Fmask"],
            "scale_meters": 30,
        },
    },
    "evapotranspiration": {
        "description": "Actual & potential evapotranspiration (ET / PET) and canopy partitioning",
        "mission": "MODIS / ECOSTRESS",
        "appeears": [
            {"product": "MOD16A2GF.061", "layer": "ET_500m", "scale_factor": 0.1, "units": "kg/m^2/8day"},
            {"product": "MOD16A2GF.061", "layer": "PET_500m", "scale_factor": 0.1, "units": "kg/m^2/8day"},
            {"product": "ECO_L3T_JET.002", "layer": "ETdaily", "scale_factor": 1.0, "units": "mm/day"},
            {"product": "ECO_L3T_JET.002", "layer": "PTJPLSMinst", "scale_factor": 1.0, "units": "W/m^2"},
        ],
        "earthaccess": {
            "modis_short_name": "MOD16A2GF",
            "modis_version": "061",
            "ecostress_short_name": "ECO_L3T_JET",
            "ecostress_version": "003",
        },
        "gee": {
            "collection": "MODIS/061/MOD16A2GF",
            "bands": ["ET", "PET", "LE", "PLE"],
            "scale_factor": 0.1,
            "scale_meters": 500,
        },
    },
    "plant_water_stress": {
        "description": "Evaporative Stress Index (ESI = ET/PET) and Water Use Efficiency (WUE)",
        "mission": "ECOSTRESS",
        "appeears": [
            {"product": "ECO_L4T_ESI.002", "layer": "ESIavg", "scale_factor": 1.0, "units": "ratio_0_to_1"},
            {"product": "ECO_L4T_WUE.002", "layer": "WUEavg", "scale_factor": 1.0, "units": "gC/kgH2O"},
        ],
        "earthaccess": {
            "esi_short_name": "ECO_L4T_ESI",
            "wue_short_name": "ECO_L4T_WUE",
            "version": "003",
        },
    },
    "land_cover": {
        "description": "Annual global land cover classification and historical cropland masks",
        "mission": "Landsat, MODIS Land Cover",
        "appeears": [
            {"product": "MCD12Q1.061", "layer": "LC_Type1"},  # IGBP (12=Croplands, 14=Crop/Natural Mosaic)
            {"product": "MCD12Q1.061", "layer": "LC_Prop2"},  # FAO LCCS Land Use
        ],
        "earthaccess": {
            "short_name": "MCD12Q1",
            "version": "061",
        },
        "gee": {
            "collection": "MODIS/061/MCD12Q1",
            "bands": ["LC_Type1", "LC_Type2", "LC_Prop2"],
            "scale_meters": 500,
        },
    },
    "flood_drought": {
        "description": "Weekly GRACE/GRACE-FO groundwater & soil moisture drought percentiles + GPM extremes",
        "mission": "GRACE, GPM",
        "earthaccess": {
            "short_name": "GRACEDADM_CLSM025GL_7D",
            "version": "3.0",
            "variables": [
                "gws_inst",    # Groundwater Storage Percentile (%)
                "rtzsm_inst",  # Root Zone Soil Moisture Percentile (%)
                "sfsm_inst",   # Surface Soil Moisture Percentile (%)
            ],
            "units": "percentile_0_to_100",
        },
    },
    "land_surface_temp": {
        "description": "Daytime and nighttime Land Surface Temperature (LST) for canopy heat stress",
        "mission": "MODIS / ECOSTRESS",
        "appeears": [
            {"product": "MOD11A2.061", "layer": "LST_Day_1km", "scale_factor": 0.02, "offset": -273.15, "units": "C"},
            {"product": "MOD11A2.061", "layer": "LST_Night_1km", "scale_factor": 0.02, "offset": -273.15, "units": "C"},
        ],
        "earthaccess": {
            "modis_short_name": "MOD11A2",
            "modis_version": "061",
            "ecostress_short_name": "ECO_L2T_LSTE",
            "ecostress_version": "003",
        },
        "gee": {
            "collection": "MODIS/061/MOD11A2",
            "bands": ["LST_Day_1km", "LST_Night_1km"],
            "scale_factor": 0.02,
            "offset": -273.15,
            "scale_meters": 1000,
        },
    },
    "elevation_terrain": {
        "description": "Digital Elevation Model (DEM), slope, and aspect for runoff/drainage modeling",
        "mission": "SRTM / NASADEM",
        "appeears": [
            {"product": "SRTMGL1_NC.003", "layer": "SRTMGL1_DEM", "units": "meters"},
        ],
        "earthaccess": {
            "short_name": "NASADEM_HGT",
            "version": "001",
        },
        "gee": {
            "collection": "USGS/SRTMGL1_003",
            "bands": ["elevation"],
            "scale_meters": 30,
        },
    },
}


# =============================================================================
# 2. BACKEND 1: NASA POWER AGROCLIMATOLOGY CLIENT (NO AUTHENTICATION REQUIRED)
# =============================================================================


class NASAPowerClient:
    """
    Client for the NASA POWER (Prediction Of Worldwide Energy Resources) REST API
    configured for the Agroclimatology (`AG`) community.

    Covers:
    - Temperature, humidity, solar radiation, PAR, and wind speed (POWER / MERRA-2)
    - Daily bias-corrected rainfall (`PRECTOTCORR`)
    - Surface, root-zone, and profile soil wetness (`GWETTOP`, `GWETROOT`, `GWETPROF`)
    - Derived flood, dry-spell, and heat-stress indicators
    """

    BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

    DEFAULT_PARAMETERS = (
        "T2M",
        "T2M_MAX",
        "T2M_MIN",
        "RH2M",
        "QV2M",
        "ALLSKY_SFC_SW_DWN",
        "ALLSKY_SFC_PAR_TOT",
        "WS2M",
        "PRECTOTCORR",
        "GWETTOP",
        "GWETROOT",
        "GWETPROF",
    )

    def __init__(self, timeout: int = 45) -> None:
        self.timeout = timeout
        self.session = requests.Session()

    def fetch_daily_agro(
        self,
        location: FarmLocation,
        start_date: str,
        end_date: str,
        parameters: Optional[Sequence[str]] = None,
        add_derived_indicators: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch daily agroclimatology & soil wetness records for a `FarmLocation`.

        Parameters
        ----------
        location : FarmLocation
            Target farm location containing latitude and longitude.
        start_date : str
            Start date in 'YYYY-MM-DD' or 'YYYYMMDD' format.
        end_date : str
            End date in 'YYYY-MM-DD' or 'YYYYMMDD' format.
        parameters : sequence of str, optional
            Custom list of NASA POWER parameter codes (defaults to all 12 core
            weather, radiation, rainfall, and soil moisture variables).
        add_derived_indicators : bool, default True
            If True, calculates rolling rainfall totals, consecutive dry days (CDD),
            vapor pressure deficit (VPD kPa), and extreme heat/rain flags.

        Returns
        -------
        pd.DataFrame
            Daily DataFrame indexed by `date`.
        """
        params_list = list(parameters or self.DEFAULT_PARAMETERS)
        start_clean = start_date.replace("-", "")
        end_clean = end_date.replace("-", "")

        query = {
            "parameters": ",".join(params_list),
            "community": "AG",
            "longitude": location.longitude,
            "latitude": location.latitude,
            "start": start_clean,
            "end": end_clean,
            "format": "JSON",
        }

        logger.info(
            "Querying NASA POWER API for farm '%s' (%.4f, %.4f) [%s to %s]...",
            location.farm_id,
            location.latitude,
            location.longitude,
            start_clean,
            end_clean,
        )
        resp = self.session.get(self.BASE_URL, params=query, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()

        records = data.get("properties", {}).get("parameter", {})
        df = pd.DataFrame(records)
        df.index = pd.to_datetime(df.index, format="%Y%m%d")
        df.index.name = "date"
        df = df.replace(-999.0, np.nan)

        if add_derived_indicators and not df.empty:
            df = self.compute_agronomic_indicators(df)

        df["farm_id"] = location.farm_id
        return df

    @staticmethod
    def compute_agronomic_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute derived flood, drought, and crop-stress features from daily
        weather and soil moisture columns.
        """
        out = df.copy()

        # 1. Flood & Drought Rainfall Metrics
        if "PRECTOTCORR" in out.columns:
            rain = out["PRECTOTCORR"].fillna(0.0)
            out["precip_7d_sum_mm"] = rain.rolling(7, min_periods=1).sum()
            out["precip_30d_sum_mm"] = rain.rolling(30, min_periods=1).sum()

            # Consecutive Dry Days (CDD < 1.0 mm/day)
            is_dry = (rain < 1.0).astype(int)
            dry_groups = is_dry.ne(is_dry.shift()).cumsum()
            out["consecutive_dry_days"] = is_dry.groupby(dry_groups).cumsum()
            out["heavy_rain_event_flag"] = (rain >= 50.0).astype(int)

        # 2. Root-Zone Drought & Waterlogging Flags
        if "GWETROOT" in out.columns:
            out["rootzone_drought_flag"] = (out["GWETROOT"] < 0.30).astype(int)
            out["rootzone_waterlogging_flag"] = (out["GWETROOT"] > 0.85).astype(int)

        # 3. Vapor Pressure Deficit (VPD in kPa) & Heat Stress
        if {"T2M", "RH2M"}.issubset(out.columns):
            # Saturation vapor pressure (Tetens equation, kPa)
            es = 0.61078 * np.exp((17.27 * out["T2M"]) / (out["T2M"] + 237.3))
            ea = es * (out["RH2M"] / 100.0)
            out["vpd_kpa"] = np.maximum(0.0, es - ea)

        if "T2M_MAX" in out.columns:
            out["heat_stress_day_flag"] = (out["T2M_MAX"] >= 35.0).astype(int)

        return out


# =============================================================================
# 3. BACKEND 2: NASA AppEEARS REST API CLIENT (POINT & POLYGON TIME SERIES)
# =============================================================================


class AppEEARSClient:
    """
    Client for NASA's Application for Extracting and Exploring Analysis Ready
    Samples (AppEEARS) REST API (`https://appeears.earthdatacloud.nasa.gov/api`).

    Ideal for extracting server-side point or polygon time series across:
    - Vegetation Health: `MOD13Q1.061` (NDVI, EVI), `HLSL30.020`, `HLSS30.020`
    - Evapotranspiration: `MOD16A2GF.061` (`ET_500m`, `PET_500m`), `ECO_L3T_JET.002`
    - Plant Water Stress: `ECO_L4T_ESI.002` (`ESIavg`), `ECO_L4T_WUE.002` (`WUEavg`)
    - Soil Moisture: `SPL3SMP_E.006` (AM/PM 9 km soil moisture)
    - Land Cover: `MCD12Q1.061` (`LC_Type1`, `LC_Prop2`)
    - Land Surface Temperature: `MOD11A2.061` (`LST_Day_1km`, `LST_Night_1km`)
    - Elevation: `SRTMGL1_NC.003` (`SRTMGL1_DEM`)
    """

    BASE_URL = "https://appeears.earthdatacloud.nasa.gov/api"

    # Curated default bundle covering all AppEEARS-accessible Field Shift layers
    FIELD_SHIFT_POINT_LAYERS: List[Dict[str, str]] = [
        # Vegetation Health (MODIS 250m + HLS 30m)
        {"product": "MOD13Q1.061", "layer": "_250m_16_days_NDVI"},
        {"product": "MOD13Q1.061", "layer": "_250m_16_days_EVI"},
        {"product": "HLSL30.020", "layer": "B05"},
        {"product": "HLSL30.020", "layer": "B04"},
        {"product": "HLSL30.020", "layer": "Fmask"},
        # Evapotranspiration & Plant Stress (MODIS 500m + ECOSTRESS 70m)
        {"product": "MOD16A2GF.061", "layer": "ET_500m"},
        {"product": "MOD16A2GF.061", "layer": "PET_500m"},
        {"product": "ECO_L3T_JET.002", "layer": "ETdaily"},
        {"product": "ECO_L4T_ESI.002", "layer": "ESIavg"},
        {"product": "ECO_L4T_WUE.002", "layer": "WUEavg"},
        # Soil Moisture (SMAP Enhanced L3 9km)
        {"product": "SPL3SMP_E.006", "layer": "Soil_Moisture_Retrieval_Data_AM_soil_moisture"},
        {"product": "SPL3SMP_E.006", "layer": "Soil_Moisture_Retrieval_Data_PM_soil_moisture_pm"},
        # Land Cover / Crop History (MODIS 500m)
        {"product": "MCD12Q1.061", "layer": "LC_Type1"},
        {"product": "MCD12Q1.061", "layer": "LC_Prop2"},
        # Land Surface Temperature (MODIS 1km)
        {"product": "MOD11A2.061", "layer": "LST_Day_1km"},
        {"product": "MOD11A2.061", "layer": "LST_Night_1km"},
        # Static Elevation (SRTM 30m)
        {"product": "SRTMGL1_NC.003", "layer": "SRTMGL1_DEM"},
    ]

    def __init__(
        self,
        username: Optional[str] = None,
        password: Optional[str] = None,
        timeout: int = 30,
    ) -> None:
        self.username = username or os.environ.get("EARTHDATA_USERNAME")
        self.password = password or os.environ.get("EARTHDATA_PASSWORD")
        self.timeout = timeout
        self._token: Optional[str] = None

    def authenticate(self) -> str:
        """
        Obtain a Bearer token from NASA AppEEARS using Earthdata Login credentials.
        """
        if not self.username or not self.password:
            raise ValueError(
                "NASA Earthdata credentials required. Set EARTHDATA_USERNAME and "
                "EARTHDATA_PASSWORD environment variables or pass them to AppEEARSClient."
            )
        resp = requests.post(
            f"{self.BASE_URL}/login",
            auth=(self.username, self.password),
            timeout=self.timeout,
        )
        resp.raise_for_status()
        self._token = resp.json()["token"]
        return self._token

    @property
    def headers(self) -> Dict[str, str]:
        if not self._token:
            self.authenticate()
        return {"Authorization": f"Bearer {self._token}"}

    @staticmethod
    def _to_appeears_date(date_str: str) -> str:
        """Convert 'YYYY-MM-DD' or 'YYYYMMDD' to AppEEARS 'MM-DD-YYYY' format."""
        dt = pd.to_datetime(date_str)
        return dt.strftime("%m-%d-%Y")

    def submit_point_task(
        self,
        locations: Sequence[FarmLocation],
        start_date: str,
        end_date: str,
        task_name: str = "FieldShift_MultiSensor_Point_Extract",
        layers: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        Submit a batch point extraction task for one or more `FarmLocation` objects.

        Returns
        -------
        str
            The submitted AppEEARS `task_id`.
        """
        coords_payload = [
            {
                "id": loc.farm_id,
                "latitude": loc.latitude,
                "longitude": loc.longitude,
                "category": loc.farmer_priority,
            }
            for loc in locations
        ]

        payload = {
            "task_type": "point",
            "task_name": task_name,
            "params": {
                "dates": [
                    {
                        "startDate": self._to_appeears_date(start_date),
                        "endDate": self._to_appeears_date(end_date),
                    }
                ],
                "layers": layers or self.FIELD_SHIFT_POINT_LAYERS,
                "coordinates": coords_payload,
            },
        }

        resp = requests.post(
            f"{self.BASE_URL}/task",
            json=payload,
            headers=self.headers,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        task_id = resp.json()["task_id"]
        logger.info("Submitted AppEEARS point task '%s' (task_id=%s)", task_name, task_id)
        return task_id

    def wait_and_download_csvs(
        self,
        task_id: str,
        output_dir: Union[str, Path],
        poll_interval_sec: int = 20,
        max_wait_sec: int = 3600,
    ) -> List[Path]:
        """
        Poll an AppEEARS task until completion and download all resulting CSV files.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        elapsed = 0
        while elapsed < max_wait_sec:
            status_resp = requests.get(
                f"{self.BASE_URL}/status/{task_id}",
                headers=self.headers,
                timeout=self.timeout,
            )
            status_resp.raise_for_status()
            status = status_resp.json().get("status")
            logger.info("AppEEARS task %s status: %s", task_id, status)
            if status == "done":
                break
            if status in ("error", "expired"):
                raise RuntimeError(f"AppEEARS task {task_id} terminated with status: {status}")
            time.sleep(poll_interval_sec)
            elapsed += poll_interval_sec
        else:
            raise TimeoutError(f"AppEEARS task {task_id} timed out after {max_wait_sec}s.")

        bundle_resp = requests.get(
            f"{self.BASE_URL}/bundle/{task_id}",
            headers=self.headers,
            timeout=self.timeout,
        )
        bundle_resp.raise_for_status()
        files = bundle_resp.json().get("files", [])

        downloaded_csvs: List[Path] = []
        for file_meta in files:
            file_name = file_meta.get("file_name", "")
            file_id = file_meta.get("file_id", "")
            if file_name.endswith(".csv"):
                dl_resp = requests.get(
                    f"{self.BASE_URL}/bundle/{task_id}/{file_id}",
                    headers=self.headers,
                    allow_redirects=True,
                    stream=True,
                    timeout=self.timeout,
                )
                dl_resp.raise_for_status()
                dest = out_path / Path(file_name).name
                with open(dest, "wb") as f:
                    for chunk in dl_resp.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                downloaded_csvs.append(dest)
                logger.info("Downloaded AppEEARS CSV: %s", dest)

        return downloaded_csvs


# =============================================================================
# 4. BACKEND 3: NASA `earthaccess` + `xarray` (DIRECT GRANULE & CLOUD STREAMING)
# =============================================================================


class EarthAccessClient:
    """
    Client built on NASA's official `earthaccess` library and `xarray` for direct
    search, streaming, and spatial subsetting of NASA Earthdata Cloud granules.

    Particularly suited for:
    - GPM / IMERG Daily Rainfall (`GPM_3IMERGDF` v07)
    - SMAP L4 3-Hourly Surface & Root-Zone Soil Moisture (`SPL4SMGP` v008)
    - SMAP Enhanced L3 Daily Soil Moisture (`SPL3SMP_E` v006)
    - GRACE / GRACE-FO Weekly Drought Percentiles (`GRACEDADM_CLSM025GL_7D` v3.0)
    - Raw granules for `MOD13Q1`, `MOD16A2GF`, `HLSL30`, `ECO_L3T_JET`, `M2SDNXSLV`
    """

    def __init__(self, persist_login: bool = True) -> None:
        self.persist_login = persist_login
        self._authenticated = False

    def ensure_login(self) -> None:
        if not self._authenticated:
            import earthaccess

            earthaccess.login(strategy="all", persist=self.persist_login)
            self._authenticated = True

    def search_and_open_granules(
        self,
        short_name: str,
        start_date: str,
        end_date: str,
        bounding_box: Tuple[float, float, float, float],
        version: Optional[str] = None,
        download_dir: Optional[Union[str, Path]] = None,
    ) -> List[Any]:
        """
        Search NASA CMR for granules matching the collection, date range, and
        bounding box, and return file-like objects (streamed) or local file paths.
        """
        import earthaccess

        self.ensure_login()
        search_kwargs: Dict[str, Any] = {
            "short_name": short_name,
            "bounding_box": bounding_box,
            "temporal": (start_date, end_date),
        }
        if version:
            search_kwargs["version"] = version

        granules = earthaccess.search_data(**search_kwargs)
        if not granules:
            raise ValueError(
                f"No granules found for {short_name} (version={version}) in {bounding_box} "
                f"from {start_date} to {end_date}."
            )

        logger.info("Found %d granules for %s.", len(granules), short_name)
        if download_dir is not None:
            Path(download_dir).mkdir(parents=True, exist_ok=True)
            return earthaccess.download(granules, local_path=str(download_dir))
        return earthaccess.open(granules)

    def fetch_gpm_imerg_rainfall(
        self,
        location: FarmLocation,
        start_date: str,
        end_date: str,
        pad_degrees: float = 0.25,
    ) -> pd.DataFrame:
        """
        Stream GPM IMERG Final Daily Precipitation (`GPM_3IMERGDF` v07, 0.1 deg)
        and extract the daily time series for `location`.
        """
        import xarray as xr

        files = self.search_and_open_granules(
            short_name="GPM_3IMERGDF",
            version="07",
            start_date=start_date,
            end_date=end_date,
            bounding_box=location.bounding_box(pad_degrees=pad_degrees),
        )
        ds = xr.open_mfdataset(files, combine="by_coords")
        point_ds = ds[["precipitation", "probabilityLiquidPrecipitation"]].sel(
            lat=location.latitude,
            lon=location.longitude,
            method="nearest",
        )
        df = point_ds.to_dataframe().reset_index()
        df = df.rename(
            columns={
                "time": "date",
                "precipitation": "gpm_precip_mm_day",
                "probabilityLiquidPrecipitation": "gpm_liquid_precip_prob_pct",
            }
        )
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date")[["gpm_precip_mm_day", "gpm_liquid_precip_prob_pct"]]

    def fetch_smap_l4_soil_moisture(
        self,
        location: FarmLocation,
        start_date: str,
        end_date: str,
        pad_degrees: float = 0.25,
    ) -> pd.DataFrame:
        """
        Stream SMAP Level-4 (`SPL4SMGP` v008, 9 km 3-hourly) surface and root-zone
        soil moisture and extract the nearest EASE-Grid cell for `location`.
        """
        import xarray as xr

        files = self.search_and_open_granules(
            short_name="SPL4SMGP",
            version="008",
            start_date=start_date,
            end_date=end_date,
            bounding_box=location.bounding_box(pad_degrees=pad_degrees),
        )
        ds_geo = xr.open_mfdataset(
            files,
            group="Geophysical_Data",
            engine="h5netcdf",
            combine="nested",
            concat_dim="time",
        )
        ds_coords = xr.open_dataset(files[0], engine="h5netcdf")[["cell_lat", "cell_lon"]]

        dist_sq = (ds_coords["cell_lat"] - location.latitude) ** 2 + (
            ds_coords["cell_lon"] - location.longitude
        ) ** 2
        min_idx = np.unravel_index(np.nanargmin(dist_sq.values), dist_sq.shape)
        y_dim, x_dim = dist_sq.dims[-2], dist_sq.dims[-1]

        vars_to_pull = ["sm_surface", "sm_rootzone", "sm_profile", "sm_surface_wetness"]
        point_ds = ds_geo[vars_to_pull].isel({y_dim: min_idx[-2], x_dim: min_idx[-1]})
        df = point_ds.to_dataframe().reset_index()
        if "time" in df.columns:
            df["date"] = pd.to_datetime(df["time"])
            df = df.set_index("date").resample("1D").mean(numeric_only=True)
        return df

    def fetch_grace_drought_indicators(
        self,
        location: FarmLocation,
        start_date: str,
        end_date: str,
        pad_degrees: float = 0.5,
    ) -> pd.DataFrame:
        """
        Stream weekly GRACE/GRACE-FO Data Assimilation Drought Indicators
        (`GRACEDADM_CLSM025GL_7D` v3.0, 0.25 deg) and extract groundwater (`gws_inst`),
        root-zone (`rtzsm_inst`), and surface (`sfsm_inst`) wetness percentiles.
        """
        import xarray as xr

        files = self.search_and_open_granules(
            short_name="GRACEDADM_CLSM025GL_7D",
            version="3.0",
            start_date=start_date,
            end_date=end_date,
            bounding_box=location.bounding_box(pad_degrees=pad_degrees),
        )
        ds = xr.open_mfdataset(files, combine="by_coords")
        point_ds = ds[["gws_inst", "rtzsm_inst", "sfsm_inst"]].sel(
            lat=location.latitude,
            lon=location.longitude,
            method="nearest",
        )
        df = point_ds.to_dataframe().reset_index()
        df = df.rename(
            columns={
                "time": "date",
                "gws_inst": "grace_groundwater_pctl",
                "rtzsm_inst": "grace_rootzone_sm_pctl",
                "sfsm_inst": "grace_surface_sm_pctl",
            }
        )
        df["date"] = pd.to_datetime(df["date"])
        df = df.replace(-999.0, np.nan)
        return df.set_index("date")[
            ["grace_groundwater_pctl", "grace_rootzone_sm_pctl", "grace_surface_sm_pctl"]
        ]


# =============================================================================
# 5. BACKEND 4: GOOGLE EARTH ENGINE PYTHON API CLIENT (CLOUD SPATIAL REDUCER)
# =============================================================================


class EarthEngineClient:
    """
    Cloud-native extractor using the Google Earth Engine Python API (`ee`).
    Computes spatial means over farm coordinates or field polygons across:
    - Rainfall: `NASA/GPM_L3/IMERG_V07`
    - Soil Moisture: `NASA/SMAP/SPL4SMGP/008`
    - Vegetation Health: `MODIS/061/MOD13Q1` (250m) & `NASA/HLS/HLSL30/v002` (30m)
    - Evapotranspiration: `MODIS/061/MOD16A2GF` (500m)
    - Land Surface Temperature: `MODIS/061/MOD11A2` (1km)
    - Land Cover / Crop History: `MODIS/061/MCD12Q1` (500m)
    - Elevation & Terrain (Slope/Aspect): `USGS/SRTMGL1_003` (30m)
    """

    def __init__(self, project_id: str) -> None:
        self.project_id = project_id
        self._initialized = False

    def ensure_initialized(self) -> None:
        if not self._initialized:
            import ee

            ee.Initialize(project=self.project_id)
            self._initialized = True

    def _build_geometry(self, location: FarmLocation):
        import ee

        self.ensure_initialized()
        if location.polygon_coords:
            return ee.Geometry.Polygon([location.polygon_coords])
        pt = ee.Geometry.Point([location.longitude, location.latitude])
        if location.buffer_meters > 0:
            return pt.buffer(location.buffer_meters)
        return pt

    def _reduce_collection(
        self,
        collection_id: str,
        bands: List[str],
        location: FarmLocation,
        start_date: str,
        end_date: str,
        scale_meters: float,
    ) -> pd.DataFrame:
        import ee

        geom = self._build_geometry(location)
        col = (
            ee.ImageCollection(collection_id)
            .filterDate(start_date, end_date)
            .select(bands)
        )

        def to_feature(img: ee.Image) -> ee.Feature:
            stats = img.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geom,
                scale=scale_meters,
                bestEffort=True,
            )
            return ee.Feature(None, stats).set(
                "date", img.date().format("YYYY-MM-dd")
            )

        fc = col.map(to_feature).getInfo()
        rows = [f["properties"] for f in fc.get("features", [])]
        df = pd.DataFrame(rows)
        if df.empty:
            return pd.DataFrame(columns=bands, index=pd.DatetimeIndex([], name="date"))
        df["date"] = pd.to_datetime(df["date"])
        return df.groupby("date").mean(numeric_only=True)

    def fetch_modis_ndvi_evi(
        self, location: FarmLocation, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Fetch 16-day 250m MODIS NDVI and EVI (`MODIS/061/MOD13Q1`), scaled to [-1, 1]."""
        df = self._reduce_collection(
            collection_id="MODIS/061/MOD13Q1",
            bands=["NDVI", "EVI"],
            location=location,
            start_date=start_date,
            end_date=end_date,
            scale_meters=250,
        )
        if not df.empty:
            df["modis_ndvi"] = df["NDVI"] * 0.0001
            df["modis_evi"] = df["EVI"] * 0.0001
            df = df[["modis_ndvi", "modis_evi"]]
        return df

    def fetch_hls_ndvi_evi(
        self, location: FarmLocation, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """
        Fetch 30m Harmonized Landsat Sentinel-2 (`NASA/HLS/HLSL30/v002`) surface
        reflectance and compute cloud-screened field-level NDVI and EVI.
        """
        df = self._reduce_collection(
            collection_id="NASA/HLS/HLSL30/v002",
            bands=["B5", "B4", "B2"],
            location=location,
            start_date=start_date,
            end_date=end_date,
            scale_meters=30,
        )
        if not df.empty:
            nir = df["B5"]
            red = df["B4"]
            blue = df["B2"]
            df["hls_ndvi_30m"] = (nir - red) / (nir + red + 1e-6)
            df["hls_evi_30m"] = 2.5 * (nir - red) / (nir + 6.0 * red - 7.5 * blue + 1.0 + 1e-6)
            df = df[["hls_ndvi_30m", "hls_evi_30m"]]
        return df

    def fetch_modis_evapotranspiration(
        self, location: FarmLocation, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """
        Fetch 8-day 500m gap-filled Evapotranspiration & Potential ET (`MODIS/061/MOD16A2GF`),
        scaled by 0.1 (kg/m^2/8day) plus Evaporative Stress Ratio (ET / PET).
        """
        df = self._reduce_collection(
            collection_id="MODIS/061/MOD16A2GF",
            bands=["ET", "PET"],
            location=location,
            start_date=start_date,
            end_date=end_date,
            scale_meters=500,
        )
        if not df.empty:
            df["modis_et_mm_8d"] = df["ET"] * 0.1
            df["modis_pet_mm_8d"] = df["PET"] * 0.1
            df["evaporative_stress_ratio"] = df["modis_et_mm_8d"] / (
                df["modis_pet_mm_8d"] + 1e-6
            )
            df = df[["modis_et_mm_8d", "modis_pet_mm_8d", "evaporative_stress_ratio"]]
        return df

    def fetch_modis_lst(
        self, location: FarmLocation, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """
        Fetch 8-day 1km Land Surface Temperature (`MODIS/061/MOD11A2`) converted to Celsius.
        """
        df = self._reduce_collection(
            collection_id="MODIS/061/MOD11A2",
            bands=["LST_Day_1km", "LST_Night_1km"],
            location=location,
            start_date=start_date,
            end_date=end_date,
            scale_meters=1000,
        )
        if not df.empty:
            df["lst_day_c"] = df["LST_Day_1km"] * 0.02 - 273.15
            df["lst_night_c"] = df["LST_Night_1km"] * 0.02 - 273.15
            df = df[["lst_day_c", "lst_night_c"]]
        return df

    def fetch_smap_soil_moisture(
        self, location: FarmLocation, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """
        Fetch daily-averaged SMAP L4 9km surface and root-zone soil moisture
        (`NASA/SMAP/SPL4SMGP/008`).
        """
        return self._reduce_collection(
            collection_id="NASA/SMAP/SPL4SMGP/008",
            bands=["sm_surface", "sm_rootzone", "sm_profile", "sm_surface_wetness"],
            location=location,
            start_date=start_date,
            end_date=end_date,
            scale_meters=9000,
        )

    def fetch_static_terrain_and_landcover(
        self, location: FarmLocation, year: int = 2024
    ) -> Dict[str, float]:
        """
        Fetch static 30m SRTM elevation, slope, and aspect (`USGS/SRTMGL1_003`)
        alongside MODIS land cover class (`MODIS/061/MCD12Q1`).
        """
        import ee

        geom = self._build_geometry(location)
        dem = ee.Image("USGS/SRTMGL1_003")
        terrain = ee.Terrain.products(dem).select(["elevation", "slope", "aspect"])
        terrain_stats = terrain.reduceRegion(
            reducer=ee.Reducer.mean(), geometry=geom, scale=30, bestEffort=True
        ).getInfo()

        lc_col = (
            ee.ImageCollection("MODIS/061/MCD12Q1")
            .filterDate(f"{year - 1}-01-01", f"{year + 1}-12-31")
            .first()
        )
        lc_stats: Dict[str, Any] = {}
        if lc_col is not None:
            lc_stats = (
                lc_col.select(["LC_Type1", "LC_Prop2"])
                .reduceRegion(
                    reducer=ee.Reducer.mode(), geometry=geom, scale=500, bestEffort=True
                )
                .getInfo()
            )

        return {
            "elevation_m": terrain_stats.get("elevation", np.nan),
            "slope_deg": terrain_stats.get("slope", np.nan),
            "aspect_deg": terrain_stats.get("aspect", np.nan),
            "modis_igbp_landcover_class": lc_stats.get("LC_Type1", np.nan),
            "modis_fao_landuse_class": lc_stats.get("LC_Prop2", np.nan),
        }


# =============================================================================
# 6. UNIFIED ORCHESTRATOR: FIELD SHIFT MULTI-SOURCE FEATURE BUILDER
# =============================================================================


class FieldShiftFeatureBuilder:
    """
    High-level orchestrator that combines multi-sensor NASA observations onto a
    unified daily time-series table and seasonal summary table for Field Shift
    crop-rotation and water/soil-health modeling.
    """

    def __init__(
        self,
        power_client: Optional[NASAPowerClient] = None,
        earthaccess_client: Optional[EarthAccessClient] = None,
        gee_client: Optional[EarthEngineClient] = None,
        appeears_client: Optional[AppEEARSClient] = None,
    ) -> None:
        self.power = power_client or NASAPowerClient()
        self.earthaccess = earthaccess_client
        self.gee = gee_client
        self.appeears = appeears_client

    def build_farm_training_table(
        self,
        location: FarmLocation,
        start_date: str,
        end_date: str,
        include_gee_features: bool = False,
        include_earthaccess_features: bool = False,
        interpolate_remote_sensing: bool = True,
    ) -> pd.DataFrame:
        """
        Build a daily feature matrix for a `FarmLocation` between `start_date` and
        `end_date` ('YYYY-MM-DD').

        1. Always fetches the daily baseline from NASA POWER (Temperature, Humidity,
           Solar Radiation, PAR, Wind, Rainfall, Surface/Root-Zone Soil Wetness, and
           derived flood/drought/VPD indicators).
        2. Optionally merges Google Earth Engine remote sensing time series
           (MODIS/HLS NDVI & EVI, MODIS ET/PET, MODIS LST, SMAP L4 soil moisture,
           and static SRTM terrain/MODIS land cover).
        3. Optionally merges NASA Earthdata Cloud (`earthaccess`) streams for
           GPM IMERG daily precipitation, SMAP L4 soil moisture, and weekly GRACE
           groundwater/soil moisture drought percentiles.
        """
        # Step 1: Baseline daily weather, radiation, rainfall & soil wetness
        master_df = self.power.fetch_daily_agro(
            location=location,
            start_date=start_date,
            end_date=end_date,
            add_derived_indicators=True,
        )

        # Step 2: Merge Google Earth Engine remote sensing features if configured
        if include_gee_features:
            if self.gee is None:
                raise ValueError(
                    "Pass an initialized EarthEngineClient(project_id=...) to use include_gee_features=True."
                )
            logger.info("Merging Google Earth Engine remote-sensing layers for '%s'...", location.farm_id)
            ndvi_df = self.gee.fetch_modis_ndvi_evi(location, start_date, end_date)
            hls_df = self.gee.fetch_hls_ndvi_evi(location, start_date, end_date)
            et_df = self.gee.fetch_modis_evapotranspiration(location, start_date, end_date)
            lst_df = self.gee.fetch_modis_lst(location, start_date, end_date)
            smap_df = self.gee.fetch_smap_soil_moisture(location, start_date, end_date)

            for remote_df in (ndvi_df, hls_df, et_df, lst_df, smap_df):
                if not remote_df.empty:
                    master_df = master_df.join(remote_df, how="left")
                    if interpolate_remote_sensing:
                        cols = list(remote_df.columns)
                        master_df[cols] = master_df[cols].interpolate(method="time", limit=16)

            static_meta = self.gee.fetch_static_terrain_and_landcover(location)
            for k, v in static_meta.items():
                master_df[k] = v

        # Step 3: Merge direct NASA Earthdata Cloud (earthaccess) features if configured
        if include_earthaccess_features:
            if self.earthaccess is None:
                self.earthaccess = EarthAccessClient()
            logger.info("Merging NASA earthaccess streams (GPM, SMAP L4, GRACE) for '%s'...", location.farm_id)
            gpm_df = self.earthaccess.fetch_gpm_imerg_rainfall(location, start_date, end_date)
            smap_ea_df = self.earthaccess.fetch_smap_l4_soil_moisture(location, start_date, end_date)
            grace_df = self.earthaccess.fetch_grace_drought_indicators(location, start_date, end_date)

            for ea_df in (gpm_df, smap_ea_df, grace_df):
                if not ea_df.empty:
                    master_df = master_df.join(ea_df, how="left")
                    if interpolate_remote_sensing:
                        cols = list(ea_df.columns)
                        master_df[cols] = master_df[cols].interpolate(method="time", limit=7)

        # Attach farm context columns from PDF specification
        master_df["latitude"] = location.latitude
        master_df["longitude"] = location.longitude
        master_df["irrigation_access"] = location.irrigation_access
        master_df["farmer_priority"] = location.farmer_priority

        return master_df

    @staticmethod
    def summarize_growing_season(daily_df: pd.DataFrame) -> Dict[str, float]:
        """
        Aggregate a daily farm DataFrame into seasonal summary features suitable
        for training or querying a crop-rotation recommendation model.
        """
        summary: Dict[str, float] = {}
        if daily_df.empty:
            return summary

        numeric_Mean_cols = [
            "T2M",
            "T2M_MAX",
            "T2M_MIN",
            "RH2M",
            "ALLSKY_SFC_SW_DWN",
            "GWETTOP",
            "GWETROOT",
            "GWETPROF",
            "vpd_kpa",
            "sm_surface",
            "sm_rootzone",
            "modis_ndvi",
            "modis_evi",
            "hls_ndvi_30m",
            "evaporative_stress_ratio",
            "lst_day_c",
            "grace_groundwater_pctl",
            "grace_rootzone_sm_pctl",
        ]
        for col in numeric_Mean_cols:
            if col in daily_df.columns:
                summary[f"seasonal_mean_{col}"] = float(daily_df[col].mean())

        if "PRECTOTCORR" in daily_df.columns:
            summary["seasonal_total_precip_mm"] = float(daily_df["PRECTOTCORR"].sum())
        if "consecutive_dry_days" in daily_df.columns:
            summary["max_consecutive_dry_days"] = float(daily_df["consecutive_dry_days"].max())
        if "heat_stress_day_flag" in daily_df.columns:
            summary["total_heat_stress_days"] = float(daily_df["heat_stress_day_flag"].sum())
        if "heavy_rain_event_flag" in daily_df.columns:
            summary["total_heavy_rain_days"] = float(daily_df["heavy_rain_event_flag"].sum())
        if "modis_et_mm_8d" in daily_df.columns:
            # Approximate seasonal cumulative ET from daily-interpolated 8-day values
            summary["seasonal_cumulative_et_mm"] = float((daily_df["modis_et_mm_8d"] / 8.0).sum())

        return summary


def list_supported_datasets() -> pd.DataFrame:
    """
    Return a clean tabular overview of all NASA datasets registered in
    `DATASET_REGISTRY` for quick inspection.
    """
    rows = []
    for category, meta in DATASET_REGISTRY.items():
        rows.append(
            {
                "category": category,
                "mission": meta.get("mission", ""),
                "description": meta.get("description", ""),
                "has_nasa_power": "power_parameters" in meta,
                "has_appeears": "appeears" in meta,
                "has_earthaccess": any(k.startswith("earthaccess") for k in meta),
                "has_gee": any(k.startswith("gee") for k in meta),
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("=== Field Shift NASA Dataset Registry ===")
    registry_df = list_supported_datasets()
    print(registry_df.to_string(index=False))

    # Example farm definition (e.g., agricultural region in Bangladesh)
    demo_farm = FarmLocation(
        farm_id="farm_bd_001",
        latitude=24.3745,
        longitude=88.6042,
        buffer_meters=250.0,
        irrigation_access=True,
        farmer_priority="water_conservation",
    )
    print(f"\nConfigured FarmLocation: {demo_farm}")
    print("Bounding Box (pad=0.05 deg):", demo_farm.bounding_box())
