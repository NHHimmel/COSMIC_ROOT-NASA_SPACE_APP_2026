from field_shift_nasa_data import FarmLocation, FieldShiftFeatureBuilder

def main():
    # Compute seasonal aggregates for crop-rotation model input
    seasonal_features = builder.summarize_growing_season(daily_df)
    print(daily_df[["T2M", "RH2M", "ALLSKY_SFC_SW_DWN", "PRECTOTCORR", "GWETROOT", "vpd_kpa"]].head())
    print("Seasonal Summary:", seasonal_features)

# Define target farm location and farmer metadata
farm = FarmLocation(
    farm_id="rajshahi_plot_12",
    latitude=24.3745,
    longitude=88.6042,
    buffer_meters=250.0,
    irrigation_access=True,
    farmer_priority="water_conservation",
)

# Build daily training table (POWER weather, radiation, rainfall, soil wetness + stress flags)
builder = FieldShiftFeatureBuilder()
daily_df = builder.build_farm_training_table(
    location=farm,
    start_date="2024-01-01",
    end_date="2024-12-31",
    include_gee_features=False,
    include_earthaccess_features=False,
)


if __name__ == "__main__":
    main()
