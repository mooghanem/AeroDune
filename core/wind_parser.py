import os
import numpy as np
import pandas as pd


def generate_synthetic_wind_data(num_days=365):
    """
    Generates realistic daily wind metrics for North Sinai.
    Dominant wind direction: North/North-West (300° - 340°).
    """
    np.random.seed(42)
    doys = np.arange(1, num_days + 1)
    
    # Rayleigh-like distribution for wind speeds with seasonal storms
    base_speed = np.random.weibull(a=2.0, size=num_days) * 5.2 + 2.5
    
    # Introduce periodic severe sand storm events (WS10M > 6.5 m/s)
    storm_mask = np.random.rand(num_days) < 0.25
    base_speed[storm_mask] += np.random.uniform(3.0, 7.0, size=np.sum(storm_mask))

    # Wind direction centered around 320 degrees (North-West)
    wind_dir = (320.0 + np.random.normal(0, 25, size=num_days)) % 360.0

    df = pd.DataFrame({
        'YEAR': 2025,
        'DOY': doys,
        'WS10M': np.round(base_speed, 2),
        'WD10M': np.round(wind_dir, 1)
    })
    return df


def parse_nasa_wind_csv(file_path="data/sinai_wind_data.csv"):
    """
    Parses NASA POWER daily meteorological CSV files.
    Automatically skips metadata headers and handles missing flags (-999).
    """
    if not os.path.exists(file_path):
        print(f"[Wind Parser] File '{file_path}' not found. Generating synthetic wind timeline.")
        return generate_synthetic_wind_data()

    try:
        # Find header start line
        skip_lines = 0
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for i, line in enumerate(f):
                if "-END HEADER-" in line or "YEAR,DOY" in line or "YEAR" in line and "WS10M" in line:
                    skip_lines = i + 1
                    break
        
        df = pd.read_csv(file_path, skiprows=skip_lines)
        df.columns = df.columns.str.strip().str.upper()

        required = ['WS10M', 'WD10M']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required wind column: {col}")

        # Handle missing data flags (-999)
        df['WS10M'] = df['WS10M'].replace(-999, np.nan)
        df['WD10M'] = df['WD10M'].replace(-999, np.nan)

        df['WS10M'] = df['WS10M'].fillna(df['WS10M'].mean())
        df['WD10M'] = df['WD10M'].fillna(315.0) # Default NW wind

        if 'DOY' not in df.columns:
            df['DOY'] = np.arange(1, len(df) + 1)

        print(f"[Wind Parser] Successfully loaded {len(df)} wind records from '{file_path}'.")
        return df

    except Exception as e:
        print(f"[Wind Parser] Error reading CSV: {e}. Falling back to synthetic wind data.")
        return generate_synthetic_wind_data()
