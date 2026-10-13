import os
import glob
import numpy as np

try:
    import rasterio
    from rasterio.merge import merge
    from rasterio.transform import Affine
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False


def generate_synthetic_sinai_dem(grid_shape=(120, 160), pixel_size=30.0):
    """
    Generates a realistic synthetic DEM matrix of North Sinai coastal dunes & inland ridges.
    Used as an automatic fallback if GeoTIFF files are missing.
    """
    ny, nx = grid_shape
    y = np.linspace(0, 10, ny)
    x = np.linspace(0, 15, nx)
    X, Y = np.meshgrid(x, y)

    # Base elevation sloping upwards towards inland south
    base = Y * 8.0 + 5.0
    
    # Transverse dune field patterns
    dunes = 12.0 * np.sin(0.8 * X - 0.4 * Y) * np.cos(0.5 * Y)
    
    # Barchan dune mounds
    mounds = 15.0 * np.exp(-((X - 7.0)**2 + (Y - 4.0)**2) / 3.0)
    
    elevation = base + dunes + mounds
    elevation = np.maximum(elevation, 0.0) # Sea level floor

    # Mock spatial transform (North Sinai coordinates ~31.0 N, 33.5 E)
    transform = Affine(pixel_size / 111000.0, 0.0, 33.0, 0.0, -pixel_size / 111000.0, 31.2)
    
    return elevation, transform


def load_and_mosaic_dem(data_folder="data", grid_shape=(120, 160)):
    """
    Loads and mosaics all GeoTIFF (.tif) tiles found inside data_folder.
    Returns:
        mosaic (2D np.ndarray): Merged elevation matrix (Z)
        transform (Affine): Spatial geo-transform metadata
    """
    if not RASTERIO_AVAILABLE:
        print("[DEM Handler] Rasterio not installed. Using synthetic DEM.")
        return generate_synthetic_sinai_dem(grid_shape=grid_shape)

    tif_files = glob.glob(os.path.join(data_folder, "*.tif")) + glob.glob(os.path.join(data_folder, "*.tiff"))
    
    if not tif_files:
        print(f"[DEM Handler] No .tif files found in '{data_folder}'. Generating synthetic Sinai DEM.")
        return generate_synthetic_sinai_dem(grid_shape=grid_shape)

    try:
        src_files_to_mosaic = [rasterio.open(fp) for fp in tif_files]
        mosaic, out_trans = merge(src_files_to_mosaic)
        
        # Close handles
        for src in src_files_to_mosaic:
            src.close()
            
        elevation_matrix = mosaic[0].astype(np.float32)
        # Handle nodata values (-32768 or negative values)
        elevation_matrix[elevation_matrix < -100] = 0.0
        
        print(f"[DEM Handler] Successfully mosaiced {len(tif_files)} GeoTIFF tiles. Grid shape: {elevation_matrix.shape}")
        return elevation_matrix, out_trans

    except Exception as e:
        print(f"[DEM Handler] Error during DEM mosaicing: {e}. Falling back to synthetic DEM.")
        return generate_synthetic_sinai_dem(grid_shape=grid_shape)
