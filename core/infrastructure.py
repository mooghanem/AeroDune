import numpy as np

try:
    import osmnx as ox
    from shapely.geometry import LineString, Point
    OSMNX_AVAILABLE = True
except ImportError:
    OSMNX_AVAILABLE = False


def generate_synthetic_highway_mask(grid_shape):
    """
    Generates a diagonal raster line simulating the Qantara-El Arish Highway
    across the grid matrix when OSMnx network data is unavailable.
    """
    ny, nx = grid_shape
    mask = np.zeros(grid_shape, dtype=np.uint8)

    # Draw diagonal road across grid
    for col in range(nx):
        row = int(0.35 * ny + 0.25 * ny * (col / nx))
        if 0 <= row < ny:
            mask[max(0, row-1):min(ny, row+2), col] = 1 # 3-pixel wide road

    return mask


def fetch_and_rasterize_highways(dem_shape, transform=None, bbox=(31.0, 31.4, 33.2, 33.8)):
    """
    Fetches real road networks from OpenStreetMap for North Sinai and rasterizes them
    onto a binary mask matching the DEM dimensions.
    """
    if not OSMNX_AVAILABLE:
        print("[Infrastructure] OSMnx package not installed. Using synthetic highway overlay.")
        return generate_synthetic_highway_mask(dem_shape)

    try:
        north, south, east, west = bbox
        graph = ox.graph_from_bbox(bbox=(north, south, east, west), network_type='drive')
        edges = ox.graph_to_gdfs(graph, nodes=False)

        mask = np.zeros(dem_shape, dtype=np.uint8)
        ny, nx = dem_shape

        # Approximate coordinate mapping to matrix grid
        for _, row in edges.iterrows():
            geom = row['geometry']
            if isinstance(geom, LineString):
                coords = list(geom.coords)
                for lon, lat in coords:
                    # Convert lat/lon to matrix index
                    c = int((lon - west) / (east - west) * nx)
                    r = int((north - lat) / (north - south) * ny)
                    if 0 <= r < ny and 0 <= c < nx:
                        mask[max(0, r-1):min(ny, r+2), max(0, c-1):min(nx, c+2)] = 1

        print(f"[Infrastructure] Successfully fetched and rasterized OSM road networks.")
        return mask

    except Exception as e:
        print(f"[Infrastructure] Failed to query OSMnx data: {e}. Falling back to synthetic road.")
        return generate_synthetic_highway_mask(dem_shape)
