# English translation of the original Spanish delivery (comments and messages only; code unchanged). Original: generated_code/phase2/t1/t1_gemini31pro.py
from pathlib import Path
import geopandas as gpd
import xarray as xr
import rioxarray
from shapely.geometry import mapping


def procesar_terraclimate():
    # -------------------------------------------------------------------------
    # 0. Configuration of paths and files
    # -------------------------------------------------------------------------
    path_ppt = Path("D:/2027-ARTICULOS/CAPITULO-SOFTWARE/ppt/")
    path_pet = Path("D:/2027-ARTICULOS/CAPITULO-SOFTWARE/pet/")
    path_shp = Path("D:/2027-ARTICULOS/CAPITULO-SOFTWARE/Noroeste/Noroeste.shp")

    out_ppt_path = "ppt_noroeste_shp_1981_2025.nc"
    out_pet_path = "pet_noroeste_shp_1981_2025.nc"

    # Get and sort file paths by year (1981 to 2025)
    files_ppt = sorted(list(path_ppt.glob("TerraClimate_ppt_*.nc")))
    files_pet = sorted(list(path_pet.glob("TerraClimate_pet_*.nc")))

    if len(files_ppt) != 45 or len(files_pet) != 45:
        print(f"Warning: Found {len(files_ppt)} ppt files and {len(files_pet)} pet files (45 were expected).")

    # -------------------------------------------------------------------------
    # 1 and 2. Load and merge datasets along time
    # -------------------------------------------------------------------------
    print("Loading and merging precipitation (ppt) NetCDF files...")
    ds_ppt = xr.open_mfdataset(files_ppt, combine="by_coords")

    print("Loading and merging evapotranspiration (pet) NetCDF files...")
    ds_pet = xr.open_mfdataset(files_pet, combine="by_coords")

    # -------------------------------------------------------------------------
    # Prepare CRS and spatial coordinates for rioxarray
    # -------------------------------------------------------------------------
    # Make sure the spatial coordinates are declared correctly for rioxarray
    ds_ppt = ds_ppt.rio.write_crs("EPSG:4326")
    ds_pet = ds_pet.rio.write_crs("EPSG:4326")

    ds_ppt = ds_ppt.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
    ds_pet = ds_pet.rio.set_spatial_dims(x_dim="lon", y_dim="lat")

    # -------------------------------------------------------------------------
    # 3. Read shapefile and clip by the REAL POLYGON
    # -------------------------------------------------------------------------
    print("\nReading the shapefile 'Noroeste.shp'...")
    gdf = gpd.read_file(path_shp)

    # Make sure the shapefile is in EPSG:4326 (WGS84)
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    geometries = gdf.geometry.values

    print("Clipping and masking the datasets by the exact polygon geometry...")
    # clip_box=False ensures that cells outside the exact polygon are set to NaN
    ds_ppt_clipped = ds_ppt.rio.clip(geometries, crs=gdf.crs, drop=True, clip_box=False)
    ds_pet_clipped = ds_pet.rio.clip(geometries, crs=gdf.crs, drop=True, clip_box=False)

    # -------------------------------------------------------------------------
    # 4. Explicit checks
    # -------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("VERIFICATION REPORT")
    print("=" * 50)

    # Chronological order check
    is_ppt_sorted = ds_ppt_clipped.indexes["time"].is_monotonic_increasing
    is_pet_sorted = ds_pet_clipped.indexes["time"].is_monotonic_increasing
    print(f"PPT - Dates in ascending chronological order?: {is_ppt_sorted}")
    print(f"PET - Dates in ascending chronological order?: {is_pet_sorted}")

    # Duplicate dates check
    has_ppt_dup = ds_ppt_clipped.indexes["time"].has_duplicates
    has_pet_dup = ds_pet_clipped.indexes["time"].has_duplicates
    print(f"PPT - Contains duplicate dates?: {has_ppt_dup}")
    print(f"PET - Contains duplicate dates?: {has_pet_dup}")

    # Consistency check between both datasets
    same_time_len = len(ds_ppt_clipped.time) == len(ds_pet_clipped.time)
    same_lat = ds_ppt_clipped["lat"].equals(ds_pet_clipped["lat"])
    same_lon = ds_ppt_clipped["lon"].equals(ds_pet_clipped["lon"])

    print(f"Do both datasets have the same number of time steps?: {same_time_len}")
    print(f"Do both datasets match EXACTLY in the Latitude coordinate?: {same_lat}")
    print(f"Do both datasets match EXACTLY in the Longitude coordinate?: {same_lon}")

    # -------------------------------------------------------------------------
    # 5. Resulting dimensions and printout
    # -------------------------------------------------------------------------
    n_time = len(ds_ppt_clipped.time)
    n_lat = len(ds_ppt_clipped.lat)
    n_lon = len(ds_ppt_clipped.lon)

    print("\n" + "=" * 50)
    print("RESULTING DIMENSIONS AND TIME STEPS")
    print("=" * 50)
    print(f"Total time steps (months): {n_time} (expected: {45 * 12})")
    print(f"Spatial grid dimensions (Latitude x Longitude): {n_lat} x {n_lon}")
    print(f"Shape of the clipped ppt variable: {ds_ppt_clipped['ppt'].shape}")
    print(f"Shape of the clipped pet variable: {ds_pet_clipped['pet'].shape}")

    # -------------------------------------------------------------------------
    # 6. Save results
    # -------------------------------------------------------------------------
    print("\nSaving clipped precipitation NetCDF...")
    ds_ppt_clipped.to_netcdf(out_ppt_path)

    print("Saving clipped evapotranspiration NetCDF...")
    ds_pet_clipped.to_netcdf(out_pet_path)

    print("\nProcess completed successfully!")


if __name__ == "__main__":
    procesar_terraclimate()
