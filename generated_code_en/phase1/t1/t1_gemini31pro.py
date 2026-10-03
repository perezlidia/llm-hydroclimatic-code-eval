# English translation of the original Spanish delivery (comments and messages only; code unchanged). Original: generated_code/phase1/t1/t1_gemini31pro.py
from pathlib import Path
import xarray as xr


def procesar_terraclimate():
    # 1. Define paths and search pattern
    input_dir = Path("D:/2027-ARTICULOS/CAPITULO-SOFTWARE/ppt/")
    output_path = input_dir / "TerraClimate_ppt_1981_2025_cortado.nc"

    # Get the list of files and sort it by year
    archivos = sorted(list(input_dir.glob("TerraClimate_ppt_*.nc")))

    if not archivos:
        raise FileNotFoundError(
            f"No NetCDF files were found in {input_dir}"
        )

    print(f"Loading {len(archivos)} NetCDF files...")

    # Open and merge all files along the time dimension
    # combining='by_coords' automatically aligns the 'time' coordinate
    ds = xr.open_mfdataset(archivos, combine="by_coords", parallel=False)

    # 2. Define geographic bounds
    lat_min, lat_max = 22.5, 32.7
    lon_min, lon_max = -117.2, -105.0

    # Ensure a correct crop regardless of whether 'lat' is ascending or descending
    if ds.lat[0] > ds.lat[-1]:  # Descending latitude (e.g., from 90 to -90)
        ds_cropped = ds.sel(
            lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max)
        )
    else:  # Ascending latitude (e.g., from -90 to 90)
        ds_cropped = ds.sel(
            lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max)
        )

    # 3. Time verification and sorting
    # Ensure strict chronological order
    ds_cropped = ds_cropped.sortby("time")

    # Check for duplicates in the time dimension
    tiempos = ds_cropped["time"].values
    if len(tiempos) != len(set(tiempos)):
        print(
            "⚠️ Warning: Duplicate dates were detected. Removing duplicates..."
        )
        _, index = np.unique(ds_cropped["time"], return_index=True)
        ds_cropped = ds_cropped.isel(time=index)
    else:
        print("✓ Confirmed: There are no duplicate dates and the order is correct.")

    # 4. Print dimensions and information of the resulting Dataset
    num_pasos_tiempo = ds_cropped.sizes["time"]
    num_lat = ds_cropped.sizes["lat"]
    num_lon = ds_cropped.sizes["lon"]

    print("\n--- SUMMARY OF THE RESULTING DATASET ---")
    print(f"Total time steps: {num_pasos_tiempo} months")
    print(
        f"Date range: {str(ds_cropped.time.values[0])[:10]} to {str(ds_cropped.time.values[-1])[:10]}"
    )
    print(
        f"Spatial dimensions (Lat x Lon): {num_lat} x {num_lon} grid points"
    )
    print(
        f"Latitude extent: [{ds_cropped.lat.min().item():.2f}°, {ds_cropped.lat.max().item():.2f}°]"
    )
    print(
        f"Longitude extent: [{ds_cropped.lon.min().item():.2f}°, {ds_cropped.lon.max().item():.2f}°]"
    )
    print("--------------------------------------\n")

    # 5. Save to a new NetCDF file
    print(f"Saving result to: {output_path}")
    ds_cropped.to_netcdf(output_path)
    print("Process completed successfully!")


if __name__ == "__main__":
    procesar_terraclimate()