from pathlib import Path
import geopandas as gpd
import xarray as xr
import rioxarray
from shapely.geometry import mapping


def procesar_terraclimate():
    # -------------------------------------------------------------------------
    # 0. Configuración de rutas y archivos
    # -------------------------------------------------------------------------
    path_ppt = Path("D:/2027-ARTICULOS/CAPITULO-SOFTWARE/ppt/")
    path_pet = Path("D:/2027-ARTICULOS/CAPITULO-SOFTWARE/pet/")
    path_shp = Path("D:/2027-ARTICULOS/CAPITULO-SOFTWARE/Noroeste/Noroeste.shp")

    out_ppt_path = "ppt_noroeste_shp_1981_2025.nc"
    out_pet_path = "pet_noroeste_shp_1981_2025.nc"

    # Obtener y ordenar rutas de archivos por año (1981 a 2025)
    files_ppt = sorted(list(path_ppt.glob("TerraClimate_ppt_*.nc")))
    files_pet = sorted(list(path_pet.glob("TerraClimate_pet_*.nc")))

    if len(files_ppt) != 45 or len(files_pet) != 45:
        print(f"Advertencia: Se encontraron {len(files_ppt)} archivos ppt y {len(files_pet)} archivos pet (se esperaban 45).")

    # -------------------------------------------------------------------------
    # 1 y 2. Cargar y unificar datasets a lo largo del tiempo
    # -------------------------------------------------------------------------
    print("Cargando y unificando archivos NetCDF de precipitación (ppt)...")
    ds_ppt = xr.open_mfdataset(files_ppt, combine="by_coords")

    print("Cargando y unificando archivos NetCDF de evapotranspiración (pet)...")
    ds_pet = xr.open_mfdataset(files_pet, combine="by_coords")

    # -------------------------------------------------------------------------
    # Preparar CRS y coordenadas espaciales para rioxarray
    # -------------------------------------------------------------------------
    # Asegurar que las coordenadas espaciales estén declaradas correctamente para rioxarray
    ds_ppt = ds_ppt.rio.write_crs("EPSG:4326")
    ds_pet = ds_pet.rio.write_crs("EPSG:4326")
    
    ds_ppt = ds_ppt.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
    ds_pet = ds_pet.rio.set_spatial_dims(x_dim="lon", y_dim="lat")

    # -------------------------------------------------------------------------
    # 3. Leer shapefile y realizar el recorte por el POLÍGONO REAL
    # -------------------------------------------------------------------------
    print("\nLeyendo el shapefile 'Noroeste.shp'...")
    gdf = gpd.read_file(path_shp)

    # Asegurar que el shapefile esté en EPSG:4326 (WGS84)
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    geometries = gdf.geometry.values

    print("Recortando y enmascarando los datasets por la geometría exacta del polígono...")
    # clip_box=False asegura que las celdas fuera del polígono exacto se asignen como NaN
    ds_ppt_clipped = ds_ppt.rio.clip(geometries, crs=gdf.crs, drop=True, clip_box=False)
    ds_pet_clipped = ds_pet.rio.clip(geometries, crs=gdf.crs, drop=True, clip_box=False)

    # -------------------------------------------------------------------------
    # 4. Verificaciones explícitas
    # -------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("REPORTE DE VERIFICACIÓN")
    print("=" * 50)

    # Verificación de orden cronológico
    is_ppt_sorted = ds_ppt_clipped.indexes["time"].is_monotonic_increasing
    is_pet_sorted = ds_pet_clipped.indexes["time"].is_monotonic_increasing
    print(f"PPT - ¿Fechas en orden cronológico ascendente?: {is_ppt_sorted}")
    print(f"PET - ¿Fechas en orden cronológico ascendente?: {is_pet_sorted}")

    # Verificación de fechas duplicadas
    has_ppt_dup = ds_ppt_clipped.indexes["time"].has_duplicates
    has_pet_dup = ds_pet_clipped.indexes["time"].has_duplicates
    print(f"PPT - ¿Contiene fechas duplicadas?: {has_ppt_dup}")
    print(f"PET - ¿Contiene fechas duplicadas?: {has_pet_dup}")

    # Verificación de consistencia entre ambos datasets
    same_time_len = len(ds_ppt_clipped.time) == len(ds_pet_clipped.time)
    same_lat = ds_ppt_clipped["lat"].equals(ds_pet_clipped["lat"])
    same_lon = ds_ppt_clipped["lon"].equals(ds_pet_clipped["lon"])

    print(f"¿Ambos datasets tienen el mismo número de pasos temporales?: {same_time_len}")
    print(f"¿Ambos datasets coinciden EXACTAMENTE en la coordenada Latitud?: {same_lat}")
    print(f"¿Ambos datasets coinciden EXACTAMENTE en la coordenada Longitud?: {same_lon}")

    # -------------------------------------------------------------------------
    # 5. Dimensiones resultantes e impresión
    # -------------------------------------------------------------------------
    n_time = len(ds_ppt_clipped.time)
    n_lat = len(ds_ppt_clipped.lat)
    n_lon = len(ds_ppt_clipped.lon)

    print("\n" + "=" * 50)
    print("DIMENSIONES Y PASOS DE TIEMPO RESULTANTES")
    print("=" * 50)
    print(f"Total de pasos de tiempo (meses): {n_time} (esperados: {45 * 12})")
    print(f"Dimensiones de la grilla espacial (Latitud x Longitud): {n_lat} x {n_lon}")
    print(f"Forma de la variable ppt recortada: {ds_ppt_clipped['ppt'].shape}")
    print(f"Forma de la variable pet recortada: {ds_pet_clipped['pet'].shape}")

    # -------------------------------------------------------------------------
    # 6. Guardar resultados
    # -------------------------------------------------------------------------
    print("\nGuardando NetCDF de precipitación recortado...")
    ds_ppt_clipped.to_netcdf(out_ppt_path)

    print("Guardando NetCDF de evapotranspiración recortada...")
    ds_pet_clipped.to_netcdf(out_pet_path)

    print("\n¡Proceso finalizado con éxito!")


if __name__ == "__main__":
    procesar_terraclimate()