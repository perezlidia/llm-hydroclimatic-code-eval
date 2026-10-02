from pathlib import Path
import xarray as xr


def procesar_terraclimate():
    # 1. Definir rutas y patrón de búsqueda
    input_dir = Path("D:/2027-ARTICULOS/CAPITULO-SOFTWARE/ppt/")
    output_path = input_dir / "TerraClimate_ppt_1981_2025_cortado.nc"

    # Obtener y ordenar la lista de archivos por año
    archivos = sorted(list(input_dir.glob("TerraClimate_ppt_*.nc")))

    if not archivos:
        raise FileNotFoundError(
            f"No se encontraron archivos NetCDF en {input_dir}"
        )

    print(f"Cargando {len(archivos)} archivos NetCDF...")

    # Abrir y unificar todos los archivos a lo largo de la dimensión temporal
    # combining='by_coords' alinea automáticamente la coordenada 'time'
    ds = xr.open_mfdataset(archivos, combine="by_coords", parallel=False)

    # 2. Definir límites geográficos
    lat_min, lat_max = 22.5, 32.7
    lon_min, lon_max = -117.2, -105.0

    # Asegurar recorte correcto independientemente de si 'lat' es ascendente o descendente
    if ds.lat[0] > ds.lat[-1]:  # Latitud descendente (p. ej., de 90 a -90)
        ds_cropped = ds.sel(
            lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max)
        )
    else:  # Latitud ascendente (p. ej., de -90 a 90)
        ds_cropped = ds.sel(
            lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max)
        )

    # 3. Verificación y ordenamiento temporal
    # Asegurar orden cronológico estricto
    ds_cropped = ds_cropped.sortby("time")

    # Verificar duplicados en la dimensión temporal
    tiempos = ds_cropped["time"].values
    if len(tiempos) != len(set(tiempos)):
        print(
            "⚠️ Advertencia: Se detectaron fechas duplicadas. Eliminando duplicados..."
        )
        _, index = np.unique(ds_cropped["time"], return_index=True)
        ds_cropped = ds_cropped.isel(time=index)
    else:
        print("✓ Confirmado: No hay fechas duplicadas y el orden es correcto.")

    # 4. Imprimir dimensiones e información del Dataset resultante
    num_pasos_tiempo = ds_cropped.sizes["time"]
    num_lat = ds_cropped.sizes["lat"]
    num_lon = ds_cropped.sizes["lon"]

    print("\n--- RESUMEN DEL DATASET RESULTANTE ---")
    print(f"Pasos de tiempo totales: {num_pasos_tiempo} meses")
    print(
        f"Rango de fechas: {str(ds_cropped.time.values[0])[:10]} a {str(ds_cropped.time.values[-1])[:10]}"
    )
    print(
        f"Dimensiones espaciales (Lat x Lon): {num_lat} x {num_lon} puntos de grilla"
    )
    print(
        f"Extensión Latitud: [{ds_cropped.lat.min().item():.2f}°, {ds_cropped.lat.max().item():.2f}°]"
    )
    print(
        f"Extensión Longitud: [{ds_cropped.lon.min().item():.2f}°, {ds_cropped.lon.max().item():.2f}°]"
    )
    print("--------------------------------------\n")

    # 5. Guardar en un nuevo archivo NetCDF
    print(f"Guardando resultado en: {output_path}")
    ds_cropped.to_netcdf(output_path)
    print("¡Proceso completado con éxito!")


if __name__ == "__main__":
    procesar_terraclimate()