from pathlib import Path
import xarray as xr


def cargar_netcdf_climatico(ruta_archivo: str | Path) -> xr.Dataset:
    """Carga y valida un archivo NetCDF de datos climáticos.

    Verifica existencia, validez de formato, dimensiones requeridas (lat, lon,
    time)
    y la presencia de variables clave ('ppt' o 'pet').
    """
    path = Path(ruta_archivo)

    # 1. Verificar existencia del archivo
    if not path.is_file():
        raise FileNotFoundError(
            f"El archivo especificado no existe o la ruta es inválida: '{path}'"
        )

    # 2. Verificar que sea un NetCDF válido
    try:
        ds = xr.open_dataset(path)
    except Exception as e:
        raise ValueError(
            f"El archivo '{path.name}' no es un archivo NetCDF válido o está dañado. "
            f"Detalle técnico: {e}"
        ) from e

    # 3. Verificar dimensiones esperadas
    dimensiones_requeridas = {"lat", "lon", "time"}
    dims_presentes = set(ds.dims)
    dims_faltantes = dimensiones_requeridas - dims_presentes

    if dims_faltantes:
        ds.close()
        raise ValueError(
            f"El archivo '{path.name}' no cumple con la estructura dimensional. "
            f"Faltan las siguientes dimensiones: {list(dims_faltantes)}. "
            f"Dimensiones encontradas: {list(dims_presentes)}"
        )

    # 4. Verificar que contenga al menos una de las variables objetivo
    variables_requeridas = {"ppt", "pet"}
    vars_presentes = set(ds.data_vars)

    if not variables_requeridas.intersection(vars_presentes):
        ds.close()
        raise ValueError(
            f"El archivo '{path.name}' no contiene ninguna variable climática esperada ('ppt' o 'pet'). "
            f"Variables encontradas: {list(vars_presentes)}"
        )

    return ds


# ==============================================================================
# BLOQUE DE PRUEBA
# ==============================================================================
if __name__ == "__main__":
    print("--- INICIANDO PRUEBAS DE CARGA SEGURA ---\n")

    # Pruebas a realizar
    pruebas = [
        ("Caso A: Archivo inexistente", "archivo_falso_1981_2025.nc", None),
        ("Caso B: Texto renombrado a NetCDF", "pet_noroeste_fake.nc", "Texto plano no válido"),
    ]

    for nombre_prueba, nombre_archivo, contenido_falso in pruebas:
        path_prueba = Path(nombre_archivo)
        print(f"Ejecutando {nombre_prueba}...")

        # Si requiere un archivo de texto corrupto, lo creamos
        if contenido_falso:
            path_prueba.write_text(contenido_falso, encoding="utf-8")

        try:
            ds = cargar_netcdf_climatico(path_prueba)
            print("  ❌ ERROR EN PRUEBA: El archivo se cargó cuando debía fallar.")
            ds.close()
        except (FileNotFoundError, ValueError) as err:
            print(f"  ✅ CAPTURADO CORRECTAMENTE:\n     --> {err}\n")
        finally:
            # Limpieza del archivo falso creado para la prueba
            if contenido_falso and path_prueba.exists():
                path_prueba.unlink()

    print("--- FIN DE LAS PRUEBAS ---")