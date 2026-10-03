# English translation of the original Spanish delivery (comments and messages only; code unchanged). Original: generated_code/phase2/t6/t6_gemini31pro.py
from pathlib import Path
import xarray as xr


def cargar_netcdf_climatico(ruta_archivo: str | Path) -> xr.Dataset:
    """Loads and validates a NetCDF climate data file.

    Checks existence, format validity, required dimensions (lat, lon,
    time)
    and the presence of key variables ('ppt' or 'pet').
    """
    path = Path(ruta_archivo)

    # 1. Check that the file exists
    if not path.is_file():
        raise FileNotFoundError(
            f"The specified file does not exist or the path is invalid: '{path}'"
        )

    # 2. Check that it is a valid NetCDF
    try:
        ds = xr.open_dataset(path)
    except Exception as e:
        raise ValueError(
            f"The file '{path.name}' is not a valid NetCDF file or is damaged. "
            f"Technical detail: {e}"
        ) from e

    # 3. Check expected dimensions
    dimensiones_requeridas = {"lat", "lon", "time"}
    dims_presentes = set(ds.dims)
    dims_faltantes = dimensiones_requeridas - dims_presentes

    if dims_faltantes:
        ds.close()
        raise ValueError(
            f"The file '{path.name}' does not comply with the dimensional structure. "
            f"The following dimensions are missing: {list(dims_faltantes)}. "
            f"Dimensions found: {list(dims_presentes)}"
        )

    # 4. Check that it contains at least one of the target variables
    variables_requeridas = {"ppt", "pet"}
    vars_presentes = set(ds.data_vars)

    if not variables_requeridas.intersection(vars_presentes):
        ds.close()
        raise ValueError(
            f"The file '{path.name}' does not contain any expected climate variable ('ppt' or 'pet'). "
            f"Variables found: {list(vars_presentes)}"
        )

    return ds


# ==============================================================================
# TEST BLOCK
# ==============================================================================
if __name__ == "__main__":
    print("--- STARTING SAFE LOADING TESTS ---\n")

    # Tests to run
    pruebas = [
        ("Case A: Nonexistent file", "archivo_falso_1981_2025.nc", None),
        ("Case B: Text renamed to NetCDF", "pet_noroeste_fake.nc", "Invalid plain text"),
    ]

    for nombre_prueba, nombre_archivo, contenido_falso in pruebas:
        path_prueba = Path(nombre_archivo)
        print(f"Running {nombre_prueba}...")

        # If it requires a corrupt text file, we create it
        if contenido_falso:
            path_prueba.write_text(contenido_falso, encoding="utf-8")

        try:
            ds = cargar_netcdf_climatico(path_prueba)
            print("  ❌ TEST ERROR: The file was loaded when it should have failed.")
            ds.close()
        except (FileNotFoundError, ValueError) as err:
            print(f"  ✅ CORRECTLY CAUGHT:\n     --> {err}\n")
        finally:
            # Cleanup of the fake file created for the test
            if contenido_falso and path_prueba.exists():
                path_prueba.unlink()

    print("--- END OF TESTS ---")