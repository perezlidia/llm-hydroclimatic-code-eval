# English translation of the original Spanish delivery (comments and messages only; code unchanged). Original: generated_code/phase1/t6/t6_gemini31pro_part2_tests.py
import os


def ejecutar_pruebas():
    # --- Case a: Nonexistent file ---
    archivo_inexistente = "archivo_fantasma.nc"

    print("--- Test A: Trying to load nonexistent file ---")
    try:
        cargar_netcdf_seguro(archivo_inexistente)
    except (FileNotFoundError, ValueError) as e:
        print(f"SUCCESS - Error caught correctly:\n  -> {e}\n")

    # --- Case b: Plain text file with .nc extension ---
    archivo_corrupto = "datos_falsos.nc"

    # Create fake text file
    with open(archivo_corrupto, "w", encoding="utf-8") as f:
        f.write("This is a plain text file disguised as NetCDF.")

    print("--- Test B: Trying to load text file renamed to .nc ---")
    try:
        cargar_netcdf_seguro(archivo_corrupto)
    except (FileNotFoundError, ValueError) as e:
        print(f"SUCCESS - Error caught correctly:\n  -> {e}\n")
    finally:
        # Cleanup of the temporary test file
        if os.path.exists(archivo_corrupto):
            os.remove(archivo_corrupto)


if __name__ == "__main__":
    ejecutar_pruebas()