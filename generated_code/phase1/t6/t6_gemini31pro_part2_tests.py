import os


def ejecutar_pruebas():
    # --- Caso a: Archivo inexistente ---
    archivo_inexistente = "archivo_fantasma.nc"

    print("--- Prueba A: Intentando cargar archivo inexistente ---")
    try:
        cargar_netcdf_seguro(archivo_inexistente)
    except (FileNotFoundError, ValueError) as e:
        print(f"ÉXITO - Error capturado correctamente:\n  -> {e}\n")

    # --- Caso b: Archivo de texto plano con extensión .nc ---
    archivo_corrupto = "datos_falsos.nc"

    # Crear archivo de texto falso
    with open(archivo_corrupto, "w", encoding="utf-8") as f:
        f.write("Este es un archivo de texto plano disfrazado de NetCDF.")

    print("--- Prueba B: Intentando cargar archivo de texto renombrado a .nc ---")
    try:
        cargar_netcdf_seguro(archivo_corrupto)
    except (FileNotFoundError, ValueError) as e:
        print(f"ÉXITO - Error capturado correctamente:\n  -> {e}\n")
    finally:
        # Limpieza del archivo temporal de prueba
        if os.path.exists(archivo_corrupto):
            os.remove(archivo_corrupto)


if __name__ == "__main__":
    ejecutar_pruebas()