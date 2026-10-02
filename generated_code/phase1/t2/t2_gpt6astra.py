#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Limpia precipitación mensual sin rellenar el océano.

Instalar:
    python -m pip install numpy xarray netCDF4 "dask[array]"

Ejecutar:
    python limpiar_ppt_terraclimate.py "entrada.nc" "salida_limpia.nc"

Supuestos y comportamiento:
* La variable (por defecto, ppt) está en mm/mes y tiene las dimensiones
  time y dos dimensiones espaciales, en cualquier orden.
* Una celda con NaN en TODA la serie se considera océano según el supuesto
  del usuario. Solo los datos no permiten distinguirla de tierra que nunca
  tuvo observaciones; para esa distinción haría falta una máscara externa.
* La máscara se obtiene del original, ANTES de descartar valores negativos.
  Las celdas con datos únicamente negativos siguen siendo problemáticas,
  no se reclasifican como océano.
* La climatología usa exclusivamente valores finitos >= 0 del mismo mes y
  celda en la serie original. Cero es precipitación válida.
* También se consideran inválidos los infinitos, si los hubiera.
* Si una celda no tiene observaciones válidas para un mes que necesita
  imputación, se reporta el problema y NO se escribe un resultado incompleto.
* No se crean fechas ausentes ni se cambian las coordenadas o la cuadrícula.
* Se escribe un archivo NUEVO con ppt, las otras variables originales,
  mascara_tierra (1=tierra, 0=océano) y ppt_imputado (1=valor corregido).
* Se usa float64 sin el empaquetado entero original para conservar los
  promedios calculados. Dask procesa bloques espaciales para limitar la RAM.

Referencia de xarray para la media por grupos:
https://docs.xarray.dev/en/stable/generated/xarray.core.groupby.DataArrayGroupBy.mean.html
"""

import argparse
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import dask
import numpy as np
import xarray as xr


def validar_ppt(ppt):
    """Exige datos numéricos y una observación como máximo por año/mes."""
    if ppt.ndim != 3 or "time" not in ppt.dims:
        raise ValueError("ppt debe tener time y dos dimensiones espaciales.")
    if any(n == 0 for n in ppt.sizes.values()):
        raise ValueError("El dataset tiene una dimensión vacía.")
    if ppt.dtype.kind not in "fiu":
        raise ValueError("ppt debe contener valores numéricos reales.")
    if "time" not in ppt.coords or ppt.time.dims != ("time",):
        raise ValueError("Falta la coordenada temporal unidimensional time.")
    if bool(ppt.time.isnull().any().item()):
        raise ValueError("La coordenada time contiene fechas faltantes.")
    try:
        meses = np.asarray(ppt.time.dt.month.values, dtype=np.int64)
        anios = np.asarray(ppt.time.dt.year.values, dtype=np.int64)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("No se pudieron decodificar las fechas de time.") from exc
    consecutivos = anios * 12 + meses
    if np.any(np.diff(consecutivos) <= 0):
        raise ValueError(
            "time debe estar en orden cronológico, sin meses duplicados."
        )
    ausentes = int(np.maximum(np.diff(consecutivos) - 1, 0).sum())
    if ausentes:
        print(
            f"Aviso: faltan {ausentes} meses completos en la coordenada time. "
            "Este script solo limpia los pasos temporales existentes.",
            flush=True,
        )


def limpiar_ppt(ppt):
    """Devuelve precipitación limpia, máscara, bandera de imputación y reporte."""
    validar_ppt(ppt)

    # 1. Máscara fija del ORIGINAL. No usar los NaN tras retirar negativos.
    tierra = ppt.notnull().any(dim="time")
    oceano = ~tierra

    # 2. Problemas solo en tierra. Los tres conteos son excluyentes.
    finitos = np.isfinite(ppt)
    nan_tierra = tierra & ppt.isnull()
    negativos = tierra & finitos & (ppt < 0)
    infinitos = tierra & np.isinf(ppt)
    invalidos = nan_tierra | negativos | infinitos
    validos = finitos & (ppt >= 0)

    # 3. Media histórica del MISMO mes y celda, sin valores inválidos.
    # Cada observación original válida participa una vez; no se reutilizan
    # valores imputados para recalcular la climatología.
    base = ppt.astype("float64").where(validos)
    climatologia = base.groupby("time.month").mean(dim="time", skipna=True)
    por_fecha = climatologia.sel(month=ppt.time.dt.month).drop_vars("month")

    # 4. Cambiar exclusivamente los valores inválidos en tierra.
    limpio = xr.where(invalidos, por_fecha, ppt).transpose(*ppt.dims)
    limpio.name = ppt.name
    limpio.attrs = dict(ppt.attrs)
    limpio.attrs.pop("actual_range", None)  # Podría incluir negativos antiguos.
    limpio.encoding = {}  # Evita reutilizar scale_factor/add_offset enteros.

    # 5. Conteos de celdas y observaciones (celda/fecha), sin confundirlos.
    nan_despues = tierra & limpio.isnull()
    resumen = xr.Dataset(
        {
            "celdas_tierra": tierra.sum(),
            "celdas_oceano": oceano.sum(),
            "celdas_tierra_con_problemas": invalidos.any("time").sum(),
            "nan_inesperados_antes": nan_tierra.sum(),
            "negativos_antes": negativos.sum(),
            "infinitos_antes": infinitos.sum(),
            "valores_imputados": (invalidos & np.isfinite(limpio)).sum(),
            "nan_inesperados_despues": nan_despues.sum(),
            "celdas_sin_resolver": nan_despues.any("time").sum(),
            "negativos_despues": (tierra & (limpio < 0)).sum(),
            "infinitos_despues": (tierra & np.isinf(limpio)).sum(),
            "valores_no_nan_en_oceano": (oceano & limpio.notnull()).sum(),
            "valores_validos_alterados": (validos & (limpio != ppt)).sum(),
        }
    ).compute()
    reporte = {nombre: int(valor.item()) for nombre, valor in resumen.data_vars.items()}

    print("\nREPORTE DE CONTROL DE CALIDAD", flush=True)
    for nombre, valor in reporte.items():
        print(f"  {nombre}: {valor:,}", flush=True)

    if reporte["valores_no_nan_en_oceano"]:
        raise RuntimeError("La verificación detectó cambios en el océano.")
    print("Océano verificado: todas sus observaciones siguen siendo NaN.", flush=True)
    if reporte["valores_validos_alterados"]:
        raise RuntimeError("La verificación detectó cambios en valores válidos.")
    if reporte["nan_inesperados_despues"]:
        raise ValueError(
            f"Quedan {reporte['nan_inesperados_despues']:,} valores sin resolver "
            f"en {reporte['celdas_sin_resolver']:,} celdas de tierra: "
            "no hay observaciones válidas del mismo mes para calcular "
            "su climatología. No se guardó un archivo de salida."
        )
    if reporte["negativos_despues"] or reporte["infinitos_despues"]:
        raise RuntimeError("La verificación detectó valores inválidos residuales.")
    print("Tierra verificada: 0 NaN inesperados y 0 valores inválidos.", flush=True)

    tierra = tierra.astype("int8").rename("mascara_tierra")
    tierra.attrs = {
        "long_name": "Mascara inferida de la serie original: 1 tierra, 0 oceano",
        "flag_values": np.array([0, 1], dtype="int8"),
        "flag_meanings": "ocean land",
        "comment": "Ocean means all observations were NaN in the input series.",
    }
    imputado = invalidos.astype("int8").transpose(*ppt.dims).rename("ppt_imputado")
    imputado.attrs = {
        "long_name": "Valores corregidos mediante climatologia mensual local",
        "flag_values": np.array([0, 1], dtype="int8"),
        "flag_meanings": "unchanged imputed",
        "comment": "Ocean cells always have flag 0; consult mascara_tierra.",
    }
    return limpio, tierra, imputado, reporte


def procesar_archivo(entrada, salida, variable="ppt"):
    """Procesa por bloques, valida y guarda sin modificar el archivo original."""
    entrada, salida = Path(entrada).expanduser(), Path(salida).expanduser()
    if not entrada.is_file():
        raise FileNotFoundError(f"No existe el archivo de entrada: {entrada}")
    if entrada.resolve() == salida.resolve():
        raise ValueError("La salida debe ser un archivo distinto de la entrada.")
    if salida.exists():
        raise FileExistsError(f"La salida ya existe; elige otro nombre: {salida}")

    # Decodifica _FillValue, missing_value, scale_factor y add_offset antes
    # de buscar NaN o negativos. Un solo trabajador limita la memoria usada.
    with dask.config.set(scheduler="single-threaded"):
        with xr.open_dataset(entrada, decode_times=True, mask_and_scale=True) as ds:
            if variable not in ds.data_vars:
                raise ValueError(f"No existe la variable {variable!r} en la entrada.")
            for nombre in ("mascara_tierra", "ppt_imputado"):
                if nombre in ds.variables:
                    raise ValueError(f"La entrada ya contiene la variable {nombre!r}.")
            bloques = {d: (-1 if d == "time" else 64) for d in ds[variable].dims}
            ppt = ds[variable].chunk(bloques)
            print(f"Dimensiones de {variable}: {dict(ppt.sizes)}", flush=True)
            print("Calculando máscaras, climatología y verificaciones...", flush=True)
            limpio, tierra, imputado, reporte = limpiar_ppt(ppt)

            resultado = ds.copy(deep=False)
            resultado[variable] = limpio
            resultado["mascara_tierra"] = tierra
            resultado["ppt_imputado"] = imputado
            momento = datetime.now(timezone.utc).isoformat(timespec="seconds")
            nota = (
                f"{momento}: cleaned {variable} using per-cell calendar-month "
                "means of original finite nonnegative data; ocean NaNs preserved."
            )
            resultado.attrs = dict(ds.attrs)
            resultado.attrs["history"] = (str(ds.attrs.get("history", "")) + "\n" + nota).strip()
            for nombre, valor in reporte.items():
                resultado.attrs[f"qc_{nombre}"] = str(valor)

            codificacion = {
                variable: {
                    "dtype": "float64", "_FillValue": np.nan,
                    "zlib": True, "complevel": 4,
                },
                "mascara_tierra": {"dtype": "int8", "_FillValue": None, "zlib": True},
                "ppt_imputado": {"dtype": "int8", "_FillValue": None, "zlib": True},
            }
            salida.parent.mkdir(parents=True, exist_ok=True)
            # El nombre definitivo solo aparece cuando termina la escritura.
            descriptor, temporal = tempfile.mkstemp(suffix=".nc", dir=salida.parent)
            os.close(descriptor)
            try:
                print("Guardando el NetCDF verificado...", flush=True)
                resultado.to_netcdf(
                    temporal, engine="netcdf4", format="NETCDF4", encoding=codificacion
                )
                os.replace(temporal, salida)
            finally:
                if os.path.exists(temporal):
                    os.remove(temporal)
    print(f"Archivo guardado: {salida.resolve()}", flush=True)
    return reporte


def main():
    parser = argparse.ArgumentParser(
        description="Imputa NaN y negativos en tierra con climatología mensual; conserva el océano."
    )
    parser.add_argument("entrada", type=Path, help="Archivo NetCDF original.")
    parser.add_argument("salida", type=Path, help="Archivo NetCDF nuevo con precipitación limpia.")
    parser.add_argument("--variable", default="ppt", help="Nombre de la variable (por defecto: ppt).")
    args = parser.parse_args()
    try:
        procesar_archivo(args.entrada, args.salida, args.variable)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"\nERROR: {exc}\n")


if __name__ == "__main__":
    main()
