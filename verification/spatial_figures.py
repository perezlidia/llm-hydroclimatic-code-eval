"""
Spatial comparison figures for Phase 2 (chapter Figures 3, 4 and 6).

    T1 -> Fig. 3: polygon clip of the models vs. the ArcMap reference
    T2 -> Fig. 4: detection and imputation of invalid values
    T4 -> Fig. 6: water balance, difference of each model vs. the ArcMap reference

Run in the working directory:
    python spatial_figures.py            # all figures
    python spatial_figures.py T4         # only one (T1, T2 or T4)

Outputs PNG/PDF figures in ./figures/ and GeoTIFF difference rasters (for QGIS/ArcMap).
If an input is missing, that figure is skipped with a message.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                      # non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, TwoSlopeNorm
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FuncFormatter
import numpy as np
import xarray as xr
import rioxarray  # noqa: F401  (registers the .rio accessor)

from config import BASE, ARCMAP, FIG, SHP, PPT, T2_INPUT, MODELS, DPI, T1_ARCMAP, T4_ARCMAP

ORIGINAL_T2 = PPT            # clean clip (T1 verified identical for the three models)


# ------------------------------------------------------------------ utilities
def export_result(da, task, model):
    """Stores a DataArray produced by a model script as <task>_<model>.nc"""
    out = BASE / f"{task}_{model}.nc"
    da.to_dataset(name="value").to_netcdf(out)
    print(f"Saved {out}  dims={dict(da.sizes)}")


def load(path, var=None):
    """Loads .nc or .tif and normalizes to dims (time?, lat, lon) with ascending lat/lon."""
    path = Path(path)
    if path.suffix.lower() in (".tif", ".tiff"):
        da = rioxarray.open_rasterio(path, masked=True)
        if "band" in da.dims:
            da = da.squeeze("band", drop=True) if da.sizes["band"] == 1 else da.rename(band="time")
        da = da.rename(x="lon", y="lat")
    else:
        ds = xr.open_dataset(path)
        ren = {k: v for k, v in {"latitude": "lat", "longitude": "lon", "y": "lat", "x": "lon"}.items()
               if k in ds.dims}
        ds = ds.rename(ren)
        if var is None:
            preferred = [v for v in ("value", "ppt", "pet") if v in ds.data_vars]
            var = preferred[0] if preferred else next(
                v for v in ds.data_vars if {"lat", "lon"} <= set(ds[v].dims))
        da = ds[var]
    da = da.drop_vars([c for c in ("spatial_ref", "crs") if c in da.coords], errors="ignore")
    return da.sortby("lat").sortby("lon")


def find(task, model):
    """Locates the output of a model (or of the ArcMap reference) for a task."""
    candidates = [BASE / f"{task}_{model}{ext}" for ext in (".tif", ".nc")]
    if model == "arcmap":
        # T1: one year of the ArcMap clip is enough for the spatial footprint
        candidates = [{"T1": T1_ARCMAP, "T4": T4_ARCMAP}.get(task, ARCMAP / f"{task}_arcmap.tif")]
    elif task == "T1":
        candidates.insert(0, BASE / f"T1_{model}_ppt.nc")
    for p in candidates:
        if p.exists():
            return p
    return None


def align(da, ref, tol=1e-3):
    """Places da on the grid of ref by matching cell centres (no interpolation)."""
    return da.reindex(lat=ref.lat, lon=ref.lon, method="nearest", tolerance=tol)


def outline(ax):
    if SHP.exists():
        import geopandas as gpd
        gpd.read_file(SHP).to_crs(4326).boundary.plot(ax=ax, color="black", linewidth=0.6)


def to_geotiff(da, path):
    # descending lat (north up) so that QGIS/ArcMap do not display the raster flipped
    (da.sortby("lat", ascending=False)
       .rio.set_spatial_dims(x_dim="lon", y_dim="lat")
       .rio.write_crs("EPSG:4326")
       .rio.to_raster(path))
    print(f"GeoTIFF: {path}")


def save(fig, name):
    """Saves PNG and PDF; if a file is open in another program, warns and continues."""
    for ext in ("png", "pdf"):
        path = FIG / f"{name}.{ext}"
        try:
            fig.savefig(path, dpi=DPI, bbox_inches="tight")
            print(f"  Saved: {path.name}")
        except OSError as e:
            print(f"  Could NOT save {path.name}: it is probably open in another program. [{e}]")
    plt.close(fig)


def presence(da):
    return da.notnull().any("time") if "time" in da.dims else da.notnull()


def background(ax, land):
    """Study area in light grey plus state outlines."""
    land.where(land).plot(ax=ax, cmap=ListedColormap(["#e8e8e8"]), add_colorbar=False)
    outline(ax)
    ax.set_aspect("equal")
    ax.set_xlabel("Longitude (°)"); ax.set_ylabel("Latitude (°)")


def points(ax, field, s=7, **kw):
    """Draws each cell as a point: ~4 km cells would not be visible as a raster at page size."""
    st = field.stack(p=("lat", "lon")).dropna("p")
    if "color" in kw:
        return ax.scatter(st["lon"].values, st["lat"].values, s=s, linewidths=0, **kw)
    return ax.scatter(st["lon"].values, st["lat"].values, c=st.values, s=s, linewidths=0, **kw)


# ------------------------------------------------------------------ Fig. 3: T1 clip
def figure_t1(window_deg=1.5):
    """Overview map plus zoom on the area with most 'ArcMap only' cells."""
    ref_p = find("T1", "arcmap")
    if ref_p is None:
        print(f"[T1] ArcMap clip not found in {ARCMAP} -> skipped"); return
    ref = presence(load(ref_p))

    cats = {}
    print("[T1] Border cells, ArcMap vs. models")
    for m, label in MODELS.items():
        p = find("T1", m)
        if p is None:
            print(f"  {label}: no file"); continue
        mdl = align(presence(load(p)).astype(float), ref).fillna(0).astype(bool)
        cat = xr.where(ref & mdl, 1, 0) + xr.where(ref & ~mdl, 2, 0) + xr.where(~ref & mdl, 3, 0)
        cats[m] = cat
        print(f"  {label}: both={int((cat == 1).sum())}  ArcMap only={int((cat == 2).sum())}  "
              f"model only={int((cat == 3).sum())}")
    if not cats:
        return
    first = next(iter(cats.values()))
    same = all(np.array_equal(c.values, first.values) for c in cats.values())
    print(f"  The {len(cats)} models give the same map: {'YES' if same else 'NO'}")
    cat = first
    n_both, n_arc, n_mod = (int((cat == k).sum()) for k in (1, 2, 3))

    res = float(abs(cat.lat.diff("lat")).mean())
    n = max(3, int(round(window_deg / res)))
    density = (cat == 2).astype(int).rolling(lat=n, lon=n, center=True, min_periods=1).sum()
    i, j = np.unravel_index(int(np.nanargmax(density.values)), density.shape)
    lat0, lon0, half = float(cat.lat[i]), float(cat.lon[j]), window_deg / 2
    print(f"  Zoom centred at lat {lat0:.2f}, lon {lon0:.2f}")

    cmap = ListedColormap(["white", "#9ecae1", "#e6550d", "#31a354"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True,
                                   gridspec_kw={"width_ratios": [1.35, 1]})
    cat.plot(ax=ax1, cmap=cmap, vmin=-0.5, vmax=3.5, add_colorbar=False)
    cat.plot(ax=ax2, cmap=cmap, vmin=-0.5, vmax=3.5, add_colorbar=False,
             edgecolors="#bbbbbb", linewidth=0.15)
    for ax in (ax1, ax2):
        outline(ax)
        ax.set_aspect("equal")
        ax.set_xlabel("Longitude (°)"); ax.set_ylabel("Latitude (°)")
    ax1.add_patch(Rectangle((lon0 - half, lat0 - half), window_deg, window_deg,
                            fill=False, edgecolor="black", linewidth=1.3))
    ax2.set_xlim(lon0 - half, lon0 + half); ax2.set_ylim(lat0 - half, lat0 + half)
    ax1.set_title("(a) Study area", loc="left", fontsize=12)
    ax2.set_title("(b) Detail of the border strip", loc="left", fontsize=12)
    handles = [Patch(color="#9ecae1", label=f"Both procedures (n = {n_both:,})"),
               Patch(color="#e6550d", label=f"ArcMap only (n = {n_arc:,})")]
    if n_mod:
        handles.append(Patch(color="#31a354", label=f"Models only (n = {n_mod:,})"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=11,
               frameon=False, bbox_to_anchor=(0.5, -0.06))
    save(fig, "fig3_t1_polygon_clip")


# ------------------------------------------------------------------ Fig. 4: T2 cleaning
def figure_t2():
    if not (ORIGINAL_T2.exists() and T2_INPUT.exists()):
        print(f"[T2] Missing {ORIGINAL_T2} or {T2_INPUT} -> skipped"); return
    orig = load(ORIGINAL_T2)
    corr = align(load(T2_INPUT), orig)
    land = presence(orig)
    all_nan = land & corr.isnull().all("time")        # NaN in all 540 months
    neg = land & (corr < 0).any("time")                # cells with negative values
    damaged = (corr.isnull() | (corr < 0)) & land

    print("[T2] Corrupted input")
    print(f"  study-area cells (clean T1 clip): {int(land.sum())}")
    print(f"  cells NaN in the whole series: {int(all_nan.sum())}")
    print(f"  cells with negative values: {int(neg.sum())} "
          f"(negative records: {int(((corr < 0) & land).sum())})")

    # Reference: error of imputing with the TRUE climatology of each cell (limit of the method)
    clim = orig.groupby("time.month").mean("time")
    ideal = clim.sel(month=orig["time"].dt.month).drop_vars("month")
    err_ideal = abs(ideal - orig).where(damaged).mean("time")
    print(f"  method limit (true climatology) in negative cells: "
          f"{float(err_ideal.where(neg).mean()):.2f} mm/month")

    results = []
    print("[T2] Cleaning result per model")
    for m, label in MODELS.items():
        p, in_memory = find("T2", m), False
        if p is None and (BASE / f"T2_{m}_in_memory.nc").exists():
            p, in_memory = BASE / f"T2_{m}_in_memory.nc", True     # computed but NOT saved by the model
        if p is None:
            print(f"  {label}: no file"); continue
        clean = align(load(p), orig)
        bad = ((clean.isnull() | (clean < 0)) & land).sum("time").where(land)
        err = abs(clean - orig).where(damaged).mean("time")
        results.append(dict(label=label, bad=bad, err=err, clean=clean, in_memory=in_memory))
        e = err.where(neg)
        e_txt = f"{float(e.mean()):.2f} mm/month" if bool(e.notnull().any()) else "no imputation"
        print(f"  {label}{' (in memory, not saved)' if in_memory else ''}: "
              f"still invalid cells={int((bad > 0).sum())} "
              f"[all-NaN={int(((bad > 0) & all_nan).sum())}, negative={int(((bad > 0) & neg).sum())}]  "
              f"remaining negatives={int(((clean < 0) & land).sum())}  error in negative cells={e_txt}")

    ncol = 1 + len(results)
    letters = "abcdefghij"
    fig, axs = plt.subplots(2, ncol, figsize=(5.4 * ncol, 9.2), constrained_layout=True, squeeze=False)

    def layer(ax, mask, color, text, s=7, zorder=2):
        n = int(mask.sum())
        if n:
            points(ax, mask.where(mask).astype(float), color=color, s=s, zorder=zorder,
                   label=f"{text} ({n})")

    C_NAN, C_NEG, C_OK, C_DET = "#b8a9d6", "#d7301f", "#1a9850", "#404040"
    background(axs[0, 0], land)
    layer(axs[0, 0], all_nan, "#5e3c99", "NaN in the whole series", s=6)
    layer(axs[0, 0], neg, "#e66101", "Negative in the whole series", s=14, zorder=3)
    axs[0, 0].legend(loc="lower left", fontsize=9, markerscale=2)
    axs[0, 0].set_title(f"({letters[0]})", loc="left", fontsize=13, fontweight="bold")   # corrupted input
    for k, r in enumerate(results, start=1):
        ax, clean = axs[0, k], r["clean"]
        background(ax, land)
        invalid = r["bad"] > 0
        still_neg = neg & (clean < 0).any("time")
        detected_not_imputed = neg & invalid & ~still_neg
        layer(ax, all_nan & invalid, C_NAN, "NaN not detected", s=5)
        layer(ax, still_neg, C_NEG, "Negative, not corrected", s=14, zorder=3)
        layer(ax, detected_not_imputed, C_DET, "Negative, detected but not imputed", s=14, zorder=3)
        layer(ax, neg & ~invalid, C_OK, "Negative, corrected", s=14, zorder=3)
        layer(ax, invalid & ~all_nan & ~neg, "black", "Other invalid", s=10, zorder=3)
        ax.legend(loc="lower left", fontsize=9, markerscale=2)
        state = "detected, no output file" if r["in_memory"] else "after cleaning"
        print(f"  panel ({letters[k]}): {r['label']}, {state}")
        ax.set_title(f"({letters[k]})", loc="left", fontsize=13, fontweight="bold")

    ref = err_ideal.where(neg)
    errs = [r["err"].where(neg) for r in results]
    vals = np.concatenate([e.values[np.isfinite(e.values)] for e in [ref] + errs])
    vmax = float(np.nanpercentile(vals, 98)) if vals.size else 1.0
    background(axs[1, 0], land)
    im = points(axs[1, 0], ref, s=12, cmap="viridis", vmin=0, vmax=vmax)
    print(f"  panel ({letters[ncol]}): method limit (true climatology), mean = {float(ref.mean()):.2f} mm/month")
    axs[1, 0].set_title(f"({letters[ncol]})", loc="left", fontsize=13, fontweight="bold")
    for k, (r, err) in enumerate(zip(results, errs), start=1):
        ax = axs[1, k]
        background(ax, land)
        if not bool(err.notnull().any()):
            ax.text(0.5, 0.5, "No imputation:\nthe model declined\nto produce a result", ha="center",
                    va="center", transform=ax.transAxes, fontsize=12,
                    bbox=dict(boxstyle="round", facecolor="white", edgecolor="#999999"))
            print(f"  panel ({letters[ncol + k]}): {r['label']}, no imputation")
            ax.set_title(f"({letters[ncol + k]})", loc="left", fontsize=13, fontweight="bold")
            continue
        im = points(ax, err, s=12, cmap="viridis", vmin=0, vmax=vmax)
        print(f"  panel ({letters[ncol + k]}): {r['label']}, imputation error, mean = {float(err.mean()):.2f} mm/month")
        ax.set_title(f"({letters[ncol + k]})", loc="left", fontsize=13, fontweight="bold")
    fig.colorbar(im, ax=list(axs[1, :]), label="Mean |imputed − original| (mm/month)",
                 shrink=0.8, extend="max")

    xmin, xmax = float(land.lon.min()) - 0.3, float(land.lon.max()) + 0.3
    ymin, ymax = float(land.lat.min()) - 0.3, float(land.lat.max()) + 0.3
    for row in range(2):
        for c in range(ncol):
            ax = axs[row, c]
            ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
            ax.set_xticks([]); ax.set_yticks([])          # no coordinates
            ax.set_xlabel(""); ax.set_ylabel("")
    save(fig, "fig4_t2_invalid_values")


# ------------------------------------------------------------------ Fig. 6: T4 water balance
def figure_t4():
    """(a) ArcMap water balance + (b-d) difference of each model vs. ArcMap, 2x2 layout."""
    ref_p = find("T4", "arcmap")
    if ref_p is None:
        print(f"[T4] T4_arcmap.tif not found in {ARCMAP} -> skipped"); return
    ref = load(ref_p)

    diffs, borders, labels = [], [], []
    print("[T4] Model − ArcMap difference (common cells)")
    for m, label in MODELS.items():
        p = find("T4", m)
        if p is None:
            print(f"  {label}: no file"); continue
        da = align(load(p), ref)
        diff = da - ref
        border = ref.notnull() & da.isnull()             # cells included only by ArcMap
        v = diff.values[np.isfinite(diff.values)]
        print(f"  {label}: common cells={v.size}  ArcMap only={int(border.sum())}  "
              f"mean={v.mean():.7f}  max|diff|={np.abs(v).max():.7f}  "
              f"identical (diff=0)={int((v == 0).sum())}  dtype={load(p).dtype}")
        to_geotiff(diff, BASE / f"diff_T4_{m}_vs_arcmap.tif")
        diffs.append(diff); borders.append(border); labels.append(label)
    if not diffs:
        return

    fig, axs = plt.subplots(2, 2, figsize=(10.5, 9.4), constrained_layout=True)
    panels = [axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]]
    xmin, xmax = float(ref.lon.min()) - 0.3, float(ref.lon.max()) + 0.3
    ymin, ymax = float(ref.lat.min()) - 0.3, float(ref.lat.max()) + 0.3

    ax = panels[0]
    im_ref = ref.plot(ax=ax, cmap="BrBG", add_colorbar=False,
                      vmin=float(np.nanpercentile(ref, 2)), vmax=float(np.nanpercentile(ref, 98)))
    fig.colorbar(im_ref, ax=ax, label="Water balance (mm/year)", orientation="horizontal",
                 shrink=0.85, pad=0.02, aspect=30, extend="both")
    ax.set_title("(a) Mean multi-year water balance,\nArcMap reference", loc="left", fontsize=11)

    lim = max(float(abs(d).max()) for d in diffs) or 1e-6
    norm = TwoSlopeNorm(0, -lim, lim)
    im_diff = None
    for k, (diff, border, label) in enumerate(zip(diffs, borders, labels), start=1):
        ax = panels[k]
        im_diff = diff.plot(ax=ax, cmap="RdBu", norm=norm, add_colorbar=False)
        points(ax, border.where(border).astype(float), color="#555555", s=3, zorder=3,
               label="ArcMap only (border)")
        mx = float(abs(diff).max())
        detail = ("identical to ArcMap (difference = 0)" if mx == 0
                  else f"max |diff.| = {mx * 1e4:.1f} × 10⁻⁴ mm/year")
        ax.set_title(f"({'abcd'[k]}) {label} − ArcMap\n{detail}", loc="left", fontsize=11)
        ax.legend(loc="lower left", fontsize=8, markerscale=3)
    fig.colorbar(im_diff, ax=panels[1:], label="Difference (× 10⁻⁴ mm/year), panels (b)–(d)",
                 shrink=0.6, format=FuncFormatter(lambda v, _: f"{v * 1e4:.0f}".replace("-", "−")))

    for i, ax in enumerate(panels):
        outline(ax)
        ax.set_aspect("equal")
        ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
        ax.set_xlabel("Longitude (°)" if i >= 2 else "")
        ax.set_ylabel("Latitude (°)" if i % 2 == 0 else "")
    save(fig, "fig6_t4_water_balance")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    FIG.mkdir(exist_ok=True)
    FIGURES = {"T1": figure_t1, "T2": figure_t2, "T4": figure_t4}
    requested = [a.upper() for a in sys.argv[1:]] or list(FIGURES)
    for key in requested:
        if key in FIGURES:
            FIGURES[key]()
        else:
            print(f"Unknown figure: {key}. Options: {', '.join(FIGURES)}")
    print(f"Done. Figures in {FIG}")
