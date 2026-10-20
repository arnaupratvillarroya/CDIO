import argparse
import pickle
import sys
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from scipy.interpolate import splev, splprep
from shapely.geometry import LineString, MultiPoint, Point

# Ajustar sys.path para poder importar shoreline_utils desde la carpeta utils/
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR / "utils"))

import shoreline_utils as utils

SPATIAL_SPACING = 10   # Distància en metres entre transectes al llarg de la costa
TRANSECT_LENGTH = 100  # Longitud total del transecte en metres


def closest_ground_truth_file(ground_truth_dir, target_date):
    """Troba el fitxer CSV de ground truth més proper temporalment a target_date."""
    base = Path(ground_truth_dir).expanduser().resolve()
    if not base.is_dir():
        print(f"Error: {base} no és un directori vàlid.", file=sys.stderr)
        sys.exit(1)

    dated_files = []
    for csv_file in base.glob("*.csv"):
        date = utils.extract_date(csv_file.name)
        if date is not None:
            dated_files.append((date, csv_file))

    if not dated_files:
        raise FileNotFoundError(f"No s'han trobat fitxers CSV amb data a {base}")

    return min(dated_files, key=lambda row: abs(row[0] - target_date))


def load_reference(shorelines, sorted_dates_str, ground_truth):
    """Carrega o determina la línia de costa de referència (Ground Truth o la primera data)."""
    if ground_truth is None:
        reference_date = sorted_dates_str[0]
        reference_coords = shorelines[reference_date]
        return reference_date, reference_coords, 1, False

    first_date = datetime.strptime(sorted_dates_str[0], "%Y-%m-%d").date()
    reference_date, gt_csv_path = closest_ground_truth_file(ground_truth, first_date)
    reference_coords = np.loadtxt(gt_csv_path, delimiter="\t", skiprows=1)

    gt_coords = utils.load_coords_csv(gt_csv_path)
    utils.export_coords_to_kmz(gt_coords, gt_csv_path.with_suffix(".kmz"))

    return reference_date.isoformat(), reference_coords, 0, True


def signed_intersection_distance(intersection, px, py, normal):
    """Calcula la distància amb signe entre el punt del transecte i la intersecció amb la costa."""
    if intersection.is_empty:
        return np.nan

    if isinstance(intersection, MultiPoint):
        distances = [Point(px, py).distance(point) for point in intersection.geoms]
        closest_pt = intersection.geoms[int(np.argmin(distances))]
        signed_vector = np.array([closest_pt.x - px, closest_pt.y - py])
        distance = np.linalg.norm(signed_vector)
        if np.dot(signed_vector, normal) < 0:
            distance = -distance
        return distance

    if isinstance(intersection, Point):
        signed_vector = np.array([intersection.x - px, intersection.y - py])
        distance = np.linalg.norm(signed_vector)
        if np.dot(signed_vector, normal) < 0:
            distance = -distance
        return distance

    return np.nan


def load_project_aoi(project_dir):
    """Carrega el polígon AOI (polygon.geojson) i el reprojecta al CRS del projecte."""
    geojson_path = project_dir / "data" / "Maps_geojson" / "polygon.geojson"
    mask_dir = project_dir / "data" / "output" / "estimated_waterbodies_images"
    first_mask = next(mask_dir.glob("*.tif"), None) if mask_dir.exists() else None

    if not geojson_path.exists():
        print(f"Avís: No s'ha trobat {geojson_path}. S'analitzarà tota la línia sense mascareta AOI.")
        return None

    aoi = gpd.read_file(geojson_path).set_crs(4326, allow_override=True)

    if first_mask is not None:
        with rasterio.open(first_mask) as src:
            return aoi.to_crs(src.crs)

    return aoi


def main(ground_truth=None, method="none", show_plot=True):
    # 1. Definició de rutes basada en ROOT_DIR i el mètode
    output_dir = ROOT_DIR / "data" / "output" / "estimated_erosion_accretion" / f"estimated_erosion_accretion_{method}"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = ROOT_DIR / "data" / "output" / "estimated_waterbodies_edges" / f"all_shorelines_{method}.csv"
    if not csv_path.exists():
        # Fallback al CSV general
        csv_path = ROOT_DIR / "data" / "output" / "estimated_waterbodies_edges" / "all_shorelines.csv"

    if not csv_path.exists():
        print(f"Error: No s'ha trobat el fitxer CSV a {csv_path}", file=sys.stderr)
        sys.exit(1)

    # 2. Carregar dades
    df = pd.read_csv(csv_path)
    shorelines = {date: group[["x", "y"]].values for date, group in df.groupby("date")}
    sorted_dates_str = sorted(shorelines.keys())

    if not sorted_dates_str:
        print("Error: El fitxer CSV no conté dades.", file=sys.stderr)
        sys.exit(1)

    # 3. Determinar línia de referència
    reference_date, reference_coords, start_index, ordered = load_reference(
        shorelines, sorted_dates_str, ground_truth
    )

    x = reference_coords[:, 0]
    y = reference_coords[:, 1]

    # Mantenir l'ordre espacial eliminant només punts exactament duplicats
    _, unique_indices = np.unique(np.column_stack((x, y)), axis=0, return_index=True)
    sorted_indices = np.sort(unique_indices)
    x_unique = x[sorted_indices]
    y_unique = y[sorted_indices]

    # 4. Construir o carregar de cache la LineString de referència
    reference_date_underscores = str(reference_date).replace("-", "_")
    line_path = output_dir / f"ref_{reference_date_underscores}.pkl"

    if line_path.exists() and line_path.stat().st_size > 0:
        with open(line_path, "rb") as f:
            reference_line = pickle.load(f)
    else:
        reference_line = utils.build_ordered_linestring(np.column_stack((x, y)), ordered)
        aoi = load_project_aoi(ROOT_DIR / "data")
        if aoi is not None and not reference_line.is_empty:
            intersection_res = reference_line.intersection(aoi.unary_union)
            if not intersection_res.is_empty:
                reference_line = intersection_res

        with open(line_path, "wb") as f:
            pickle.dump(reference_line, f)

    if reference_line.is_empty:
        print("Error: La línia de referència està buida.", file=sys.stderr)
        sys.exit(1)

    # 5. Generar transectes al llarg de la línia suavitzada (B-Spline)
    shoreline_length = reference_line.length
    n_points = max(2, int(shoreline_length / SPATIAL_SPACING))

    tck, _ = splprep([x_unique, y_unique], s=0)
    unew = np.linspace(0, 1, num=n_points)
    x_smooth, y_smooth = splev(unew, tck)
    dx, dy = splev(unew, tck, der=1)

    # Preparar gràfica
    fig, ax = plt.subplots(figsize=(12, 8))
    if hasattr(reference_line, 'xy'):
        x_ref, y_ref = reference_line.xy
        ax.plot(x_ref, y_ref, "k--", label=f"Línia referència ({reference_date})", alpha=0.5)
    ax.plot(x_smooth, y_smooth, "b", label=f"Línia suavitzada ({reference_date})", alpha=0.7)

    # Dibuixar transectes
    for i in range(len(unew)):
        px, py = x_smooth[i], y_smooth[i]
        tx, ty = dx[i], dy[i]
        normal = np.array([-ty, tx])
        norm_val = np.linalg.norm(normal)
        if norm_val > 0:
            normal /= norm_val

        start = (px - normal[0] * TRANSECT_LENGTH / 2, py - normal[1] * TRANSECT_LENGTH / 2)
        end = (px + normal[0] * TRANSECT_LENGTH / 2, py + normal[1] * TRANSECT_LENGTH / 2)
        ax.plot([start[0], end[0]], [start[1], end[1]], "g-", alpha=0.2)

    # 6. Processar la resta de dates
    distance_results = []
    colors = ["r", "m", "c", "y"]
    dist_reference = 0

    for idx, date in enumerate(sorted_dates_str[start_index:]):
        coords = shorelines[date]
        if len(coords) < 2:
            print(f"Ometent la data {date} per falta de punts suficients.")
            continue

        print(f"Processant línia de costa per a la data: {date}...")
        date_underscores = date.replace("-", "_")
        line_cache_path = output_dir / f"line_{date_underscores}.pkl"

        if line_cache_path.exists() and line_cache_path.stat().st_size > 0:
            with open(line_cache_path, "rb") as f:
                line = pickle.load(f)
        else:
            line = utils.build_ordered_linestring(coords, False)
            with open(line_cache_path, "wb") as f:
                pickle.dump(line, f)

        if hasattr(line, 'xy'):
            x_line, y_line = line.xy
            ax.plot(
                x_line,
                y_line,
                "--",
                label=date,
                color=colors[idx % len(colors)],
                alpha=0.8
            )

        for i in range(len(unew)):
            px, py = x_smooth[i], y_smooth[i]
            if i > 0:
                dist_reference += np.sqrt((px - x_smooth[i - 1]) ** 2 + (py - y_smooth[i - 1]) ** 2)

            tx, ty = dx[i], dy[i]
            normal = np.array([-ty, tx])
            norm_val = np.linalg.norm(normal)
            if norm_val > 0:
                normal /= norm_val

            start = (px - normal[0] * TRANSECT_LENGTH / 2, py - normal[1] * TRANSECT_LENGTH / 2)
            end = (px + normal[0] * TRANSECT_LENGTH / 2, py + normal[1] * TRANSECT_LENGTH / 2)
            transect = LineString([start, end])

            distance = signed_intersection_distance(transect.intersection(line), px, py, normal)
            distance_results.append(
                {
                    "date": date,
                    "transect_id": i,
                    "reference_x": px,
                    "reference_y": py,
                    "distance_m": distance,
                }
            )

    # 7. Resum de mètriques i exportació
    if distance_results:
        avg_spacing = dist_reference / max(1, (len(unew) - 1) * max(1, len(sorted_dates_str[start_index:])))
        print(f"Espaiat mitjà entre transectes: {avg_spacing:.2f} metres")

    distance_df = pd.DataFrame(distance_results)
    total_transects = distance_df["transect_id"].nunique() if not distance_df.empty else 0

    print(f"Número de transectes calculats: {len(unew)}")
    if not distance_df.empty:
        valid_count = distance_df.loc[distance_df['distance_m'].notna(), 'transect_id'].nunique()
        print(f"Transectes vàlids amb almenys una distància mesurada: {valid_count}")
        print(f"Primera data analitzada: {distance_df['date'].min()}")
        print(f"Última data analitzada: {distance_df['date'].max()}")
        print(f"Total de dates diferents: {distance_df['date'].nunique()}")

        summary_df = (
            distance_df.groupby("date")
            .agg(
                avg_distance_m=("distance_m", "mean"),
                median_distance_m=("distance_m", "median"),
                std_distance_m=("distance_m", "std"),
                max_distance_m=("distance_m", "max"),
                min_distance_m=("distance_m", "min"),
                valid_points=("distance_m", lambda values: values.notna().sum()),
            )
            .reset_index()
        )

        summary_df["coverage_ratio"] = summary_df["valid_points"] / total_transects
        summary_df["range_distance_m"] = summary_df["max_distance_m"] - summary_df["min_distance_m"]

        summary_df.to_csv(output_dir / "distances_summary.csv", index=False)
        distance_df.to_csv(output_dir / "shoreline_distances.csv", index=False)
        print(f"\n✔ Resultats d'erosió exportats a {output_dir}")

    # 8. Renderitzar la figura
    ax.set_title(f"Geometria de la Costa i Transectes ({method})")
    ax.set_xlabel("X (metres, projectat)")
    ax.set_ylabel("Y (metres, projectat)")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    ax.axis("equal")
    plt.grid(True)
    plt.tight_layout()

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estima l'erosió i acumulació de sorra en la línia de costa.")
    parser.add_argument(
        "-g",
        "--ground-truth",
        help="Ruta a la carpeta amb els fitxers CSV de ground truth.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="none",
        help="Mètode de denoising utilitzat."
    )
    parser.add_argument("--no-plot", action="store_true", help="Executa sense obrir la finestra de la gràfica.")
    args = parser.parse_args()
    main(ground_truth=args.ground_truth, method=args.method, show_plot=not args.no_plot)