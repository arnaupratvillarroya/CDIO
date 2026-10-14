import argparse
import sys
import re
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import splev, splprep
from shapely.geometry import LineString, MultiPoint, Point

import shoreline_utils as utils

SPATIAL_SPACING = 10
TRANSECT_LENGTH = 300


def extract_date_from_file(csv_file):
    if hasattr(utils, "extract_date"):
        try:
            date_val = utils.extract_date(csv_file.name)
            if date_val is not None:
                if isinstance(date_val, str):
                    return datetime.strptime(date_val.replace("-", ""), "%Y%m%d").date()
                elif isinstance(date_val, datetime):
                    return date_val.date()
                return date_val
        except Exception:
            pass

    match = re.search(r'(\d{4}-?\d{2}-?\d{2})', csv_file.name)
    if match:
        raw = match.group(1).replace('-', '')
        if len(raw) == 8:
            return datetime.strptime(raw, "%Y%m%d").date()
    return None


def closest_ground_truth_file(ground_truth_dir, target_date):
    base = Path(ground_truth_dir).expanduser().resolve()
    if not base.is_dir():
        print(f"Error: {base} no és un directori", file=sys.stderr)
        sys.exit(1)

    dated_files = []
    for csv_file in base.glob("*.csv"):
        date = extract_date_from_file(csv_file)
        if date is not None:
            dated_files.append((date, csv_file))

    if not dated_files:
        raise FileNotFoundError(f"No s'han trobat fitxers CSV amb data vàlida a {base}")

    return min(dated_files, key=lambda row: abs(row[0] - target_date))


def load_reference_coords(gt_csv_path):
    try:
        coords = np.loadtxt(gt_csv_path, delimiter="\t", skiprows=1)
        if coords.ndim == 2 and coords.shape[1] >= 2 and not np.isnan(coords).all():
            return coords[:, :2]
    except Exception:
        pass

    for sep in [',', ';', r'\s+', '\t']:
        try:
            df = pd.read_csv(gt_csv_path, sep=sep, engine='python')
            num_df = df.select_dtypes(include=[np.number])
            if len(num_df.columns) >= 2:
                coords = num_df.iloc[:, :2].dropna().values
                if len(coords) > 0:
                    return coords
        except Exception:
            continue

    raise ValueError(f"No s'han pogut carregar les coordenades X,Y de: {gt_csv_path}")


def signed_intersection_distance(intersection, px, py, normal):
    if isinstance(intersection, MultiPoint):
        distances = [Point(px, py).distance(point) for point in intersection.geoms]
        return distances[int(np.argmin(distances))]

    if isinstance(intersection, Point):
        signed_vector = np.array([intersection.x - px, intersection.y - py])
        distance = np.linalg.norm(signed_vector)
        if np.dot(signed_vector, normal) < 0:
            distance = -distance
        return distance

    return np.nan


def safe_build_ordered_linestring(coords, is_ordered=False):
    """Executa build_ordered_linestring evitant incompatibilitats de paràmetres només per paraula clau."""
    try:
        return utils.build_ordered_linestring(coords, is_ordered)
    except TypeError:
        return utils.build_ordered_linestring(coords)


def main(project, ground_truth, date, show_plot=True):
    target_date = datetime.strptime(date, "%Y-%m-%d").date()
    project_dir = utils.project_path(project)
    output_dir = project_dir / "output" / "estimated_error"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = project_dir / "output" / "estimated_waterbodies_edges" / "all_shorelines.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No s'ha trobat el fitxer {csv_path}")

    df = pd.read_csv(csv_path)
    df['date_str'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    shorelines = {d: group[["x", "y"]].values for d, group in df.groupby("date_str")}

    if date not in shorelines:
        available = list(shorelines.keys())
        raise ValueError(f"La data {date} no està present a {csv_path}.\nDates disponibles: {available}")

    reference_date, gt_csv_path = closest_ground_truth_file(ground_truth, target_date)
    print(f"Fitxer Ground Truth seleccionat: {gt_csv_path.name} (Data GT: {reference_date})")

    reference_coords = load_reference_coords(gt_csv_path)

    try:
        if hasattr(utils, "load_coords_csv") and hasattr(utils, "export_coords_to_kmz"):
            gt_coords = utils.load_coords_csv(gt_csv_path)
            utils.export_coords_to_kmz(gt_coords, gt_csv_path.with_suffix(".kmz"))
    except Exception as e:
        print(f"Avís (KMZ): No s'ha pogut generar el fitxer KMZ ({e})")

    x = reference_coords[:, 0]
    y = reference_coords[:, 1]
    xy_unique = np.unique(np.column_stack((x, y)), axis=0)
    x_unique = xy_unique[:, 0]
    y_unique = xy_unique[:, 1]

    reference_line = safe_build_ordered_linestring(np.column_stack((x, y)), is_ordered=True)
    shoreline_length = reference_line.length
    n_points = int(shoreline_length / SPATIAL_SPACING)

    tck, _ = splprep([x_unique, y_unique], s=0)
    unew = np.linspace(0, 1, num=n_points)
    x_smooth, y_smooth = splev(unew, tck)
    dx, dy = splev(unew, tck, der=1)

    fig, ax = plt.subplots(figsize=(12, 8))
    x_ref, y_ref = reference_line.xy
    ax.plot(x_ref, y_ref, "k--", label=f"Reference shoreline ({reference_date})", alpha=0.5)
    ax.plot(x_smooth, y_smooth, "b", label=f"Smoothed shoreline ({reference_date})", alpha=0.7)

    for i in range(len(unew)):
        px, py = x_smooth[i], y_smooth[i]
        tx, ty = dx[i], dy[i]
        normal = np.array([-ty, tx])
        normal /= np.linalg.norm(normal)
        start = (px - normal[0] * TRANSECT_LENGTH / 2, py - normal[1] * TRANSECT_LENGTH / 2)
        end = (px + normal[0] * TRANSECT_LENGTH / 2, py + normal[1] * TRANSECT_LENGTH / 2)
        ax.plot([start[0], end[0]], [start[1], end[1]], "g-", alpha=0.3)

    print(f"Ordenant la línia de costa per a {date}...")
    line = safe_build_ordered_linestring(shorelines[date], is_ordered=False)
    print("Fet")

    distance_results = []
    x_line, y_line = line.xy
    ax.plot(x_line, y_line, "k--", label=date)

    for i in range(len(unew)):
        px, py = x_smooth[i], y_smooth[i]
        tx, ty = dx[i], dy[i]
        normal = np.array([-ty, tx])
        normal /= np.linalg.norm(normal)
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

    distance_df = pd.DataFrame(distance_results)
    valid_distances = distance_df["distance_m"].dropna()

    if len(valid_distances) > 0:
        rmse = np.sqrt((valid_distances ** 2).mean())
        print(f"\n" + "=" * 40)
        print(f"RMSE: {rmse:.2f} metres")
        print(f"Mostres vàlides: {len(valid_distances)}")
        print(f"Data de referència GT: {reference_date}")
        print("=" * 40 + "\n")
    else:
        print("\n[AVÍS] No s'han trobat interseccions vàlides entre els transectes i la línia estimada.\n")

    distance_df.to_csv(output_dir / "shoreline_distances.csv", index=False)

    ax.set_title("Geometria de Línia de Costa i Interseccions de Transectes")
    ax.set_xlabel("X (metres, projectat)")
    ax.set_ylabel("Y (metres, projectat)")
    ax.legend()
    ax.axis("equal")
    plt.grid(True)

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Estimació de l'error de la línia de costa contra dades de Ground Truth.")
    parser.add_argument("-p", "--project", required=True, help="Ruta al projecte o nom del directori de dades.")
    parser.add_argument("-g", "--ground-truth", required=True, help="Ruta a la carpeta de Ground Truth.")
    parser.add_argument("-d", "--date", required=True, help="Data de Sentinel-2 a comparar (YYYY-MM-DD).")
    parser.add_argument("--no-plot", action="store_true", help="Generar CSV sense mostrar gràfics.")
    args = parser.parse_args()

    main(args.project, args.ground_truth, args.date, show_plot=not args.no_plot)