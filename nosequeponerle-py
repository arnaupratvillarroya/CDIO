import sys
import re
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.interpolate import splprep, splev
from shapely.geometry import LineString, Point, MultiPoint

import shoreline_utils as utils

# Configuración
SPATIAL_SPACING = 10
TRANSECT_LENGTH = 300
GT_TARGET_DATES = ["2017-05-23", "2019-05-23", "2021-06-11"]


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
    raise ValueError(f"No se pudieron cargar coordenadas de: {gt_csv_path}")


def signed_distance(intersection, px, py, normal):
    if intersection.is_empty:
        return np.nan
    if isinstance(intersection, MultiPoint):
        distances = [Point(px, py).distance(pt) for pt in intersection.geoms]
        closest_pt = intersection.geoms[int(np.argmin(distances))]
        signed_vec = np.array([closest_pt.x - px, closest_pt.y - py])
        dist = np.linalg.norm(signed_vec)
        return -dist if np.dot(signed_vec, normal) < 0 else dist
    if isinstance(intersection, Point):
        signed_vec = np.array([intersection.x - px, intersection.y - py])
        dist = np.linalg.norm(signed_vec)
        return -dist if np.dot(signed_vec, normal) < 0 else dist
    return np.nan


def run_validation(project_dir, gt_dir):
    csv_path = project_dir / "output" / "estimated_waterbodies_edges" / "all_shorelines.csv"
    if not csv_path.exists():
        print(f"[ERROR] No se encuentra {csv_path}")
        return

    df = pd.read_csv(csv_path)
    df['date_dt'] = pd.to_datetime(df['date'])
    df['date_str'] = df['date_dt'].dt.strftime('%Y-%m-%d')
    
    available_dates = df['date_dt'].drop_duplicates().sort_values().tolist()

    validation_results = []

    print("\n" + "="*60)
    print(" 1. VALIDACIÓN CONTRA GROUND TRUTH (EVALUACIÓN RMSE)")
    print("="*60)

    for gt_date_str in GT_TARGET_DATES:
        gt_date = datetime.strptime(gt_date_str, "%Y-%m-%d")
        
        # Encontrar el archivo GT correspondiente
        gt_files = list(gt_dir.glob("*.csv"))
        matched_gt_file = None
        for f in gt_files:
            if gt_date_str in f.name or gt_date_str.replace("-", "") in f.name:
                matched_gt_file = f
                break
        
        if not matched_gt_file and gt_files:
            matched_gt_file = min(gt_files, key=lambda f: abs(
                datetime.strptime(re.search(r'\d{4}-?\d{2}-?\d{2}', f.name).group(0).replace("-", ""), "%Y%m%d") - gt_date
            ))

        if not matched_gt_file:
            print(f"[ADVERTENCIA] No se encontró CSV de GT para {gt_date_str}")
            continue

        # Encontrar la fecha de Sentinel-2 más cercana
        closest_s2_dt = min(available_dates, key=lambda d: abs(d - gt_date))
        closest_s2_str = closest_s2_dt.strftime('%Y-%m-%d')
        time_gap = abs((closest_s2_dt - gt_date).days)

        # Cargar coordenadas
        gt_coords = load_reference_coords(matched_gt_file)
        s2_coords = df[df['date_str'] == closest_s2_str][["x", "y"]].values

        # Crear líneas
        gt_line = utils.build_ordered_linestring(gt_coords, is_ordered=True)
        s2_line = utils.build_ordered_linestring(s2_coords, is_ordered=False)

        # Suavizado B-Spline de la referencia
        x, y = gt_coords[:, 0], gt_coords[:, 1]
        xy_unique = np.unique(np.column_stack((x, y)), axis=0)
        tck, _ = splprep([xy_unique[:, 0], xy_unique[:, 1]], s=0)
        n_pts = max(2, int(gt_line.length / SPATIAL_SPACING))
        unew = np.linspace(0, 1, num=n_pts)
        x_smooth, y_smooth = splev(unew, tck)
        dx, dy = splev(unew, tck, der=1)

        distances = []
        for i in range(len(unew)):
            px, py = x_smooth[i], y_smooth[i]
            tx, ty = dx[i], dy[i]
            normal = np.array([-ty, tx])
            norm_val = np.linalg.norm(normal)
            if norm_val > 0:
                normal /= norm_val

            start = (px - normal[0] * TRANSECT_LENGTH / 2, py - normal[1] * TRANSECT_LENGTH / 2)
            end = (px + normal[0] * TRANSECT_LENGTH / 2, py + normal[1] * TRANSECT_LENGTH / 2)
            transect = LineString([start, end])

            d = signed_distance(transect.intersection(s2_line), px, py, normal)
            distances.append(d)

        valid_d = pd.Series(distances).dropna()
        rmse = np.sqrt((valid_d ** 2).mean()) if len(valid_d) > 0 else np.nan

        validation_results.append({
            "Ground truth date": gt_date_str,
            "Sentinel 2 date": closest_s2_str,
            "Time gap (days)": time_gap,
            "RMSE (meters)": round(rmse, 2),
            "Status": "OK (<50m)" if rmse < 50 else "REVISAR (>=50m)"
        })

    val_df = pd.DataFrame(validation_results)
    print(val_df.to_string(index=False))
    
    return df, val_df


def generate_homework_metrics(df):
    print("\n" + "="*60)
    print(" 2. MÉTRICAS DE VALIDACIÓN Y HOMEWORK")
    print("="*60)

    dates = pd.to_datetime(df['date']).drop_duplicates().sort_values()
    
    total_points = len(df)
    valid_points = df['x'].notna().sum()
    
    # Espaciado medio entre puntos en la misma fecha
    spacings = []
    for d, group in df.groupby('date'):
        coords = group[['x', 'y']].values
        if len(coords) > 1:
            diffs = np.diff(coords, axis=0)
            dist = np.sqrt((diffs**2).sum(axis=1))
            spacings.extend(dist)
    avg_spacing = np.mean(spacings) if spacings else 0.0

    timespan_days = (dates.max() - dates.min()).days
    
    # Resolución temporal (días promedio entre capturas consecutivas)
    date_diffs = dates.diff().dt.days.dropna()
    avg_temp_res = date_diffs.mean() if len(date_diffs) > 0 else 0.0
    
    # Detección de huecos por nubes (pasos de tiempo mayores a 10 días)
    cloud_gaps = (date_diffs[date_diffs > 10]).count()

    print(f"• Number of points analyzed: {total_points}")
    print(f"• Valid data points: {valid_points}")
    print(f"• Average spacing between points: {avg_spacing:.2f} m")
    print(f"• Timespan under analysis: {timespan_days} days ({dates.min().strftime('%Y-%m-%d')} a {dates.max().strftime('%Y-%m-%d')})")
    print(f"• Temporal resolution: {avg_temp_res:.1f} days between images")
    print(f"• Gaps due to clouds (>10 days without data): {cloud_gaps} gaps")

    # Métrica de 2025 H1 (primer semestre de 2025)
    df_2025 = df[(pd.to_datetime(df['date']) >= "2025-01-01") & (pd.to_datetime(df['date']) <= "2025-06-30")]
    print(f"\n• Dates in 2025 H1: {df_2025['date'].nunique()} available dates")


if __name__ == "__main__":
    project_path = Path(".")
    gt_path = project_path / "ground_truth"
    
    if not gt_path.exists():
        gt_path.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Se ha creado la carpeta '{gt_path}'. Guarda ahí los CSVs de Atenea.")
    
    df_shorelines, val_table = run_validation(project_path, gt_path)
    if df_shorelines is not None:
        generate_homework_metrics(df_shorelines)
