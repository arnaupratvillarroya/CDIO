
from pathlib import Path
import argparse
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def run_stage(name, script, extra_args=None):
    script_path = ROOT / script

    if not script_path.exists():
        raise FileNotFoundError(
            f"No se encuentra el script necesario: {script_path}"
        )

    print("\n" + "=" * 70)
    print(f" ETAPA: {name}")
    print("=" * 70)

    cmd = [sys.executable, str(script_path)]
    if extra_args:
        cmd.extend(extra_args)

    subprocess.run(
        cmd,
        cwd=ROOT,
        check=True
    )


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline completo del proyecto CDIO con soporte para métodos de denoising."
    )

    parser.add_argument(
        "--method",
        type=str,
        default="none",
        choices=["none", "boxcar_4x4", "bilateral"],
        help="Método de denoising a aplicar y evaluar."
    )

    parser.add_argument(
        "--skip-ndwi",
        action="store_true",
        help="Salta la descarga de imágenes y el cálculo de NDWI."
    )

    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Salta la validación contra Ground Truth."
    )

    args = parser.parse_args()

    print(f"\n[CONFIG] Ejecutando pipeline completo con el método: **{args.method}**")

    # 1. Descarga + NDWI (Pasando el método)
    if not args.skip_ndwi:
        run_stage(
            f"DESCARGA DE IMÁGENES + CÁLCULO NDWI ({args.method})",
            "ndwi_n_photose.py",
            extra_args=["--method", args.method]
        )
    else:
        print("\n[INFO] Se salta la etapa de descarga + NDWI.")

    # 2. Detección de masas de agua (Pasando el método para leer la carpeta correcta)
    run_stage(
        f"DETECCIÓN DE WATERBODY ({args.method})",
        "detect_waterbody.py",
        extra_args=["--method", args.method]
    )

    # 3. Extracción de línea de costa (Pasando el método)
    run_stage(
        f"EXTRACCIÓN DE COASTLINE ({args.method})",
        "coastline_estimator.py",
        extra_args=["--method", args.method]
    )

    # 4. Generación del CSV de líneas de costa (Pasando el método)
    run_stage(
        f"GENERACIÓN DEL CSV DE SHORELINES ({args.method})",
        "generate_shorelines_csv.py",
        extra_args=["--method", args.method]
    )

    # 5. Validación (Pasando el método para que busque el CSV correcto)
    if not args.skip_validation:
        run_stage(
            f"VALIDACIÓN CONTRA GROUND TRUTH ({args.method})",
            "validate_ground_truth.py",
            extra_args=["--method", args.method]
        )
    else:
        print("\n[INFO] Se salta la etapa de validación.")

    print("\n" + "=" * 70)
    print(f" PIPELINE COMPLETADO CORRECTAMENTE ({args.method})")
    print("=" * 70)


if __name__ == "__main__":
    main()