#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ejecutor completo de S12.

Lanza, en orden, los cuatro scripts de preparación, modelado y comparación:
  1. s12_prepare_2025.py
  2. s12_parametric_forecast.py
  3. s12_lstm_forecast.py
  4. s12_compare.py

Uso:
    python s12_run_all.py

Coloca este archivo en la carpeta coastal_timeseries, junto a los otros
scripts de S12, y ejecútalo desde esa carpeta.

El proceso se detiene inmediatamente si alguno de los scripts falla.
No modifica results_s10 ni results_s11.
"""
from pathlib import Path
import subprocess
import sys
import time


BASE_DIR = Path(__file__).resolve().parent

S12_STEPS = [
    ("Preparación de las observaciones reales de 2025", "s12_prepare_2025.py"),
    ("Pronóstico paramétrico Step + FFT", "s12_parametric_forecast.py"),
    ("Pronóstico LSTM recursivo", "s12_lstm_forecast.py"),
    ("Comparación final de los modelos", "s12_compare.py"),
]


def main():
    print("=" * 72)
    print("INICIO DEL FLUJO COMPLETO DE LA SESIÓN 12")
    print("=" * 72)
    print(f"Carpeta de trabajo: {BASE_DIR}")
    print(f"Intérprete Python: {sys.executable}")

    # Comprobar todos los scripts antes de empezar para evitar una ejecución
    # parcial por un archivo que falte.
    missing = [
        filename for _, filename in S12_STEPS
        if not (BASE_DIR / filename).is_file()
    ]
    if missing:
        print("\nERROR: faltan estos archivos en coastal_timeseries:")
        for filename in missing:
            print(f" - {filename}")
        print("\nCopia todos los scripts de S12 a esta misma carpeta y vuelve a intentarlo.")
        raise SystemExit(1)

    started = time.time()

    for index, (description, filename) in enumerate(S12_STEPS, start=1):
        script_path = BASE_DIR / filename

        print("\n" + "-" * 72, flush=True)
        print(f"PASO {index}/{len(S12_STEPS)}: {description}", flush=True)
        print(f"Ejecutando: {filename}", flush=True)
        print("-" * 72, flush=True)

        try:
            subprocess.run(
                [sys.executable, str(script_path)],
                cwd=BASE_DIR,
                check=True,
            )
        except subprocess.CalledProcessError as error:
            print("\n" + "!" * 72)
            print(f"ERROR: el paso {index} ({filename}) ha terminado con código {error.returncode}.")
            print("El flujo se detiene aquí; no se ejecutan los pasos siguientes.")
            print("Corrige el error y vuelve a ejecutar python s12_run_all.py.")
            print("!" * 72)
            raise SystemExit(error.returncode)

        print(f"\nPASO {index}/{len(S12_STEPS)} COMPLETADO.", flush=True)

    elapsed = time.time() - started
    print("\n" + "=" * 72)
    print("SESIÓN 12 COMPLETADA CORRECTAMENTE")
    print(f"Tiempo total: {elapsed / 60:.1f} minutos")
    print(f"Resultados disponibles en: {BASE_DIR / 'results_s12'}")
    print("Incluye la preparación, los dos pronósticos y su comparación.")
    print("=" * 72)


if __name__ == "__main__":
    main()
