# Sessions 10-12: analisi i prediccio de la serie temporal de la costa

Requisits: `numpy pandas scipy matplotlib` (la sessio 12 afegira TensorFlow o PyTorch).

## Sessio 10
```bash
cd coastal_timeseries
# Part 1: preparacio de dades (CSV brut -> serie mensual neta)
python s10_part1_prepare.py --input ../data/input/shoreline_distances_2017_2024.csv
# Part 2: tendencia, estacionalitat (FFT), 3 models, velocitat i acceleracio
python s10_part2_analysis.py --n-cycles 3
```
Totes les sortides (CSV, JSON, figures) van a `results_s10/`.

| Fitxer | Contingut |
|---|---|
| `ts_utils.py` | metriques, models de tendencia (linear / step / sigmoid) i cicles FFT (compartit amb la sessio 12) |
| `s10_part1_prepare.py` | passos 1-13 de la Part 1 |
| `s10_part2_analysis.py` | passos 2-11 de la Part 2; desa `best_model.json` |

Convencio: distancia > 0 = erosio (la costa retrocedeix); < 0 = acrecio.
Parametres principals: `--n-transects`, `--max-nan-transect`, `--iqr-k`, `--max-nan-date`, `--freq`, `--jump`, `--n-cycles`.