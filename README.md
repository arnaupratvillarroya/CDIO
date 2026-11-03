## CDIO-1: Coastline Erosion Monitoring

¡Bienvenidos a nuestro repositorio! En este Readme os explicaremos el contenido del proyecto de la asignatura CDIO1, de la carrera de Ingeniera de Satelites de la UPC.

### Nuestro equipo:


<img width="657" height="226" alt="image" src="https://github.com/user-attachments/assets/929d6402-a966-4f2a-907d-845d2412059c" />

Somos un grupo de estudiantes apasionados por la observación de la Tierra y espacio exterior.



### Estructura del Proyecto y Códigos:

El código está organizado de forma esquematica para separar claramente las distintas fases del procesamiento de los datos. Esta es la estructura principal del repositorio:

CDIO-repository

 - 📂 data/                  # (Se genera al ejecutar) Imágenes Sentinel-2, GeoJSONs, Ground Truth y resultados
 
 - 📂 download/              # Descarga de datos y preprocesamiento
 
 - 📂 processing/            # Algoritmos de extracción de costas
 
 - 📂 evaluation/            # Validación y cálculo de errores
 
 - 📂 pipeline/              # Orquestador del proyecto
 
 - 📂 tests/                 # Pruebas unitarias de los módulos

 - 📂 coastal_timeseries/    # Módulo avanzado de series temporales y machine learning
 - 



**Descarga de Imagenes (/download)**:

En este apartado accedemos al los archivos de STAC para buscar y descargar bandas multiespectrales (Green y NIR) de Sentinel-2 con baja nubidosidad.

- search_images.py / download_gt_scenes.py: Buscan y descargan imágenes de las fechas de interes.

- nano_cloud_filter: Filtra imágenes según el porcentaje de nubosidad

**Procesamiento de Imagenes (/processing)**:

Transformamos los indices de NDWI en lineas que representan la costa.

- detect_waterbody.py: Procesa el NDWI para separar las masas de agua.

- coastline_estimator.py: Utiliza un análisis para detectar los píxeles entre tierra y agua.

- generate_shorelines_csv.py: Transforma esos píxeles en contornos ordenados y los unifica en un archivo CSV fácil de analizar.

- ndwi_n_photose.py / ndwi_denoising.py: Aplican métodos de reducción de ruido y calculan el índice NDWI utilizando Multiprocesamiento y Multithreading.

**Evaluación y Validación (/evaluation)**

Módulo encargado de medir la erosión y la precisión de nuestros algoritmos.

- calculate_erosion.py: Dibuja una lineas que siguen la forma general de la costa.
  
- validate.py / validate_ground_truth.py: Compara nuestras costas estimadas por satélite con datos medidos in situ (Ground Truth), calculando el error (RMSE).

- experiments_morphology.py: Codigo de experimentación que compara diferentes filtros de ruido con filtros que limpian y pulen las formas de las imagenes para encontrar el modelo más preciso.


**Pipeline Principal (/pipeline)**

- cdio_pipeline.py: Es el ejecutador del proyecto. Permite ejecutar todo el flujo de trabajo (descarga > procesamiento > evaluación) con un solo comando, pudiendo elegir qué técnica de corrección de ruido aplicar.
 
**Análisis de Series Temporales y Modelado**

- s10_part_prepare.py / s10_part2_analysis.py: Limpia, filtra outliers y convierte los datos en una serie temporal mensual regular, incluyendo descomposición de la tendencia y análisis estacional, evaluando modelos lineales, de escalón y sigmoides.

- s11_forecasting.py / s11_compare.py: Entrenamiento de redes neuronales recurrentes (LSTM) y comparación empírica de diversos modelos.

- s12_compare.py / s12_lstm_forecast.py / s12_parametric_forecast.py / s12_prepare.py / s12_run_all.py: Preparación de las observaciones reales para el primer semestre de 2025, ejecución de pronósticos a largo plazo y orquestador global de la sesión.

- results_s10 / results_s11 / results_s12: Carpetas generadas automáticamente que contienen todos los resultados de sus respectivas sesiones.

## Instalación y Ejecución

Sigue estos pasos para configurar el entorno y ejecutar el proyecto en tu máquina local.

### Clonar el repositorio e instalar dependencias

Te recomendamos usar un entorno virtual para no tener conflictos con otras librerías de Python. Abre tu terminal y ejecuta:

**1. Clonar el repositorio**
git clone [https://github.com/tu-usuario/nombre-del-repo.git]
cd nombre-del-repo

**2. Crear y activar el entorno virtual**
python -m venv env

**- En Mac/Linux:**
    source env/bin/activate

**- En Windows:**
    env\Scripts\activate

**3. Instalar las dependencias**
pip install -r requirements.txt

###  Ejecutar la Pipeline Principal                       
La forma más recomendable y facil de ejecutar toda la primera parte del proyecto (desde la descarga de imágenes hasta la validación de la línea de costa) es utilizar el código principal (pipeline).

Puedes elegir el método de reducción de ruido que prefieras añadiendo el argumento --method (opciones: none, boxcar_4x4, bilateral). 

Por ejemplo:

- python pipeline/cdio_pipeline.py --method bilateral

### Ejecutar Modelo de Series Temporales y Predicción (Sesiones 10-12)

La forma más recomendable y eficiente de ejecutar la segunda parte del código (que incluye la regularización de la serie temporal, entrenamiento de modelos y comparación paramétrica) es ejecutando el código general de esta sesión escribiendo la siguiente línea en la terminal:

- python coastal_timeseries/s12_run_all.py

## Resultados Visuales y Ejemplos Ilustrativos

Trabajar con imágenes satelitales es un reto. Nos podemos encontrar imágenes completamente negras, con exceso de nubes o con datos corruptos que hacen fallar los algoritmos. Sin embargo, cuando la *pipeline* procesa una escena limpia, los resultados son muy claros.

A continuación, mostramos algunos ejemplos ilustrativos de lo que consiguen nuestros códigos.

### 1. Cálculo del Índice NDWI (Índice Diferencial de Agua Normalizado)

Mediante la combinación de las bandas espectrales **Green** y **NIR** de Sentinel-2, se obtiene el mapa continuo de NDWI. Los valores positivos (azul oscuro) identifican los cuerpos de agua (mar, canales y lagunas), mientras que los valores negativos corresponden a la superficie terrestre.

<img width="1500" height="1200" alt="S2A_31TDF_20180717_0_L2A_NDWI" src="https://github.com/user-attachments/assets/71f3a4bf-ec8c-437d-96b4-89b675ce1138" />



### 2. Impacto del *Denoising* en la Extracción de la Línea de Costa

El cálculo directo del NDWI sobre imágenes crudas introduce falso ruido debido a sombra de edificios, vegetación o variaciones atmosféricas. Al extraer los bordes sin filtrado, el algoritmo detecta objetos por toda la superficie terrestre.

Al aplicar técnicas de filtrado espacial (**Boxcar 4x4** o **Bilateral**), se elimina el grano y conseguimos aislar la línea de costa real y el canal costero:

<img width="1600" height="640" alt="image" src="https://github.com/user-attachments/assets/499a5a97-c561-4044-9f84-60e99274da3d" />



### 3. Impacto de las Operaciones Morfológicas en la Extracción

Tras el filtrado de ruido y el cálculo de la máscara binaria de agua, la aplicación de operadores morfológicos es fundamental para pulir la geometría:

**Apertura (*Opening*):** Elimina el ruido de pequeños píxeles aislados en la zona terrestre.

**Cierre (*Closing*):** Rellena pequeños huecos dentro de la masa de agua.

La experimentación permite evaluar de forma cuantitativa el rendimiento de distintas configuraciones frente al **Ground Truth**:

<img width="1820" height="1040" alt="qualitative_2019-05-23" src="https://github.com/user-attachments/assets/783433f2-5f50-4028-ae01-48379e215165" />



Este refinamiento permite obtener un contorno continuo y limpio antes de convertir los píxeles a formatos vectoriales.

### 4. Modelado Avanzado de Series Temporales y Machine Learning

En esta sección se amplía de forma significativa las capacidades del proyecto, abordando el análisis evolutivo y la predicción a medio y largo plazo de la línea de costa de la zona de interés. Esta parte se divide en 3 fases:

 **Regularización y Análisis Espectral (FFT):**
Se procesa la serie temporal mensual detectando valores perdidos y filtrando valores atípicos. Se analizan también las frecuencias con la Transformada de Fourier para aislar los ciclos estacionales dominantes. 

<img width="2210" height="520" alt="11_boxplots" src="https://github.com/user-attachments/assets/dc2a9b28-9ab6-4a3a-8540-84caf99b65cb" />

<img width="1300" height="520" alt="12_fft_spectrum" src="https://github.com/user-attachments/assets/14607d61-2392-4de8-8a72-736158072a39" />



 **Modelado Paramétrico y Dinámica Costera:**
 Se ajustan los modelos de tendencia analíticos, como *linear*, *step* y *sigmoide* para capturar saltos bruscos y cuantifiacar la velocidad y aceleración de la costa ante eventos extremos.

<img width="3565" height="1769" alt="s11_model_comparison" src="https://github.com/user-attachments/assets/b7122760-6050-41d9-a9ac-3180f226ada4" />



  **Machine Learning:**
Se hace un pronóstico a medio/largo plazo mediante redes neuronales **LSTM** frente al modelo paramétrico ***Step + FFT***.

<img width="2860" height="1430" alt="s12_final_model_comparison" src="https://github.com/user-attachments/assets/88a9f5dd-be05-445e-ab12-cad1f1a7b747" />

