## CDIO-1: Coastline Erosion Monitoring

¡Bienvenidos a nuestro repositorio! En este Readme os explicaremos el contenido del proyecto de la asignatura CDIO1, de la carrera de Ingeniera de Satelites de la UPC.

### Nuestro equipo:


<img width="657" height="226" alt="image" src="https://github.com/user-attachments/assets/929d6402-a966-4f2a-907d-845d2412059c" />

Somos un grupo de estudiantes apasionados por la observación de la Tierra y espacio exterior.



### Estructura del Proyecto y Códigos:

El código está organizado de forma esquematica para separar claramente las distintas fases del procesamiento de los datos. Esta es la estructura principal del repositorio:

CDIO-repository

 - 📂 data/          # (Se genera al ejecutar) Imágenes Sentinel-2, GeoJSONs, Ground Truth y resultados
 
 - 📂 download/      # Descarga de datos y preprocesamiento
 
 - 📂 processing/    # Algoritmos de extracción de costas
 
 - 📂 evaluation/    # Validación y cálculo de errores
 
 - 📂 pipeline/      # Orquestador del proyecto
 
 - 📂 tests/         # Pruebas unitarias de los módulos



**Descarga de Imagenes (/download)**:

En este apartado accedemos al los archivos de STAC para buscar y descargar bandas multiespectrales (Green y NIR) de Sentinel-2 con baja nubidosidad.

- search_images.py / download_gt_scenes.py: Buscan y descargan imágenes de las fechas de interes.

- ndwi_n_photose.py / ndwi_denoising.py: Aplican métodos de reducción de ruido y calculan el índice NDWI utilizando Multiprocesamiento y Multithreading.


**Procesamiento de Imagenes (/processing)**:

Transformamos los indices de NDWI en lineas que representan la costa.

- detect_waterbody.py: Procesa el NDWI para separar las masas de agua.

- coastline_estimator.py: Utiliza un análisis para detectar los píxeles entre tierra y agua.

- generate_shorelines_csv.py: Transforma esos píxeles en contornos ordenados y los unifica en un archivo CSV fácil de analizar.


**Evaluación y Validación (/evaluation)**

Módulo encargado de medir la erosión y la precisión de nuestros algoritmos.

- calculate_erosion.py: Dibuja una lineas que siguen la forma general de la costa.
  
- validate.py / validate_ground_truth.py: Compara nuestras costas estimadas por satélite con datos medidos in situ (Ground Truth), calculando el error (RMSE).

- experiments_morphology.py: Codigo de experimentación que compara diferentes filtros de ruido con filtros que limpian y pulen las formas de las imagenes para encontrar el modelo más preciso.


**Pipeline Principal (/pipeline)**

- cdio_pipeline.py: Es el ejecutador del proyecto. Permite ejecutar todo el flujo de trabajo (descarga > procesamiento > evaluación) con un solo comando, pudiendo elegir qué técnica de corrección de ruido aplicar.



## Instalación y Ejecución

Sigue estos pasos para configurar el entorno y ejecutar el proyecto en tu máquina local.

### Clonar el repositorio e instalar dependencias

Te recomendamos usar un entorno virtual para no tener conflictos con otras librerías de Python. Abre tu terminal y ejecuta:

**1. Clonar el repositorio**
git clone [https://github.com/tu-usuario/nombre-del-repo.git]
cd nombre-del-repo

**2. Crear y activar el entorno virtual**
python -m venv env

**En Mac/Linux:**
source env/bin/activate
**En Windows:**
**env\Scripts\activate**

**Instalar las dependencias**
pip install -r requirements.txt

###  Ejecutar la Pipeline Principal                       
La forma más recomendable y facil de ejecutar todo el proyecto (desde la descarga de imágenes hasta la validación de la línea de costa) es utilizar el codigo principal (pipeline).

Puedes elegir el método de reducción de ruido que prefieras añadiendo el argumento --method (opciones: none, boxcar_4x4, bilateral). 

Por ejemplo:

- python pipeline/cdio_pipeline.py --method bilateral

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


