## CDIO-1: Coastline Erosion Monitoring

¡Bienvenidos a nuestro repositorio! En este Readme os explicaremos el contenido del proyecto de la asignatura CDIO1, de la carrera de Ingeniera de Satelites de la UPC.

**Nuestro equipo:**


<img width="657" height="226" alt="image" src="https://github.com/user-attachments/assets/929d6402-a966-4f2a-907d-845d2412059c" />

Somos un grupo de estudiantes apasionados por la observación de la Tierra y espacio exterior.



**Estructura del Proyecto y Códigos**:

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
  
- validate.py / validate_ground_truth.py: Compara nuestras costas estimadas por satélite con datos medidos in situ (Ground Truth), calculando el erro (RMSE).

- experiments_morphology.py: Codigo de experimentación que compara diferentes filtros de ruido con filtros que limpian y pulen las formas de las imagenes para encontrar el modelo más preciso.


**Pipeline Principal (/pipeline)**

- cdio_pipeline.py: Es el ejecutador del proyecto. Permite ejecutar todo el flujo de trabajo (descarga > procesamiento > evaluación) con un solo comando, pudiendo elegir qué técnica de corrección de ruido aplicar.



## Instalación y Ejecución

Sigue estos pasos para configurar el entorno y ejecutar el proyecto en tu máquina local.

### 1. Clonar el repositorio e instalar dependencias

Te recomendamos usar un entorno virtual para no tener conflictos con otras librerías de Python. Abre tu terminal y ejecuta:

```bash
# 1. Clonar el repositorio
git clone [https://github.com/tu-usuario/nombre-del-repo.git](https://github.com/tu-usuario/nombre-del-repo.git)
cd nombre-del-repo

# 2. Crear y activar el entorno virtual
python -m venv env

# En Mac/Linux:
source env/bin/activate
# En Windows:
# env\Scripts\activate

# 3. Instalar las dependencias
pip install -r requirements.txt



