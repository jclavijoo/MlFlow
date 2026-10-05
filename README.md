## Taller de MLOps con MLflow, MinIO, PostgreSQL y JupyterLab

Proyecto de experimentación e inferencia que registra 20 entrenamientos, almacena los datos originales y procesados en PostgreSQL y publica predicciones mediante una API que obtiene el modelo desde el registro de MLflow.

Esta versión utiliza **Wine de scikit-learn**, no el dataset de cobertura forestal de los notebooks iniciales. Wine contiene 178 observaciones, 13 características numéricas y tres clases (0, 1 y 2). El objetivo es clasificar el tipo de vino a partir de sus características químicas; no predecir su calidad. El dataset viene incluido en scikit-learn y no necesita descarga externa.

## Arquitectura

| Servicio Compose | Responsabilidad | Puerto en el servidor |
|---|---|---|
| `postgres_MLFlow` | Base `bdmlflow`: metadata de experimentos, métricas y registro de modelos | Sin publicación; 5432 interno |
| `data-db` | Base `experiment_data`: datos originales, procesados y resultados | Sin publicación; 5432 interno |
| `minio` | Bucket `mlflows3`: modelos y otros artefactos | 9000 (S3), 9001 (consola) |
| `mlflow` | Tracking Server y Model Registry | 5001 |
| `jupyter` | Ejecución del notebook de entrenamiento | 8888 |
| `api` | Inferencia con el modelo registrado | 8027 |

Jupyter registra las ejecuciones en MLflow. MLflow guarda la metadata en su PostgreSQL y los artefactos en MinIO mediante `--artifacts-destination`. La API consulta el alias `champion` del modelo registrado y descarga su versión a través de MLflow. MLflow y la API comparten la imagen `mlflow-taller:2.22.0`.

## Archivos principales

| Ruta | Contenido |
|---|---|
| `docker-compose.yaml` | Definición de los seis servicios y volúmenes |
| `mlflow/Dockerfile` | Imagen compartida por MLflow y la API |
| `notebooks/01_taller_20_experimentos.ipynb` | Notebook principal del taller |
| `notebooks/app/__init__.py` | Inicialización del paquete Python |
| `notebooks/app/training.py` | Persistencia, procesamiento, entrenamiento y selección |
| `api/api.py` | Aplicación FastAPI |
| `request.json` | Petición de ejemplo para inferencia |

Los notebooks `mlflow_examples.ipynb` y `mlflow_pipeline_example.ipynb` son ejemplos previos. No es necesario ejecutarlos para completar este flujo.

## Requisitos

- Docker Engine y Docker Compose v2.
- Acceso a Internet para descargar imágenes e instalar dependencias.
- Espacio libre en la partición que almacena Docker y containerd, también durante la construcción e instalación de paquetes.
- Acceso SSH al servidor si se trabaja remotamente.

Comprobar antes de construir:

```bash
docker --version
docker compose version
df -h /var
docker system df
```

Las versiones principales se fijan para el laboratorio: MLflow 2.22.0, Python 3.11 en el servidor MLflow/API, SQLAlchemy 2.0.40, scikit-learn 1.6.1, NumPy 1.24.4 y pandas 2.0.3. La imagen de Jupyter se define por separado en Compose.

## 1. Iniciar los servicios

Desde la carpeta `mlflow_compose`, donde está `docker-compose.yaml`:

```bash
docker compose up -d --build
docker compose ps -a
```

Comprobar los servicios:

```bash
curl --max-time 10 http://localhost:5001/health
curl --max-time 10 http://localhost:8027/health
docker compose logs --tail=30 mlflow api
```

La API debe responder a `/health` con `{"status":"alive"}`. Antes del entrenamiento, `/ready` puede devolver 503 porque todavía no existe el modelo con alias `champion`.

### Accesos locales

| Interfaz | URL | Credenciales del laboratorio |
|---|---|---|
| MLflow | http://localhost:5001 | Sin autenticación configurada |
| MinIO | http://localhost:9001 | Usuario `admin`, contraseña `supersecret` |
| JupyterLab | http://localhost:8888 | Token `valentasecret` |
| API Swagger | http://localhost:8027/docs | Sin autenticación configurada |

Las credenciales son de práctica. Esta configuración no incluye autenticación de la API ni HTTPS y no debe publicarse directamente en Internet.

### Acceso remoto por VPN y SSH

Si la VPN permite SSH pero no los puertos de las aplicaciones, ejecutar **desde el computador del usuario**, no dentro del servidor:

```bash
ssh -N -L 19001:127.0.0.1:9001 -L 15001:127.0.0.1:5001 -L 18888:127.0.0.1:8888 -L 18027:127.0.0.1:8027 estudiante@10.43.97.97
```

Mantener abierta esa terminal. Adaptar usuario e IP si cambia el servidor.

| Servicio | URL mediante el túnel |
|---|---|
| MinIO | http://localhost:19001 |
| MLflow | http://localhost:15001 |
| JupyterLab | http://localhost:18888 |
| API Swagger | http://localhost:18027/docs |

Dentro de Docker, el notebook utiliza `http://mlflow:5000` y `http://api:8000`. Son direcciones internas; no se sustituyen por las del túnel.

## 2. Ejecutar el notebook

1. Abrir JupyterLab y entrar a `work`.
2. Abrir `01_taller_20_experimentos.ipynb`.
3. Ejecutar la primera celda de instalación de dependencias.
4. Seleccionar **Kernel → Restart Kernel**.
5. Ejecutar las celdas restantes en orden, sin repetir la instalación.

**No se reinicia el kernel entre entrenamientos.** Una sola celda ejecuta las 20 combinaciones y muestra el progreso `01/20` hasta `20/20`.

El notebook verifica MinIO y crea el bucket `mlflows3` si falta. También comprueba `/health` de la API, por lo que `api/api.py` debe estar copiado y el servicio iniciado antes de continuar.

## 3. Datos y procesamiento

La base de entrenamiento `experiment_data` pertenece a una instancia PostgreSQL diferente de la utilizada por MLflow.

| Tabla | Contenido |
|---|---|
| `wine_raw` | Datos originales con `dataset_id` y `sample_id` |
| `wine_processed` | Variables transformadas, etiqueta, identificadores y partición |
| `experiment_results` | Hiperparámetros, métricas de validación, run y versión de cada candidato |

El flujo de datos es:

1. Guardar Wine en `wine_raw` y releerlo desde PostgreSQL.
2. Dividir de forma estratificada, con semilla 42: 106 filas de entrenamiento, 36 de validación y 36 de prueba.
3. Ajustar imputación por mediana y `StandardScaler` exclusivamente sobre entrenamiento.
4. Transformar las tres particiones y persistirlas en `wine_processed`.
5. Releer los datos procesados desde PostgreSQL para entrenar y validar.

Se registra el pipeline completo, incluido el preprocesador, para que la API reciba las 13 variables originales sin escalarlas previamente.

## 4. Veinte entrenamientos registrados

Se utiliza `LogisticRegression` con la siguiente cuadrícula:

| Hiperparámetro | Valores |
|---|---|
| `C` | 0.01, 0.1, 1.0, 10.0, 100.0 |
| `class_weight` | `None`, `balanced` |
| `fit_intercept` | `False`, `True` |

Total: **5 × 2 × 2 = 20 configuraciones**. El solver es `lbfgs` y el máximo de iteraciones es 3000.

Cada entrenamiento genera un run independiente con parámetros, métricas, identificador y huella del dataset, reporte de clasificación de validación, matriz de confusión, firma, ejemplo de entrada y pipeline del modelo. Cada modelo se registra como una nueva versión de `wine_classifier`.

### Selección del modelo

- Criterio principal: mayor `val_f1_macro`.
- Desempate: menor `val_log_loss` y después menor número de versión.
- Evaluación final: accuracy y F1 macro en test únicamente para el ganador.
- Publicación: asignar el alias `champion` a esa versión.

El conjunto de test no participa en la selección de hiperparámetros. No se realiza un reentrenamiento adicional con train y validación combinados.

### Dónde ver las ejecuciones

En MLflow, seleccionar **Experiments → `wine_20_experimentos`**. Aparecen los runs `logreg_01` a `logreg_20`. En **Models → `wine_classifier`** se encuentran las versiones registradas y el alias `champion`.

Técnicamente son **20 runs dentro de un experimento de MLflow**, no 20 experimentos independientes.

Cada ejecución completa del notebook genera un nuevo `dataset_id`, añade 20 runs y versiones y conserva los anteriores. El alias pasa al ganador del lote más reciente; no se compara automáticamente con ganadores de lotes anteriores. No ejecutar dos copias del notebook simultáneamente.

## 5. API de inferencia

| Método | Ruta | Función |
|---|---|---|
| GET | `/health` | Comprobar que el proceso está activo |
| GET | `/ready` | Comprobar que el modelo se puede obtener desde MLflow |
| POST | `/predict` | Obtener clases y probabilidades |
| GET | `/docs` | Abrir documentación interactiva |

La API consulta `wine_classifier@champion`, resuelve la versión y la carga con una URI `models:/wine_classifier/<version>`. Conserva el modelo en memoria y vuelve a descargarlo si cambia el alias. Si no puede resolverlo en MLflow, responde 503.

No utiliza un `.pkl` copiado manualmente ni necesita consultar la base de entrenamiento.

### Petición de ejemplo

```json
{
  "instances": [
    {
      "alcohol": 14.23,
      "malic_acid": 1.71,
      "ash": 2.43,
      "alcalinity_of_ash": 15.6,
      "magnesium": 127.0,
      "total_phenols": 2.8,
      "flavanoids": 3.06,
      "nonflavanoid_phenols": 0.28,
      "proanthocyanins": 2.29,
      "color_intensity": 5.64,
      "hue": 1.04,
      "od280/od315_of_diluted_wines": 3.92,
      "proline": 1065.0
    }
  ]
}
```

Desde el servidor, utilizando `request.json`:

```bash
curl http://localhost:8027/ready
curl -X POST http://localhost:8027/predict -H 'Content-Type: application/json' --data-binary @request.json
```

Desde el computador con el túnel abierto, usar el puerto `18027`. En PowerShell:

```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:18027/predict -ContentType 'application/json' -Body (Get-Content .\request.json -Raw)
```

La respuesta incluye `model`, `version`, `alias`, `classes`, `predictions` y `probabilities`. Las probabilidades siguen el orden de `classes`. Se admiten entre 1 y 1000 muestras; entradas incompletas, campos adicionales y números no finitos se rechazan con 422.

## 6. Verificar los requisitos del taller

| Requisito | Evidencia esperada |
|---|---|
| BD dedicada a metadata | Servicio `postgres_MLFlow`, base `bdmlflow` |
| MLflow operativo | Interfaz y `/health` |
| MinIO dedicado | Bucket `mlflows3` con artefactos |
| JupyterLab | Notebook ejecutado con salidas |
| Al menos 20 configuraciones | 20 runs terminados para el mismo `dataset_id` |
| Datos originales y procesados en otra BD | Tablas `wine_raw` y `wine_processed` en `data-db` |
| Modelos registrados | Versiones de `wine_classifier` |
| Inferencia mediante MLflow | Respuesta `/predict` con la versión ganadora |

Consultas en el servidor:

```bash
docker compose exec data-db psql -U data_user -d experiment_data -c "SELECT dataset_id, COUNT(*) FROM wine_raw GROUP BY dataset_id;"
docker compose exec data-db psql -U data_user -d experiment_data -c "SELECT dataset_id, split, COUNT(*) FROM wine_processed GROUP BY dataset_id, split;"
docker compose exec data-db psql -U data_user -d experiment_data -c "SELECT dataset_id, COUNT(*) FROM experiment_results GROUP BY dataset_id;"
```

El notebook comprueba al final los 20 runs terminados, sus versiones registradas y que las predicciones de la API coincidan con la carga directa del modelo desde MLflow. Guardar el notebook ejecutado y capturas de estas evidencias para la entrega. Este README no afirma que los entrenamientos ya se hayan ejecutado en el servidor ni proporciona métricas inventadas.

## 7. Operación y solución de problemas

### Logs

```bash
docker compose logs --tail=50 mlflow
docker compose logs --tail=50 api
docker compose logs --tail=50 jupyter
docker compose logs --tail=50 data-db
```

### Cambios de archivos

- Notebook: cerrar la pestaña sin sobrescribir el archivo nuevo y volver a abrir; no requiere reconstruir Docker.
- `training.py`: reiniciar el kernel para recargar el módulo y ejecutar las celdas necesarias.
- `api/api.py`: ejecutar `docker compose restart api`.
- Compose o Dockerfile: ejecutar `docker compose up -d --build`.

### `Could not import module "api"`

Comprobar que exista `api/api.py` junto al Compose, y que el servicio monte `./api:/api:ro` con directorio de trabajo `/api`. Si existe y falla, revisar el traceback y los permisos; en Rocky Linux también puede intervenir SELinux. No basta con reiniciar si falta el archivo.

### `/ready` devuelve 503

Comprobar MLflow, ejecutar completamente el notebook y verificar que `wine_classifier` tenga el alias `champion`. `/health` puede responder correctamente antes de tener un modelo.

### No aparecen los 20 runs

Abrir `wine_20_experimentos`, no `mlflow_tracking_examples_class`. Comprobar que se ejecutó la celda de 20 combinaciones del notebook nuevo y que no se detuvo por un error.

### `No space left on device`

```bash
df -h
df -i
docker system df
sudo du -xsh /var/lib/docker /var/lib/containerd
```

Puede estar lleno `/var` aunque `/` o `/home` tengan espacio. No repetir instalaciones hasta liberar o ampliar el almacenamiento. No borrar volúmenes si contienen datos que se necesitan conservar.

### Detener conservando los datos

```bash
docker compose down
```

Los volúmenes `minio_data`, `postgres-db-volume` y `datos-taller` conservan artefactos y bases. Los notebooks están en la carpeta montada del servidor. No añadir `-v` si se desea conservar los volúmenes. Mantener el nombre y ubicación del proyecto Compose al actualizar para reutilizar los recursos existentes.
