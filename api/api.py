import logging
import os
from threading import RLock
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('inference')
mlflow.set_tracking_uri(os.environ.get('MLFLOW_TRACKING_URI', 'http://mlflow:5000'))
MODEL_NAME = os.environ.get('MODEL_NAME', 'wine_classifier')
MODEL_ALIAS = os.environ.get('MODEL_ALIAS', 'champion')
app = FastAPI(title='Inferencia Wine mediante MLflow', version='1.0.0')
lock = RLock()
cache = {'model': None, 'version': None}

class WineSample(BaseModel):
    model_config = ConfigDict(extra='forbid')
    alcohol: FiniteFloat
    malic_acid: FiniteFloat
    ash: FiniteFloat
    alcalinity_of_ash: FiniteFloat
    magnesium: FiniteFloat
    total_phenols: FiniteFloat
    flavanoids: FiniteFloat
    nonflavanoid_phenols: FiniteFloat
    proanthocyanins: FiniteFloat
    color_intensity: FiniteFloat
    hue: FiniteFloat
    od280_od315_of_diluted_wines: FiniteFloat = Field(alias='od280/od315_of_diluted_wines')
    proline: FiniteFloat

class PredictRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    instances: list[WineSample] = Field(min_length=1, max_length=1000)

def resolve_model():
    # Resuelve el alias en cada peticion; descarga solo si cambia la version.
    with lock:
        version = str(MlflowClient().get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS).version)
        if cache['model'] is None or cache['version'] != version:
            model = mlflow.sklearn.load_model(f'models:/{MODEL_NAME}/{version}')
            cache.update(model=model, version=version)
            logger.info('Modelo cargado desde MLflow: %s version=%s', MODEL_NAME, version)
        return cache['model'], cache['version']

@app.get('/health')
def health():
    return {'status': 'alive'}

@app.get('/ready')
def ready():
    try:
        _, version = resolve_model()
        return {'status': 'ready', 'model': MODEL_NAME, 'version': version, 'alias': MODEL_ALIAS}
    except Exception:
        logger.exception('No se pudo resolver el modelo')
        raise HTTPException(503, 'Modelo no disponible. Ejecuta el notebook y verifica MLflow.')

@app.post('/predict')
def predict(body: PredictRequest):
    try:
        model, version = resolve_model()
    except Exception:
        logger.exception('MLflow no disponible o alias inexistente')
        raise HTTPException(503, 'Modelo no disponible. Ejecuta el notebook y verifica MLflow.')
    frame = pd.DataFrame([sample.model_dump(by_alias=True) for sample in body.instances])
    frame = frame[list(model.feature_names_in_)]
    predictions = model.predict(frame)
    probabilities = model.predict_proba(frame)
    logger.info('Prediccion: model=%s version=%s rows=%s', MODEL_NAME, version, len(frame))
    return {'model': MODEL_NAME, 'version': version, 'alias': MODEL_ALIAS,
            'classes': model.classes_.tolist(), 'predictions': predictions.tolist(),
            'probabilities': probabilities.tolist()}
