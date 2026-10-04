"""Funciones usadas por el notebook. No se ejecuta entrenamiento al importar."""
import hashlib
import os
import uuid
import time
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from mlflow.models import infer_signature
from mlflow.data.code_dataset_source import CodeDatasetSource
from sklearn.datasets import load_wine
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import accuracy_score, f1_score, log_loss, confusion_matrix, classification_report
from sqlalchemy import create_engine, URL, text

MODEL_NAME = 'wine_classifier'
EXPERIMENT = 'wine_20_experimentos'
SEED = 42

def data_engine():
    return create_engine(URL.create('postgresql+psycopg2', username='data_user',
        password=os.environ['DATA_PASSWORD'], host='data-db', database='experiment_data'))

def prepare_data(engine):
    dataset_id = uuid.uuid4().hex
    source = load_wine(as_frame=True)
    raw = source.frame.copy()
    raw.insert(0, 'sample_id', np.arange(len(raw)))
    raw.insert(0, 'dataset_id', dataset_id)
    # El dataframe se guarda primero y luego se relee de la BD.
    with engine.begin() as conn:
        raw.to_sql('wine_raw', conn, if_exists='append', index=False)
    with engine.connect() as conn:
        raw = pd.read_sql(text('SELECT * FROM wine_raw WHERE dataset_id=:id ORDER BY sample_id'),
                          conn, params={'id': dataset_id})
    features = list(source.feature_names)
    trainval, test = train_test_split(raw, test_size=0.2, stratify=raw.target, random_state=SEED)
    train, val = train_test_split(trainval, test_size=0.25, stratify=trainval.target, random_state=SEED)
    prep = Pipeline([('imputer', SimpleImputer(strategy='median')),
                     ('scaler', StandardScaler())]).set_output(transform='pandas')
    prep.fit(train[features])  # Solo train: no fuga desde validacion/test.
    parts = {'train': train, 'validation': val, 'test': test}
    processed = []
    for split, frame in parts.items():
        result = prep.transform(frame[features]).copy()
        result['target'] = frame.target.to_numpy()
        result['sample_id'] = frame.sample_id.to_numpy()
        result['split'] = split
        result['dataset_id'] = dataset_id
        processed.append(result)
    transformed = pd.concat(processed, ignore_index=True)
    with engine.begin() as conn:
        transformed.to_sql('wine_processed', conn, if_exists='append', index=False)
    # Entrenamiento y evaluacion leen las tablas persistidas.
    with engine.connect() as conn:
        transformed = pd.read_sql(text('SELECT * FROM wine_processed WHERE dataset_id=:id ORDER BY sample_id'),
                                  conn, params={'id': dataset_id})
    parts = {key: frame.sort_values('sample_id') for key, frame in parts.items()}
    processed_parts = {key: transformed[transformed.split == key].sort_values('sample_id') for key in parts}
    digest = hashlib.sha256(raw.drop(columns='dataset_id').to_csv(index=False).encode()).hexdigest()
    return dict(dataset_id=dataset_id, features=features, prep=prep, raw=parts,
                processed=processed_parts, digest=digest)

def parameter_grid():
    return list(ParameterGrid({'C': [0.01, 0.1, 1.0, 10.0, 100.0],
                               'class_weight': [None, 'balanced'],
                               'fit_intercept': [False, True]}))

def train_one(ctx, params, index):
    features = ctx['features']
    train = ctx['processed']['train']
    val = ctx['processed']['validation']
    raw_val = ctx['raw']['validation']
    with mlflow.start_run(run_name=f'logreg_{index:02d}') as run:
        mlflow.set_tags({'dataset_id': ctx['dataset_id'], 'dataset_sha256': ctx['digest'],
                        'source_table': 'wine_raw', 'processed_table': 'wine_processed',
                        'algorithm': 'LogisticRegression', 'selection_metric': 'val_f1_macro'})
        mlflow.log_params({**params, 'solver': 'lbfgs', 'max_iter': 3000, 'seed': SEED,
                           'imputation': 'median_train', 'scaling': 'standard_train',
                           'n_train': len(train), 'n_validation': len(val),
                           'n_test': len(ctx['processed']['test'])})
        clf = LogisticRegression(**params, solver='lbfgs', max_iter=3000, random_state=SEED)
        start = time.perf_counter()
        clf.fit(train[features], train.target)
        duration = time.perf_counter() - start
        pred = clf.predict(val[features])
        metrics = {'train_accuracy': accuracy_score(train.target, clf.predict(train[features])),
                   'val_accuracy': accuracy_score(val.target, pred),
                   'val_f1_macro': f1_score(val.target, pred, average='macro'),
                   'val_log_loss': log_loss(val.target, clf.predict_proba(val[features])),
                   'fit_seconds': duration}
        mlflow.log_metrics(metrics)
        # Pipeline completo para recibir datos CRUDOS desde la API.
        model = Pipeline([('preprocessing', ctx['prep']), ('classifier', clf)])
        assert np.array_equal(model.predict(raw_val[features]), pred)
        mlflow.log_dict({'features': features, 'classes': [0, 1, 2],
                         'dataset_id': ctx['dataset_id']}, 'schema.json')
        mlflow.log_dict({'matrix': confusion_matrix(val.target, pred).tolist(),
                         'labels': [0, 1, 2]}, 'validation_confusion_matrix.json')
        mlflow.log_dict(classification_report(val.target, pred, output_dict=True, zero_division=0),
                        'validation_report.json')
        mlflow.log_input(mlflow.data.from_pandas(train[features + ['target']],
                         source=CodeDatasetSource(tags={'database': 'experiment_data',
                             'table': 'wine_processed', 'dataset_id': ctx['dataset_id']}),
                         targets='target', name='wine_train'), context='training')
        mlflow.sklearn.log_model(model, artifact_path='model',
            signature=infer_signature(raw_val[features], model.predict(raw_val[features])),
            input_example=raw_val[features].head(2),
            pip_requirements=['mlflow==2.22.0', 'scikit-learn==1.6.1', 'pandas==2.0.3', 'numpy==1.24.4'])
        version = mlflow.register_model(f'runs:/{run.info.run_id}/model', MODEL_NAME)
        return {**params, **metrics, 'run_id': run.info.run_id, 'version': int(version.version)}

def select_and_evaluate(ctx, rows, engine):
    results = pd.DataFrame(rows).sort_values(
        ['val_f1_macro', 'val_log_loss', 'version'], ascending=[False, True, True])
    assert len(results) == 20 and results.run_id.nunique() == 20
    winner = results.iloc[0]
    version = str(int(winner.version))
    model = mlflow.sklearn.load_model(f'models:/{MODEL_NAME}/{version}')
    test = ctx['raw']['test']
    pred = model.predict(test[ctx['features']])
    test_metrics = {'test_accuracy': accuracy_score(test.target, pred),
                    'test_f1_macro': f1_score(test.target, pred, average='macro')}
    with mlflow.start_run(run_id=winner.run_id):
        mlflow.log_metrics(test_metrics)
        mlflow.log_dict({'ranking': results.to_dict(orient='records')}, 'experiment_ranking.json')
        mlflow.log_dict(classification_report(test.target, pred, output_dict=True, zero_division=0),
                        'test_report.json')
    client = MlflowClient()
    client.set_registered_model_alias(MODEL_NAME, 'champion', version)
    client.set_model_version_tag(MODEL_NAME, version, 'selection', 'max_val_f1_macro_then_min_val_log_loss')
    results['dataset_id'] = ctx['dataset_id']
    with engine.begin() as conn:
        results.to_sql('experiment_results', conn, if_exists='append', index=False)
    return results, {'model_name': MODEL_NAME, 'version': version,
                     'model_uri': f'models:/{MODEL_NAME}@champion', **test_metrics}
