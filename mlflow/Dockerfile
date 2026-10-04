FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    mlflow==2.22.0 \
    SQLAlchemy==2.0.40 \
    psycopg2-binary==2.9.10 \
    boto3 \
    numpy==1.24.4 pandas==2.0.3 scikit-learn==1.6.1 \
    fastapi==0.115.12 uvicorn==0.34.2

EXPOSE 5000

CMD ["mlflow", "server", \
     "--host", "0.0.0.0", \
     "--port", "5000", \
     "--backend-store-uri", "postgresql+psycopg2://bdmlflow:bdmlflow@postgres_MLFlow:5432/bdmlflow", \
     "--artifacts-destination", "s3://mlflows3/artifacts", \
     "--serve-artifacts"]