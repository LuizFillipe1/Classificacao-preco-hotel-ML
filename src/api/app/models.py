import io
import logging
import os

import boto3
import joblib
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
from dotenv import load_dotenv

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurar boto3 para acessar o S3
s3_client = boto3.client('s3')

load_dotenv()
# fazer download do model
AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")
MODEL_FILE_KEY = 'modelo/model.joblib'


class ModelLoader:
    def __init__(self):
        self.model = self.load_model_from_s3()

    def load_model_from_s3(self):
        try:
            logger.info(f"Baixando o modelo do bucket {AWS_BUCKET_NAME}, chave {MODEL_FILE_KEY}")
            response = s3_client.get_object(Bucket=AWS_BUCKET_NAME, Key=MODEL_FILE_KEY)
            model_data = response['Body'].read()
            model = joblib.load(io.BytesIO(model_data))
            logger.info("Modelo carregado com sucesso")
            return model
        except s3_client.exceptions.NoSuchKey:
            logger.error("Modelo não encontrado no bucket S3.")
            return None
        except s3_client.exceptions.NoSuchBucket:
            logger.error("Bucket S3 não encontrado.")
            return None
        except NoCredentialsError:
            logger.error("Credenciais AWS não encontradas.")
            return None
        except PartialCredentialsError:
            logger.error("Credenciais AWS incompletas.")
            return None
        except Exception as e:
            logger.error(f"Erro ao baixar o modelo: {e}")
            return None

model_loader = ModelLoader()
model = model_loader.model
