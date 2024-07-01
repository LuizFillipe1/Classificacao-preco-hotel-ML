from app.models import model
from fastapi import HTTPException


class PredictionController:
    @staticmethod
    def predict(features: list):
        if model is None:
            raise HTTPException(status_code=500, detail="Modelo não carregado")

        try:
            # Certificar que a entrada é um array 2D (como esperado pela maioria dos modelos sklearn)
            prediction = model.predict([features])
            # Converter os tipos numpy para tipos nativos do Python
            prediction = prediction.tolist()
            return prediction[0]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Erro ao fazer a predição: {e}")
