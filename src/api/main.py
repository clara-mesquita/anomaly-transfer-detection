"""
FastAPI application for anomaly detection model serving.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.isolation_forest import IsolationForestDetector
from src.preprocessing.scalers import TransferScaler


# Inicializar FastAPI app
app = FastAPI(
    title="Anomaly Detection API",
    description="API para detecção de anomalias em dados elétricos usando transferência de conhecimento",
    version="1.0.0"
)


# Modelos globais (carregados na inicialização)
model: Optional[IsolationForestDetector] = None
scaler: Optional[TransferScaler] = None
model_info: Dict[str, Any] = {}


# Schemas Pydantic para validação
class DataRecord(BaseModel):
    """Schema para um registro de dados."""
    Voltage: float = Field(..., description="Tensão em Volts")
    Current: float = Field(..., description="Corrente em Amperes")
    Temperature: float = Field(..., description="Temperatura em Celsius")
    Pressure: Optional[float] = Field(None, description="Pressão (opcional)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "Voltage": 230.5,
                "Current": 15.2,
                "Temperature": 45.3,
                "Pressure": 101.3
            }
        }


class PredictionRequest(BaseModel):
    """Schema para requisição de predição."""
    records: List[DataRecord] = Field(..., description="Lista de registros para análise")
    
    class Config:
        json_schema_extra = {
            "example": {
                "records": [
                    {
                        "Voltage": 230.5,
                        "Current": 15.2,
                        "Temperature": 45.3,
                        "Pressure": 101.3
                    },
                    {
                        "Voltage": 220.0,
                        "Current": 10.5,
                        "Temperature": 40.0
                    }
                ]
            }
        }


class PredictionResponse(BaseModel):
    """Schema para resposta de predição."""
    predictions: List[int] = Field(..., description="0=normal, 1=anomalia")
    anomaly_scores: List[float] = Field(..., description="Scores de anomalia (maior = mais anômalo)")
    n_anomalies: int = Field(..., description="Número de anomalias detectadas")
    
    class Config:
        json_schema_extra = {
            "example": {
                "predictions": [0, 1],
                "anomaly_scores": [0.15, 0.87],
                "n_anomalies": 1
            }
        }


@app.on_event("startup")
async def load_models():
    """
    Carrega modelos e preprocessadores na inicialização da API.
    """
    global model, scaler, model_info
    
    models_dir = Path("models_exported")
    
    try:
        # Carregar modelo
        model_path = models_dir / "isolation_forest_source.pkl"
        if model_path.exists():
            model = IsolationForestDetector.load(model_path)
            logger.info(f"✓ Modelo carregado: {model_path}")
        else:
            logger.warning(f"Modelo não encontrado: {model_path}")
        
        # Carregar scaler
        scaler_path = models_dir / "scaler_source.pkl"
        if scaler_path.exists():
            scaler = TransferScaler.load(scaler_path)
            logger.info(f"✓ Scaler carregado: {scaler_path}")
        else:
            logger.warning(f"Scaler não encontrado: {scaler_path}")
        
        # Carregar informações do modelo
        import json
        info_path = models_dir / "source_model_info.json"
        if info_path.exists():
            with open(info_path, 'r') as f:
                model_info = json.load(f)
            logger.info(f"✓ Informações carregadas: {info_path}")
        
        logger.info("=" * 50)
        logger.info("API inicializada e pronta para receber requisições")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"Erro ao carregar modelos: {e}")
        raise


@app.get("/")
async def root():
    """Endpoint raiz com informações básicas."""
    return {
        "service": "Anomaly Detection API",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": model is not None,
        "endpoints": {
            "health": "/health",
            "info": "/info",
            "predict": "/predict"
        }
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    Verifica se a API está funcionando e se os modelos estão carregados.
    """
    health_status = {
        "status": "healthy" if model is not None and scaler is not None else "unhealthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "model_fitted": model.is_fitted if model is not None else False
    }
    
    if health_status["status"] == "unhealthy":
        raise HTTPException(status_code=503, detail="Modelos não carregados corretamente")
    
    return health_status


@app.get("/info")
async def model_information():
    """
    Retorna informações sobre o modelo carregado.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo não carregado")
    
    return {
        "model_type": model_info.get("model_type", "isolation_forest"),
        "feature_names": model_info.get("feature_names", []),
        "n_features": model_info.get("n_features", 0),
        "contamination": model_info.get("contamination", 0.1),
        "model_params": model.get_params() if hasattr(model, 'get_params') else {},
        "scaler_method": scaler.method if scaler is not None else None,
        "trained_on": "Electrical Fault Detection Dataset (Source Domain)",
        "applicable_to": "Vehicle electrical data (Target Domain)"
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_anomalies(request: PredictionRequest):
    """
    Realiza predição de anomalias para novos dados.
    
    Aceita lista de registros e retorna predições + scores.
    """
    if model is None or scaler is None:
        raise HTTPException(
            status_code=503, 
            detail="Modelo ou scaler não carregados. Verifique /health"
        )
    
    try:
        # Converter records para DataFrame
        records_dict = [record.dict() for record in request.records]
        df = pd.DataFrame(records_dict)
        
        # Usar apenas features que o modelo foi treinado
        expected_features = model_info.get("feature_names", [])
        # Mapear nomes (source tem unidades, target não)
        feature_mapping = {
            'Voltage': 'Voltage (V)',
            'Current': 'Current (A)',
            'Temperature': 'Temperature (°C)'
        }
        
        # Selecionar e renomear features disponíveis
        available_features = []
        df_aligned = pd.DataFrame()
        
        for target_name, source_name in feature_mapping.items():
            if target_name in df.columns and source_name in expected_features:
                df_aligned[source_name] = df[target_name]
                available_features.append(source_name)
        
        if len(df_aligned.columns) == 0:
            raise HTTPException(
                status_code=400,
                detail=f"Nenhuma feature válida encontrada. Esperadas: {list(feature_mapping.keys())}"
            )
        
        # Preencher features faltantes com mediana (se necessário)
        for feat in expected_features:
            if feat not in df_aligned.columns:
                df_aligned[feat] = 0  # ou outro valor padrão
        
        # Garantir ordem correta das colunas
        df_aligned = df_aligned[expected_features]
        
        # Aplicar scaling
        X_scaled = scaler.transform(df_aligned)
        
        # Predição
        predictions = model.predict(X_scaled)
        scores = model.score_samples(X_scaled)
        
        # Preparar resposta
        response = PredictionResponse(
            predictions=predictions.tolist(),
            anomaly_scores=scores.tolist(),
            n_anomalies=int(np.sum(predictions))
        )
        
        logger.info(f"Predição realizada: {len(predictions)} registros, {response.n_anomalies} anomalias")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro na predição: {e}")
        raise HTTPException(status_code=500, detail=f"Erro na predição: {str(e)}")


@app.get("/statistics")
async def get_statistics():
    """
    Retorna estatísticas do scaler (útil para debugging).
    """
    if scaler is None:
        raise HTTPException(status_code=503, detail="Scaler não carregado")
    
    stats = scaler.get_statistics()
    
    # Converter numpy arrays para listas para JSON
    for key, value in stats.items():
        if isinstance(value, np.ndarray):
            stats[key] = value.tolist()
    
    return stats


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )