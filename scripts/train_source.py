"""
Script para treinar modelos no dataset fonte (Electrical Fault Detection).
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from loguru import logger

from src.data_loading.source_loaders import ElectricalFaultLoader
from src.preprocessing.scalers import TransferScaler, handle_missing_values
from src.models.isolation_forest import IsolationForestDetector


def train_source_model(
    dataset: str = 'electrical_fault',
    model_type: str = 'isolation_forest',
    contamination: float = 0.1,
    output_dir: str = 'models_exported',
    test_size: float = 0.2
):
    """
    Treina modelo no dataset fonte.
    
    Args:
        dataset: Nome do dataset fonte
        model_type: Tipo de modelo ('isolation_forest', 'kmeans')
        contamination: Proporção esperada de anomalias
        output_dir: Diretório para salvar modelos
        test_size: Proporção dos dados para teste
    """
    logger.info("=" * 60)
    logger.info("TREINAMENTO NO DATASET FONTE")
    logger.info("=" * 60)
    
    # 1. Carregar dados
    logger.info(f"Carregando dataset: {dataset}")
    if dataset == 'electrical_fault':
        loader = ElectricalFaultLoader()
        # Usar Component Health como label (Normal vs Faulty/Overheated)
        X, y = loader.load_preprocessed(label_column='Component Health')
    else:
        raise ValueError(f"Dataset não suportado: {dataset}")
    
    logger.info(f"Dados carregados: {X.shape[0]} amostras, {X.shape[1]} features")
    logger.info(f"Distribuição de labels: Normal={np.sum(y==0)}, Anomalia={np.sum(y==1)}")
    
    # 2. Tratar valores faltantes
    X = handle_missing_values(X, strategy='mean')
    
    # 3. Split treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    logger.info(f"Treino: {X_train.shape[0]} amostras")
    logger.info(f"Teste: {X_test.shape[0]} amostras")
    
    # 4. Normalização
    logger.info("Aplicando normalização...")
    scaler = TransferScaler(method='standard')
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 5. Treinar modelo
    if model_type == 'isolation_forest':
        logger.info("Treinando Isolation Forest...")
        model = IsolationForestDetector(
            n_estimators=100,
            contamination=contamination,
            random_state=42
        )
        model.fit(X_train_scaled)
    else:
        raise ValueError(f"Modelo não suportado: {model_type}")
    
    # 6. Avaliar no conjunto de teste
    logger.info("\n" + "=" * 60)
    logger.info("AVALIAÇÃO NO CONJUNTO DE TESTE")
    logger.info("=" * 60)
    
    y_pred = model.predict(X_test_scaled)
    scores = model.score_samples(X_test_scaled)
    
    # Métricas de classificação
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Anomalia']))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # ROC-AUC usando scores
    try:
        auc = roc_auc_score(y_test, scores)
        logger.info(f"ROC-AUC Score: {auc:.4f}")
    except Exception as e:
        logger.warning(f"Não foi possível calcular ROC-AUC: {e}")
    
    # 7. Salvar modelo e preprocessador
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    model_path = output_path / f"{model_type}_source.pkl"
    scaler_path = output_path / "scaler_source.pkl"
    
    model.save(model_path)
    scaler.save(scaler_path)
    
    logger.info(f"\nModelo salvo em: {model_path}")
    logger.info(f"Scaler salvo em: {scaler_path}")
    
    # 8. Salvar informações sobre features
    import json
    feature_info = {
        'feature_names': list(X.columns),
        'n_features': len(X.columns),
        'model_type': model_type,
        'contamination': contamination
    }
    
    info_path = output_path / "source_model_info.json"
    with open(info_path, 'w') as f:
        json.dump(feature_info, f, indent=2)
    
    logger.info(f"Informações salvas em: {info_path}")
    logger.info("\n✓ Treinamento concluído com sucesso!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Treina modelo de detecção de anomalias no dataset fonte"
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        default='electrical_fault',
        choices=['electrical_fault', 'skab', 'nab'],
        help='Dataset fonte para treinar'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='isolation_forest',
        choices=['isolation_forest', 'kmeans'],
        help='Tipo de modelo'
    )
    
    parser.add_argument(
        '--contamination',
        type=float,
        default=0.1,
        help='Proporção esperada de anomalias (0.0 a 0.5)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='models_exported',
        help='Diretório para salvar modelos'
    )
    
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Proporção dos dados para teste'
    )
    
    args = parser.parse_args()
    
    train_source_model(
        dataset=args.dataset,
        model_type=args.model,
        contamination=args.contamination,
        output_dir=args.output,
        test_size=args.test_size
    )