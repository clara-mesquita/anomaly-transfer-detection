"""
Script para executar experimento de transferência completo.
Compara baseline (treinado no alvo) vs transferência (treinado no fonte).
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import argparse
import numpy as np
import pandas as pd
import json
from loguru import logger

from src.data_loading.source_loaders import ElectricalFaultLoader
from src.data_loading.target_loaders import CANBusLoader
from src.preprocessing.scalers import TransferScaler
from src.preprocessing.feature_alignment import FeatureAligner, create_default_alignment
from src.models.isolation_forest import IsolationForestDetector


def run_transfer_experiment(
    source_dataset: str = 'electrical_fault',
    target_dataset: str = 'can_bus',
    output_dir: str = 'results',
    save_models: bool = True
):
    """
    Executa experimento completo de transferência.
    
    Args:
        source_dataset: Nome do dataset fonte
        target_dataset: Nome do dataset alvo
        output_dir: Diretório para salvar resultados
        save_models: Se True, salva modelos treinados
    """
    logger.info("=" * 80)
    logger.info("EXPERIMENTO DE TRANSFERÊNCIA DE CONHECIMENTO")
    logger.info("=" * 80)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # 1. Carregar dados fonte
    logger.info(f"\n[1/6] Carregando dataset fonte: {source_dataset}")
    source_loader = ElectricalFaultLoader()
    # Usar Component Health como label
    X_source, y_source = source_loader.load_preprocessed(label_column='Component Health')
    
    logger.info(f"Fonte: {X_source.shape[0]} amostras, {X_source.shape[1]} features")
    logger.info(f"Labels: Normal={np.sum(y_source==0)}, Anomalia={np.sum(y_source==1)}")
    
    # 2. Carregar dados alvo
    logger.info(f"\n[2/6] Carregando dataset alvo: {target_dataset}")
    target_loader = CANBusLoader()
    target_df = target_loader.load_preprocessed()
    
    # 3. Alinhar features
    logger.info(f"\n[3/6] Alinhando features entre fonte e alvo")
    
    # Features comuns conhecidas
    common_source = ['Voltage (V)', 'Current (A)', 'Temperature (°C)']
    common_target = ['Voltage', 'Current', 'Temperature']
    
    X_source_aligned = X_source[common_source].copy()
    X_target_aligned = target_df[common_target].copy().dropna()
    
    # Renomear alvo para match com fonte
    X_target_aligned.columns = common_source
    
    logger.info(f"Features alinhadas: {common_source}")
    logger.info(f"Alvo após alinhamento: {X_target_aligned.shape}")
    
    # 4. BASELINE: Treinar diretamente no alvo
    logger.info(f"\n[4/6] Treinando modelo BASELINE (no dataset alvo)")
    
    scaler_baseline = TransferScaler(method='standard')
    X_target_scaled_baseline = scaler_baseline.fit_transform(X_target_aligned)
    
    model_baseline = IsolationForestDetector(
        n_estimators=100,
        contamination=0.1,
        random_state=42
    )
    model_baseline.fit(X_target_scaled_baseline)
    
    predictions_baseline = model_baseline.predict(X_target_scaled_baseline)
    scores_baseline = model_baseline.score_samples(X_target_scaled_baseline)
    
    results['baseline'] = {
        'n_anomalies': int(np.sum(predictions_baseline)),
        'anomaly_rate': float(np.mean(predictions_baseline)),
        'score_mean': float(np.mean(scores_baseline)),
        'score_std': float(np.std(scores_baseline)),
        'score_min': float(np.min(scores_baseline)),
        'score_max': float(np.max(scores_baseline))
    }
    
    logger.info(f"Baseline - Anomalias: {results['baseline']['n_anomalies']} "
                f"({results['baseline']['anomaly_rate']*100:.2f}%)")
    
    # 5. TRANSFERÊNCIA: Treinar no fonte, aplicar no alvo
    logger.info(f"\n[5/6] Treinando modelo TRANSFERÊNCIA (no dataset fonte)")
    
    scaler_transfer = TransferScaler(method='standard')
    X_source_scaled = scaler_transfer.fit_transform(X_source_aligned)
    
    model_transfer = IsolationForestDetector(
        n_estimators=100,
        contamination=0.3,  # Maior porque fonte tem mais anomalias rotuladas
        random_state=42
    )
    model_transfer.fit(X_source_scaled)
    
    # Aplicar no alvo
    X_target_scaled_transfer = scaler_transfer.transform(X_target_aligned)
    predictions_transfer = model_transfer.predict(X_target_scaled_transfer)
    scores_transfer = model_transfer.score_samples(X_target_scaled_transfer)
    
    results['transfer'] = {
        'n_anomalies': int(np.sum(predictions_transfer)),
        'anomaly_rate': float(np.mean(predictions_transfer)),
        'score_mean': float(np.mean(scores_transfer)),
        'score_std': float(np.std(scores_transfer)),
        'score_min': float(np.min(scores_transfer)),
        'score_max': float(np.max(scores_transfer))
    }
    
    logger.info(f"Transferência - Anomalias: {results['transfer']['n_anomalies']} "
                f"({results['transfer']['anomaly_rate']*100:.2f}%)")
    
    # 6. Comparação e avaliação
    logger.info(f"\n[6/6] Comparando resultados")
    
    # Overlap de anomalias detectadas
    overlap = np.sum((predictions_baseline == 1) & (predictions_transfer == 1))
    results['comparison'] = {
        'overlap_anomalies': int(overlap),
        'overlap_rate': float(overlap / max(np.sum(predictions_baseline), 1)),
        'score_correlation': float(np.corrcoef(scores_baseline, scores_transfer)[0, 1])
    }
    
    logger.info(f"Overlap de anomalias: {overlap} "
                f"({results['comparison']['overlap_rate']*100:.2f}%)")
    logger.info(f"Correlação de scores: {results['comparison']['score_correlation']:.4f}")
    
    # Verificar se há labels no alvo para avaliação quantitativa
    if target_loader.has_labels():
        logger.info("\nAvaliando com labels do dataset alvo...")
        y_target = target_loader.get_labels().loc[X_target_aligned.index]
        
        # Binarizar se necessário
        if y_target.nunique() > 2:
            mode_value = y_target.mode()[0]
            y_target_binary = (y_target != mode_value).astype(int)
        else:
            y_target_binary = y_target
        
        from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
        
        # Métricas baseline
        results['baseline']['precision'] = float(precision_score(y_target_binary, predictions_baseline, zero_division=0))
        results['baseline']['recall'] = float(recall_score(y_target_binary, predictions_baseline, zero_division=0))
        results['baseline']['f1'] = float(f1_score(y_target_binary, predictions_baseline, zero_division=0))
        
        try:
            results['baseline']['roc_auc'] = float(roc_auc_score(y_target_binary, scores_baseline))
        except:
            results['baseline']['roc_auc'] = None
        
        # Métricas transferência
        results['transfer']['precision'] = float(precision_score(y_target_binary, predictions_transfer, zero_division=0))
        results['transfer']['recall'] = float(recall_score(y_target_binary, predictions_transfer, zero_division=0))
        results['transfer']['f1'] = float(f1_score(y_target_binary, predictions_transfer, zero_division=0))
        
        try:
            results['transfer']['roc_auc'] = float(roc_auc_score(y_target_binary, scores_transfer))
        except:
            results['transfer']['roc_auc'] = None
        
        logger.info(f"\nBaseline - F1: {results['baseline']['f1']:.4f}, "
                    f"Precision: {results['baseline']['precision']:.4f}, "
                    f"Recall: {results['baseline']['recall']:.4f}")
        logger.info(f"Transfer - F1: {results['transfer']['f1']:.4f}, "
                    f"Precision: {results['transfer']['precision']:.4f}, "
                    f"Recall: {results['transfer']['recall']:.4f}")
    
    # Salvar resultados
    results_file = output_path / 'transfer_experiment_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n✓ Resultados salvos em: {results_file}")
    
    # Salvar modelos se solicitado
    if save_models:
        logger.info("\nSalvando modelos...")
        model_baseline.save(output_path / 'model_baseline.pkl')
        model_transfer.save(output_path / 'model_transfer.pkl')
        scaler_baseline.save(output_path / 'scaler_baseline.pkl')
        scaler_transfer.save(output_path / 'scaler_transfer.pkl')
        logger.info("✓ Modelos salvos")
    
    logger.info("\n" + "=" * 80)
    logger.info("EXPERIMENTO CONCLUÍDO COM SUCESSO")
    logger.info("=" * 80)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Executa experimento de transferência de conhecimento"
    )
    
    parser.add_argument(
        '--source',
        type=str,
        default='electrical_fault',
        help='Dataset fonte'
    )
    
    parser.add_argument(
        '--target',
        type=str,
        default='can_bus',
        help='Dataset alvo'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='results',
        help='Diretório para resultados'
    )
    
    parser.add_argument(
        '--no-save-models',
        action='store_true',
        help='Não salvar modelos treinados'
    )
    
    args = parser.parse_args()
    
    run_transfer_experiment(
        source_dataset=args.source,
        target_dataset=args.target,
        output_dir=args.output,
        save_models=not args.no_save_models
    )