"""
Script para verificar e explorar as labels do dataset fonte.
Útil para confirmar que estamos usando a coluna correta.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
from loguru import logger

from src.data_loading.source_loaders import ElectricalFaultLoader


def verify_labels():
    """
    Verifica as colunas de labels disponíveis no dataset e suas distribuições.
    """
    logger.info("=" * 80)
    logger.info("VERIFICAÇÃO DE LABELS DO DATASET FONTE")
    logger.info("=" * 80)
    
    # Carregar dataset
    loader = ElectricalFaultLoader()
    
    try:
        df = loader.load()
    except FileNotFoundError as e:
        logger.error(str(e))
        return
    
    # Listar todas as colunas
    logger.info(f"\n📋 Todas as colunas do dataset:")
    for i, col in enumerate(df.columns, 1):
        logger.info(f"  {i}. {col} ({df[col].dtype})")
    
    # Analisar colunas categóricas relevantes para labels
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    logger.info(f"\n🏷️  Colunas categóricas (possíveis labels):")
    for col in categorical_cols:
        if any(keyword in col.lower() for keyword in ['fault', 'health', 'status', 'type']):
            logger.info(f"\n  📌 {col}:")
            value_counts = df[col].value_counts()
            for value, count in value_counts.items():
                percentage = (count / len(df)) * 100
                logger.info(f"      - {value}: {count} ({percentage:.1f}%)")
    
    # Análise específica de Component Health
    if 'Component Health' in df.columns:
        logger.info("\n" + "=" * 80)
        logger.info("ANÁLISE DETALHADA: Component Health")
        logger.info("=" * 80)
        
        health_col = df['Component Health']
        
        logger.info(f"\n📊 Distribuição completa:")
        for value, count in health_col.value_counts().items():
            percentage = (count / len(df)) * 100
            logger.info(f"  {value:15s}: {count:4d} ({percentage:5.1f}%)")
        
        # Criar labels binárias
        logger.info(f"\n🔄 Conversão para labels binárias:")
        logger.info(f"  Normal       → 0 (Não anomalia)")
        logger.info(f"  Faulty       → 1 (Anomalia)")
        logger.info(f"  Overheated   → 1 (Anomalia)")
        
        binary_labels = (health_col != 'Normal').astype(int)
        
        logger.info(f"\n✅ Resultado:")
        logger.info(f"  Classe 0 (Normal):   {sum(binary_labels == 0):4d} ({sum(binary_labels == 0)/len(df)*100:.1f}%)")
        logger.info(f"  Classe 1 (Anomalia): {sum(binary_labels == 1):4d} ({sum(binary_labels == 1)/len(df)*100:.1f}%)")
        
        # Verificar balanceamento
        imbalance_ratio = max(sum(binary_labels == 0), sum(binary_labels == 1)) / min(sum(binary_labels == 0), sum(binary_labels == 1))
        logger.info(f"\n⚖️  Razão de desbalanceamento: {imbalance_ratio:.2f}:1")
        
        if imbalance_ratio > 3:
            logger.warning("  ⚠️  Dataset desbalanceado! Considere usar:")
            logger.warning("     - Class weights no modelo")
            logger.warning("     - SMOTE para oversampling")
            logger.warning("     - Ajustar contamination adequadamente")
        else:
            logger.info("  ✓ Dataset razoavelmente balanceado")
    
    # Análise de correlação entre Fault Type e Component Health (se ambas existirem)
    if 'Component Health' in df.columns and 'Fault Type' in df.columns:
        logger.info("\n" + "=" * 80)
        logger.info("CORRELAÇÃO: Component Health vs Fault Type")
        logger.info("=" * 80)
        
        cross_tab = pd.crosstab(df['Component Health'], df['Fault Type'], margins=True)
        logger.info("\n📊 Tabela cruzada:")
        logger.info(f"\n{cross_tab}")
        
        logger.info("\n💡 Interpretação:")
        logger.info("  Esta tabela mostra como Component Health se relaciona com Fault Type")
        logger.info("  Útil para entender se as duas colunas capturam informações diferentes")
    
    # Testar carregamento com a nova implementação
    logger.info("\n" + "=" * 80)
    logger.info("TESTE: Carregamento com load_preprocessed()")
    logger.info("=" * 80)
    
    try:
        X, y = loader.load_preprocessed(label_column='Component Health')
        logger.info(f"\n✅ Sucesso!")
        logger.info(f"  Features: {X.shape}")
        logger.info(f"  Labels: {y.shape}")
        logger.info(f"  Distribuição: Normal={sum(y==0)}, Anomalia={sum(y==1)}")
        
        # Mostrar algumas amostras
        logger.info(f"\n📝 Primeiras 5 amostras:")
        sample_df = pd.concat([X.head(), y.head().rename('Label')], axis=1)
        logger.info(f"\n{sample_df}")
        
    except Exception as e:
        logger.error(f"\n❌ Erro ao carregar: {e}")
    
    logger.info("\n" + "=" * 80)
    logger.info("VERIFICAÇÃO CONCLUÍDA")
    logger.info("=" * 80)
    
    # Recomendações
    logger.info("\n💡 RECOMENDAÇÕES:")
    logger.info("  1. Use label_column='Component Health' para treinamento")
    logger.info("  2. Normal = não anomalia (0)")
    logger.info("  3. Faulty + Overheated = anomalia (1)")
    logger.info("  4. Ajuste contamination baseado na proporção real de anomalias")
    
    if 'Component Health' in df.columns:
        anomaly_rate = sum(binary_labels == 1) / len(df)
        logger.info(f"  5. Contamination sugerido: {anomaly_rate:.2f}")


if __name__ == "__main__":
    verify_labels()