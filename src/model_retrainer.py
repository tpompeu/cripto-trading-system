"""
Sistema de Trading Quantitativo - Retreinador de Modelos
Versão: 2.01
Data: 26 de agosto de 2025

Este módulo é responsável pela Fase de Manutenção:
- Retreino simples de modelos existentes com dados atualizados
- Mantém os mesmos parâmetros de estratégia que tornaram o modelo bem-sucedido
- Execução mais leve e rápida que a validação completa
- Adaptação às condições de mercado recentes sem reotimização completa
"""

import json
import logging
import os
import sys
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import lightgbm as lgb
import talib
import boto3
from botocore.exceptions import ClientError

# Importa serviço de configuração
from config_service import ConfigService

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelRetrainer:
    """
    Retreinador de modelos para estratégias em produção.
    
    Executa retreino eficiente mantendo os parâmetros ótimos descobertos
    no Walk-Forward original, mas atualizando o modelo com dados recentes.
    """
    
    def __init__(self, strategy_id: str):
        """
        Inicializa o retreinador de modelos.
        
        Args:
            strategy_id: ID da estratégia a ser retreinada
        """
        self.strategy_id = strategy_id
        
        # Serviços AWS
        self.s3 = boto3.client('s3')
        self.config_service = ConfigService()
        
        # Configurações
        self.config = self.config_service.get_model_config()
        self.trading_config = self.config_service.get_trading_config()
        
        logger.info(f"ModelRetrainer inicializado para {strategy_id}")
    
    def load_strategy_data(self) -> Dict[str, Any]:
        """
        Carrega dados da estratégia do DynamoDB.
        
        Returns:
            Dados completos da estratégia
        """
        try:
            strategy_data = self.config_service.get_strategy_state(self.strategy_id)
            
            if not strategy_data:
                raise ValueError(f"Estratégia {self.strategy_id} não encontrada")
            
            if strategy_data.get('status') != 'production':
                raise ValueError(f"Estratégia {self.strategy_id} não está em produção")
            
            if 'validation_results' not in strategy_data:
                raise ValueError(f"Parâmetros ótimos não encontrados para {self.strategy_id}")
            
            logger.info(f"Dados da estratégia carregados: {strategy_data['symbol']}_{strategy_data['timeframe']}")
            return strategy_data
            
        except Exception as e:
            logger.error(f"Erro ao carregar dados da estratégia: {str(e)}")
            raise
    
    def load_market_data(self, symbol: str, timeframe: str, months_back: int = 15) -> pd.DataFrame:
        """
        Carrega dados de mercado recentes do S3.
        
        Args:
            symbol: Par de trading
            timeframe: Timeframe
            months_back: Meses de dados históricos para carregar
            
        Returns:
            DataFrame com dados OHLCV
        """
        try:
            data_key = f"market_data/{symbol}/{timeframe}/data.parquet"
            
            response = self.s3.get_object(
                Bucket=self.trading_config['data_bucket'],
                Key=data_key
            )
            
            df = pd.read_parquet(response['Body'])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp').sort_index()
            
            # Filtra apenas dados recentes
            cutoff_date = datetime.now() - timedelta(days=months_back * 30)
            df = df[df.index >= cutoff_date]
            
            logger.info(f"Dados de mercado carregados: {len(df)} registros de {df.index[0]} a {df.index[-1]}")
            return df
            
        except Exception as e:
            logger.error(f"Erro ao carregar dados de mercado: {str(e)}")
            raise
    
    def generate_features(self, df: pd.DataFrame, optimal_params: Dict[str, Any]) -> pd.DataFrame:
        """
        Gera features usando os parâmetros ótimos descobertos.
        
        Args:
            df: DataFrame com dados OHLCV
            optimal_params: Parâmetros ótimos da estratégia
            
        Returns:
            DataFrame com features
        """
        try:
            # Importa o FeatureEngineer do model_validator
            from model_validator import FeatureEngineer
            
            feature_engineer = FeatureEngineer(self.config)
            features = feature_engineer.generate_features(df, optimal_params)
            
            logger.info(f"Features geradas: {features.shape[1]} colunas")
            return features
            
        except Exception as e:
            logger.error(f"Erro ao gerar features: {str(e)}")
            raise
    
    def generate_targets(self, df: pd.DataFrame) -> pd.Series:
        """
        Gera variável alvo usando a mesma lógica do treinamento original.
        
        Args:
            df: DataFrame com dados OHLCV
            
        Returns:
            Série com targets
        """
        try:
            # Importa o TargetGenerator do model_validator
            from model_validator import TargetGenerator
            
            target_generator = TargetGenerator(self.config)
            targets = target_generator.calculate_target(df)
            
            logger.info(f"Targets gerados: {len(targets)} registros")
            return targets
            
        except Exception as e:
            logger.error(f"Erro ao gerar targets: {str(e)}")
            raise
    
    def load_previous_model(self, strategy_data: Dict[str, Any]) -> Optional[lgb.Booster]:
        """
        Carrega modelo anterior do S3 (se existir).
        
        Args:
            strategy_data: Dados da estratégia
            
        Returns:
            Modelo anterior ou None
        """
        try:
            model_key = f"models/{self.strategy_id}/latest_model.pkl"
            
            response = self.s3.get_object(
                Bucket=self.trading_config['data_bucket'],
                Key=model_key
            )
            
            model_data = pickle.loads(response['Body'].read())
            model = model_data.get('model')
            
            if model:
                logger.info("Modelo anterior carregado com sucesso")
                return model
            
            return None
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                logger.info("Nenhum modelo anterior encontrado")
                return None
            else:
                logger.error(f"Erro ao carregar modelo anterior: {str(e)}")
                return None
        except Exception as e:
            logger.error(f"Erro inesperado ao carregar modelo: {str(e)}")
            return None
    
    def train_updated_model(self, X_train: pd.DataFrame, y_train: pd.Series, 
                           previous_model: Optional[lgb.Booster] = None) -> lgb.Booster:
        """
        Treina modelo atualizado com dados recentes.
        
        Args:
            X_train: Features de treino
            y_train: Targets de treino
            previous_model: Modelo anterior (para warm start se disponível)
            
        Returns:
            Modelo treinado
        """
        try:
            # Parâmetros do modelo (mesmos usados na validação original)
            model_params = {
                'objective': 'multiclass',
                'num_class': 3,
                'metric': 'multi_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'random_state': 42
            }
            
            # Prepara dados
            train_data = lgb.Dataset(X_train, label=y_train)
            
            # Treina modelo
            if previous_model and hasattr(previous_model, 'num_trees'):
                # Continua treinamento do modelo anterior (warm start)
                logger.info("Continuando treinamento do modelo anterior")
                model = lgb.train(
                    model_params,
                    train_data,
                    num_boost_round=100,  # Adiciona mais árvores
                    init_model=previous_model,
                    callbacks=[lgb.log_evaluation(0)]
                )
            else:
                # Treina novo modelo do zero
                logger.info("Treinando novo modelo do zero")
                model = lgb.train(
                    model_params,
                    train_data,
                    num_boost_round=500,
                    callbacks=[lgb.log_evaluation(0)]
                )
            
            logger.info(f"Modelo treinado com {model.num_trees()} árvores")
            return model
            
        except Exception as e:
            logger.error(f"Erro ao treinar modelo: {str(e)}")
            raise
    
    def validate_model_performance(self, model: lgb.Booster, X_val: pd.DataFrame, 
                                 y_val: pd.Series) -> Dict[str, Any]:
        """
        Valida performance do modelo retreinado.
        
        Args:
            model: Modelo treinado
            X_val: Features de validação
            y_val: Targets de validação
            
        Returns:
            Métricas de performance
        """
        try:
            # Predições
            y_pred = model.predict(X_val)
            y_pred_class = np.argmax(y_pred.reshape(-1, 3), axis=1) - 1  # -1, 0, 1
            
            # Métricas básicas
            accuracy = np.mean(y_pred_class == y_val)
            
            # Calcula expectância simulada
            expectancy = self._calculate_expectancy(y_val, y_pred_class)
            
            # Métricas por classe
            unique_classes = np.unique(y_val)
            class_metrics = {}
            
            for cls in unique_classes:
                if cls in [-1, 0, 1]:
                    mask = y_val == cls
                    if np.sum(mask) > 0:
                        class_acc = np.mean(y_pred_class[mask] == y_val[mask])
                        class_metrics[f'class_{int(cls)}_accuracy'] = class_acc
            
            metrics = {
                'accuracy': accuracy,
                'expectancy': expectancy,
                'total_predictions': len(y_val),
                'trading_signals': len(y_val[y_val != 0]),
                'class_metrics': class_metrics,
                'feature_importance': dict(zip(X_val.columns, model.feature_importance()))
            }
            
            logger.info(f"Performance do modelo: Accuracy={accuracy:.3f}, Expectancy={expectancy:.4f}")
            return metrics
            
        except Exception as e:
            logger.error(f"Erro ao validar modelo: {str(e)}")
            return {'accuracy': 0, 'expectancy': -999}
    
    def _calculate_expectancy(self, y_true: pd.Series, y_pred: np.ndarray) -> float:
        """
        Calcula expectância baseada em R-multiples simulados.
        
        Args:
            y_true: Targets verdadeiros
            y_pred: Predições do modelo
            
        Returns:
            Expectância em R-multiples
        """
        try:
            results = []
            
            for true_val, pred_val in zip(y_true, y_pred):
                if pred_val == 0:  # Não opera
                    continue
                
                if true_val == pred_val:  # Predição correta
                    if pred_val != 0:  # Operação executada
                        results.append(3.0)  # +3R
                else:  # Predição incorreta
                    if pred_val != 0:  # Operação executada
                        results.append(-1.0)  # -1R
            
            if len(results) == 0:
                return 0.0
            
            expectancy = np.mean(results)
            return expectancy
            
        except Exception as e:
            logger.error(f"Erro ao calcular expectância: {str(e)}")
            return 0.0
    
    def save_updated_model(self, model: lgb.Booster, metrics: Dict[str, Any], 
                          optimal_params: Dict[str, Any]):
        """
        Salva modelo atualizado no S3.
        
        Args:
            model: Modelo treinado
            metrics: Métricas de performance
            optimal_params: Parâmetros ótimos da estratégia
        """
        try:
            # Dados do modelo
            model_data = {
                'model': model,
                'strategy_id': self.strategy_id,
                'retrain_timestamp': datetime.now().isoformat(),
                'metrics': metrics,
                'optimal_params': optimal_params,
                'model_version': f"retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            }
            
            # Serializa modelo
            model_bytes = pickle.dumps(model_data)
            
            # Salva modelo atual
            current_model_key = f"models/{self.strategy_id}/latest_model.pkl"
            self.s3.put_object(
                Bucket=self.trading_config['data_bucket'],
                Key=current_model_key,
                Body=model_bytes
            )
            
            # Salva backup com timestamp
            backup_model_key = f"models/{self.strategy_id}/backups/model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            self.s3.put_object(
                Bucket=self.trading_config['data_bucket'],
                Key=backup_model_key,
                Body=model_bytes
            )
            
            logger.info("Modelo atualizado salvo com sucesso")
            
        except Exception as e:
            logger.error(f"Erro ao salvar modelo: {str(e)}")
            raise
    
    def update_strategy_metadata(self, metrics: Dict[str, Any]):
        """
        Atualiza metadados da estratégia no DynamoDB.
        
        Args:
            metrics: Métricas do modelo retreinado
        """
        try:
            updates = {
                'last_retrain_date': datetime.now().isoformat(),
                'retrain_metrics': {
                    'accuracy': metrics['accuracy'],
                    'expectancy': metrics['expectancy'],
                    'total_predictions': metrics['total_predictions'],
                    'trading_signals': metrics['trading_signals']
                },
                'model_version': f"retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            }
            
            success = self.config_service.update_strategy_state(self.strategy_id, updates)
            
            if success:
                logger.info("Metadados da estratégia atualizados")
            else:
                logger.error("Falha ao atualizar metadados da estratégia")
            
        except Exception as e:
            logger.error(f"Erro ao atualizar metadados: {str(e)}")
    
    def run_retraining(self) -> Dict[str, Any]:
        """
        Executa processo completo de retreino.
        
        Returns:
            Resultados do retreino
        """
        try:
            logger.info("=== INÍCIO DO RETREINO DE MODELO ===")
            
            # Carrega dados da estratégia
            strategy_data = self.load_strategy_data()
            symbol = strategy_data['symbol']
            timeframe = strategy_data['timeframe']
            optimal_params = strategy_data['validation_results']['best_params']
            
            # Carrega dados de mercado recentes
            df = self.load_market_data(symbol, timeframe)
            
            if len(df) < 1000:  # Precisa de dados suficientes
                raise ValueError("Dados insuficientes para retreino")
            
            # Gera features e targets
            features = self.generate_features(df, optimal_params)
            targets = self.generate_targets(df)
            
            # Alinha dados
            common_idx = features.index.intersection(targets.index)
            X = features.loc[common_idx]
            y = targets.loc[common_idx]
            
            # Remove linhas com NaN
            mask = ~(X.isna().any(axis=1) | y.isna())
            X = X[mask]
            y = y[mask]
            
            if len(X) < 500:
                raise ValueError("Dados válidos insuficientes após limpeza")
            
            # Divide dados (80% treino, 20% validação)
            split_idx = int(len(X) * 0.8)
            X_train = X.iloc[:split_idx]
            y_train = y.iloc[:split_idx]
            X_val = X.iloc[split_idx:]
            y_val = y.iloc[split_idx:]
            
            # Carrega modelo anterior
            previous_model = self.load_previous_model(strategy_data)
            
            # Treina modelo atualizado
            updated_model = self.train_updated_model(X_train, y_train, previous_model)
            
            # Valida performance
            metrics = self.validate_model_performance(updated_model, X_val, y_val)
            
            # Verifica se performance é aceitável
            if metrics['expectancy'] < -0.5:  # Threshold de degradação
                logger.warning(f"Performance degradada detectada: {metrics['expectancy']}")
                # Em produção, poderia reverter para modelo anterior ou acionar Gravedigger
            
            # Salva modelo atualizado
            self.save_updated_model(updated_model, metrics, optimal_params)
            
            # Atualiza metadados
            self.update_strategy_metadata(metrics)
            
            results = {
                'success': True,
                'strategy_id': self.strategy_id,
                'retrain_timestamp': datetime.now().isoformat(),
                'data_period': f"{df.index[0]} to {df.index[-1]}",
                'training_samples': len(X_train),
                'validation_samples': len(X_val),
                'metrics': metrics,
                'performance_acceptable': metrics['expectancy'] >= -0.5
            }
            
            logger.info("=== RETREINO DE MODELO CONCLUÍDO ===")
            return results
            
        except Exception as e:
            logger.error(f"Erro crítico no retreino: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'strategy_id': self.strategy_id,
                'timestamp': datetime.now().isoformat()
            }


def lambda_handler(event, context):
    """
    Função principal para execução no AWS Lambda.
    
    Args:
        event: Evento do Lambda
        context: Contexto de execução do Lambda
        
    Returns:
        Resultado do retreino
    """
    try:
        logger.info("=== INÍCIO DA EXECUÇÃO DO MODEL RETRAINER ===")
        logger.info(f"Event: {json.dumps(event, default=str)}")
        
        # Obtém ID da estratégia do evento
        strategy_id = event.get('strategy_id', '')
        
        if not strategy_id:
            raise ValueError("strategy_id não fornecido no evento")
        
        # Executa retreino
        retrainer = ModelRetrainer(strategy_id)
        results = retrainer.run_retraining()
        
        # Prepara resposta
        response = {
            'statusCode': 200 if results['success'] else 500,
            'body': json.dumps(results, default=str),
            'headers': {
                'Content-Type': 'application/json'
            }
        }
        
        logger.info("=== FIM DA EXECUÇÃO DO MODEL RETRAINER ===")
        return response
        
    except Exception as e:
        error_msg = f"Erro crítico no Lambda handler: {str(e)}"
        logger.error(error_msg)
        
        return {
            'statusCode': 500,
            'body': json.dumps({
                'success': False,
                'error': error_msg,
                'timestamp': datetime.now().isoformat()
            }),
            'headers': {
                'Content-Type': 'application/json'
            }
        }


def main():
    """Função principal para execução local/teste."""
    try:
        # Obtém parâmetros do ambiente ou argumentos
        strategy_id = os.environ.get('STRATEGY_ID', sys.argv[1] if len(sys.argv) > 1 else '')
        
        if not strategy_id:
            logger.error("strategy_id não fornecido")
            sys.exit(1)
        
        logger.info(f"Iniciando retreino para {strategy_id}")
        
        # Executa retreino
        retrainer = ModelRetrainer(strategy_id)
        results = retrainer.run_retraining()
        
        if results['success']:
            logger.info("Retreino concluído com sucesso")
        else:
            logger.error(f"Retreino falhou: {results.get('error', 'Erro desconhecido')}")
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"Erro crítico no retreino: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()

