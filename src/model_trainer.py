"""
Sistema de Trading Quantitativo - Model Trainer
Versão: 2.01
Descrição: Treinamento do modelo LightGBM com otimização de hiperparâmetros
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from typing import Dict, List, Tuple, Optional, Any
import logging
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import ParameterGrid
import optuna
import boto3
import joblib
import json
import os
from datetime import datetime
import awswrangler as wr

logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Classe responsável pelo treinamento do modelo LightGBM.
    Inclui otimização de hiperparâmetros e validação do modelo.
    """
    
    def __init__(self, config: Dict):
        """
        Inicializa o ModelTrainer.
        
        Args:
            config: Configurações do sistema
        """
        self.config = config
        self.model_config = config.get('model_config', {})
        self.s3_client = boto3.client('s3')
        self.bucket_name = config.get('data_bucket_name')
        
        # Configurações padrão do LightGBM
        self.default_params = {
            'objective': 'multiclass',
            'num_class': 3,  # -1, 0, 1
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Espaço de busca para otimização
        self.param_space = {
            'num_leaves': [15, 31, 63, 127],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'feature_fraction': [0.7, 0.8, 0.9, 1.0],
            'bagging_fraction': [0.7, 0.8, 0.9, 1.0],
            'min_data_in_leaf': [10, 20, 50, 100],
            'lambda_l1': [0, 0.1, 0.5, 1.0],
            'lambda_l2': [0, 0.1, 0.5, 1.0]
        }
        
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                   X_val: pd.DataFrame, y_val: pd.Series,
                   params: Optional[Dict] = None) -> Tuple[lgb.LGBMClassifier, Dict]:
        """
        Treina um modelo LightGBM.
        
        Args:
            X_train: Features de treinamento
            y_train: Target de treinamento
            X_val: Features de validação
            y_val: Target de validação
            params: Parâmetros do modelo (opcional)
            
        Returns:
            Tuple com modelo treinado e métricas
        """
        logger.info("Iniciando treinamento do modelo LightGBM")
        
        # Usa parâmetros padrão se não fornecidos
        if params is None:
            params = self.default_params.copy()
        
        # Converte targets para formato adequado (0, 1, 2 ao invés de -1, 0, 1)
        y_train_encoded = self._encode_target(y_train)
        y_val_encoded = self._encode_target(y_val)
        
        # Cria datasets LightGBM
        train_data = lgb.Dataset(X_train, label=y_train_encoded)
        val_data = lgb.Dataset(X_val, label=y_val_encoded, reference=train_data)
        
        # Configurações de treinamento
        train_params = params.copy()
        train_params.update({
            'num_boost_round': self.model_config.get('max_iterations', 1000),
            'valid_sets': [train_data, val_data],
            'valid_names': ['train', 'valid'],
            'early_stopping_rounds': self.model_config.get('early_stopping_rounds', 100),
            'verbose_eval': False
        })
        
        # Treina o modelo
        model = lgb.train(
            params=train_params,
            train_set=train_data,
            callbacks=[lgb.log_evaluation(0)]  # Silencia logs
        )
        
        # Avalia o modelo
        metrics = self._evaluate_model(model, X_train, y_train_encoded, 
                                     X_val, y_val_encoded)
        
        # Converte para LGBMClassifier para compatibilidade
        lgb_classifier = lgb.LGBMClassifier(**params)
        lgb_classifier._Booster = model
        lgb_classifier.classes_ = np.array([0, 1, 2])
        lgb_classifier.n_classes_ = 3
        
        logger.info(f"Modelo treinado. Acurácia validação: {metrics['val_accuracy']:.4f}")
        return lgb_classifier, metrics
    
    def optimize_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series,
                                X_val: pd.DataFrame, y_val: pd.Series,
                                optimization_method: str = 'optuna',
                                n_trials: int = 100) -> Tuple[Dict, float]:
        """
        Otimiza hiperparâmetros do modelo.
        
        Args:
            X_train: Features de treinamento
            y_train: Target de treinamento
            X_val: Features de validação
            y_val: Target de validação
            optimization_method: Método de otimização ('optuna' ou 'grid')
            n_trials: Número de trials para Optuna
            
        Returns:
            Tuple com melhores parâmetros e score
        """
        logger.info(f"Iniciando otimização de hiperparâmetros usando {optimization_method}")
        
        if optimization_method == 'optuna':
            return self._optimize_with_optuna(X_train, y_train, X_val, y_val, n_trials)
        elif optimization_method == 'grid':
            return self._optimize_with_grid_search(X_train, y_train, X_val, y_val)
        else:
            raise ValueError(f"Método de otimização não suportado: {optimization_method}")
    
    def _optimize_with_optuna(self, X_train: pd.DataFrame, y_train: pd.Series,
                             X_val: pd.DataFrame, y_val: pd.Series,
                             n_trials: int) -> Tuple[Dict, float]:
        """Otimização usando Optuna"""
        
        def objective(trial):
            # Define espaço de busca
            params = self.default_params.copy()
            params.update({
                'num_leaves': trial.suggest_categorical('num_leaves', self.param_space['num_leaves']),
                'learning_rate': trial.suggest_categorical('learning_rate', self.param_space['learning_rate']),
                'feature_fraction': trial.suggest_categorical('feature_fraction', self.param_space['feature_fraction']),
                'bagging_fraction': trial.suggest_categorical('bagging_fraction', self.param_space['bagging_fraction']),
                'min_data_in_leaf': trial.suggest_categorical('min_data_in_leaf', self.param_space['min_data_in_leaf']),
                'lambda_l1': trial.suggest_categorical('lambda_l1', self.param_space['lambda_l1']),
                'lambda_l2': trial.suggest_categorical('lambda_l2', self.param_space['lambda_l2'])
            })
            
            try:
                # Treina modelo com parâmetros sugeridos
                model, metrics = self.train_model(X_train, y_train, X_val, y_val, params)
                
                # Retorna métrica a ser otimizada (acurácia de validação)
                return metrics['val_accuracy']
                
            except Exception as e:
                logger.warning(f"Erro no trial {trial.number}: {str(e)}")
                return 0.0
        
        # Cria estudo Optuna
        study = optuna.create_study(direction='maximize', 
                                   sampler=optuna.samplers.TPESampler(seed=42))
        
        # Executa otimização
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        best_params = self.default_params.copy()
        best_params.update(study.best_params)
        
        logger.info(f"Melhor score: {study.best_value:.4f}")
        return best_params, study.best_value
    
    def _optimize_with_grid_search(self, X_train: pd.DataFrame, y_train: pd.Series,
                                  X_val: pd.DataFrame, y_val: pd.Series) -> Tuple[Dict, float]:
        """Otimização usando Grid Search"""
        
        # Cria grid reduzido para evitar explosão combinatória
        reduced_param_space = {
            'num_leaves': [31, 63],
            'learning_rate': [0.05, 0.1],
            'feature_fraction': [0.8, 0.9],
            'bagging_fraction': [0.8, 0.9],
            'min_data_in_leaf': [20, 50]
        }
        
        best_score = 0.0
        best_params = self.default_params.copy()
        
        # Gera todas as combinações
        param_grid = ParameterGrid(reduced_param_space)
        total_combinations = len(param_grid)
        
        logger.info(f"Testando {total_combinations} combinações de parâmetros")
        
        for i, params in enumerate(param_grid):
            try:
                # Atualiza parâmetros base
                current_params = self.default_params.copy()
                current_params.update(params)
                
                # Treina modelo
                model, metrics = self.train_model(X_train, y_train, X_val, y_val, current_params)
                
                # Verifica se é o melhor score
                if metrics['val_accuracy'] > best_score:
                    best_score = metrics['val_accuracy']
                    best_params = current_params.copy()
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Progresso: {i+1}/{total_combinations} - Melhor score: {best_score:.4f}")
                    
            except Exception as e:
                logger.warning(f"Erro na combinação {i+1}: {str(e)}")
                continue
        
        logger.info(f"Grid search concluído. Melhor score: {best_score:.4f}")
        return best_params, best_score
    
    def _encode_target(self, y: pd.Series) -> np.ndarray:
        """
        Converte target de (-1, 0, 1) para (0, 1, 2).
        
        Args:
            y: Target original
            
        Returns:
            Target codificado
        """
        mapping = {-1: 0, 0: 1, 1: 2}
        return y.map(mapping).values
    
    def _decode_target(self, y_encoded: np.ndarray) -> np.ndarray:
        """
        Converte target de (0, 1, 2) para (-1, 0, 1).
        
        Args:
            y_encoded: Target codificado
            
        Returns:
            Target original
        """
        mapping = {0: -1, 1: 0, 2: 1}
        return np.array([mapping[val] for val in y_encoded])
    
    def _evaluate_model(self, model: lgb.Booster, 
                       X_train: pd.DataFrame, y_train: np.ndarray,
                       X_val: pd.DataFrame, y_val: np.ndarray) -> Dict:
        """
        Avalia o modelo treinado.
        
        Args:
            model: Modelo LightGBM treinado
            X_train: Features de treinamento
            y_train: Target de treinamento (codificado)
            X_val: Features de validação
            y_val: Target de validação (codificado)
            
        Returns:
            Dicionário com métricas
        """
        # Predições
        train_pred = model.predict(X_train, num_iteration=model.best_iteration)
        val_pred = model.predict(X_val, num_iteration=model.best_iteration)
        
        # Converte probabilidades para classes
        train_pred_class = np.argmax(train_pred, axis=1)
        val_pred_class = np.argmax(val_pred, axis=1)
        
        # Calcula métricas
        metrics = {
            'train_accuracy': float(np.mean(train_pred_class == y_train)),
            'val_accuracy': float(np.mean(val_pred_class == y_val)),
            'best_iteration': int(model.best_iteration),
            'feature_importance': self._get_feature_importance(model, X_train.columns),
            'confusion_matrix': confusion_matrix(y_val, val_pred_class).tolist(),
            'classification_report': classification_report(y_val, val_pred_class, 
                                                         target_names=['Sell', 'Neutral', 'Buy'],
                                                         output_dict=True)
        }
        
        # Calcula AUC para cada classe (one-vs-rest)
        try:
            auc_scores = {}
            for i in range(3):
                y_binary = (y_val == i).astype(int)
                if len(np.unique(y_binary)) > 1:  # Verifica se há ambas as classes
                    auc_scores[f'auc_class_{i}'] = roc_auc_score(y_binary, val_pred[:, i])
            metrics['auc_scores'] = auc_scores
        except Exception as e:
            logger.warning(f"Erro ao calcular AUC: {str(e)}")
            metrics['auc_scores'] = {}
        
        return metrics
    
    def _get_feature_importance(self, model: lgb.Booster, 
                               feature_names: List[str]) -> Dict:
        """
        Obtém importância das features.
        
        Args:
            model: Modelo treinado
            feature_names: Nomes das features
            
        Returns:
            Dicionário com importância das features
        """
        importance = model.feature_importance(importance_type='gain')
        
        # Cria dicionário ordenado por importância
        feature_importance = dict(zip(feature_names, importance))
        feature_importance = dict(sorted(feature_importance.items(), 
                                       key=lambda x: x[1], reverse=True))
        
        return feature_importance
    
    def save_model(self, model: lgb.LGBMClassifier, metrics: Dict,
                   symbol: str, timeframe: str, 
                   split_id: Optional[int] = None) -> str:
        """
        Salva modelo treinado no S3.
        
        Args:
            model: Modelo treinado
            metrics: Métricas do modelo
            symbol: Par de trading
            timeframe: Timeframe
            split_id: ID da divisão (para Walk-Forward)
            
        Returns:
            Path do modelo salvo
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if split_id is not None:
            filename = f"model_{symbol}_{timeframe}_split{split_id}_{timestamp}"
        else:
            filename = f"model_{symbol}_{timeframe}_{timestamp}"
        
        # Salva modelo
        model_path = f"/tmp/{filename}.joblib"
        joblib.dump(model, model_path)
        
        # Salva métricas
        metrics_path = f"/tmp/{filename}_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        
        try:
            # Upload modelo para S3
            model_s3_key = f"models/{filename}.joblib"
            self.s3_client.upload_file(model_path, self.bucket_name, model_s3_key)
            
            # Upload métricas para S3
            metrics_s3_key = f"models/{filename}_metrics.json"
            self.s3_client.upload_file(metrics_path, self.bucket_name, metrics_s3_key)
            
            # Remove arquivos temporários
            os.remove(model_path)
            os.remove(metrics_path)
            
            model_s3_path = f"s3://{self.bucket_name}/{model_s3_key}"
            logger.info(f"Modelo salvo em: {model_s3_path}")
            return model_s3_path
            
        except Exception as e:
            logger.error(f"Erro ao salvar modelo: {str(e)}")
            raise
    
    def load_model(self, model_path: str) -> lgb.LGBMClassifier:
        """
        Carrega modelo do S3.
        
        Args:
            model_path: Path do modelo no S3
            
        Returns:
            Modelo carregado
        """
        try:
            # Parse S3 path
            bucket = model_path.split('/')[2]
            key = '/'.join(model_path.split('/')[3:])
            
            # Download para arquivo temporário
            local_path = f"/tmp/model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
            self.s3_client.download_file(bucket, key, local_path)
            
            # Carrega modelo
            model = joblib.load(local_path)
            
            # Remove arquivo local
            os.remove(local_path)
            
            logger.info(f"Modelo carregado de: {model_path}")
            return model
            
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {str(e)}")
            raise
    
    def predict(self, model: lgb.LGBMClassifier, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Faz predições com o modelo.
        
        Args:
            model: Modelo treinado
            X: Features para predição
            
        Returns:
            Tuple com classes preditas e probabilidades
        """
        # Predições de probabilidade
        probabilities = model._Booster.predict(X, num_iteration=model._Booster.best_iteration)
        
        # Classes preditas
        predicted_classes = np.argmax(probabilities, axis=1)
        
        # Decodifica classes para formato original (-1, 0, 1)
        predicted_classes = self._decode_target(predicted_classes)
        
        return predicted_classes, probabilities
    
    def calculate_trading_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                 returns: Optional[np.ndarray] = None) -> Dict:
        """
        Calcula métricas específicas para trading.
        
        Args:
            y_true: Targets verdadeiros
            y_pred: Predições do modelo
            returns: Retornos dos ativos (opcional)
            
        Returns:
            Dicionário com métricas de trading
        """
        # Converte para formato original se necessário
        if set(np.unique(y_true)) == {0, 1, 2}:
            y_true = self._decode_target(y_true)
        if set(np.unique(y_pred)) == {0, 1, 2}:
            y_pred = self._decode_target(y_pred)
        
        metrics = {}
        
        # Métricas por classe
        for signal_class, class_name in [(-1, 'sell'), (0, 'neutral'), (1, 'buy')]:
            mask_true = y_true == signal_class
            mask_pred = y_pred == signal_class
            
            if mask_true.any():
                precision = np.sum(mask_true & mask_pred) / np.sum(mask_pred) if mask_pred.any() else 0
                recall = np.sum(mask_true & mask_pred) / np.sum(mask_true)
                
                metrics[f'{class_name}_precision'] = precision
                metrics[f'{class_name}_recall'] = recall
                metrics[f'{class_name}_f1'] = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Métricas gerais
        metrics['accuracy'] = np.mean(y_true == y_pred)
        metrics['signal_distribution'] = {
            'sell_ratio': np.mean(y_pred == -1),
            'neutral_ratio': np.mean(y_pred == 0),
            'buy_ratio': np.mean(y_pred == 1)
        }
        
        # Se retornos estão disponíveis, calcula métricas financeiras
        if returns is not None:
            strategy_returns = returns * y_pred  # Retorno da estratégia
            
            metrics['strategy_return'] = np.sum(strategy_returns)
            metrics['strategy_sharpe'] = np.mean(strategy_returns) / np.std(strategy_returns) if np.std(strategy_returns) > 0 else 0
            metrics['hit_rate'] = np.mean(strategy_returns > 0)
            metrics['max_drawdown'] = self._calculate_max_drawdown(np.cumsum(strategy_returns))
        
        return metrics
    
    def _calculate_max_drawdown(self, cumulative_returns: np.ndarray) -> float:
        """Calcula o máximo drawdown"""
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - peak) / peak
        return np.min(drawdown)

