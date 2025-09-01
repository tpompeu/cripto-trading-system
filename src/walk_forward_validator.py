"""
Sistema de Trading Quantitativo - Walk-Forward Validator
Versão: 2.01
Descrição: Validação Walk-Forward para descoberta e validação de estratégias
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
import boto3
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from feature_engineering import FeatureEngineer, create_target_variable
from data_processor import DataProcessor
from data_validator import DataValidator
from model_trainer import ModelTrainer

logger = logging.getLogger(__name__)

class WalkForwardValidator:
    """
    Classe responsável pela validação Walk-Forward.
    Implementa o processo de descoberta de estratégias através de validação temporal rigorosa.
    """
    
    def __init__(self, config: Dict):
        """
        Inicializa o WalkForwardValidator.
        
        Args:
            config: Configurações do sistema
        """
        self.config = config
        self.s3_client = boto3.client('s3')
        self.dynamodb = boto3.resource('dynamodb')
        self.bucket_name = config.get('data_bucket_name')
        
        # Configurações do Walk-Forward
        self.wf_config = config.get('walk_forward_config', {
            'train_months': 12,
            'validation_months': 3,
            'step_months': 3,
            'min_samples_train': 1000,
            'min_samples_val': 100
        })
        
        # Inicializa componentes
        self.feature_engineer = FeatureEngineer(config)
        self.data_processor = DataProcessor(config)
        self.data_validator = DataValidator(config)
        self.model_trainer = ModelTrainer(config)
        
        # Tabela DynamoDB para checkpoints
        self.checkpoint_table_name = config.get('checkpoint_table_name', 'trading-system-checkpoints')
        
    def run_walk_forward_validation(self, symbol: str, timeframe: str,
                                   parameter_combinations: List[Dict],
                                   job_id: Optional[str] = None) -> Dict:
        """
        Executa validação Walk-Forward completa.
        
        Args:
            symbol: Par de trading
            timeframe: Timeframe dos dados
            parameter_combinations: Lista de combinações de parâmetros para testar
            job_id: ID do job (para checkpoints)
            
        Returns:
            Dicionário com resultados da validação
        """
        if job_id is None:
            job_id = f"wf_{symbol}_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Iniciando Walk-Forward validation para {symbol} {timeframe}")
        logger.info(f"Job ID: {job_id}")
        logger.info(f"Testando {len(parameter_combinations)} combinações de parâmetros")
        
        # Verifica se há checkpoint existente
        checkpoint = self._load_checkpoint(job_id)
        
        if checkpoint:
            logger.info(f"Checkpoint encontrado. Retomando do parâmetro {checkpoint['current_param_index']}")
            start_param_index = checkpoint['current_param_index']
            results = checkpoint['results']
        else:
            logger.info("Iniciando validação do zero")
            start_param_index = 0
            results = {
                'job_id': job_id,
                'symbol': symbol,
                'timeframe': timeframe,
                'start_time': datetime.now().isoformat(),
                'parameter_results': [],
                'best_strategy': None,
                'validation_summary': {}
            }
        
        try:
            # Carrega dados
            data = self._load_and_prepare_data(symbol, timeframe)
            
            # Cria divisões Walk-Forward
            splits = self.data_processor.create_walk_forward_splits(
                data,
                train_months=self.wf_config['train_months'],
                validation_months=self.wf_config['validation_months'],
                step_months=self.wf_config['step_months']
            )
            
            logger.info(f"Criadas {len(splits)} divisões Walk-Forward")
            
            # Testa cada combinação de parâmetros
            for param_index in range(start_param_index, len(parameter_combinations)):
                param_combo = parameter_combinations[param_index]
                
                logger.info(f"Testando parâmetros {param_index + 1}/{len(parameter_combinations)}: {param_combo}")
                
                # Valida combinação de parâmetros
                param_results = self._validate_parameter_combination(
                    data, splits, param_combo, symbol, timeframe
                )
                
                param_results['parameter_index'] = param_index
                param_results['parameters'] = param_combo
                results['parameter_results'].append(param_results)
                
                # Salva checkpoint
                self._save_checkpoint(job_id, param_index + 1, results)
                
                logger.info(f"Parâmetros {param_index + 1} concluídos. Score médio: {param_results['avg_score']:.4f}")
            
            # Finaliza validação
            results['end_time'] = datetime.now().isoformat()
            results = self._finalize_validation_results(results)
            
            # Salva resultados finais
            self._save_final_results(results, symbol, timeframe)
            
            # Remove checkpoint
            self._delete_checkpoint(job_id)
            
            logger.info("Walk-Forward validation concluída")
            return results
            
        except Exception as e:
            logger.error(f"Erro na validação Walk-Forward: {str(e)}")
            # Salva checkpoint em caso de erro
            self._save_checkpoint(job_id, start_param_index, results, error=str(e))
            raise
    
    def _load_and_prepare_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Carrega e prepara dados para validação"""
        logger.info("Carregando e preparando dados")
        
        # Carrega dados brutos
        raw_data = self.data_processor.load_data(symbol, timeframe)
        
        # Valida dados
        validation_results = self.data_validator.validate_ohlcv_data(raw_data)
        if not validation_results['validation_passed']:
            raise ValueError(f"Dados não passaram na validação: {validation_results['errors']}")
        
        # Limpa dados
        clean_data = self.data_processor.clean_data(raw_data)
        
        logger.info(f"Dados preparados: {len(clean_data)} registros")
        return clean_data
    
    def _validate_parameter_combination(self, data: pd.DataFrame, splits: List[Dict],
                                      param_combo: Dict, symbol: str, timeframe: str) -> Dict:
        """
        Valida uma combinação específica de parâmetros.
        
        Args:
            data: Dados preparados
            splits: Divisões Walk-Forward
            param_combo: Combinação de parâmetros
            symbol: Par de trading
            timeframe: Timeframe
            
        Returns:
            Resultados da validação para esta combinação
        """
        split_results = []
        
        for split_info in splits:
            try:
                # Gera features com parâmetros específicos
                split_data = self._generate_features_for_split(data, split_info, param_combo)
                
                # Cria variável target
                split_data = create_target_variable(split_data, param_combo.get('atr_multiplier', 1.0))
                
                # Prepara features
                split_data, feature_columns = self.data_processor.prepare_features(
                    split_data, self.config.get('feature_config', {})
                )
                
                # Normaliza features
                split_data = self.data_processor.normalize_features(
                    split_data, feature_columns, fit_scaler=True
                )
                
                # Obtém dados da divisão
                X_train, X_val, y_train, y_val = self.data_processor.get_split_data(
                    split_data, split_info, feature_columns
                )
                
                # Verifica se há dados suficientes
                if len(X_train) < self.wf_config['min_samples_train'] or len(X_val) < self.wf_config['min_samples_val']:
                    logger.warning(f"Dados insuficientes na divisão {split_info['split_id']}")
                    continue
                
                # Treina modelo
                model, metrics = self.model_trainer.train_model(X_train, y_train, X_val, y_val)
                
                # Calcula métricas de trading
                y_pred, _ = self.model_trainer.predict(model, X_val)
                trading_metrics = self.model_trainer.calculate_trading_metrics(y_val.values, y_pred)
                
                # Combina métricas
                split_result = {
                    'split_id': split_info['split_id'],
                    'train_period': f"{split_info['train_start']} to {split_info['train_end']}",
                    'val_period': f"{split_info['val_start']} to {split_info['val_end']}",
                    'train_samples': len(X_train),
                    'val_samples': len(X_val),
                    'model_metrics': metrics,
                    'trading_metrics': trading_metrics,
                    'score': metrics['val_accuracy']  # Score principal para otimização
                }
                
                split_results.append(split_result)
                
            except Exception as e:
                logger.warning(f"Erro na divisão {split_info['split_id']}: {str(e)}")
                continue
        
        # Calcula estatísticas agregadas
        if split_results:
            scores = [result['score'] for result in split_results]
            param_results = {
                'split_results': split_results,
                'avg_score': np.mean(scores),
                'std_score': np.std(scores),
                'min_score': np.min(scores),
                'max_score': np.max(scores),
                'successful_splits': len(split_results),
                'total_splits': len(splits)
            }
        else:
            param_results = {
                'split_results': [],
                'avg_score': 0.0,
                'std_score': 0.0,
                'min_score': 0.0,
                'max_score': 0.0,
                'successful_splits': 0,
                'total_splits': len(splits)
            }
        
        return param_results
    
    def _generate_features_for_split(self, data: pd.DataFrame, split_info: Dict,
                                   param_combo: Dict) -> pd.DataFrame:
        """
        Gera features para uma divisão específica usando parâmetros dados.
        
        Args:
            data: Dados base
            split_info: Informações da divisão
            param_combo: Combinação de parâmetros
            
        Returns:
            DataFrame com features geradas
        """
        # Filtra dados até o final da validação (evita look-ahead bias)
        end_date = split_info['val_end']
        split_data = data[data.index <= end_date].copy()
        
        # Atualiza configuração do feature engineer com parâmetros específicos
        feature_config = self.config.copy()
        
        # Atualiza parâmetros dos indicadores
        if 'ichimoku_params' in param_combo:
            feature_config['ichimoku_params'] = param_combo['ichimoku_params']
        if 'rsi_period' in param_combo:
            feature_config['rsi_period'] = param_combo['rsi_period']
        if 'atr_period' in param_combo:
            feature_config['atr_period'] = param_combo['atr_period']
        
        # Cria feature engineer com parâmetros específicos
        feature_engineer = FeatureEngineer(feature_config)
        
        # Gera features
        split_data = feature_engineer.generate_all_features(split_data)
        
        return split_data
    
    def _finalize_validation_results(self, results: Dict) -> Dict:
        """Finaliza e analisa resultados da validação"""
        logger.info("Finalizando resultados da validação")
        
        if not results['parameter_results']:
            logger.warning("Nenhum resultado de parâmetro encontrado")
            return results
        
        # Encontra melhor estratégia
        best_strategy = max(results['parameter_results'], key=lambda x: x['avg_score'])
        results['best_strategy'] = best_strategy
        
        # Calcula estatísticas gerais
        all_scores = [param['avg_score'] for param in results['parameter_results']]
        
        results['validation_summary'] = {
            'total_parameter_combinations': len(results['parameter_results']),
            'best_avg_score': best_strategy['avg_score'],
            'best_parameters': best_strategy['parameters'],
            'score_statistics': {
                'mean': np.mean(all_scores),
                'std': np.std(all_scores),
                'min': np.min(all_scores),
                'max': np.max(all_scores),
                'median': np.median(all_scores)
            },
            'successful_validations': len([s for s in all_scores if s > 0])
        }
        
        logger.info(f"Melhor estratégia encontrada com score: {best_strategy['avg_score']:.4f}")
        return results
    
    def _save_checkpoint(self, job_id: str, current_param_index: int, 
                        results: Dict, error: Optional[str] = None):
        """Salva checkpoint no DynamoDB"""
        try:
            table = self.dynamodb.Table(self.checkpoint_table_name)
            
            checkpoint_data = {
                'job_id': job_id,
                'current_param_index': current_param_index,
                'results': results,
                'timestamp': datetime.now().isoformat(),
                'status': 'error' if error else 'running'
            }
            
            if error:
                checkpoint_data['error'] = error
            
            table.put_item(Item=checkpoint_data)
            logger.debug(f"Checkpoint salvo para job {job_id}")
            
        except Exception as e:
            logger.warning(f"Erro ao salvar checkpoint: {str(e)}")
    
    def _load_checkpoint(self, job_id: str) -> Optional[Dict]:
        """Carrega checkpoint do DynamoDB"""
        try:
            table = self.dynamodb.Table(self.checkpoint_table_name)
            response = table.get_item(Key={'job_id': job_id})
            
            if 'Item' in response:
                checkpoint = response['Item']
                logger.info(f"Checkpoint carregado para job {job_id}")
                return checkpoint
            
            return None
            
        except Exception as e:
            logger.warning(f"Erro ao carregar checkpoint: {str(e)}")
            return None
    
    def _delete_checkpoint(self, job_id: str):
        """Remove checkpoint do DynamoDB"""
        try:
            table = self.dynamodb.Table(self.checkpoint_table_name)
            table.delete_item(Key={'job_id': job_id})
            logger.debug(f"Checkpoint removido para job {job_id}")
            
        except Exception as e:
            logger.warning(f"Erro ao remover checkpoint: {str(e)}")
    
    def _save_final_results(self, results: Dict, symbol: str, timeframe: str):
        """Salva resultados finais no S3"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"walk_forward_results_{symbol}_{timeframe}_{timestamp}.json"
        s3_key = f"validation_results/{filename}"
        
        try:
            # Converte para JSON
            results_json = json.dumps(results, indent=2, default=str)
            
            # Upload para S3
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=results_json,
                ContentType='application/json'
            )
            
            s3_path = f"s3://{self.bucket_name}/{s3_key}"
            logger.info(f"Resultados finais salvos em: {s3_path}")
            
        except Exception as e:
            logger.error(f"Erro ao salvar resultados finais: {str(e)}")
            raise
    
    def generate_parameter_combinations(self, param_ranges: Dict) -> List[Dict]:
        """
        Gera todas as combinações de parâmetros para testar.
        
        Args:
            param_ranges: Dicionário com ranges de parâmetros
            
        Returns:
            Lista de combinações de parâmetros
        """
        logger.info("Gerando combinações de parâmetros")
        
        combinations = []
        
        # Parâmetros do Ichimoku
        ichimoku_ranges = param_ranges.get('ichimoku', {
            'tenkan_period': [9, 12, 15],
            'kijun_period': [26, 30, 34],
            'senkou_b_period': [52, 60, 68]
        })
        
        # Parâmetros do RSI
        rsi_periods = param_ranges.get('rsi_period', [14, 21, 28])
        
        # Parâmetros do ATR
        atr_periods = param_ranges.get('atr_period', [14, 20, 26])
        atr_multipliers = param_ranges.get('atr_multiplier', [1.0, 1.5, 2.0])
        
        # Gera todas as combinações
        for tenkan in ichimoku_ranges['tenkan_period']:
            for kijun in ichimoku_ranges['kijun_period']:
                for senkou_b in ichimoku_ranges['senkou_b_period']:
                    for rsi_period in rsi_periods:
                        for atr_period in atr_periods:
                            for atr_mult in atr_multipliers:
                                combination = {
                                    'ichimoku_params': {
                                        'tenkan_period': tenkan,
                                        'kijun_period': kijun,
                                        'senkou_b_period': senkou_b,
                                        'displacement': kijun  # Displacement = kijun_period
                                    },
                                    'rsi_period': rsi_period,
                                    'atr_period': atr_period,
                                    'atr_multiplier': atr_mult
                                }
                                combinations.append(combination)
        
        logger.info(f"Geradas {len(combinations)} combinações de parâmetros")
        return combinations
    
    def resume_validation(self, job_id: str) -> Dict:
        """
        Retoma uma validação Walk-Forward interrompida.
        
        Args:
            job_id: ID do job a ser retomado
            
        Returns:
            Resultados da validação
        """
        logger.info(f"Retomando validação Walk-Forward: {job_id}")
        
        # Carrega checkpoint
        checkpoint = self._load_checkpoint(job_id)
        
        if not checkpoint:
            raise ValueError(f"Checkpoint não encontrado para job {job_id}")
        
        if checkpoint.get('status') == 'error':
            logger.warning(f"Job {job_id} estava em estado de erro: {checkpoint.get('error')}")
        
        # Extrai informações do checkpoint
        results = checkpoint['results']
        symbol = results['symbol']
        timeframe = results['timeframe']
        current_param_index = checkpoint['current_param_index']
        
        # Gera combinações de parâmetros (assumindo mesmo range)
        param_ranges = self.config.get('parameter_ranges', {})
        parameter_combinations = self.generate_parameter_combinations(param_ranges)
        
        # Retoma validação
        return self.run_walk_forward_validation(
            symbol, timeframe, parameter_combinations, job_id
        )

