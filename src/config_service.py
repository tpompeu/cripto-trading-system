"""
Sistema de Trading Quantitativo - Serviço de Configuração Unificada
Versão: 2.01
Data: 26 de agosto de 2025

Este módulo fornece uma interface unificada para acessar configurações de:
- AWS Systems Manager Parameter Store (parâmetros dinâmicos)
- DynamoDB (estados das estratégias)
- S3 (configuração estática)
- Variáveis de ambiente
"""

import json
import logging
import os
from typing import Dict, List, Optional, Any, Union
from decimal import Decimal
from datetime import datetime
import boto3
from botocore.exceptions import ClientError

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ConfigService:
    """
    Serviço centralizado de configuração para o sistema de trading.
    
    Agrega configurações de múltiplas fontes e fornece interface
    unificada com cache para otimizar performance.
    """
    
    def __init__(self):
        """Inicializa o serviço de configuração."""
        self.ssm = boto3.client('ssm')
        self.s3 = boto3.client('s3')
        self.dynamodb = boto3.resource('dynamodb')
        
        # Cache de configurações
        self._cache = {}
        self._cache_ttl = {}
        self._cache_duration = 300  # 5 minutos
        
        # Configurações básicas
        self.project_name = os.environ.get('PROJECT_NAME', 'CryptoTradingSystem')
        self.data_bucket = os.environ.get('DATA_BUCKET', '')
        
        # Tabelas DynamoDB
        self.strategy_table = self.dynamodb.Table(f"{self.project_name}-Strategies")
        
        logger.info("ConfigService inicializado com sucesso")
    
    def _is_cache_valid(self, key: str) -> bool:
        """
        Verifica se o cache para uma chave ainda é válido.
        
        Args:
            key: Chave do cache
            
        Returns:
            True se o cache é válido
        """
        if key not in self._cache_ttl:
            return False
        
        return datetime.now().timestamp() < self._cache_ttl[key]
    
    def _set_cache(self, key: str, value: Any, ttl_seconds: Optional[int] = None):
        """
        Define valor no cache com TTL.
        
        Args:
            key: Chave do cache
            value: Valor a ser armazenado
            ttl_seconds: TTL em segundos (usa padrão se None)
        """
        self._cache[key] = value
        ttl = ttl_seconds or self._cache_duration
        self._cache_ttl[key] = datetime.now().timestamp() + ttl
    
    def _get_cache(self, key: str) -> Optional[Any]:
        """
        Obtém valor do cache se válido.
        
        Args:
            key: Chave do cache
            
        Returns:
            Valor do cache ou None se inválido
        """
        if self._is_cache_valid(key):
            return self._cache.get(key)
        return None
    
    def get_ssm_parameter(self, parameter_name: str, default_value: Any = None, 
                         use_cache: bool = True) -> Any:
        """
        Obtém parâmetro do AWS Systems Manager Parameter Store.
        
        Args:
            parameter_name: Nome do parâmetro (ex: '/trading_system/risk/risk_per_trade')
            default_value: Valor padrão se parâmetro não existir
            use_cache: Se deve usar cache
            
        Returns:
            Valor do parâmetro
        """
        cache_key = f"ssm_{parameter_name}"
        
        # Verifica cache
        if use_cache:
            cached_value = self._get_cache(cache_key)
            if cached_value is not None:
                return cached_value
        
        try:
            response = self.ssm.get_parameter(Name=parameter_name)
            value = response['Parameter']['Value']
            
            # Tenta fazer parse JSON
            try:
                parsed_value = json.loads(value)
            except json.JSONDecodeError:
                parsed_value = value
            
            # Armazena no cache
            if use_cache:
                self._set_cache(cache_key, parsed_value)
            
            return parsed_value
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'ParameterNotFound':
                logger.warning(f"Parâmetro {parameter_name} não encontrado, usando valor padrão")
                return default_value
            else:
                logger.error(f"Erro ao buscar parâmetro {parameter_name}: {str(e)}")
                return default_value
        except Exception as e:
            logger.error(f"Erro inesperado ao buscar parâmetro {parameter_name}: {str(e)}")
            return default_value
    
    def get_s3_config(self, config_key: str, default_value: Any = None, 
                     use_cache: bool = True) -> Any:
        """
        Obtém configuração do S3.
        
        Args:
            config_key: Chave da configuração (ex: 'config.json')
            default_value: Valor padrão se não encontrado
            use_cache: Se deve usar cache
            
        Returns:
            Configuração do S3
        """
        cache_key = f"s3_{config_key}"
        
        # Verifica cache
        if use_cache:
            cached_value = self._get_cache(cache_key)
            if cached_value is not None:
                return cached_value
        
        try:
            response = self.s3.get_object(
                Bucket=self.data_bucket,
                Key=f"config/{config_key}"
            )
            
            content = response['Body'].read().decode('utf-8')
            
            # Tenta fazer parse JSON
            try:
                parsed_content = json.loads(content)
            except json.JSONDecodeError:
                parsed_content = content
            
            # Armazena no cache
            if use_cache:
                self._set_cache(cache_key, parsed_content)
            
            return parsed_content
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                logger.warning(f"Configuração {config_key} não encontrada no S3")
                return default_value
            else:
                logger.error(f"Erro ao buscar configuração {config_key}: {str(e)}")
                return default_value
        except Exception as e:
            logger.error(f"Erro inesperado ao buscar configuração {config_key}: {str(e)}")
            return default_value
    
    def get_strategy_state(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """
        Obtém estado de uma estratégia do DynamoDB.
        
        Args:
            strategy_id: ID da estratégia
            
        Returns:
            Estado da estratégia ou None se não encontrada
        """
        try:
            response = self.strategy_table.get_item(Key={'strategy_id': strategy_id})
            return response.get('Item')
        except ClientError as e:
            logger.error(f"Erro ao buscar estratégia {strategy_id}: {str(e)}")
            return None
    
    def update_strategy_state(self, strategy_id: str, updates: Dict[str, Any]) -> bool:
        """
        Atualiza estado de uma estratégia no DynamoDB.
        
        Args:
            strategy_id: ID da estratégia
            updates: Campos a serem atualizados
            
        Returns:
            True se bem-sucedido
        """
        try:
            # Prepara expressão de atualização
            update_expression = "SET "
            expression_values = {}
            expression_names = {}
            
            for key, value in updates.items():
                attr_name = f"#{key}"
                attr_value = f":{key}"
                
                update_expression += f"{attr_name} = {attr_value}, "
                expression_names[attr_name] = key
                expression_values[attr_value] = value
            
            # Remove vírgula final
            update_expression = update_expression.rstrip(', ')
            
            # Adiciona timestamp de atualização
            expression_names['#updated_at'] = 'updated_at'
            expression_values[':updated_at'] = datetime.now().isoformat()
            update_expression += ", #updated_at = :updated_at"
            
            self.strategy_table.update_item(
                Key={'strategy_id': strategy_id},
                UpdateExpression=update_expression,
                ExpressionAttributeNames=expression_names,
                ExpressionAttributeValues=expression_values
            )
            
            return True
            
        except ClientError as e:
            logger.error(f"Erro ao atualizar estratégia {strategy_id}: {str(e)}")
            return False
    
    def get_trading_config(self) -> Dict[str, Any]:
        """
        Obtém configuração completa de trading.
        
        Returns:
            Configuração consolidada
        """
        config = {
            # Configurações básicas
            'project_name': self.project_name,
            'data_bucket': self.data_bucket,
            
            # Símbolos e timeframes
            'symbols': self.get_ssm_parameter('/trading_system/symbols', ['BTCUSDT', 'ETHUSDT']),
            'timeframes': self.get_ssm_parameter('/trading_system/timeframes', ['1h', '4h']),
            
            # Configurações de risco
            'risk_per_trade': float(self.get_ssm_parameter('/trading_system/risk/risk_per_trade', '0.01')),
            'risk_exposition': float(self.get_ssm_parameter('/trading_system/risk/risk_exposition', '0.03')),
            
            # Configurações operacionais
            'master_trading_switch': self.get_ssm_parameter('/trading_system/ops/master_trading_switch', 'false').lower() == 'true',
            
            # Configurações de custo
            'monthly_cost_limit': float(self.get_ssm_parameter('/trading_system/cost/monthly_cost_limit', '10.0')),
            'spot_min_discount_perc': float(self.get_ssm_parameter('/trading_system/cost/spot_min_discount_perc', '0.95')),
            'spot_instance_type': self.get_ssm_parameter('/trading_system/cost/spot_instance_type', ['c6i.large']),
            
            # Configurações de ambiente
            'github_repo': self.get_ssm_parameter('/trading_system/environment/github_repo', ''),
            'binance_api_key': self.get_ssm_parameter('/trading_system/environment/binance_api_key', ''),
            'binance_api_secret': self.get_ssm_parameter('/trading_system/environment/binance_api_secret', ''),
        }
        
        return config
    
    def get_model_config(self) -> Dict[str, Any]:
        """
        Obtém configuração de modelo e validação.
        
        Returns:
            Configuração de modelo
        """
        # Tenta buscar do S3 primeiro, depois valores padrão
        s3_config = self.get_s3_config('config.json', {})
        
        default_config = {
            "parameter_ranges": {
                "ichimoku": {
                    "tenkan_sen": {"start": 8, "end": 10, "step": 1},
                    "kijun_sen": {"start": 24, "end": 28, "step": 2},
                    "senkou_span_b": {"start": 48, "end": 56, "step": 4}
                },
                "rsi": {
                    "period": {"start": 7, "end": 28, "step": 7}
                },
                "atr": {
                    "period": {"start": 7, "end": 28, "step": 7}
                }
            },
            "model_hyperparameters": {
                "LightGBM": {
                    "learning_rate": {"type": "float", "min": 0.01, "max": 0.3, "log": True},
                    "num_leaves": {"type": "int", "min": 20, "max": 100},
                    "min_data_in_leaf": {"type": "int", "min": 10, "max": 50},
                    "n_estimators": {"type": "int", "min": 50, "max": 500}
                }
            },
            "walk_forward_config": {
                "training_months": 12,
                "validation_months": 3,
                "step_months": 3
            },
            "incubation_promotion": {
                "expectancy_threshold": 0.01,
                "min_trades": 10,
                "period": "last_30_trades"
            },
            "production_promotion": {
                "minimum_duration_months": 1,
                "expectancy_threshold": 0.01,
                "min_trades": 10,
                "period": "last_30_trades"
            },
            "gravedigger": {
                "minimum_duration_months": 3,
                "min_trades": 10,
                "expectancy_threshold": 0.0
            },
            "circuit_breaker": {
                "daily": 0.05,
                "weekly": 0.10,
                "monthly": 0.15,
                "auto_reset_threshold": 0.5,
                "breakeven_alert_threshold": 0.5
            },
            "break_even": {
                "min_multiplier": 1.0,
                "max_multiplier": 1.5,
                "historical_atr_window": 252
            },
            "checkpoint_config": {
                "checkpoint_interval": "after_each_validation_window",
                "s3_checkpoint_path": f"s3://{self.data_bucket}/walk_forward_checkpoints/",
                "max_resume_attempts": 3,
                "checkpoint_retention_days": 30
            },
            "monte_carlo_config": {
                "n_simulations": 1000,
                "confidence_level": 0.95,
                "p_value_threshold": 0.05,
                "random_seed": 42
            }
        }
        
        # Merge configurações do S3 com padrões
        if s3_config:
            self._deep_merge(default_config, s3_config)
        
        return default_config
    
    def _deep_merge(self, base_dict: Dict, update_dict: Dict):
        """
        Faz merge profundo de dicionários.
        
        Args:
            base_dict: Dicionário base (será modificado)
            update_dict: Dicionário com atualizações
        """
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_merge(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def get_binance_credentials(self) -> Dict[str, str]:
        """
        Obtém credenciais da Binance de forma segura.
        
        Returns:
            Credenciais da Binance
        """
        return {
            'api_key': self.get_ssm_parameter('/trading_system/environment/binance_api_key', ''),
            'api_secret': self.get_ssm_parameter('/trading_system/environment/binance_api_secret', '')
        }
    
    def is_trading_enabled(self) -> bool:
        """
        Verifica se o trading está habilitado.
        
        Returns:
            True se trading está habilitado
        """
        return self.get_ssm_parameter('/trading_system/ops/master_trading_switch', 'false').lower() == 'true'
    
    def get_risk_parameters(self) -> Dict[str, float]:
        """
        Obtém parâmetros de risco.
        
        Returns:
            Parâmetros de risco
        """
        return {
            'risk_per_trade': float(self.get_ssm_parameter('/trading_system/risk/risk_per_trade', '0.01')),
            'risk_exposition': float(self.get_ssm_parameter('/trading_system/risk/risk_exposition', '0.03'))
        }
    
    def clear_cache(self):
        """Limpa todo o cache de configurações."""
        self._cache.clear()
        self._cache_ttl.clear()
        logger.info("Cache de configurações limpo")
    
    def get_all_strategies(self) -> List[Dict[str, Any]]:
        """
        Obtém todas as estratégias do DynamoDB.
        
        Returns:
            Lista de estratégias
        """
        try:
            response = self.strategy_table.scan()
            return response.get('Items', [])
        except ClientError as e:
            logger.error(f"Erro ao buscar estratégias: {str(e)}")
            return []


def lambda_handler(event, context):
    """
    Função principal para execução no AWS Lambda.
    
    Args:
        event: Evento do Lambda
        context: Contexto de execução do Lambda
        
    Returns:
        Configurações solicitadas
    """
    try:
        logger.info("=== INÍCIO DA EXECUÇÃO DO CONFIG SERVICE ===")
        logger.info(f"Event: {json.dumps(event, default=str)}")
        
        config_service = ConfigService()
        
        # Determina que configuração retornar baseado no evento
        config_type = event.get('config_type', 'trading')
        
        if config_type == 'trading':
            config = config_service.get_trading_config()
        elif config_type == 'model':
            config = config_service.get_model_config()
        elif config_type == 'risk':
            config = config_service.get_risk_parameters()
        elif config_type == 'binance':
            config = config_service.get_binance_credentials()
        else:
            config = {
                'trading': config_service.get_trading_config(),
                'model': config_service.get_model_config(),
                'risk': config_service.get_risk_parameters()
            }
        
        response = {
            'statusCode': 200,
            'body': json.dumps(config, default=str),
            'headers': {
                'Content-Type': 'application/json'
            }
        }
        
        logger.info("=== FIM DA EXECUÇÃO DO CONFIG SERVICE ===")
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


# Para uso como módulo
def get_config_service() -> ConfigService:
    """
    Factory function para obter instância do ConfigService.
    
    Returns:
        Instância configurada do ConfigService
    """
    return ConfigService()


# Para execução local/teste
if __name__ == "__main__":
    # Configura variáveis de ambiente para teste local
    os.environ['PROJECT_NAME'] = 'CryptoTradingSystem'
    os.environ['DATA_BUCKET'] = 'crypto-trading-data-test'
    
    # Testa o serviço
    config_service = ConfigService()
    
    print("=== CONFIGURAÇÃO DE TRADING ===")
    trading_config = config_service.get_trading_config()
    print(json.dumps(trading_config, indent=2, default=str))
    
    print("\n=== CONFIGURAÇÃO DE MODELO ===")
    model_config = config_service.get_model_config()
    print(json.dumps(model_config, indent=2, default=str))

