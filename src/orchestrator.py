"""
Sistema de Trading Quantitativo - Orquestrador
Versão: 2.01
Data: 26 de agosto de 2025

Este módulo é o "cérebro" do sistema, responsável por:
- Monitorar a performance das estratégias em produção
- Promover estratégias da incubação para produção
- Acionar o Gravedigger quando necessário
- Verificar se é hora de retreino simples
- Enfileirar jobs de validação completa (Walk-Forward)
- Gerenciar o ciclo de vida das estratégias
"""

import json
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from decimal import Decimal
from enum import Enum

import boto3
from botocore.exceptions import ClientError

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StrategyStatus(Enum):
    """Estados possíveis de uma estratégia."""
    DISCOVERY = "discovery"          # Em processo de Walk-Forward
    MONTE_CARLO = "monte_carlo"      # Em teste de Monte Carlo
    INCUBATION = "incubation"        # Paper trading
    PRODUCTION = "production"        # Trading real
    GRAVEDIGGER = "gravedigger"      # Performance degradou, precisa revalidação
    DISABLED = "disabled"            # Desativada permanentemente


class TradingOrchestrator:
    """
    Orquestrador principal do sistema de trading.
    
    Gerencia o ciclo de vida completo das estratégias, desde a descoberta
    até a operação em produção, incluindo monitoramento e manutenção.
    """
    
    def __init__(self):
        """Inicializa o orquestrador com clientes AWS."""
        self.dynamodb = boto3.resource('dynamodb')
        self.sqs = boto3.client('sqs')
        self.ssm = boto3.client('ssm')
        self.s3 = boto3.client('s3')
        self.lambda_client = boto3.client('lambda')
        
        # Carrega configurações
        self.config = self._load_configuration()
        
        # Tabelas DynamoDB
        self.strategy_table = self.dynamodb.Table(f"{self.config['project_name']}-Strategies")
        
        logger.info("TradingOrchestrator inicializado com sucesso")
    
    def _load_configuration(self) -> Dict[str, Any]:
        """
        Carrega configurações do AWS Systems Manager Parameter Store.
        
        Returns:
            Dict: Configurações do sistema
        """
        try:
            config = {
                'project_name': os.environ.get('PROJECT_NAME', 'CryptoTradingSystem'),
                'data_bucket': os.environ.get('DATA_BUCKET', ''),
                'validation_queue_url': os.environ.get('VALIDATION_QUEUE_URL', ''),
            }
            
            # Parâmetros do SSM
            ssm_params = [
                '/trading_system/symbols',
                '/trading_system/timeframes',
                '/trading_system/ops/master_trading_switch',
                '/trading_system/risk/risk_per_trade',
                '/trading_system/risk/risk_exposition',
                '/trading_system/cost/monthly_cost_limit',
                '/trading_system/cost/spot_min_discount_perc'
            ]
            
            for param_name in ssm_params:
                try:
                    response = self.ssm.get_parameter(Name=param_name)
                    param_key = param_name.split('/')[-1]
                    
                    # Parse JSON se possível
                    try:
                        config[param_key] = json.loads(response['Parameter']['Value'])
                    except json.JSONDecodeError:
                        config[param_key] = response['Parameter']['Value']
                        
                except ClientError as e:
                    if e.response['Error']['Code'] == 'ParameterNotFound':
                        logger.warning(f"Parâmetro {param_name} não encontrado")
                        # Valores padrão
                        defaults = {
                            'symbols': ['BTCUSDT', 'ETHUSDT'],
                            'timeframes': ['1h', '4h'],
                            'master_trading_switch': 'true',
                            'risk_per_trade': '0.01',
                            'risk_exposition': '0.03',
                            'monthly_cost_limit': '10.0',
                            'spot_min_discount_perc': '0.95'
                        }
                        param_key = param_name.split('/')[-1]
                        config[param_key] = defaults.get(param_key, '')
                    else:
                        raise
            
            logger.info(f"Configuração carregada: {list(config.keys())}")
            return config
            
        except Exception as e:
            logger.error(f"Erro ao carregar configuração: {str(e)}")
            raise
    
    def _get_strategy_key(self, symbol: str, timeframe: str) -> str:
        """
        Gera chave única para uma estratégia.
        
        Args:
            symbol: Par de trading (ex: 'BTCUSDT')
            timeframe: Timeframe (ex: '1h')
            
        Returns:
            Chave única da estratégia
        """
        return f"{symbol}_{timeframe}"
    
    def _get_strategy_from_db(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """
        Busca uma estratégia no DynamoDB.
        
        Args:
            strategy_id: ID da estratégia
            
        Returns:
            Dados da estratégia ou None se não encontrada
        """
        try:
            response = self.strategy_table.get_item(Key={'strategy_id': strategy_id})
            return response.get('Item')
        except ClientError as e:
            logger.error(f"Erro ao buscar estratégia {strategy_id}: {str(e)}")
            return None
    
    def _update_strategy_in_db(self, strategy_id: str, updates: Dict[str, Any]) -> bool:
        """
        Atualiza uma estratégia no DynamoDB.
        
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
            
            logger.info(f"Estratégia {strategy_id} atualizada: {list(updates.keys())}")
            return True
            
        except ClientError as e:
            logger.error(f"Erro ao atualizar estratégia {strategy_id}: {str(e)}")
            return False
    
    def _create_strategy_in_db(self, strategy_id: str, symbol: str, timeframe: str, 
                              status: StrategyStatus = StrategyStatus.DISCOVERY) -> bool:
        """
        Cria uma nova estratégia no DynamoDB.
        
        Args:
            strategy_id: ID da estratégia
            symbol: Par de trading
            timeframe: Timeframe
            status: Status inicial
            
        Returns:
            True se bem-sucedido
        """
        try:
            now = datetime.now().isoformat()
            
            item = {
                'strategy_id': strategy_id,
                'symbol': symbol,
                'timeframe': timeframe,
                'status': status.value,
                'created_at': now,
                'updated_at': now,
                'last_validation_date': None,
                'last_retrain_date': None,
                'performance_metrics': {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'expectancy': Decimal('0.0'),
                    'max_drawdown': Decimal('0.0'),
                    'profit_factor': Decimal('0.0')
                },
                'validation_results': {},
                'monte_carlo_results': {},
                'incubation_results': {},
                'production_results': {}
            }
            
            self.strategy_table.put_item(Item=item)
            logger.info(f"Nova estratégia criada: {strategy_id}")
            return True
            
        except ClientError as e:
            logger.error(f"Erro ao criar estratégia {strategy_id}: {str(e)}")
            return False
    
    def _calculate_strategy_performance(self, strategy_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Calcula métricas de performance de uma estratégia.
        
        Args:
            strategy_data: Dados da estratégia
            
        Returns:
            Métricas calculadas
        """
        try:
            # Busca dados de trades no S3 ou DynamoDB
            # Por simplicidade, usa dados mockados aqui
            # Em produção, buscaria dados reais de performance
            
            metrics = {
                'expectancy': 0.0,
                'total_trades': 0,
                'winning_trades': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0
            }
            
            # TODO: Implementar cálculo real baseado em dados históricos
            # Por enquanto, retorna métricas zeradas
            
            return metrics
            
        except Exception as e:
            logger.error(f"Erro ao calcular performance: {str(e)}")
            return {}
    
    def _should_trigger_gravedigger(self, strategy_data: Dict[str, Any]) -> bool:
        """
        Verifica se uma estratégia deve ser enviada para o Gravedigger.
        
        Args:
            strategy_data: Dados da estratégia
            
        Returns:
            True se deve acionar o Gravedigger
        """
        try:
            # Critérios para acionar o Gravedigger
            performance = strategy_data.get('performance_metrics', {})
            
            # Expectância negativa
            expectancy = float(performance.get('expectancy', 0))
            if expectancy <= 0:
                logger.warning(f"Estratégia {strategy_data['strategy_id']} com expectância negativa: {expectancy}")
                return True
            
            # Drawdown excessivo
            max_drawdown = float(performance.get('max_drawdown', 0))
            if max_drawdown > 0.15:  # 15%
                logger.warning(f"Estratégia {strategy_data['strategy_id']} com drawdown excessivo: {max_drawdown}")
                return True
            
            # Número mínimo de trades para análise
            total_trades = int(performance.get('total_trades', 0))
            if total_trades < 10:
                return False  # Não há dados suficientes ainda
            
            # Profit factor baixo
            profit_factor = float(performance.get('profit_factor', 0))
            if profit_factor < 1.1:
                logger.warning(f"Estratégia {strategy_data['strategy_id']} com profit factor baixo: {profit_factor}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Erro ao verificar Gravedigger: {str(e)}")
            return False
    
    def _should_promote_from_incubation(self, strategy_data: Dict[str, Any]) -> bool:
        """
        Verifica se uma estratégia deve ser promovida da incubação para produção.
        
        Args:
            strategy_data: Dados da estratégia
            
        Returns:
            True se deve ser promovida
        """
        try:
            # Critérios de promoção baseados no config.json
            incubation_results = strategy_data.get('incubation_results', {})
            
            # Duração mínima na incubação
            created_at = datetime.fromisoformat(strategy_data.get('created_at', ''))
            days_in_incubation = (datetime.now() - created_at).days
            
            if days_in_incubation < 30:  # Mínimo 30 dias
                return False
            
            # Expectância positiva
            expectancy = float(incubation_results.get('expectancy', 0))
            if expectancy <= 0.01:  # Threshold do config
                return False
            
            # Número mínimo de trades
            total_trades = int(incubation_results.get('total_trades', 0))
            if total_trades < 10:
                return False
            
            logger.info(f"Estratégia {strategy_data['strategy_id']} qualificada para promoção")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao verificar promoção: {str(e)}")
            return False
    
    def _should_retrain_model(self, strategy_data: Dict[str, Any]) -> bool:
        """
        Verifica se é hora de retreinar o modelo de uma estratégia.
        
        Args:
            strategy_data: Dados da estratégia
            
        Returns:
            True se deve retreinar
        """
        try:
            last_retrain = strategy_data.get('last_retrain_date')
            if not last_retrain:
                return True  # Nunca foi retreinado
            
            last_retrain_date = datetime.fromisoformat(last_retrain)
            months_since_retrain = (datetime.now() - last_retrain_date).days / 30
            
            # Retreina a cada 3 meses (step_months do config)
            return months_since_retrain >= 3
            
        except Exception as e:
            logger.error(f"Erro ao verificar retreino: {str(e)}")
            return False
    
    def _enqueue_validation_job(self, strategy_id: str, symbol: str, timeframe: str) -> bool:
        """
        Enfileira um job de validação completa (Walk-Forward) na SQS.
        
        Args:
            strategy_id: ID da estratégia
            symbol: Par de trading
            timeframe: Timeframe
            
        Returns:
            True se enfileirado com sucesso
        """
        try:
            message = {
                'job_type': 'walk_forward_validation',
                'strategy_id': strategy_id,
                'symbol': symbol,
                'timeframe': timeframe,
                'timestamp': datetime.now().isoformat(),
                'priority': 'high' if 'BTC' in symbol else 'normal'
            }
            
            response = self.sqs.send_message(
                QueueUrl=self.config['validation_queue_url'],
                MessageBody=json.dumps(message),
                MessageGroupId=strategy_id,  # FIFO queue
                MessageDeduplicationId=f"{strategy_id}_{int(datetime.now().timestamp())}"
            )
            
            logger.info(f"Job de validação enfileirado: {strategy_id}")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao enfileirar job de validação: {str(e)}")
            return False
    
    def _trigger_model_retraining(self, strategy_id: str) -> bool:
        """
        Aciona o retreino simples de um modelo.
        
        Args:
            strategy_id: ID da estratégia
            
        Returns:
            True se acionado com sucesso
        """
        try:
            payload = {
                'strategy_id': strategy_id,
                'retrain_type': 'simple',
                'timestamp': datetime.now().isoformat()
            }
            
            # Invoca a Lambda de retreino
            response = self.lambda_client.invoke(
                FunctionName=f"{self.config['project_name']}-ModelRetrainer",
                InvocationType='Event',  # Assíncrono
                Payload=json.dumps(payload)
            )
            
            logger.info(f"Retreino acionado para estratégia: {strategy_id}")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao acionar retreino: {str(e)}")
            return False
    
    def monitor_production_strategies(self) -> Dict[str, Any]:
        """
        Monitora estratégias em produção e aciona Gravedigger se necessário.
        
        Returns:
            Resultado do monitoramento
        """
        results = {
            'strategies_monitored': 0,
            'gravedigger_triggered': 0,
            'retraining_triggered': 0,
            'errors': []
        }
        
        try:
            # Busca todas as estratégias em produção
            response = self.strategy_table.scan(
                FilterExpression='#status = :status',
                ExpressionAttributeNames={'#status': 'status'},
                ExpressionAttributeValues={':status': StrategyStatus.PRODUCTION.value}
            )
            
            for strategy in response.get('Items', []):
                results['strategies_monitored'] += 1
                strategy_id = strategy['strategy_id']
                
                try:
                    # Calcula performance atual
                    performance = self._calculate_strategy_performance(strategy)
                    
                    # Atualiza métricas no DynamoDB
                    self._update_strategy_in_db(strategy_id, {
                        'performance_metrics': performance
                    })
                    
                    # Verifica se deve acionar Gravedigger
                    if self._should_trigger_gravedigger(strategy):
                        self._update_strategy_in_db(strategy_id, {
                            'status': StrategyStatus.GRAVEDIGGER.value
                        })
                        
                        # Enfileira job de revalidação
                        symbol, timeframe = strategy_id.split('_')
                        self._enqueue_validation_job(strategy_id, symbol, timeframe)
                        
                        results['gravedigger_triggered'] += 1
                        logger.warning(f"Gravedigger acionado para: {strategy_id}")
                    
                    # Verifica se precisa retreinar
                    elif self._should_retrain_model(strategy):
                        self._trigger_model_retraining(strategy_id)
                        
                        self._update_strategy_in_db(strategy_id, {
                            'last_retrain_date': datetime.now().isoformat()
                        })
                        
                        results['retraining_triggered'] += 1
                        logger.info(f"Retreino acionado para: {strategy_id}")
                
                except Exception as e:
                    error_msg = f"Erro ao monitorar estratégia {strategy_id}: {str(e)}"
                    results['errors'].append(error_msg)
                    logger.error(error_msg)
            
            logger.info(f"Monitoramento concluído: {results}")
            return results
            
        except Exception as e:
            error_msg = f"Erro no monitoramento de produção: {str(e)}"
            results['errors'].append(error_msg)
            logger.error(error_msg)
            return results
    
    def promote_incubation_strategies(self) -> Dict[str, Any]:
        """
        Promove estratégias qualificadas da incubação para produção.
        
        Returns:
            Resultado das promoções
        """
        results = {
            'strategies_evaluated': 0,
            'strategies_promoted': 0,
            'errors': []
        }
        
        try:
            # Busca estratégias em incubação
            response = self.strategy_table.scan(
                FilterExpression='#status = :status',
                ExpressionAttributeNames={'#status': 'status'},
                ExpressionAttributeValues={':status': StrategyStatus.INCUBATION.value}
            )
            
            for strategy in response.get('Items', []):
                results['strategies_evaluated'] += 1
                strategy_id = strategy['strategy_id']
                
                try:
                    if self._should_promote_from_incubation(strategy):
                        # Promove para produção
                        self._update_strategy_in_db(strategy_id, {
                            'status': StrategyStatus.PRODUCTION.value,
                            'promoted_to_production_at': datetime.now().isoformat()
                        })
                        
                        results['strategies_promoted'] += 1
                        logger.info(f"Estratégia promovida para produção: {strategy_id}")
                
                except Exception as e:
                    error_msg = f"Erro ao avaliar estratégia {strategy_id}: {str(e)}"
                    results['errors'].append(error_msg)
                    logger.error(error_msg)
            
            logger.info(f"Promoção de incubação concluída: {results}")
            return results
            
        except Exception as e:
            error_msg = f"Erro na promoção de incubação: {str(e)}"
            results['errors'].append(error_msg)
            logger.error(error_msg)
            return results
    
    def initialize_missing_strategies(self) -> Dict[str, Any]:
        """
        Inicializa estratégias que ainda não existem no sistema.
        
        Returns:
            Resultado da inicialização
        """
        results = {
            'strategies_created': 0,
            'strategies_existing': 0,
            'errors': []
        }
        
        try:
            symbols = self.config.get('symbols', [])
            timeframes = self.config.get('timeframes', [])
            
            for symbol in symbols:
                for timeframe in timeframes:
                    strategy_id = self._get_strategy_key(symbol, timeframe)
                    
                    try:
                        # Verifica se já existe
                        existing = self._get_strategy_from_db(strategy_id)
                        
                        if existing:
                            results['strategies_existing'] += 1
                        else:
                            # Cria nova estratégia
                            if self._create_strategy_in_db(strategy_id, symbol, timeframe):
                                # Enfileira job de validação inicial
                                self._enqueue_validation_job(strategy_id, symbol, timeframe)
                                results['strategies_created'] += 1
                    
                    except Exception as e:
                        error_msg = f"Erro ao inicializar estratégia {strategy_id}: {str(e)}"
                        results['errors'].append(error_msg)
                        logger.error(error_msg)
            
            logger.info(f"Inicialização de estratégias concluída: {results}")
            return results
            
        except Exception as e:
            error_msg = f"Erro na inicialização de estratégias: {str(e)}"
            results['errors'].append(error_msg)
            logger.error(error_msg)
            return results
    
    def run_orchestration(self) -> Dict[str, Any]:
        """
        Executa o ciclo completo de orquestração.
        
        Returns:
            Resultado consolidado da orquestração
        """
        start_time = datetime.now()
        
        results = {
            'start_time': start_time.isoformat(),
            'master_trading_switch': self.config.get('master_trading_switch', 'false'),
            'initialization_results': {},
            'monitoring_results': {},
            'promotion_results': {},
            'errors': [],
            'success': True
        }
        
        try:
            logger.info("=== INÍCIO DA ORQUESTRAÇÃO ===")
            
            # Verifica se o trading está habilitado
            if self.config.get('master_trading_switch', 'false').lower() != 'true':
                logger.warning("Master trading switch está desabilitado")
                results['success'] = False
                results['errors'].append("Master trading switch desabilitado")
                return results
            
            # 1. Inicializa estratégias faltantes
            logger.info("Inicializando estratégias faltantes...")
            results['initialization_results'] = self.initialize_missing_strategies()
            
            # 2. Monitora estratégias em produção
            logger.info("Monitorando estratégias em produção...")
            results['monitoring_results'] = self.monitor_production_strategies()
            
            # 3. Promove estratégias da incubação
            logger.info("Promovendo estratégias da incubação...")
            results['promotion_results'] = self.promote_incubation_strategies()
            
            # Consolida erros
            all_errors = []
            all_errors.extend(results['initialization_results'].get('errors', []))
            all_errors.extend(results['monitoring_results'].get('errors', []))
            all_errors.extend(results['promotion_results'].get('errors', []))
            
            results['errors'] = all_errors
            results['success'] = len(all_errors) == 0
            
            # Finaliza
            end_time = datetime.now()
            results['end_time'] = end_time.isoformat()
            results['duration_seconds'] = (end_time - start_time).total_seconds()
            
            logger.info(f"Orquestração concluída: {results['success']}")
            return results
            
        except Exception as e:
            error_msg = f"Erro crítico na orquestração: {str(e)}"
            logger.error(error_msg)
            results['errors'].append(error_msg)
            results['success'] = False
            return results


def lambda_handler(event, context):
    """
    Função principal para execução no AWS Lambda.
    
    Args:
        event: Evento do Lambda
        context: Contexto de execução do Lambda
        
    Returns:
        Resultado da orquestração
    """
    try:
        logger.info("=== INÍCIO DA EXECUÇÃO DO ORQUESTRADOR ===")
        logger.info(f"Event: {json.dumps(event, default=str)}")
        
        # Inicializa o orquestrador
        orchestrator = TradingOrchestrator()
        
        # Executa a orquestração
        results = orchestrator.run_orchestration()
        
        # Prepara resposta
        response = {
            'statusCode': 200 if results['success'] else 500,
            'body': json.dumps(results, default=str),
            'headers': {
                'Content-Type': 'application/json'
            }
        }
        
        logger.info("=== FIM DA EXECUÇÃO DO ORQUESTRADOR ===")
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


# Para execução local/teste
if __name__ == "__main__":
    # Configura variáveis de ambiente para teste local
    os.environ['PROJECT_NAME'] = 'CryptoTradingSystem'
    os.environ['DATA_BUCKET'] = 'crypto-trading-data-test'
    os.environ['VALIDATION_QUEUE_URL'] = 'https://sqs.us-east-1.amazonaws.com/123456789012/test-queue'
    
    # Executa o orquestrador
    orchestrator = TradingOrchestrator()
    results = orchestrator.run_orchestration()
    
    print("=== RESULTADOS DA ORQUESTRAÇÃO ===")
    print(json.dumps(results, indent=2, default=str))

