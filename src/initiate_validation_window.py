"""
Sistema de Trading Quantitativo - Iniciador de Janela de Validação
Versão: 2.01
Data: 26 de agosto de 2025

Este módulo é responsável por:
- Iniciar janelas de oportunidade para alocação de instâncias Spot
- Monitorar fila SQS de jobs de validação
- Acionar o launch_spot_validator quando há jobs pendentes
- Otimizar timing para melhor disponibilidade e preços Spot
"""

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

import boto3
from botocore.exceptions import ClientError

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ValidationWindowInitiator:
    """
    Iniciador de janelas de validação para instâncias Spot.
    
    Monitora a fila de jobs e otimiza o timing de lançamento
    de instâncias para maximizar disponibilidade e minimizar custos.
    """
    
    def __init__(self):
        """Inicializa o iniciador com clientes AWS."""
        self.sqs = boto3.client('sqs')
        self.lambda_client = boto3.client('lambda')
        self.ec2 = boto3.client('ec2')
        self.ssm = boto3.client('ssm')
        self.dynamodb = boto3.resource('dynamodb')
        
        # Carrega configurações
        self.config = self._load_configuration()
        
        # Tabela para tracking de janelas
        self.windows_table = self.dynamodb.Table(f"{self.config['project_name']}-ValidationWindows")
        
        logger.info("ValidationWindowInitiator inicializado com sucesso")
    
    def _load_configuration(self) -> Dict[str, Any]:
        """
        Carrega configurações do AWS Systems Manager Parameter Store.
        
        Returns:
            Dict: Configurações do sistema
        """
        try:
            config = {
                'project_name': os.environ.get('PROJECT_NAME', 'CryptoTradingSystem'),
                'validation_queue_url': os.environ.get('VALIDATION_QUEUE_URL', ''),
                'spot_launcher_function': os.environ.get('SPOT_LAUNCHER_FUNCTION', ''),
            }
            
            # Parâmetros do SSM
            ssm_params = [
                '/trading_system/cost/spot_instance_type',
                '/trading_system/cost/monthly_cost_limit'
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
                            'spot_instance_type': ['c6i.large', 'c5.xlarge'],
                            'monthly_cost_limit': '10.0'
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
    
    def _get_queue_metrics(self) -> Dict[str, int]:
        """
        Obtém métricas da fila SQS de validação.
        
        Returns:
            Métricas da fila
        """
        try:
            response = self.sqs.get_queue_attributes(
                QueueUrl=self.config['validation_queue_url'],
                AttributeNames=[
                    'ApproximateNumberOfMessages',
                    'ApproximateNumberOfMessagesNotVisible',
                    'ApproximateNumberOfMessagesDelayed'
                ]
            )
            
            attributes = response.get('Attributes', {})
            metrics = {
                'messages_available': int(attributes.get('ApproximateNumberOfMessages', 0)),
                'messages_in_flight': int(attributes.get('ApproximateNumberOfMessagesNotVisible', 0)),
                'messages_delayed': int(attributes.get('ApproximateNumberOfMessagesDelayed', 0))
            }
            
            logger.info(f"Métricas da fila: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Erro ao obter métricas da fila: {str(e)}")
            return {'messages_available': 0, 'messages_in_flight': 0, 'messages_delayed': 0}
    
    def _get_spot_availability_score(self) -> float:
        """
        Calcula score de disponibilidade de instâncias Spot.
        
        Returns:
            Score de 0.0 a 1.0 (1.0 = melhor disponibilidade)
        """
        try:
            instance_types = self.config.get('spot_instance_type', ['c6i.large'])
            total_score = 0.0
            
            for instance_type in instance_types:
                try:
                    # Consulta histórico de preços Spot
                    response = self.ec2.describe_spot_price_history(
                        InstanceTypes=[instance_type],
                        ProductDescriptions=['Linux/UNIX'],
                        MaxResults=10,
                        StartTime=datetime.now() - timedelta(hours=24)
                    )
                    
                    prices = [float(item['SpotPrice']) for item in response.get('SpotPriceHistory', [])]
                    
                    if prices:
                        # Score baseado na estabilidade de preços
                        avg_price = sum(prices) / len(prices)
                        price_variance = sum((p - avg_price) ** 2 for p in prices) / len(prices)
                        
                        # Menor variância = melhor score
                        stability_score = max(0.0, 1.0 - (price_variance / avg_price))
                        total_score += stability_score
                    
                except Exception as e:
                    logger.warning(f"Erro ao avaliar {instance_type}: {str(e)}")
                    total_score += 0.5  # Score neutro
            
            # Média dos scores
            final_score = total_score / len(instance_types) if instance_types else 0.5
            logger.info(f"Score de disponibilidade Spot: {final_score:.2f}")
            
            return final_score
            
        except Exception as e:
            logger.error(f"Erro ao calcular disponibilidade Spot: {str(e)}")
            return 0.5  # Score neutro em caso de erro
    
    def _is_optimal_time_window(self) -> bool:
        """
        Verifica se é um momento ótimo para lançar instâncias Spot.
        
        Returns:
            True se é momento ótimo
        """
        try:
            now = datetime.now()
            hour = now.hour
            
            # Horários com melhor disponibilidade Spot (baseado em padrões históricos)
            # Evita horários de pico de uso comercial
            optimal_hours = [
                (0, 6),   # Madrugada
                (14, 16), # Meio da tarde
                (22, 24)  # Final da noite
            ]
            
            is_optimal = any(start <= hour < end for start, end in optimal_hours)
            
            # Considera também disponibilidade atual
            availability_score = self._get_spot_availability_score()
            
            # Combina horário ótimo com disponibilidade
            is_good_time = is_optimal or availability_score > 0.7
            
            logger.info(f"Janela ótima: {is_good_time} (hora: {hour}, score: {availability_score:.2f})")
            return is_good_time
            
        except Exception as e:
            logger.error(f"Erro ao verificar janela ótima: {str(e)}")
            return True  # Em caso de erro, permite lançamento
    
    def _should_initiate_validation(self) -> bool:
        """
        Decide se deve iniciar processo de validação.
        
        Returns:
            True se deve iniciar
        """
        try:
            # Verifica se há jobs na fila
            metrics = self._get_queue_metrics()
            has_pending_jobs = metrics['messages_available'] > 0
            
            if not has_pending_jobs:
                logger.info("Nenhum job pendente na fila")
                return False
            
            # Verifica se não há muitos jobs já em processamento
            max_concurrent = 3  # Máximo de instâncias simultâneas
            if metrics['messages_in_flight'] >= max_concurrent:
                logger.info(f"Muitos jobs em processamento: {metrics['messages_in_flight']}")
                return False
            
            # Verifica janela ótima de tempo
            if not self._is_optimal_time_window():
                logger.info("Não é janela ótima para lançamento")
                return False
            
            logger.info("Condições favoráveis para iniciar validação")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao decidir iniciação: {str(e)}")
            return False
    
    def _invoke_spot_launcher(self) -> Dict[str, Any]:
        """
        Invoca a Lambda de lançamento de instâncias Spot.
        
        Returns:
            Resultado da invocação
        """
        try:
            function_name = self.config.get('spot_launcher_function', 
                                          f"{self.config['project_name']}-LaunchSpotValidator")
            
            # Payload vazio - a Lambda lerá da fila SQS
            payload = {
                'source': 'validation_window_initiator',
                'timestamp': datetime.now().isoformat()
            }
            
            response = self.lambda_client.invoke(
                FunctionName=function_name,
                InvocationType='Event',  # Assíncrono
                Payload=json.dumps(payload)
            )
            
            logger.info(f"Spot launcher invocado: {function_name}")
            return {
                'success': True,
                'function_name': function_name,
                'status_code': response.get('StatusCode', 0)
            }
            
        except Exception as e:
            logger.error(f"Erro ao invocar spot launcher: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _record_validation_window(self, initiated: bool, metrics: Dict[str, Any]) -> bool:
        """
        Registra informações sobre a janela de validação.
        
        Args:
            initiated: Se a validação foi iniciada
            metrics: Métricas coletadas
            
        Returns:
            True se registrado com sucesso
        """
        try:
            item = {
                'window_id': f"window_{int(datetime.now().timestamp())}",
                'timestamp': datetime.now().isoformat(),
                'initiated': initiated,
                'queue_metrics': metrics,
                'spot_availability_score': self._get_spot_availability_score(),
                'hour_of_day': datetime.now().hour,
                'day_of_week': datetime.now().weekday()
            }
            
            self.windows_table.put_item(Item=item)
            logger.info(f"Janela de validação registrada: {item['window_id']}")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao registrar janela: {str(e)}")
            return False
    
    def run_validation_window_check(self) -> Dict[str, Any]:
        """
        Executa verificação completa da janela de validação.
        
        Returns:
            Resultado da verificação
        """
        start_time = datetime.now()
        
        results = {
            'start_time': start_time.isoformat(),
            'queue_metrics': {},
            'spot_availability_score': 0.0,
            'validation_initiated': False,
            'launcher_result': {},
            'errors': [],
            'success': True
        }
        
        try:
            logger.info("=== INÍCIO DA VERIFICAÇÃO DE JANELA DE VALIDAÇÃO ===")
            
            # Coleta métricas da fila
            results['queue_metrics'] = self._get_queue_metrics()
            
            # Avalia disponibilidade Spot
            results['spot_availability_score'] = self._get_spot_availability_score()
            
            # Decide se deve iniciar validação
            should_initiate = self._should_initiate_validation()
            
            if should_initiate:
                # Invoca spot launcher
                launcher_result = self._invoke_spot_launcher()
                results['launcher_result'] = launcher_result
                results['validation_initiated'] = launcher_result.get('success', False)
                
                if not launcher_result.get('success', False):
                    results['errors'].append(f"Falha ao invocar spot launcher: {launcher_result.get('error', '')}")
            else:
                logger.info("Condições não favoráveis para iniciar validação")
            
            # Registra janela para análise histórica
            self._record_validation_window(results['validation_initiated'], results['queue_metrics'])
            
            # Finaliza
            end_time = datetime.now()
            results['end_time'] = end_time.isoformat()
            results['duration_seconds'] = (end_time - start_time).total_seconds()
            results['success'] = len(results['errors']) == 0
            
            logger.info(f"Verificação de janela concluída: {results}")
            return results
            
        except Exception as e:
            error_msg = f"Erro crítico na verificação de janela: {str(e)}"
            logger.error(error_msg)
            results['errors'].append(error_msg)
            results['success'] = False
            return results


def lambda_handler(event, context):
    """
    Função principal para execução no AWS Lambda.
    
    Args:
        event: Evento do Lambda (EventBridge Scheduler)
        context: Contexto de execução do Lambda
        
    Returns:
        Resultado da verificação
    """
    try:
        logger.info("=== INÍCIO DA EXECUÇÃO DO VALIDATION WINDOW INITIATOR ===")
        logger.info(f"Event: {json.dumps(event, default=str)}")
        
        # Inicializa o iniciador
        initiator = ValidationWindowInitiator()
        
        # Executa verificação
        results = initiator.run_validation_window_check()
        
        # Prepara resposta
        response = {
            'statusCode': 200 if results['success'] else 500,
            'body': json.dumps(results, default=str),
            'headers': {
                'Content-Type': 'application/json'
            }
        }
        
        logger.info("=== FIM DA EXECUÇÃO DO VALIDATION WINDOW INITIATOR ===")
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
    os.environ['VALIDATION_QUEUE_URL'] = 'https://sqs.ap-southeast-1.amazonaws.com/123456789012/test-queue'
    os.environ['SPOT_LAUNCHER_FUNCTION'] = 'CryptoTradingSystem-LaunchSpotValidator'
    
    # Executa o iniciador
    initiator = ValidationWindowInitiator()
    results = initiator.run_validation_window_check()
    
    print("=== RESULTADOS ===")
    print(json.dumps(results, indent=2, default=str))

