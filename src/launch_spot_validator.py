"""
Sistema de Trading Quantitativo - Lançador de Instâncias Spot
Versão: 2.01
Data: 26 de agosto de 2025

Este módulo é responsável por:
- Receber mensagens da fila SQS com jobs de validação
- Verificar limites de custo mensal
- Provisionar instâncias EC2 Spot para execução de validação
- Gerenciar o ciclo de vida das instâncias
"""

import json
import logging
import os
import base64
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from decimal import Decimal

import boto3
from botocore.exceptions import ClientError

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SpotInstanceLauncher:
    """
    Gerenciador de instâncias EC2 Spot para validação de estratégias.
    
    Responsável por provisionar instâncias sob demanda, respeitando
    limites de custo e otimizando para máxima economia.
    """
    
    def __init__(self):
        """Inicializa o lançador com clientes AWS."""
        self.ec2 = boto3.client('ec2')
        self.ssm = boto3.client('ssm')
        self.ce = boto3.client('ce')  # Cost Explorer
        self.sqs = boto3.client('sqs')
        self.dynamodb = boto3.resource('dynamodb')
        
        # Carrega configurações
        self.config = self._load_configuration()
        
        # Tabela para tracking de instâncias
        self.instances_table = self.dynamodb.Table(f"{self.config['project_name']}-SpotInstances")
        
        logger.info("SpotInstanceLauncher inicializado com sucesso")
    
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
                'vpc_id': os.environ.get('VPC_ID', ''),
                'subnet_id': os.environ.get('SUBNET_ID', ''),
                'security_group_id': os.environ.get('SECURITY_GROUP_ID', ''),
                'key_pair_name': os.environ.get('KEY_PAIR_NAME', ''),
            }
            
            # Parâmetros do SSM
            ssm_params = [
                '/trading_system/cost/monthly_cost_limit',
                '/trading_system/cost/spot_min_discount_perc',
                '/trading_system/cost/spot_instance_type',
                '/trading_system/environment/github_repo'
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
                            'monthly_cost_limit': '10.0',
                            'spot_min_discount_perc': '0.95',
                            'spot_instance_type': ['c6i.large', 'c5.xlarge', 'c6a.2xlarge', 'm6i.large', 'm5.xlarge'],
                            'github_repo': 'git@github.com:tpompeu/cripto-trading-system.git'
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
    
    def _get_monthly_costs(self) -> float:
        """
        Consulta o AWS Cost Explorer para obter custos do mês atual.
        
        Returns:
            Custo acumulado do mês em USD
        """
        try:
            # Período do mês atual
            now = datetime.now()
            start_of_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            end_of_month = (start_of_month + timedelta(days=32)).replace(day=1) - timedelta(days=1)
            
            response = self.ce.get_cost_and_usage(
                TimePeriod={
                    'Start': start_of_month.strftime('%Y-%m-%d'),
                    'End': end_of_month.strftime('%Y-%m-%d')
                },
                Granularity='MONTHLY',
                Metrics=['BlendedCost'],
                GroupBy=[
                    {
                        'Type': 'TAG',
                        'Key': 'Project'
                    }
                ]
            )
            
            total_cost = 0.0
            for result in response.get('ResultsByTime', []):
                for group in result.get('Groups', []):
                    # Filtra apenas custos do projeto
                    if 'CryptoTradingSystem' in group.get('Keys', []):
                        amount = group.get('Metrics', {}).get('BlendedCost', {}).get('Amount', '0')
                        total_cost += float(amount)
            
            logger.info(f"Custo mensal atual: ${total_cost:.2f}")
            return total_cost
            
        except Exception as e:
            logger.error(f"Erro ao consultar custos: {str(e)}")
            # Em caso de erro, assume custo zero para não bloquear
            return 0.0
    
    def _estimate_instance_cost(self, instance_type: str, hours: int = 24) -> float:
        """
        Estima o custo de uma instância Spot por período.
        
        Args:
            instance_type: Tipo da instância (ex: 'c6i.large')
            hours: Número de horas estimadas
            
        Returns:
            Custo estimado em USD
        """
        try:
            # Preços aproximados por hora (Spot pricing varia)
            # Estes são preços de referência, o real pode ser menor
            spot_prices = {
                'c6i.large': 0.05,
                'c5.xlarge': 0.08,
                'c6a.2xlarge': 0.15,
                'm6i.large': 0.05,
                'm5.xlarge': 0.08
            }
            
            base_price = spot_prices.get(instance_type, 0.10)  # Default
            estimated_cost = base_price * hours
            
            logger.info(f"Custo estimado para {instance_type} por {hours}h: ${estimated_cost:.2f}")
            return estimated_cost
            
        except Exception as e:
            logger.error(f"Erro ao estimar custo: {str(e)}")
            return 10.0  # Estimativa conservadora
    
    def _can_launch_instance(self, instance_type: str) -> bool:
        """
        Verifica se pode lançar uma instância respeitando limites de custo.
        
        Args:
            instance_type: Tipo da instância
            
        Returns:
            True se pode lançar
        """
        try:
            # Custo atual do mês
            current_cost = self._get_monthly_costs()
            
            # Custo estimado da nova instância
            estimated_cost = self._estimate_instance_cost(instance_type)
            
            # Limite mensal
            monthly_limit = float(self.config.get('monthly_cost_limit', 10.0))
            
            # Verifica se não excede o limite
            total_projected = current_cost + estimated_cost
            
            if total_projected > monthly_limit:
                logger.warning(f"Limite de custo excedido: ${total_projected:.2f} > ${monthly_limit:.2f}")
                return False
            
            logger.info(f"Custo OK: ${total_projected:.2f} <= ${monthly_limit:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao verificar limite de custo: {str(e)}")
            return False
    
    def _get_spot_price_history(self, instance_type: str) -> float:
        """
        Consulta histórico de preços Spot para uma instância.
        
        Args:
            instance_type: Tipo da instância
            
        Returns:
            Preço Spot atual
        """
        try:
            response = self.ec2.describe_spot_price_history(
                InstanceTypes=[instance_type],
                ProductDescriptions=['Linux/UNIX'],
                MaxResults=1
            )
            
            if response.get('SpotPriceHistory'):
                price = float(response['SpotPriceHistory'][0]['SpotPrice'])
                logger.info(f"Preço Spot atual para {instance_type}: ${price:.4f}/hora")
                return price
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Erro ao consultar preço Spot: {str(e)}")
            return 0.0
    
    def _select_best_instance_type(self) -> Optional[str]:
        """
        Seleciona o melhor tipo de instância baseado em preço e disponibilidade.
        
        Returns:
            Tipo de instância selecionado ou None
        """
        try:
            instance_types = self.config.get('spot_instance_type', ['c6i.large'])
            min_discount = float(self.config.get('spot_min_discount_perc', 0.95))
            
            best_option = None
            best_price = float('inf')
            
            for instance_type in instance_types:
                # Verifica limite de custo
                if not self._can_launch_instance(instance_type):
                    continue
                
                # Consulta preço Spot atual
                spot_price = self._get_spot_price_history(instance_type)
                
                if spot_price > 0 and spot_price < best_price:
                    best_option = instance_type
                    best_price = spot_price
            
            if best_option:
                logger.info(f"Melhor opção selecionada: {best_option} (${best_price:.4f}/hora)")
            else:
                logger.warning("Nenhuma instância disponível dentro dos critérios")
            
            return best_option
            
        except Exception as e:
            logger.error(f"Erro ao selecionar instância: {str(e)}")
            return None
    
    def _create_user_data_script(self, job_data: Dict[str, Any]) -> str:
        """
        Cria script user_data para configurar a instância.
        
        Args:
            job_data: Dados do job de validação
            
        Returns:
            Script user_data codificado em base64
        """
        try:
            script = f"""#!/bin/bash
set -e

# Logs de inicialização
exec > >(tee /var/log/user-data.log|logger -t user-data -s 2>/dev/console) 2>&1
echo "Iniciando configuração da instância Spot..."

# Atualiza sistema
yum update -y
yum install -y git python3 python3-pip htop

# Instala AWS CLI v2
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Configura Python
python3 -m pip install --upgrade pip
pip3 install boto3 pandas numpy scikit-learn lightgbm ta-lib requests

# Cria diretório de trabalho
mkdir -p /opt/trading-system
cd /opt/trading-system

# Clona repositório (assumindo que a instância tem acesso via IAM role)
# Em produção, usar AWS CodeCommit ou S3 para código
aws s3 cp s3://{self.config['data_bucket']}/code/model_validator.py ./
aws s3 cp s3://{self.config['data_bucket']}/code/random_signal_simulator.py ./
aws s3 cp s3://{self.config['data_bucket']}/code/config_service.py ./

# Configura variáveis de ambiente
export PROJECT_NAME="{self.config['project_name']}"
export DATA_BUCKET="{self.config['data_bucket']}"
export STRATEGY_ID="{job_data.get('strategy_id', '')}"
export SYMBOL="{job_data.get('symbol', '')}"
export TIMEFRAME="{job_data.get('timeframe', '')}"
export JOB_TYPE="{job_data.get('job_type', '')}"

# Executa o job de validação
echo "Iniciando job de validação: $STRATEGY_ID"
python3 model_validator.py

# Sinaliza conclusão
echo "Job concluído com sucesso"

# Auto-termina a instância após conclusão
shutdown -h +5
"""
            
            # Codifica em base64
            encoded_script = base64.b64encode(script.encode('utf-8')).decode('utf-8')
            return encoded_script
            
        except Exception as e:
            logger.error(f"Erro ao criar user_data script: {str(e)}")
            return ""
    
    def _launch_spot_instance(self, job_data: Dict[str, Any]) -> Optional[str]:
        """
        Lança uma instância EC2 Spot para executar validação.
        
        Args:
            job_data: Dados do job de validação
            
        Returns:
            ID da instância lançada ou None
        """
        try:
            # Seleciona melhor tipo de instância
            instance_type = self._select_best_instance_type()
            if not instance_type:
                logger.error("Nenhum tipo de instância disponível")
                return None
            
            # Cria user_data script
            user_data = self._create_user_data_script(job_data)
            if not user_data:
                logger.error("Falha ao criar user_data script")
                return None
            
            # Configuração da instância
            launch_spec = {
                'ImageId': 'ami-0c02fb55956c7d316',  # Amazon Linux 2 (us-east-1)
                'InstanceType': instance_type,
                'KeyName': self.config.get('key_pair_name', ''),
                'SecurityGroupIds': [self.config['security_group_id']],
                'SubnetId': self.config['subnet_id'],
                'UserData': user_data,
                'IamInstanceProfile': {
                    'Name': f"{self.config['project_name']}-SpotInstanceProfile"
                },
                'TagSpecifications': [
                    {
                        'ResourceType': 'instance',
                        'Tags': [
                            {'Key': 'Name', 'Value': f"{self.config['project_name']}-Validator-{job_data.get('strategy_id', '')}"},
                            {'Key': 'Project', 'Value': self.config['project_name']},
                            {'Key': 'Purpose', 'Value': 'ModelValidation'},
                            {'Key': 'StrategyId', 'Value': job_data.get('strategy_id', '')},
                            {'Key': 'JobType', 'Value': job_data.get('job_type', '')},
                            {'Key': 'LaunchedAt', 'Value': datetime.now().isoformat()}
                        ]
                    }
                ]
            }
            
            # Solicita instância Spot
            response = self.ec2.request_spot_instances(
                SpotPrice='0.50',  # Preço máximo que aceita pagar
                LaunchSpecification=launch_spec,
                Type='one-time',
                InstanceCount=1
            )
            
            spot_request_id = response['SpotInstanceRequests'][0]['SpotInstanceRequestId']
            logger.info(f"Solicitação Spot criada: {spot_request_id}")
            
            # Aguarda instância ser lançada
            waiter = self.ec2.get_waiter('spot_instance_request_fulfilled')
            waiter.wait(
                SpotInstanceRequestIds=[spot_request_id],
                WaiterConfig={'Delay': 15, 'MaxAttempts': 20}
            )
            
            # Obtém ID da instância
            response = self.ec2.describe_spot_instance_requests(
                SpotInstanceRequestIds=[spot_request_id]
            )
            
            instance_id = response['SpotInstanceRequests'][0].get('InstanceId')
            
            if instance_id:
                logger.info(f"Instância Spot lançada: {instance_id}")
                
                # Registra na tabela de tracking
                self._track_instance(instance_id, job_data, instance_type)
                
                return instance_id
            else:
                logger.error("Falha ao obter ID da instância")
                return None
            
        except Exception as e:
            logger.error(f"Erro ao lançar instância Spot: {str(e)}")
            return None
    
    def _track_instance(self, instance_id: str, job_data: Dict[str, Any], instance_type: str):
        """
        Registra instância na tabela de tracking.
        
        Args:
            instance_id: ID da instância
            job_data: Dados do job
            instance_type: Tipo da instância
        """
        try:
            item = {
                'instance_id': instance_id,
                'strategy_id': job_data.get('strategy_id', ''),
                'job_type': job_data.get('job_type', ''),
                'instance_type': instance_type,
                'launched_at': datetime.now().isoformat(),
                'status': 'running',
                'job_data': job_data
            }
            
            self.instances_table.put_item(Item=item)
            logger.info(f"Instância {instance_id} registrada no tracking")
            
        except Exception as e:
            logger.error(f"Erro ao registrar instância: {str(e)}")
    
    def process_validation_job(self, message_body: str) -> Dict[str, Any]:
        """
        Processa um job de validação da fila SQS.
        
        Args:
            message_body: Corpo da mensagem SQS
            
        Returns:
            Resultado do processamento
        """
        try:
            job_data = json.loads(message_body)
            strategy_id = job_data.get('strategy_id', '')
            
            logger.info(f"Processando job de validação: {strategy_id}")
            
            # Lança instância Spot
            instance_id = self._launch_spot_instance(job_data)
            
            if instance_id:
                return {
                    'success': True,
                    'instance_id': instance_id,
                    'strategy_id': strategy_id,
                    'message': 'Instância Spot lançada com sucesso'
                }
            else:
                return {
                    'success': False,
                    'strategy_id': strategy_id,
                    'message': 'Falha ao lançar instância Spot'
                }
            
        except Exception as e:
            logger.error(f"Erro ao processar job: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'message': 'Erro no processamento do job'
            }


def lambda_handler(event, context):
    """
    Função principal para execução no AWS Lambda.
    
    Args:
        event: Evento do Lambda (mensagens SQS)
        context: Contexto de execução do Lambda
        
    Returns:
        Resultado do processamento
    """
    try:
        logger.info("=== INÍCIO DA EXECUÇÃO DO SPOT LAUNCHER ===")
        logger.info(f"Event: {json.dumps(event, default=str)}")
        
        launcher = SpotInstanceLauncher()
        results = []
        
        # Processa cada mensagem SQS
        for record in event.get('Records', []):
            message_body = record.get('body', '{}')
            result = launcher.process_validation_job(message_body)
            results.append(result)
        
        # Prepara resposta
        success_count = sum(1 for r in results if r.get('success', False))
        total_count = len(results)
        
        response = {
            'statusCode': 200 if success_count == total_count else 207,  # 207 = Multi-Status
            'body': json.dumps({
                'processed': total_count,
                'successful': success_count,
                'failed': total_count - success_count,
                'results': results
            }),
            'headers': {
                'Content-Type': 'application/json'
            }
        }
        
        logger.info("=== FIM DA EXECUÇÃO DO SPOT LAUNCHER ===")
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
    
    # Simula evento SQS
    test_event = {
        'Records': [
            {
                'body': json.dumps({
                    'job_type': 'walk_forward_validation',
                    'strategy_id': 'BTCUSDT_1h',
                    'symbol': 'BTCUSDT',
                    'timeframe': '1h',
                    'timestamp': datetime.now().isoformat()
                })
            }
        ]
    }
    
    # Executa o launcher
    result = lambda_handler(test_event, None)
    print("=== RESULTADO ===")
    print(json.dumps(result, indent=2, default=str))

