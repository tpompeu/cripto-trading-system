#!/usr/bin/env python3
"""
Script para configurar parâmetros no AWS Systems Manager Parameter Store.

Este script lê o arquivo config.json e cria/atualiza os parâmetros necessários
no Parameter Store para o sistema de trading funcionar corretamente.

Uso:
    python3 setup_parameters.py [--dry-run] [--region us-east-1] [--profile PROFILE]
"""

import argparse
import json
import sys
from typing import Dict, Any

import boto3
from botocore.exceptions import ClientError


class ParameterStoreManager:
    """Gerencia parâmetros no AWS Systems Manager Parameter Store."""
    
    def __init__(self, region: str = 'us-east-1', dry_run: bool = False, profile: str = None):
        """
        Inicializa o gerenciador de parâmetros.
        
        Args:
            region: Região AWS
            dry_run: Se True, apenas mostra o que seria feito sem executar
            profile: Perfil AWS CLI a ser usado
        """
        self.region = region
        self.dry_run = dry_run
        self.profile = profile
        
        if not dry_run:
            if profile:
                session = boto3.Session(profile_name=profile, region_name=region)
                self.ssm_client = session.client('ssm')
            else:
                self.ssm_client = boto3.client('ssm', region_name=region)
        
        print(f"ParameterStoreManager inicializado - Região: {region}, Dry Run: {dry_run}, Profile: {profile or 'default'}")
    
    def load_config(self, config_file: str = 'config.json') -> Dict[str, Any]:
        """
        Carrega configurações do arquivo JSON.
        
        Args:
            config_file: Caminho para o arquivo de configuração
            
        Returns:
            Dicionário com as configurações
        """
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            print(f"✓ Configuração carregada de {config_file}")
            return config
        except FileNotFoundError:
            print(f"❌ Arquivo {config_file} não encontrado")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"❌ Erro ao decodificar JSON: {e}")
            sys.exit(1)
    
    def put_parameter(self, name: str, value: str, parameter_type: str = 'String', 
                     description: str = '', overwrite: bool = True) -> bool:
        """
        Cria ou atualiza um parâmetro no Parameter Store.
        
        Args:
            name: Nome do parâmetro
            value: Valor do parâmetro
            parameter_type: Tipo do parâmetro (String, StringList, SecureString)
            description: Descrição do parâmetro
            overwrite: Se deve sobrescrever parâmetros existentes
            
        Returns:
            True se bem-sucedido, False caso contrário
        """
        if self.dry_run:
            print(f"[DRY RUN] Criaria parâmetro: {name} = {value[:50]}...")
            return True
        
        try:
            # Primeiro, verifica se o parâmetro já existe
            try:
                self.ssm_client.get_parameter(Name=name, WithDecryption=False)
                parameter_exists = True
            except ClientError as e:
                if e.response['Error']['Code'] == 'ParameterNotFound':
                    parameter_exists = False
                else:
                    raise e
            
            if parameter_exists:
                # Se o parâmetro existe, atualiza sem tags
                self.ssm_client.put_parameter(
                    Name=name,
                    Value=value,
                    Type=parameter_type,
                    Description=description,
                    Overwrite=True
                )
                print(f"✓ Parâmetro atualizado: {name}")
                
                # Adiciona tags separadamente se necessário
                try:
                    self.ssm_client.add_tags_to_resource(
                        ResourceType='Parameter',
                        ResourceId=name,
                        Tags=[
                            {'Key': 'Project', 'Value': 'CryptoTradingSystem'},
                            {'Key': 'Environment', 'Value': 'Production'},
                            {'Key': 'ManagedBy', 'Value': 'setup_parameters.py'}
                        ]
                    )
                    print(f"✓ Tags adicionadas ao parâmetro: {name}")
                except ClientError as e:
                    print(f"⚠️  Não foi possível adicionar tags ao parâmetro {name}: {e}")
                    
            else:
                # Se o parâmetro não existe, cria com tags
                self.ssm_client.put_parameter(
                    Name=name,
                    Value=value,
                    Type=parameter_type,
                    Description=description,
                    Overwrite=False,
                    Tags=[
                        {
                            'Key': 'Project',
                            'Value': 'CryptoTradingSystem'
                        },
                        {
                            'Key': 'Environment',
                            'Value': 'Production'
                        },
                        {
                            'Key': 'ManagedBy',
                            'Value': 'setup_parameters.py'
                        }
                    ]
                )
                print(f"✓ Parâmetro criado: {name}")
                
            return True
            
        except ClientError as e:
            print(f"❌ Erro ao criar/atualizar parâmetro {name}: {e}")
            return False
    
    def setup_trading_parameters(self, config: Dict[str, Any]) -> bool:
        """
        Configura todos os parâmetros necessários para o sistema de trading.
        
        Args:
            config: Dicionário com as configurações
            
        Returns:
            True se todos os parâmetros foram criados com sucesso
        """
        success = True
        
        # Determina o nome do bucket S3 baseado na conta AWS
        try:
            if not self.dry_run:
                if self.profile:
                    session = boto3.Session(profile_name=self.profile, region_name=self.region)
                    sts_client = session.client('sts')
                else:
                    sts_client = boto3.client('sts', region_name=self.region)
                
                account_id = sts_client.get_caller_identity()['Account']
                data_bucket = f"{config['aws_resources']['data_bucket_prefix']}-{account_id}-{self.region}"
            else:
                data_bucket = f"{config['aws_resources']['data_bucket_prefix']}-123456789012-{self.region}"
        except Exception as e:
            print(f"❌ Erro ao obter Account ID: {e}")
            data_bucket = f"{config['aws_resources']['data_bucket_prefix']}-default-{self.region}"
        
        # Lista de parâmetros para criar
        parameters = [
            # Configurações básicas do sistema
            {
                'name': '/trading_system/data_bucket',
                'value': data_bucket,
                'description': 'Nome do bucket S3 para armazenamento de dados'
            },
            {
                'name': '/trading_system/symbols',
                'value': json.dumps(config['trading_config']['symbols']),
                'description': 'Lista de símbolos de criptomoedas para trading'
            },
            {
                'name': '/trading_system/timeframes',
                'value': json.dumps(config['trading_config']['timeframes']),
                'description': 'Lista de timeframes para coleta de dados'
            },
            {
                'name': '/trading_system/lookback_days',
                'value': str(config['trading_config']['lookback_days']),
                'description': 'Número de días de histórico para baixar'
            },
            
            # Configurações da API Binance
            {
                'name': '/trading_system/binance/base_url',
                'value': config['binance_api']['base_url'],
                'description': 'URL base da API da Binance'
            },
            {
                'name': '/trading_system/binance/rate_limit',
                'value': str(config['binance_api']['rate_limit_requests_per_minute']),
                'description': 'Limite de requisições por minuto da API Binance'
            },
            {
                'name': '/trading_system/binance/max_retries',
                'value': str(config['binance_api']['max_retries']),
                'description': 'Número máximo de tentativas para requisições'
            },
            
            # Configurações de processamento de dados
            {
                'name': '/trading_system/data/file_formats',
                'value': json.dumps(config['data_processing']['file_formats']),
                'description': 'Formatos de arquivo para salvar dados'
            },
            {
                'name': '/trading_system/data/retention_days',
                'value': str(config['data_processing']['data_retention_days']),
                'description': 'Dias de retenção de dados no S3'
            },
            
            # Configurações de monitoramento
            {
                'name': '/trading_system/monitoring/log_level',
                'value': config['monitoring']['log_level'],
                'description': 'Nível de logging do sistema'
            },
            {
                'name': '/trading_system/monitoring/log_retention_days',
                'value': str(config['monitoring']['cloudwatch_log_retention_days']),
                'description': 'Dias de retenção de logs no CloudWatch'
            }
        ]
        
        # Cria todos os parâmetros
        print(f"\n📝 Configurando {len(parameters)} parâmetros...")
        for param in parameters:
            if not self.put_parameter(
                name=param['name'],
                value=param['value'],
                description=param['description']
            ):
                success = False
        
        return success
    
    def setup_secure_parameters(self) -> bool:
        """
        Configura parâmetros seguros (SecureString) para credenciais.
        
        Nota: As credenciais da Binance devem ser configuradas manualmente
        por questões de segurança.
        
        Returns:
            True se bem-sucedido
        """
        print("\n🔐 Configurando parâmetros seguros...")
        
        # Parâmetros seguros que devem ser configurados manualmente
        secure_params = [
            '/trading_system/binance/api_key',
            '/trading_system/binance/api_secret'
        ]
        
        if self.dry_run:
            for param in secure_params:
                print(f"[DRY RUN] Parâmetro seguro deve ser configurado manualmente: {param}")
            return True
        
        # Verifica se os parâmetros seguros já existem
        existing_params = []
        for param in secure_params:
            try:
                self.ssm_client.get_parameter(Name=param, WithDecryption=False)
                existing_params.append(param)
                print(f"✓ Parâmetro seguro já existe: {param}")
            except ClientError as e:
                if e.response['Error']['Code'] == 'ParameterNotFound':
                    print(f"⚠️  Parâmetro seguro não encontrado: {param}")
                else:
                    print(f"❌ Erro ao verificar parâmetro {param}: {e}")
        
        # Instruções para configuração manual
        if len(existing_params) < len(secure_params):
            print("\n📋 INSTRUÇÕES PARA CONFIGURAÇÃO MANUAL:")
            print("Execute os comandos abaixo para configurar as credenciais da Binance:")
            print()
            
            for param in secure_params:
                if param not in existing_params:
                    param_name = param.split('/')[-1].upper()
                    profile_flag = f" --profile {self.profile}" if self.profile else ""
                    region_flag = f" --region {self.region}" if self.region != 'us-east-1' else ""
                    
                    print(f"aws ssm put-parameter{profile_flag}{region_flag} \\")
                    print(f"  --name '{param}' \\")
                    print(f"  --value 'SUA_{param_name}_AQUI' \\")
                    print(f"  --type 'SecureString' \\")
                    print(f"  --description 'Credencial da Binance - {param_name}' \\")
                    print(f"  --tags 'Key=Project,Value=CryptoTradingSystem' 'Key=Environment,Value=Production' 'Key=ManagedBy,Value=setup_parameters.py'")
                    print()
        
        return True
    
    def verify_parameters(self, config: Dict[str, Any]) -> bool:
        """
        Verifica se todos os parâmetros foram criados corretamente.
        
        Args:
            config: Configuração original
            
        Returns:
            True se todos os parâmetros estão corretos
        """
        if self.dry_run:
            print("\n[DRY RUN] Verificação de parâmetros seria executada")
            return True
        
        print("\n🔍 Verificando parâmetros criados...")
        
        # Lista de parâmetros para verificar
        params_to_check = [
            '/trading_system/data_bucket',
            '/trading_system/symbols',
            '/trading_system/timeframes',
            '/trading_system/lookback_days'
        ]
        
        all_good = True
        for param_name in params_to_check:
            try:
                response = self.ssm_client.get_parameter(Name=param_name)
                value = response['Parameter']['Value']
                print(f"✓ {param_name}: {value[:50]}...")
            except ClientError as e:
                print(f"❌ Erro ao verificar {param_name}: {e}")
                all_good = False
        
        return all_good


def main():
    """Função principal do script."""
    parser = argparse.ArgumentParser(
        description='Configura parâmetros no AWS Systems Manager Parameter Store'
    )
    parser.add_argument(
        '--dry-run', 
        action='store_true',
        help='Mostra o que seria feito sem executar'
    )
    parser.add_argument(
        '--region',
        default='us-east-1',
        help='Região AWS (padrão: us-east-1)'
    )
    parser.add_argument(
        '--profile',
        default=None,
        help='Perfil AWS CLI a ser usado (opcional)'
    )
    parser.add_argument(
        '--config',
        default='config.json',
        help='Arquivo de configuração (padrão: config.json)'
    )
    
    args = parser.parse_args()
    
    print("🚀 Configurador de Parâmetros do Sistema de Trading")
    print("=" * 50)
    
    # Inicializa o gerenciador
    manager = ParameterStoreManager(
        region=args.region, 
        dry_run=args.dry_run, 
        profile=args.profile
    )
    
    # Carrega configurações
    config = manager.load_config(args.config)
    
    # Configura parâmetros
    success = True
    
    # Parâmetros básicos
    if not manager.setup_trading_parameters(config):
        success = False
    
    # Parâmetros seguros
    if not manager.setup_secure_parameters():
        success = False
    
    # Verificação
    if not manager.verify_parameters(config):
        success = False
    
    # Resultado final
    print("\n" + "=" * 50)
    if success:
        print("✅ Configuração concluída com sucesso!")
        if not args.dry_run:
            print("\n📝 PRÓXIMOS PASSOS:")
            print("1. Configure as credenciais da Binance manualmente (veja instruções acima)")
            print("2. Execute o deploy da infraestrutura atualizada")
            print("3. Teste a função Lambda data_downloader")
    else:
        print("❌ Configuração concluída com erros!")
        sys.exit(1)


if __name__ == '__main__':
    main()