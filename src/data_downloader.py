"""
Sistema de Trading Quantitativo - Data Downloader
Versão: 2.01
Data: 25 de agosto de 2025

Este módulo é responsável por baixar dados históricos e em tempo real da Binance,
processá-los e armazená-los no S3 para uso posterior pelo sistema de trading.

Funcionalidades:
- Download de dados OHLCV históricos da Binance
- Processamento e limpeza dos dados
- Armazenamento estruturado no S3
- Logging detalhado para auditoria
- Tratamento robusto de erros e retry automático
"""

import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import boto3
import pandas as pd
import requests
from botocore.exceptions import ClientError
import awswrangler as wr

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BinanceDataDownloader:
    """
    Classe principal para download de dados da Binance.
    
    Esta classe gerencia a conexão com a API da Binance, o download de dados
    históricos e em tempo real, e o armazenamento estruturado no S3.
    """
    
    def __init__(self):
        """Inicializa o downloader com configurações da AWS e Binance."""
        self.s3_client = boto3.client('s3')
        self.ssm_client = boto3.client('ssm')
        
        # Carrega configurações do Systems Manager Parameter Store
        self.config = self._load_configuration()
        
        # URLs da API da Binance
        self.base_url = "https://data.binance.com"
        self.klines_endpoint = "/api/v3/klines"
        
        # Configurações de retry
        self.max_retries = 3
        self.retry_delay = 5  # segundos
        
        logger.info("BinanceDataDownloader inicializado com sucesso")
    
    def _load_configuration(self) -> Dict:
        """
        Carrega configurações do AWS Systems Manager Parameter Store.
        
        Returns:
            Dict: Dicionário com as configurações do sistema
        """
        try:
            # Parâmetros básicos do sistema
            params_to_load = [
                '/trading_system/data_bucket',
                '/trading_system/symbols',
                '/trading_system/timeframes',
                '/trading_system/lookback_days'
            ]
            
            config = {}
            for param_name in params_to_load:
                try:
                    response = self.ssm_client.get_parameter(Name=param_name)
                    param_key = param_name.split('/')[-1]  # Pega apenas o nome final
                    
                    # Tenta fazer parse JSON se possível, senão mantém como string
                    try:
                        config[param_key] = json.loads(response['Parameter']['Value'])
                    except json.JSONDecodeError:
                        config[param_key] = response['Parameter']['Value']
                        
                except ClientError as e:
                    if e.response['Error']['Code'] == 'ParameterNotFound':
                        logger.warning(f"Parâmetro {param_name} não encontrado, usando valor padrão")
                        # Define valores padrão
                        defaults = {
                            'data_bucket': os.environ.get('DATA_BUCKET', 'crypto-trading-data-default'),
                            'symbols': ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT', 'LINKUSDT'],
                            'timeframes': ['1h', '4h', '1d'],
                            'lookback_days': 365
                        }
                        param_key = param_name.split('/')[-1]
                        config[param_key] = defaults.get(param_key)
                    else:
                        raise
            
            logger.info(f"Configuração carregada: {list(config.keys())}")
            return config
            
        except Exception as e:
            logger.error(f"Erro ao carregar configuração: {str(e)}")
            # Configuração de fallback
            return {
                'data_bucket': os.environ.get('DATA_BUCKET', 'crypto-trading-data-default'),
                'symbols': ['BTCUSDT', 'ETHUSDT'],
                'timeframes': ['1h', '4h'],
                'lookback_days': 30
            }
    
    def _make_request_with_retry(self, url: str, params: Dict) -> Optional[List]:
        """
        Faz requisição HTTP com retry automático.
        
        Args:
            url: URL da API
            params: Parâmetros da requisição
            
        Returns:
            Lista com os dados da API ou None em caso de erro
        """
        for attempt in range(self.max_retries):
            try:
            
                print(f"Fazendo requisição para: {url} com params: {params}")
                response = requests.get(url, params=params, timeout=30)
                print(f"Status code: {response.status_code}")
                print(f"Response text: {response.text[:200]}...")  # Primeiros 200 caracteres
            
                response.raise_for_status()
              
                data = response.json()
                logger.debug(f"Requisição bem-sucedida: {len(data)} registros")
                return data
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Tentativa {attempt + 1} falhou: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))  # Backoff exponencial
                else:
                    logger.error(f"Todas as tentativas falharam para {url}")
                    return None
    
    def download_historical_data(self, symbol: str, interval: str, 
                                lookback_days: int) -> Optional[pd.DataFrame]:
        """
        Baixa dados históricos OHLCV da Binance.
        
        Args:
            symbol: Par de trading (ex: 'BTCUSDT')
            interval: Timeframe (ex: '1h', '4h', '1d')
            lookback_days: Número de dias para buscar no histórico
            
        Returns:
            DataFrame com os dados OHLCV ou None em caso de erro
        """
        try:
            # Calcula timestamps
            end_time = int(datetime.now().timestamp() * 1000)
            start_time = int((datetime.now() - timedelta(days=lookback_days)).timestamp() * 1000)
            
            # Parâmetros da requisição
            params = {
                'symbol': symbol,
                'interval': interval,
                'startTime': start_time,
                'endTime': end_time,
                'limit': 1000  # Máximo permitido pela Binance
            }
            
            url = f"{self.base_url}{self.klines_endpoint}"
            logger.info(f"Baixando dados: {symbol} - {interval} - {lookback_days} dias")
            
            # Faz a requisição
            data = self._make_request_with_retry(url, params)
            if not data:
                return None
            
            # Converte para DataFrame
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Processa os dados
            df = self._process_ohlcv_data(df)
            
            logger.info(f"Dados processados: {len(df)} registros para {symbol}-{interval}")
            return df
            
        except Exception as e:
            logger.error(f"Erro ao baixar dados históricos para {symbol}-{interval}: {str(e)}")
            return None
    
    def _process_ohlcv_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Processa e limpa os dados OHLCV baixados.
        
        Args:
            df: DataFrame bruto da Binance
            
        Returns:
            DataFrame processado e limpo
        """
        try:
            # Converte timestamp para datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
            
            # Converte colunas numéricas
            numeric_columns = ['open', 'high', 'low', 'close', 'volume', 
                             'quote_asset_volume', 'number_of_trades',
                             'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
            
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remove duplicatas e ordena por timestamp
            df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
            
            # Remove linhas com valores nulos críticos
            df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
            
            # Adiciona colunas calculadas
            df['price_change'] = df['close'] - df['open']
            df['price_change_pct'] = (df['close'] - df['open']) / df['open'] * 100
            df['high_low_spread'] = df['high'] - df['low']
            df['vwap'] = df['quote_asset_volume'] / df['volume']  # Volume Weighted Average Price
            
            # Remove colunas desnecessárias
            df = df.drop(['ignore'], axis=1)
            
            # Reset index
            df = df.reset_index(drop=True)
            
            logger.debug(f"Dados processados: {len(df)} registros válidos")
            return df
            
        except Exception as e:
            logger.error(f"Erro ao processar dados OHLCV: {str(e)}")
            raise
    
    def save_to_s3(self, df: pd.DataFrame, symbol: str, interval: str) -> bool:
        try:
            current_date = datetime.now().strftime('%Y-%m-%d')
            s3_path = f"s3://{self.config['data_bucket']}/historical_data/{symbol}/{interval}/{current_date}"

            # Salva em Parquet
            wr.s3.to_parquet(
                df=df,
                path=f"{s3_path}.parquet",
                index=False
            )

            # Salva em CSV (opcional)
            wr.s3.to_csv(
                df=df,
                path=f"{s3_path}.csv",
                index=False
            )

            logger.info(f"Dados salvos no S3: {s3_path}[.parquet/.csv]")
            return True

        except Exception as e:
            logger.error(f"Erro ao salvar dados no S3: {str(e)}")
            return False
    
    def run_data_collection(self) -> Dict[str, any]:
        """
        Executa o processo completo de coleta de dados.
        
        Returns:
            Dicionário com o resultado da execução
        """
        start_time = datetime.now()
        results = {
            'start_time': start_time.isoformat(),
            'symbols_processed': [],
            'symbols_failed': [],
            'total_records': 0,
            'errors': []
        }
        
        try:
            symbols = self.config['symbols']
            timeframes = self.config['timeframes']
            lookback_days = self.config['lookback_days']
            
            logger.info(f"Iniciando coleta de dados para {len(symbols)} símbolos e {len(timeframes)} timeframes")
            
            for symbol in symbols:
                symbol_success = True
                symbol_records = 0
                
                for interval in timeframes:
                    try:
                        # Download dos dados
                        df = self.download_historical_data(symbol, interval, lookback_days)
                        
                        if df is not None and len(df) > 0:
                            # Salva no S3
                            if self.save_to_s3(df, symbol, interval):
                                symbol_records += len(df)
                                logger.info(f"✓ {symbol}-{interval}: {len(df)} registros")
                            else:
                                symbol_success = False
                                error_msg = f"Falha ao salvar {symbol}-{interval} no S3"
                                results['errors'].append(error_msg)
                                logger.error(error_msg)
                        else:
                            symbol_success = False
                            error_msg = f"Nenhum dado obtido para {symbol}-{interval}"
                            results['errors'].append(error_msg)
                            logger.warning(error_msg)
                            
                    except Exception as e:
                        symbol_success = False
                        error_msg = f"Erro ao processar {symbol}-{interval}: {str(e)}"
                        results['errors'].append(error_msg)
                        logger.error(error_msg)
                
                # Atualiza resultados
                if symbol_success:
                    results['symbols_processed'].append(symbol)
                    results['total_records'] += symbol_records
                else:
                    results['symbols_failed'].append(symbol)
            
            # Finaliza
            end_time = datetime.now()
            results['end_time'] = end_time.isoformat()
            results['duration_seconds'] = (end_time - start_time).total_seconds()
            results['success'] = len(results['symbols_failed']) == 0
            
            logger.info(f"Coleta finalizada: {len(results['symbols_processed'])} sucessos, "
                       f"{len(results['symbols_failed'])} falhas, "
                       f"{results['total_records']} registros totais")
            
            return results
            
        except Exception as e:
            error_msg = f"Erro crítico na coleta de dados: {str(e)}"
            logger.error(error_msg)
            results['errors'].append(error_msg)
            results['success'] = False
            return results


def lambda_handler(event, context):
    """
    Função principal para execução no AWS Lambda.
    
    Args:
        event: Evento do Lambda (pode vir do EventBridge Scheduler)
        context: Contexto de execução do Lambda
        
    Returns:
        Dicionário com o resultado da execução
    """
    try:
        logger.info("=== INÍCIO DA EXECUÇÃO DO DATA DOWNLOADER ===")
        logger.info(f"Event: {json.dumps(event, default=str)}")
        
        # Inicializa o downloader
        downloader = BinanceDataDownloader()
        
        # Executa a coleta de dados
        results = downloader.run_data_collection()
        
        # Prepara resposta
        response = {
            'statusCode': 200 if results['success'] else 500,
            'body': json.dumps(results, default=str),
            'headers': {
                'Content-Type': 'application/json'
            }
        }
        
        logger.info("=== FIM DA EXECUÇÃO DO DATA DOWNLOADER ===")
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
    os.environ['DATA_BUCKET'] = 'crypto-trading-data-test'
    
    # Executa o downloader
    downloader = BinanceDataDownloader()
    results = downloader.run_data_collection()
    
    print("=== RESULTADOS ===")
    print(json.dumps(results, indent=2, default=str))

