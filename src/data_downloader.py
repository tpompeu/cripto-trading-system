"""
Sistema de Trading Quantitativo - Data Downloader
Versão: 3.1 - Histórico dinâmico baseado em SSM Parameter Store
"""

import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

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
    def __init__(self):
        self.s3_client = boto3.client('s3')
        self.ssm_client = boto3.client('ssm')
        self.config = self._load_configuration()
        
        self.base_url = "https://api.binance.com"
        self.klines_endpoint = "/api/v3/klines"
        self.max_retries = 3
        self.retry_delay = 5
        
        # Carrega lookback_days do SSM Parameter Store
        self.lookback_days = self._get_lookback_days()
        logger.info(f"BinanceDataDownloader inicializado com {self.lookback_days} dias de histórico")

    def _get_lookback_days(self) -> int:
        """Obtém o número de dias de histórico do Parameter Store"""
        try:
            response = self.ssm_client.get_parameter(Name='/trading_system/lookback_days')
            lookback_days = int(response['Parameter']['Value'])
            logger.info(f"Lookback days configurado: {lookback_days}")
            return lookback_days
        except ClientError as e:
            if e.response['Error']['Code'] == 'ParameterNotFound':
                logger.warning("Parâmetro lookback_days não encontrado, usando valor padrão de 1825 dias (5 anos)")
                return 1825
            else:
                logger.error(f"Erro ao acessar Parameter Store: {str(e)}")
                return 1825
        except Exception as e:
            logger.error(f"Erro ao obter lookback_days: {str(e)}")
            return 1825

    def _load_configuration(self) -> Dict:
        """Carrega configurações do SSM Parameter Store"""
        try:
            params_to_load = [
                '/trading_system/data_bucket',
                '/trading_system/symbols',
                '/trading_system/timeframes'
            ]
            
            config = {}
            for param_name in params_to_load:
                try:
                    response = self.ssm_client.get_parameter(Name=param_name)
                    param_key = param_name.split('/')[-1]
                    try:
                        config[param_key] = json.loads(response['Parameter']['Value'])
                    except json.JSONDecodeError:
                        config[param_key] = response['Parameter']['Value']
                except ClientError as e:
                    if e.response['Error']['Code'] == 'ParameterNotFound':
                        defaults = {
                            'data_bucket': os.environ.get('DATA_BUCKET', 'crypto-trading-data-default'),
                            'symbols': ['BTCUSDT', 'ETHUSDT'],
                            'timeframes': ['1h', '4h', '1d']
                        }
                        config[param_name.split('/')[-1]] = defaults.get(param_name.split('/')[-1])
            return config
        except Exception as e:
            logger.error(f"Erro ao carregar configuração: {str(e)}")
            return {
                'data_bucket': os.environ.get('DATA_BUCKET', 'crypto-trading-data-default'),
                'symbols': ['BTCUSDT', 'ETHUSDT'],
                'timeframes': ['1h', '4h', '1d']
            }

    def _make_request_with_retry(self, url: str, params: Dict) -> Optional[List]:
        """Faz requisição HTTP com retry automático"""
        for attempt in range(self.max_retries):
            try:
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                logger.warning(f"Tentativa {attempt + 1} falhou: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))
                else:
                    logger.error(f"Todas as tentativas falharam para {url}")
                    return None

    def _get_existing_data(self, symbol: str, interval: str) -> Tuple[Optional[pd.DataFrame], Optional[datetime]]:
        """Retorna dados existentes e a data mais antiga"""
        try:
            s3_path = f"s3://{self.config['data_bucket']}/historical_data/{symbol}/{interval}/dynamic_history.parquet"
            
            if wr.s3.does_object_exist(s3_path):
                df = wr.s3.read_parquet(s3_path)
                if not df.empty:
                    oldest_date = df['timestamp'].min()
                    newest_date = df['timestamp'].max()
                    logger.info(f"Dados existentes: {len(df)} registros, de {oldest_date} até {newest_date}")
                    return df, oldest_date
            return None, None
            
        except Exception as e:
            logger.warning(f"Erro ao carregar dados existentes: {str(e)}")
            return None, None

    def _download_historical_range(self, symbol: str, interval: str, 
                                 start_time: int, end_time: int) -> Optional[pd.DataFrame]:
        """Baixa dados de um range específico com paginação"""
        all_data = []
        current_end = end_time
        
        while current_end > start_time:
            params = {
                'symbol': symbol,
                'interval': interval,
                'endTime': current_end,
                'limit': 1000
            }
            
            data = self._make_request_with_retry(f"{self.base_url}{self.klines_endpoint}", params)
            if not data:
                break
                
            all_data.extend(data)
            
            # Atualiza para o próximo batch (mais antigo)
            if data:
                current_end = data[0][0] - 1
            else:
                break
            
            time.sleep(0.1)  # Rate limiting
        
        if not all_data:
            return None
            
        # Converte para DataFrame
        df = pd.DataFrame(all_data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        return self._process_ohlcv_data(df)

    def _process_ohlcv_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Processa e limpa os dados OHLCV"""
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
            
            numeric_columns = ['open', 'high', 'low', 'close', 'volume', 
                             'quote_asset_volume', 'number_of_trades',
                             'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
            
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
            df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
            df = df.drop(['ignore'], axis=1)
            df = df.reset_index(drop=True)
            
            return df
        except Exception as e:
            logger.error(f"Erro ao processar dados OHLCV: {str(e)}")
            raise

    def _ensure_dynamic_history(self, symbol: str, interval: str) -> bool:
        """Garante que temos exatamente lookback_days de dados históricos"""
        try:
            # 1. Carrega dados existentes
            existing_df, oldest_date = self._get_existing_data(symbol, interval)
            
            # Datas de referência
            now = datetime.now()
            history_start_date = now - timedelta(days=self.lookback_days)
            
            # 2. Caso 1: Arquivo não existe
            if existing_df is None:
                logger.info(f"Caso 1: Criando histórico de {self.lookback_days} dias para {symbol}-{interval}")
                start_time = int(history_start_date.timestamp() * 1000)
                end_time = int(now.timestamp() * 1000)
                
                new_data = self._download_historical_range(symbol, interval, start_time, end_time)
                if new_data is None:
                    return False
                    
                return self._save_final_data(new_data, symbol, interval, history_start_date, now)
            
            # 3. Caso 2: Existe mas não tem histórico completo
            elif oldest_date > history_start_date:
                logger.info(f"Caso 2: Completando histórico para {symbol}-{interval}")
                missing_days = (oldest_date - history_start_date).days
                logger.info(f"Faltam {missing_days} dias para completar {self.lookback_days} dias")
                
                # Baixa dados faltantes
                start_time = int(history_start_date.timestamp() * 1000)
                end_time = int((oldest_date - timedelta(days=1)).timestamp() * 1000)
                
                missing_data = self._download_historical_range(symbol, interval, start_time, end_time)
                if missing_data is None:
                    return False
                
                # Combina com dados existentes
                combined_data = pd.concat([missing_data, existing_df], ignore_index=True)
                combined_data = combined_data.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
                
                return self._save_final_data(combined_data, symbol, interval, history_start_date, now)
            
            # 4. Caso 3: Já tem histórico completo - mantém atualizado
            else:
                logger.info(f"Caso 3: Mantendo histórico atualizado para {symbol}-{interval}")
                
                # Remove dados mais antigos que o período configurado
                cutoff_date = now - timedelta(days=self.lookback_days)
                updated_data = existing_df[existing_df['timestamp'] >= cutoff_date].copy()
                
                # Baixa dados dos últimos dias para garantir que está atualizado
                last_date_in_data = updated_data['timestamp'].max()
                if last_date_in_data < now - timedelta(days=1):
                    logger.info(f"Baixando dados recentes de {last_date_in_data} até agora")
                    start_time = int(last_date_in_data.timestamp() * 1000) + 1
                    end_time = int(now.timestamp() * 1000)
                    
                    recent_data = self._download_historical_range(symbol, interval, start_time, end_time)
                    if recent_data is not None:
                        updated_data = pd.concat([updated_data, recent_data], ignore_index=True)
                        updated_data = updated_data.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
                
                return self._save_final_data(updated_data, symbol, interval, history_start_date, now)
                
        except Exception as e:
            logger.error(f"Erro ao garantir histórico dinâmico: {str(e)}")
            return False

    def _save_final_data(self, df: pd.DataFrame, symbol: str, interval: str, 
                        start_date: datetime, end_date: datetime) -> bool:
        """Salva o dataset final garantindo qualidade"""
        try:
            # Filtra para o período exato configurado
            df = df[df['timestamp'] >= start_date]
            df = df[df['timestamp'] <= end_date]
            
            # Remove duplicatas e ordena
            df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
            
            # Salva em Parquet
            s3_path = f"s3://{self.config['data_bucket']}/historical_data/{symbol}/{interval}/dynamic_history.parquet"
            wr.s3.to_parquet(df=df, path=s3_path, index=False)
            
            logger.info(f"Histórico de {self.lookback_days} dias salvo: {s3_path} ({len(df)} registros)")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao salvar dados finais: {str(e)}")
            return False

    def run_data_collection(self) -> Dict[str, any]:
        """Executa a coleta para todos os símbolos e timeframes"""
        start_time = datetime.now()
        results = {
            'start_time': start_time.isoformat(),
            'lookback_days': self.lookback_days,
            'symbols_processed': [],
            'symbols_failed': [],
            'total_records': 0,
            'errors': []
        }
        
        try:
            symbols = self.config['symbols']
            timeframes = self.config['timeframes']
            
            logger.info(f"Iniciando manutenção de histórico ({self.lookback_days} dias) para {len(symbols)} símbolos")
            
            for symbol in symbols:
                symbol_success = True
                
                for interval in timeframes:
                    try:
                        success = self._ensure_dynamic_history(symbol, interval)
                        if not success:
                            symbol_success = False
                            error_msg = f"Falha ao processar {symbol}-{interval}"
                            results['errors'].append(error_msg)
                            
                    except Exception as e:
                        symbol_success = False
                        error_msg = f"Erro em {symbol}-{interval}: {str(e)}"
                        results['errors'].append(error_msg)
                        logger.error(error_msg)
                
                if symbol_success:
                    results['symbols_processed'].append(symbol)
                else:
                    results['symbols_failed'].append(symbol)
            
            # Estimativa de registros
            results['total_records'] = len(symbols) * len(timeframes) * self.lookback_days * 24
            
            end_time = datetime.now()
            results['end_time'] = end_time.isoformat()
            results['duration_seconds'] = (end_time - start_time).total_seconds()
            results['success'] = len(results['symbols_failed']) == 0
            
            logger.info(f"Manutenção concluída: {len(results['symbols_processed'])} sucessos, {len(results['symbols_failed'])} falhas")
            return results
            
        except Exception as e:
            error_msg = f"Erro crítico: {str(e)}"
            logger.error(error_msg)
            results['errors'].append(error_msg)
            results['success'] = False
            return results

def lambda_handler(event, context):
    """Handler principal da Lambda"""
    try:
        logger.info("=== INÍCIO DA MANUTENÇÃO DO HISTÓRICO DINÂMICO ===")
        
        downloader = BinanceDataDownloader()
        results = downloader.run_data_collection()
        
        response = {
            'statusCode': 200 if results['success'] else 500,
            'body': json.dumps(results, default=str),
            'headers': {'Content-Type': 'application/json'}
        }
        
        logger.info("=== FIM DA MANUTENÇÃO ===")
        return response
        
    except Exception as e:
        error_msg = f"Erro no handler: {str(e)}"
        logger.error(error_msg)
        return {
            'statusCode': 500,
            'body': json.dumps({'error': error_msg}),
            'headers': {'Content-Type': 'application/json'}
        }

# Para teste local
if __name__ == "__main__":
    os.environ['DATA_BUCKET'] = 'crypto-trading-data-test'
    
    downloader = BinanceDataDownloader()
    results = downloader.run_data_collection()
    
    print(json.dumps(results, indent=2, default=str))