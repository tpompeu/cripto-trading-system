"""
Sistema de Trading Quantitativo - Data Processor
Versão: 2.01
Descrição: Processamento e preparação dos dados para machine learning
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
import boto3
import awswrangler as wr
from datetime import datetime, timedelta
import joblib
import os

logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Classe responsável pelo processamento e preparação dos dados para ML.
    Inclui limpeza, normalização, feature selection e divisão temporal dos dados.
    """
    
    def __init__(self, config: Dict):
        """
        Inicializa o DataProcessor.
        
        Args:
            config: Configurações do sistema
        """
        self.config = config
        self.s3_client = boto3.client('s3')
        self.bucket_name = config.get('data_bucket_name')
        self.scaler = None
        self.feature_columns = None
        
    def load_data(self, symbol: str, timeframe: str, 
                  start_date: Optional[str] = None, 
                  end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Carrega dados do S3.
        
        Args:
            symbol: Par de trading (ex: BTCUSDT)
            timeframe: Timeframe dos dados (ex: 1h)
            start_date: Data de início (YYYY-MM-DD)
            end_date: Data de fim (YYYY-MM-DD)
            
        Returns:
            DataFrame com os dados carregados
        """
        logger.info(f"Carregando dados para {symbol} {timeframe}")
        
        try:
            # Constrói o path no S3
            s3_path = f"s3://{self.bucket_name}/raw_data/symbol={symbol}/interval={timeframe}/"
            
            # Carrega dados usando awswrangler
            df = wr.s3.read_parquet(
                path=s3_path,
                dataset=True,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            # Converte timestamp para datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp').sort_index()
            
            # Filtra por datas se especificado
            if start_date:
                df = df[df.index >= start_date]
            if end_date:
                df = df[df.index <= end_date]
                
            logger.info(f"Dados carregados: {len(df)} registros de {df.index.min()} a {df.index.max()}")
            return df
            
        except Exception as e:
            logger.error(f"Erro ao carregar dados: {str(e)}")
            raise
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Limpa e valida os dados.
        
        Args:
            df: DataFrame com dados brutos
            
        Returns:
            DataFrame limpo
        """
        logger.info("Iniciando limpeza dos dados")
        
        initial_rows = len(df)
        
        # Remove duplicatas
        df = df.drop_duplicates()
        
        # Remove linhas com valores nulos em colunas críticas
        critical_columns = ['open', 'high', 'low', 'close']
        df = df.dropna(subset=critical_columns)
        
        # Valida consistência OHLC
        invalid_ohlc = (
            (df['high'] < df['low']) |
            (df['high'] < df['open']) |
            (df['high'] < df['close']) |
            (df['low'] > df['open']) |
            (df['low'] > df['close']) |
            (df['open'] <= 0) |
            (df['high'] <= 0) |
            (df['low'] <= 0) |
            (df['close'] <= 0)
        )
        
        if invalid_ohlc.any():
            logger.warning(f"Removendo {invalid_ohlc.sum()} registros com OHLC inválido")
            df = df[~invalid_ohlc]
        
        # Remove outliers extremos (preços que variam mais de 50% em um período)
        price_change = df['close'].pct_change().abs()
        extreme_outliers = price_change > 0.5
        
        if extreme_outliers.any():
            logger.warning(f"Removendo {extreme_outliers.sum()} outliers extremos")
            df = df[~extreme_outliers]
        
        # Preenche volume nulo com 0 se necessário
        if 'volume' in df.columns:
            df['volume'] = df['volume'].fillna(0)
        
        final_rows = len(df)
        logger.info(f"Limpeza concluída: {initial_rows} -> {final_rows} registros ({final_rows/initial_rows*100:.1f}%)")
        
        return df
    
    def prepare_features(self, df: pd.DataFrame, 
                        feature_config: Dict) -> Tuple[pd.DataFrame, List[str]]:
        """
        Prepara features para o modelo.
        
        Args:
            df: DataFrame com features geradas
            feature_config: Configuração das features
            
        Returns:
            Tuple com DataFrame preparado e lista de colunas de features
        """
        logger.info("Preparando features para o modelo")
        
        # Remove colunas que não são features
        exclude_columns = ['open', 'high', 'low', 'close', 'volume', 'target']
        feature_columns = [col for col in df.columns if col not in exclude_columns]
        
        # Remove features com muitos valores nulos
        null_threshold = feature_config.get('max_null_ratio', 0.1)
        for col in feature_columns.copy():
            null_ratio = df[col].isnull().sum() / len(df)
            if null_ratio > null_threshold:
                logger.warning(f"Removendo feature {col} (null ratio: {null_ratio:.3f})")
                feature_columns.remove(col)
        
        # Remove features com variância muito baixa
        variance_threshold = feature_config.get('min_variance', 1e-6)
        for col in feature_columns.copy():
            if df[col].var() < variance_threshold:
                logger.warning(f"Removendo feature {col} (baixa variância: {df[col].var():.6f})")
                feature_columns.remove(col)
        
        # Remove features altamente correlacionadas
        correlation_threshold = feature_config.get('max_correlation', 0.95)
        feature_df = df[feature_columns]
        corr_matrix = feature_df.corr().abs()
        
        # Encontra pares altamente correlacionados
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        to_remove = [column for column in upper_triangle.columns 
                    if any(upper_triangle[column] > correlation_threshold)]
        
        if to_remove:
            logger.info(f"Removendo {len(to_remove)} features altamente correlacionadas")
            feature_columns = [col for col in feature_columns if col not in to_remove]
        
        # Preenche valores nulos restantes
        df[feature_columns] = df[feature_columns].fillna(method='ffill').fillna(0)
        
        # Remove linhas com infinitos
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        
        self.feature_columns = feature_columns
        logger.info(f"Features preparadas: {len(feature_columns)} colunas selecionadas")
        
        return df, feature_columns
    
    def normalize_features(self, df: pd.DataFrame, feature_columns: List[str],
                          scaler_type: str = 'robust', fit_scaler: bool = True) -> pd.DataFrame:
        """
        Normaliza as features.
        
        Args:
            df: DataFrame com features
            feature_columns: Lista de colunas de features
            scaler_type: Tipo de scaler ('standard' ou 'robust')
            fit_scaler: Se deve treinar o scaler
            
        Returns:
            DataFrame com features normalizadas
        """
        logger.info(f"Normalizando features usando {scaler_type} scaler")
        
        if fit_scaler or self.scaler is None:
            if scaler_type == 'standard':
                self.scaler = StandardScaler()
            elif scaler_type == 'robust':
                self.scaler = RobustScaler()
            else:
                raise ValueError(f"Scaler type não suportado: {scaler_type}")
            
            # Treina o scaler
            self.scaler.fit(df[feature_columns])
            logger.info("Scaler treinado")
        
        # Aplica normalização
        df_normalized = df.copy()
        df_normalized[feature_columns] = self.scaler.transform(df[feature_columns])
        
        return df_normalized
    
    def create_walk_forward_splits(self, df: pd.DataFrame, 
                                  train_months: int = 12,
                                  validation_months: int = 3,
                                  step_months: int = 3) -> List[Dict]:
        """
        Cria divisões para Walk-Forward validation.
        
        Args:
            df: DataFrame com dados
            train_months: Meses para treinamento
            validation_months: Meses para validação
            step_months: Meses para avançar entre janelas
            
        Returns:
            Lista de dicionários com informações das divisões
        """
        logger.info("Criando divisões Walk-Forward")
        
        splits = []
        start_date = df.index.min()
        end_date = df.index.max()
        
        current_date = start_date
        
        while True:
            # Define janela de treinamento
            train_start = current_date
            train_end = train_start + timedelta(days=train_months * 30)
            
            # Define janela de validação
            val_start = train_end
            val_end = val_start + timedelta(days=validation_months * 30)
            
            # Verifica se há dados suficientes
            if val_end > end_date:
                break
            
            # Filtra dados para esta divisão
            train_data = df[(df.index >= train_start) & (df.index < train_end)]
            val_data = df[(df.index >= val_start) & (df.index < val_end)]
            
            if len(train_data) < 100 or len(val_data) < 50:
                logger.warning(f"Dados insuficientes para divisão {len(splits)+1}")
                break
            
            split_info = {
                'split_id': len(splits) + 1,
                'train_start': train_start,
                'train_end': train_end,
                'val_start': val_start,
                'val_end': val_end,
                'train_size': len(train_data),
                'val_size': len(val_data)
            }
            
            splits.append(split_info)
            
            # Avança para próxima janela
            current_date += timedelta(days=step_months * 30)
        
        logger.info(f"Criadas {len(splits)} divisões Walk-Forward")
        return splits
    
    def get_split_data(self, df: pd.DataFrame, split_info: Dict,
                      feature_columns: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, 
                                                          pd.Series, pd.Series]:
        """
        Obtém dados de uma divisão específica.
        
        Args:
            df: DataFrame completo
            split_info: Informações da divisão
            feature_columns: Colunas de features
            
        Returns:
            Tuple com X_train, X_val, y_train, y_val
        """
        # Filtra dados de treinamento
        train_mask = ((df.index >= split_info['train_start']) & 
                     (df.index < split_info['train_end']))
        train_data = df[train_mask]
        
        # Filtra dados de validação
        val_mask = ((df.index >= split_info['val_start']) & 
                   (df.index < split_info['val_end']))
        val_data = df[val_mask]
        
        # Separa features e target
        X_train = train_data[feature_columns]
        y_train = train_data['target']
        X_val = val_data[feature_columns]
        y_val = val_data['target']
        
        return X_train, X_val, y_train, y_val
    
    def save_processed_data(self, df: pd.DataFrame, symbol: str, 
                           timeframe: str, suffix: str = "") -> str:
        """
        Salva dados processados no S3.
        
        Args:
            df: DataFrame processado
            symbol: Par de trading
            timeframe: Timeframe
            suffix: Sufixo para o nome do arquivo
            
        Returns:
            Path do arquivo salvo
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"processed_data_{symbol}_{timeframe}_{timestamp}{suffix}.parquet"
        s3_path = f"s3://{self.bucket_name}/processed_data/{filename}"
        
        try:
            wr.s3.to_parquet(
                df=df,
                path=s3_path,
                compression='snappy'
            )
            logger.info(f"Dados processados salvos em: {s3_path}")
            return s3_path
            
        except Exception as e:
            logger.error(f"Erro ao salvar dados processados: {str(e)}")
            raise
    
    def save_scaler(self, symbol: str, timeframe: str) -> str:
        """
        Salva o scaler treinado.
        
        Args:
            symbol: Par de trading
            timeframe: Timeframe
            
        Returns:
            Path do scaler salvo
        """
        if self.scaler is None:
            raise ValueError("Scaler não foi treinado")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"scaler_{symbol}_{timeframe}_{timestamp}.joblib"
        local_path = f"/tmp/{filename}"
        s3_path = f"s3://{self.bucket_name}/models/scalers/{filename}"
        
        try:
            # Salva localmente primeiro
            joblib.dump(self.scaler, local_path)
            
            # Upload para S3
            self.s3_client.upload_file(local_path, self.bucket_name, 
                                     f"models/scalers/{filename}")
            
            # Remove arquivo local
            os.remove(local_path)
            
            logger.info(f"Scaler salvo em: {s3_path}")
            return s3_path
            
        except Exception as e:
            logger.error(f"Erro ao salvar scaler: {str(e)}")
            raise
    
    def load_scaler(self, scaler_path: str):
        """
        Carrega um scaler salvo.
        
        Args:
            scaler_path: Path do scaler no S3
        """
        try:
            # Parse S3 path
            bucket = scaler_path.split('/')[2]
            key = '/'.join(scaler_path.split('/')[3:])
            
            # Download para arquivo temporário
            local_path = f"/tmp/scaler_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
            self.s3_client.download_file(bucket, key, local_path)
            
            # Carrega scaler
            self.scaler = joblib.load(local_path)
            
            # Remove arquivo local
            os.remove(local_path)
            
            logger.info(f"Scaler carregado de: {scaler_path}")
            
        except Exception as e:
            logger.error(f"Erro ao carregar scaler: {str(e)}")
            raise
    
    def validate_data_quality(self, df: pd.DataFrame, 
                             feature_columns: List[str]) -> Dict:
        """
        Valida a qualidade dos dados processados.
        
        Args:
            df: DataFrame processado
            feature_columns: Colunas de features
            
        Returns:
            Dicionário com métricas de qualidade
        """
        logger.info("Validando qualidade dos dados")
        
        quality_metrics = {
            'total_rows': len(df),
            'total_features': len(feature_columns),
            'null_ratio': df[feature_columns].isnull().sum().sum() / (len(df) * len(feature_columns)),
            'infinite_values': np.isinf(df[feature_columns]).sum().sum(),
            'target_distribution': df['target'].value_counts().to_dict() if 'target' in df.columns else {},
            'date_range': {
                'start': df.index.min().isoformat(),
                'end': df.index.max().isoformat()
            }
        }
        
        # Verifica features com problemas
        problematic_features = []
        for col in feature_columns:
            if df[col].isnull().all():
                problematic_features.append(f"{col}: todos valores nulos")
            elif df[col].var() == 0:
                problematic_features.append(f"{col}: variância zero")
            elif np.isinf(df[col]).any():
                problematic_features.append(f"{col}: valores infinitos")
        
        quality_metrics['problematic_features'] = problematic_features
        
        # Calcula score de qualidade
        quality_score = 1.0
        if quality_metrics['null_ratio'] > 0.05:
            quality_score -= 0.3
        if quality_metrics['infinite_values'] > 0:
            quality_score -= 0.2
        if len(problematic_features) > 0:
            quality_score -= 0.2
        if len(df) < 1000:
            quality_score -= 0.3
        
        quality_metrics['quality_score'] = max(0, quality_score)
        
        logger.info(f"Score de qualidade: {quality_score:.2f}")
        return quality_metrics

