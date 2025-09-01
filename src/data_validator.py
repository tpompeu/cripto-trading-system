"""
Sistema de Trading Quantitativo - Data Validator
Versão: 2.01
Descrição: Validação de qualidade e integridade dos dados
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta
import boto3
import json

logger = logging.getLogger(__name__)

class DataValidator:
    """
    Classe responsável pela validação de qualidade e integridade dos dados.
    Verifica consistência, completude e detecta anomalias nos dados de mercado.
    """
    
    def __init__(self, config: Dict):
        """
        Inicializa o DataValidator.
        
        Args:
            config: Configurações do sistema
        """
        self.config = config
        self.validation_rules = config.get('validation_rules', {})
        self.s3_client = boto3.client('s3')
        
    def validate_ohlcv_data(self, df: pd.DataFrame) -> Dict:
        """
        Valida dados OHLCV básicos.
        
        Args:
            df: DataFrame com dados OHLCV
            
        Returns:
            Dicionário com resultados da validação
        """
        logger.info("Validando dados OHLCV")
        
        validation_results = {
            'timestamp': datetime.now().isoformat(),
            'total_rows': len(df),
            'validation_passed': True,
            'errors': [],
            'warnings': [],
            'metrics': {}
        }
        
        # Verifica colunas obrigatórias
        required_columns = ['open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            validation_results['errors'].append(f"Colunas obrigatórias ausentes: {missing_columns}")
            validation_results['validation_passed'] = False
            return validation_results
        
        # Verifica se há dados
        if len(df) == 0:
            validation_results['errors'].append("Dataset vazio")
            validation_results['validation_passed'] = False
            return validation_results
        
        # Validações específicas
        self._validate_price_consistency(df, validation_results)
        self._validate_data_completeness(df, validation_results)
        self._validate_price_ranges(df, validation_results)
        self._validate_temporal_consistency(df, validation_results)
        self._validate_volume_data(df, validation_results)
        self._detect_price_anomalies(df, validation_results)
        
        # Calcula métricas gerais
        validation_results['metrics'] = self._calculate_data_metrics(df)
        
        logger.info(f"Validação concluída. Passou: {validation_results['validation_passed']}")
        return validation_results
    
    def _validate_price_consistency(self, df: pd.DataFrame, results: Dict):
        """Valida consistência dos preços OHLC"""
        logger.debug("Validando consistência de preços")
        
        # High deve ser >= Open, Close, Low
        high_violations = (
            (df['high'] < df['open']) |
            (df['high'] < df['close']) |
            (df['high'] < df['low'])
        )
        
        if high_violations.any():
            count = high_violations.sum()
            results['errors'].append(f"High inconsistente em {count} registros")
            results['validation_passed'] = False
        
        # Low deve ser <= Open, Close, High
        low_violations = (
            (df['low'] > df['open']) |
            (df['low'] > df['close']) |
            (df['low'] > df['high'])
        )
        
        if low_violations.any():
            count = low_violations.sum()
            results['errors'].append(f"Low inconsistente em {count} registros")
            results['validation_passed'] = False
        
        # Preços devem ser positivos
        negative_prices = (
            (df['open'] <= 0) |
            (df['high'] <= 0) |
            (df['low'] <= 0) |
            (df['close'] <= 0)
        )
        
        if negative_prices.any():
            count = negative_prices.sum()
            results['errors'].append(f"Preços não positivos em {count} registros")
            results['validation_passed'] = False
    
    def _validate_data_completeness(self, df: pd.DataFrame, results: Dict):
        """Valida completude dos dados"""
        logger.debug("Validando completude dos dados")
        
        # Verifica valores nulos
        null_counts = df[['open', 'high', 'low', 'close']].isnull().sum()
        
        for col, null_count in null_counts.items():
            if null_count > 0:
                ratio = null_count / len(df)
                if ratio > 0.01:  # Mais de 1% de nulos
                    results['errors'].append(f"Muitos valores nulos em {col}: {null_count} ({ratio:.2%})")
                    results['validation_passed'] = False
                else:
                    results['warnings'].append(f"Valores nulos em {col}: {null_count}")
        
        # Verifica gaps temporais (se index é datetime)
        if isinstance(df.index, pd.DatetimeIndex):
            self._validate_temporal_gaps(df, results)
    
    def _validate_temporal_gaps(self, df: pd.DataFrame, results: Dict):
        """Valida gaps temporais nos dados"""
        logger.debug("Validando gaps temporais")
        
        if len(df) < 2:
            return
        
        # Calcula diferenças temporais
        time_diffs = df.index.to_series().diff().dropna()
        
        # Identifica o intervalo mais comum (modo)
        most_common_interval = time_diffs.mode().iloc[0] if len(time_diffs.mode()) > 0 else None
        
        if most_common_interval is None:
            results['warnings'].append("Não foi possível determinar intervalo temporal padrão")
            return
        
        # Tolerância para variação (5% do intervalo padrão)
        tolerance = most_common_interval * 0.05
        
        # Identifica gaps significativos
        significant_gaps = time_diffs[time_diffs > most_common_interval + tolerance]
        
        if len(significant_gaps) > 0:
            gap_count = len(significant_gaps)
            max_gap = significant_gaps.max()
            results['warnings'].append(
                f"Encontrados {gap_count} gaps temporais significativos. Maior gap: {max_gap}"
            )
            
            # Se muitos gaps, pode ser um problema
            if gap_count > len(df) * 0.05:  # Mais de 5% dos dados
                results['errors'].append(f"Muitos gaps temporais: {gap_count}")
                results['validation_passed'] = False
    
    def _validate_price_ranges(self, df: pd.DataFrame, results: Dict):
        """Valida ranges de preços razoáveis"""
        logger.debug("Validando ranges de preços")
        
        # Calcula estatísticas de preços
        price_stats = df[['open', 'high', 'low', 'close']].describe()
        
        # Verifica se há preços extremamente altos ou baixos
        for col in ['open', 'high', 'low', 'close']:
            q99 = df[col].quantile(0.99)
            q01 = df[col].quantile(0.01)
            
            # Outliers extremos (mais de 10x o percentil 99)
            extreme_high = df[col] > q99 * 10
            extreme_low = df[col] < q01 * 0.1
            
            if extreme_high.any():
                count = extreme_high.sum()
                results['warnings'].append(f"Preços extremamente altos em {col}: {count} registros")
            
            if extreme_low.any():
                count = extreme_low.sum()
                results['warnings'].append(f"Preços extremamente baixos em {col}: {count} registros")
    
    def _validate_temporal_consistency(self, df: pd.DataFrame, results: Dict):
        """Valida consistência temporal"""
        logger.debug("Validando consistência temporal")
        
        if not isinstance(df.index, pd.DatetimeIndex):
            results['warnings'].append("Index não é datetime, pulando validação temporal")
            return
        
        # Verifica se dados estão ordenados
        if not df.index.is_monotonic_increasing:
            results['errors'].append("Dados não estão ordenados cronologicamente")
            results['validation_passed'] = False
        
        # Verifica duplicatas temporais
        duplicate_timestamps = df.index.duplicated()
        if duplicate_timestamps.any():
            count = duplicate_timestamps.sum()
            results['errors'].append(f"Timestamps duplicados: {count}")
            results['validation_passed'] = False
        
        # Verifica se há dados futuros
        now = datetime.now()
        future_data = df.index > now
        if future_data.any():
            count = future_data.sum()
            results['warnings'].append(f"Dados futuros encontrados: {count} registros")
    
    def _validate_volume_data(self, df: pd.DataFrame, results: Dict):
        """Valida dados de volume se disponíveis"""
        if 'volume' not in df.columns:
            return
        
        logger.debug("Validando dados de volume")
        
        # Volume deve ser não negativo
        negative_volume = df['volume'] < 0
        if negative_volume.any():
            count = negative_volume.sum()
            results['errors'].append(f"Volume negativo em {count} registros")
            results['validation_passed'] = False
        
        # Verifica volumes extremamente altos
        volume_q99 = df['volume'].quantile(0.99)
        extreme_volume = df['volume'] > volume_q99 * 100
        
        if extreme_volume.any():
            count = extreme_volume.sum()
            results['warnings'].append(f"Volumes extremamente altos: {count} registros")
        
        # Verifica se há muitos zeros
        zero_volume = df['volume'] == 0
        zero_ratio = zero_volume.sum() / len(df)
        
        if zero_ratio > 0.1:  # Mais de 10% com volume zero
            results['warnings'].append(f"Muitos registros com volume zero: {zero_ratio:.2%}")
    
    def _detect_price_anomalies(self, df: pd.DataFrame, results: Dict):
        """Detecta anomalias nos preços"""
        logger.debug("Detectando anomalias de preços")
        
        # Calcula retornos
        returns = df['close'].pct_change().dropna()
        
        # Detecta retornos extremos
        return_std = returns.std()
        extreme_returns = np.abs(returns) > return_std * 5  # 5 desvios padrão
        
        if extreme_returns.any():
            count = extreme_returns.sum()
            max_return = returns[extreme_returns].abs().max()
            results['warnings'].append(
                f"Retornos extremos detectados: {count} registros (máximo: {max_return:.2%})"
            )
        
        # Detecta sequências de preços idênticos (possível erro)
        for col in ['open', 'high', 'low', 'close']:
            identical_sequences = self._find_identical_sequences(df[col])
            if identical_sequences:
                max_sequence = max(identical_sequences)
                if max_sequence > 10:  # Mais de 10 períodos idênticos
                    results['warnings'].append(
                        f"Sequência longa de preços idênticos em {col}: {max_sequence} períodos"
                    )
    
    def _find_identical_sequences(self, series: pd.Series) -> List[int]:
        """Encontra sequências de valores idênticos"""
        sequences = []
        current_sequence = 1
        
        for i in range(1, len(series)):
            if series.iloc[i] == series.iloc[i-1]:
                current_sequence += 1
            else:
                if current_sequence > 1:
                    sequences.append(current_sequence)
                current_sequence = 1
        
        # Adiciona última sequência se necessário
        if current_sequence > 1:
            sequences.append(current_sequence)
        
        return sequences
    
    def _calculate_data_metrics(self, df: pd.DataFrame) -> Dict:
        """Calcula métricas gerais dos dados"""
        metrics = {
            'row_count': len(df),
            'date_range': {
                'start': df.index.min().isoformat() if isinstance(df.index, pd.DatetimeIndex) else None,
                'end': df.index.max().isoformat() if isinstance(df.index, pd.DatetimeIndex) else None,
                'days': (df.index.max() - df.index.min()).days if isinstance(df.index, pd.DatetimeIndex) else None
            },
            'price_statistics': {
                'close_mean': float(df['close'].mean()),
                'close_std': float(df['close'].std()),
                'close_min': float(df['close'].min()),
                'close_max': float(df['close'].max()),
                'daily_return_mean': float(df['close'].pct_change().mean()),
                'daily_return_std': float(df['close'].pct_change().std())
            },
            'data_quality': {
                'null_ratio': float(df[['open', 'high', 'low', 'close']].isnull().sum().sum() / (len(df) * 4)),
                'zero_prices': int((df[['open', 'high', 'low', 'close']] == 0).sum().sum()),
                'negative_prices': int((df[['open', 'high', 'low', 'close']] < 0).sum().sum())
            }
        }
        
        # Adiciona métricas de volume se disponível
        if 'volume' in df.columns:
            metrics['volume_statistics'] = {
                'volume_mean': float(df['volume'].mean()),
                'volume_std': float(df['volume'].std()),
                'volume_min': float(df['volume'].min()),
                'volume_max': float(df['volume'].max()),
                'zero_volume_ratio': float((df['volume'] == 0).sum() / len(df))
            }
        
        return metrics
    
    def validate_features(self, df: pd.DataFrame, feature_columns: List[str]) -> Dict:
        """
        Valida features geradas.
        
        Args:
            df: DataFrame com features
            feature_columns: Lista de colunas de features
            
        Returns:
            Dicionário com resultados da validação
        """
        logger.info("Validando features geradas")
        
        validation_results = {
            'timestamp': datetime.now().isoformat(),
            'total_features': len(feature_columns),
            'validation_passed': True,
            'errors': [],
            'warnings': [],
            'feature_metrics': {}
        }
        
        for feature in feature_columns:
            if feature not in df.columns:
                validation_results['errors'].append(f"Feature ausente: {feature}")
                validation_results['validation_passed'] = False
                continue
            
            feature_series = df[feature]
            
            # Métricas da feature
            feature_metrics = {
                'null_count': int(feature_series.isnull().sum()),
                'null_ratio': float(feature_series.isnull().sum() / len(feature_series)),
                'infinite_count': int(np.isinf(feature_series).sum()),
                'unique_values': int(feature_series.nunique()),
                'variance': float(feature_series.var()) if feature_series.dtype in ['float64', 'int64'] else None,
                'mean': float(feature_series.mean()) if feature_series.dtype in ['float64', 'int64'] else None,
                'std': float(feature_series.std()) if feature_series.dtype in ['float64', 'int64'] else None
            }
            
            validation_results['feature_metrics'][feature] = feature_metrics
            
            # Validações específicas
            if feature_metrics['null_ratio'] > 0.1:
                validation_results['warnings'].append(f"Feature {feature} tem muitos nulos: {feature_metrics['null_ratio']:.2%}")
            
            if feature_metrics['infinite_count'] > 0:
                validation_results['errors'].append(f"Feature {feature} tem valores infinitos: {feature_metrics['infinite_count']}")
                validation_results['validation_passed'] = False
            
            if feature_metrics['variance'] is not None and feature_metrics['variance'] == 0:
                validation_results['warnings'].append(f"Feature {feature} tem variância zero")
            
            if feature_metrics['unique_values'] == 1:
                validation_results['warnings'].append(f"Feature {feature} tem apenas um valor único")
        
        return validation_results
    
    def validate_target_variable(self, df: pd.DataFrame) -> Dict:
        """
        Valida a variável target.
        
        Args:
            df: DataFrame com variável target
            
        Returns:
            Dicionário com resultados da validação
        """
        logger.info("Validando variável target")
        
        validation_results = {
            'timestamp': datetime.now().isoformat(),
            'validation_passed': True,
            'errors': [],
            'warnings': [],
            'target_metrics': {}
        }
        
        if 'target' not in df.columns:
            validation_results['errors'].append("Variável target não encontrada")
            validation_results['validation_passed'] = False
            return validation_results
        
        target = df['target']
        
        # Métricas do target
        target_distribution = target.value_counts().to_dict()
        validation_results['target_metrics'] = {
            'total_samples': len(target),
            'null_count': int(target.isnull().sum()),
            'distribution': target_distribution,
            'class_balance': {
                'buy_ratio': target_distribution.get(1, 0) / len(target),
                'sell_ratio': target_distribution.get(-1, 0) / len(target),
                'neutral_ratio': target_distribution.get(0, 0) / len(target)
            }
        }
        
        # Validações
        if target.isnull().any():
            null_count = target.isnull().sum()
            validation_results['warnings'].append(f"Target tem valores nulos: {null_count}")
        
        # Verifica se target tem valores válidos (-1, 0, 1)
        valid_values = {-1, 0, 1}
        invalid_values = set(target.dropna().unique()) - valid_values
        
        if invalid_values:
            validation_results['errors'].append(f"Target tem valores inválidos: {invalid_values}")
            validation_results['validation_passed'] = False
        
        # Verifica balanceamento das classes
        min_class_ratio = min(validation_results['target_metrics']['class_balance'].values())
        if min_class_ratio < 0.05:  # Menos de 5% para alguma classe
            validation_results['warnings'].append(f"Classes muito desbalanceadas. Menor classe: {min_class_ratio:.2%}")
        
        return validation_results
    
    def save_validation_report(self, validation_results: Dict, 
                              symbol: str, timeframe: str) -> str:
        """
        Salva relatório de validação no S3.
        
        Args:
            validation_results: Resultados da validação
            symbol: Par de trading
            timeframe: Timeframe
            
        Returns:
            Path do relatório salvo
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"validation_report_{symbol}_{timeframe}_{timestamp}.json"
        
        bucket_name = self.config.get('data_bucket_name')
        s3_key = f"validation_reports/{filename}"
        
        try:
            # Converte para JSON
            report_json = json.dumps(validation_results, indent=2, default=str)
            
            # Upload para S3
            self.s3_client.put_object(
                Bucket=bucket_name,
                Key=s3_key,
                Body=report_json,
                ContentType='application/json'
            )
            
            s3_path = f"s3://{bucket_name}/{s3_key}"
            logger.info(f"Relatório de validação salvo em: {s3_path}")
            return s3_path
            
        except Exception as e:
            logger.error(f"Erro ao salvar relatório de validação: {str(e)}")
            raise

