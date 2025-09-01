"""
Sistema de Trading Quantitativo - Feature Engineering
Versão: 2.01
Descrição: Geração de features baseadas em Ichimoku, RSI, ATR e Candlestick
"""

import numpy as np
import pandas as pd
import talib
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Classe responsável pela geração de features para o modelo de trading.
    Implementa features baseadas em Ichimoku Kinko Hyo, RSI, ATR e padrões de candlestick.
    """
    
    def __init__(self, config: Dict):
        """
        Inicializa o FeatureEngineer com configurações.
        
        Args:
            config: Dicionário com configurações dos indicadores
        """
        self.config = config
        self.ichimoku_params = config.get('ichimoku_params', {
            'tenkan_period': 9,
            'kijun_period': 26,
            'senkou_b_period': 52,
            'displacement': 26
        })
        self.rsi_period = config.get('rsi_period', 14)
        self.atr_period = config.get('atr_period', 14)
        
    def generate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Gera todas as features para o dataset.
        
        Args:
            df: DataFrame com dados OHLCV
            
        Returns:
            DataFrame com todas as features adicionadas
        """
        logger.info("Iniciando geração de features")
        
        # Copia o DataFrame para não modificar o original
        result_df = df.copy()
        
        # Gera features de cada categoria
        result_df = self._add_ichimoku_features(result_df)
        result_df = self._add_rsi_features(result_df)
        result_df = self._add_atr_features(result_df)
        result_df = self._add_candlestick_features(result_df)
        result_df = self._add_volume_features(result_df)
        
        logger.info(f"Features geradas: {len(result_df.columns) - len(df.columns)} novas colunas")
        return result_df
    
    def _add_ichimoku_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona features baseadas no Ichimoku Kinko Hyo"""
        logger.debug("Gerando features do Ichimoku")
        
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        # Parâmetros do Ichimoku
        tenkan_period = self.ichimoku_params['tenkan_period']
        kijun_period = self.ichimoku_params['kijun_period']
        senkou_b_period = self.ichimoku_params['senkou_b_period']
        displacement = self.ichimoku_params['displacement']
        
        # Cálculo das linhas do Ichimoku
        tenkan_sen = (pd.Series(high).rolling(tenkan_period).max() + 
                     pd.Series(low).rolling(tenkan_period).min()) / 2
        
        kijun_sen = (pd.Series(high).rolling(kijun_period).max() + 
                    pd.Series(low).rolling(kijun_period).min()) / 2
        
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(displacement)
        
        senkou_span_b = ((pd.Series(high).rolling(senkou_b_period).max() + 
                         pd.Series(low).rolling(senkou_b_period).min()) / 2).shift(displacement)
        
        chikou_span = pd.Series(close).shift(-displacement)
        
        # Features do Ichimoku
        df['tenkan_sen'] = tenkan_sen
        df['kijun_sen'] = kijun_sen
        df['senkou_span_a'] = senkou_span_a
        df['senkou_span_b'] = senkou_span_b
        df['chikou_span'] = chikou_span
        
        # Features derivadas
        cloud_top = np.maximum(senkou_span_a, senkou_span_b)
        cloud_bottom = np.minimum(senkou_span_a, senkou_span_b)
        
        df['price_above_cloud'] = (close > cloud_top).astype(int)
        df['price_below_cloud'] = (close < cloud_bottom).astype(int)
        df['price_inside_cloud'] = ((close >= cloud_bottom) & (close <= cloud_top)).astype(int)
        df['cloud_green'] = (senkou_span_a > senkou_span_b).astype(int)
        df['cloud_thickness'] = (senkou_span_a - senkou_span_b) / close
        
        # Cruzamentos
        tenkan_prev = tenkan_sen.shift(1)
        kijun_prev = kijun_sen.shift(1)
        
        bullish_cross = (tenkan_sen > kijun_sen) & (tenkan_prev <= kijun_prev)
        bearish_cross = (tenkan_sen < kijun_sen) & (tenkan_prev >= kijun_prev)
        
        df['tenkan_kijun_cross'] = np.where(bullish_cross, 1, 
                                          np.where(bearish_cross, -1, 0))
        
        # Posição do preço vs Kijun
        df['price_vs_kijun'] = np.where(close > kijun_sen * 1.001, 1,
                                       np.where(close < kijun_sen * 0.999, -1, 0))
        
        # Posição da Chikou Span
        high_26_ago = pd.Series(high).shift(displacement)
        low_26_ago = pd.Series(low).shift(displacement)
        
        df['chikou_position'] = np.where(chikou_span > high_26_ago, 1,
                                        np.where(chikou_span < low_26_ago, -1, 0))
        
        # Distâncias normalizadas pelo ATR
        atr = talib.ATR(high, low, close, timeperiod=self.atr_period)
        
        cloud_edge = np.where(close > cloud_top, cloud_top,
                             np.where(close < cloud_bottom, cloud_bottom,
                                     np.where(close - cloud_bottom < cloud_top - close, 
                                             cloud_bottom, cloud_top)))
        
        df['dist_price_to_cloud'] = (close - cloud_edge) / atr
        df['dist_price_to_kijun'] = (close - kijun_sen) / atr
        
        # Kumo Twist (mudança de cor da nuvem)
        cloud_color_prev = (senkou_span_a.shift(1) > senkou_span_b.shift(1)).astype(int)
        cloud_color_curr = (senkou_span_a > senkou_span_b).astype(int)
        df['kumo_twist'] = (cloud_color_prev != cloud_color_curr).astype(int)
        
        return df
    
    def _add_rsi_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona features baseadas no RSI"""
        logger.debug("Gerando features do RSI")
        
        close = df['close'].values
        
        # RSI básico
        rsi = talib.RSI(close, timeperiod=self.rsi_period)
        df['rsi'] = rsi
        
        # Condições de sobrecompra/sobrevenda
        df['rsi_overbought'] = (rsi > 70).astype(int)
        df['rsi_oversold'] = (rsi < 30).astype(int)
        
        # Divergências (simplificadas)
        price_series = pd.Series(close)
        rsi_series = pd.Series(rsi)
        
        # Detecta topos e fundos locais
        price_peaks = self._find_peaks(price_series, window=5)
        price_troughs = self._find_troughs(price_series, window=5)
        rsi_peaks = self._find_peaks(rsi_series, window=5)
        rsi_troughs = self._find_troughs(rsi_series, window=5)
        
        # Divergência de alta (preço faz fundo mais baixo, RSI faz fundo mais alto)
        bullish_div = self._detect_bullish_divergence(price_troughs, rsi_troughs, window=20)
        df['rsi_bullish_div'] = bullish_div.astype(int)
        
        # Divergência de baixa (preço faz topo mais alto, RSI faz topo mais baixo)
        bearish_div = self._detect_bearish_divergence(price_peaks, rsi_peaks, window=20)
        df['rsi_bearish_div'] = bearish_div.astype(int)
        
        # Inclinação do RSI
        df['rsi_slope'] = (rsi_series - rsi_series.shift(3)) / 3
        
        # Volatilidade do RSI
        df['rsi_std'] = rsi_series.rolling(14).std()
        
        # Rejeição na linha central
        rsi_prev = rsi_series.shift(1)
        rejection_up = (rsi_prev < 50) & (rsi > 50) & (rsi < 55)
        rejection_down = (rsi_prev > 50) & (rsi < 50) & (rsi > 45)
        
        df['rsi_midline_rejection'] = np.where(rejection_up, 1,
                                              np.where(rejection_down, -1, 0))
        
        # Zona de momentum
        df['rsi_in_momentum_zone'] = np.where(rsi > 60, 1,
                                             np.where(rsi < 40, -1, 0))
        
        # Tempo em zona extrema
        extreme_zone = (rsi > 70) | (rsi < 30)
        df['time_in_extreme_zone'] = extreme_zone.groupby((~extreme_zone).cumsum()).cumsum()
        
        return df
    
    def _add_atr_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona features baseadas no ATR"""
        logger.debug("Gerando features do ATR")
        
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        # ATR básico
        atr = talib.ATR(high, low, close, timeperiod=self.atr_period)
        df['atr'] = atr
        
        # Volatilidade relativa
        df['atr_pct'] = (atr / close) * 100
        
        # ATR relativo ao máximo histórico
        atr_series = pd.Series(atr)
        df['atr_expanding'] = atr / atr_series.rolling(252).max()
        
        # Amplitude do preço normalizada
        df['price_atr_ratio'] = (high - low) / atr
        
        # Correlação ATR-Volume (se volume disponível)
        if 'volume' in df.columns:
            volume_series = pd.Series(df['volume'].values)
            df['atr_volume'] = atr_series.rolling(10).corr(volume_series.rolling(10))
        else:
            df['atr_volume'] = 0
        
        # Compressão/expansão da volatilidade
        atr_5 = talib.ATR(high, low, close, timeperiod=5)
        atr_20 = talib.ATR(high, low, close, timeperiod=20)
        df['compression_ratio'] = atr_5 / atr_20
        
        # Regime de volatilidade
        atr_pct_series = pd.Series(df['atr_pct'].values)
        low_vol_threshold = atr_pct_series.rolling(252).quantile(0.33)
        high_vol_threshold = atr_pct_series.rolling(252).quantile(0.67)
        
        df['atr_regime'] = np.where(df['atr_pct'] < low_vol_threshold, 0,
                                   np.where(df['atr_pct'] > high_vol_threshold, 2, 1))
        
        # Breakout de volatilidade
        atr_mean = atr_series.rolling(20).mean()
        atr_std = atr_series.rolling(20).std()
        df['atr_breakout'] = (atr > (atr_mean + 2 * atr_std)).astype(int)
        
        # Volatilidade extremamente baixa
        low_vol_percentile = atr_pct_series.rolling(252).quantile(0.1)
        df['is_low_volatility'] = (df['atr_pct'] < low_vol_percentile).astype(int)
        
        # Tendência da volatilidade
        atr_sma_5 = atr_series.rolling(5).mean()
        atr_sma_20 = atr_series.rolling(20).mean()
        df['atr_trend'] = (atr_sma_5 > atr_sma_20).astype(int)
        
        return df
    
    def _add_candlestick_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona features baseadas em padrões de candlestick"""
        logger.debug("Gerando features de candlestick")
        
        open_price = df['open'].values
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        # Proporções do corpo e sombras
        body_size = np.abs(close - open_price)
        total_range = high - low
        
        # Evita divisão por zero
        total_range = np.where(total_range == 0, 0.0001, total_range)
        
        df['body_size_ratio'] = body_size / total_range
        df['upper_shadow_ratio'] = (high - np.maximum(open_price, close)) / total_range
        df['lower_shadow_ratio'] = (np.minimum(open_price, close) - low) / total_range
        
        # Padrões de candlestick usando TA-Lib
        df['hammer'] = talib.CDLHAMMER(open_price, high, low, close)
        df['doji'] = talib.CDLDOJI(open_price, high, low, close)
        df['engulfing'] = talib.CDLENGULFING(open_price, high, low, close)
        df['morning_star'] = talib.CDLMORNINGSTAR(open_price, high, low, close)
        df['evening_star'] = talib.CDLEVENINGSTAR(open_price, high, low, close)
        df['shooting_star'] = talib.CDLSHOOTINGSTAR(open_price, high, low, close)
        df['hanging_man'] = talib.CDLHANGINGMAN(open_price, high, low, close)
        
        # Normaliza os padrões para 0, 1, -1
        pattern_columns = ['hammer', 'doji', 'engulfing', 'morning_star', 
                          'evening_star', 'shooting_star', 'hanging_man']
        
        for col in pattern_columns:
            df[col] = np.where(df[col] > 0, 1, np.where(df[col] < 0, -1, 0))
        
        # Gap analysis
        prev_close = pd.Series(close).shift(1)
        df['gap_up'] = ((open_price > prev_close * 1.002) & 
                       (low > prev_close)).astype(int)
        df['gap_down'] = ((open_price < prev_close * 0.998) & 
                         (high < prev_close)).astype(int)
        
        # Momentum intrabar
        df['intrabar_momentum'] = (close - open_price) / total_range
        
        return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona features baseadas em volume (se disponível)"""
        if 'volume' not in df.columns:
            logger.warning("Volume não disponível, pulando features de volume")
            return df
            
        logger.debug("Gerando features de volume")
        
        volume = df['volume'].values
        close = df['close'].values
        
        # Volume relativo
        volume_series = pd.Series(volume)
        volume_sma_20 = volume_series.rolling(20).mean()
        df['volume_ratio'] = volume / volume_sma_20
        
        # On-Balance Volume
        df['obv'] = talib.OBV(close, volume)
        
        # Volume-Price Trend
        df['vpt'] = ((close - pd.Series(close).shift(1)) / pd.Series(close).shift(1) * volume).cumsum()
        
        # Accumulation/Distribution Line
        df['ad_line'] = talib.AD(df['high'].values, df['low'].values, close, volume)
        
        return df
    
    def _find_peaks(self, series: pd.Series, window: int = 5) -> pd.Series:
        """Encontra picos locais em uma série"""
        peaks = pd.Series(False, index=series.index)
        for i in range(window, len(series) - window):
            if series.iloc[i] == series.iloc[i-window:i+window+1].max():
                peaks.iloc[i] = True
        return peaks
    
    def _find_troughs(self, series: pd.Series, window: int = 5) -> pd.Series:
        """Encontra vales locais em uma série"""
        troughs = pd.Series(False, index=series.index)
        for i in range(window, len(series) - window):
            if series.iloc[i] == series.iloc[i-window:i+window+1].min():
                troughs.iloc[i] = True
        return troughs
    
    def _detect_bullish_divergence(self, price_troughs: pd.Series, 
                                  rsi_troughs: pd.Series, window: int = 20) -> pd.Series:
        """Detecta divergência de alta"""
        divergence = pd.Series(False, index=price_troughs.index)
        
        price_trough_indices = price_troughs[price_troughs].index
        rsi_trough_indices = rsi_troughs[rsi_troughs].index
        
        for i, idx in enumerate(price_trough_indices[1:], 1):
            # Procura por trough anterior dentro da janela
            prev_troughs = price_trough_indices[price_trough_indices < idx]
            prev_troughs = prev_troughs[prev_troughs >= idx - window]
            
            if len(prev_troughs) > 0:
                prev_idx = prev_troughs[-1]
                
                # Verifica se há troughs correspondentes no RSI
                rsi_current = rsi_trough_indices[abs(rsi_trough_indices - idx) <= 2]
                rsi_previous = rsi_trough_indices[abs(rsi_trough_indices - prev_idx) <= 2]
                
                if len(rsi_current) > 0 and len(rsi_previous) > 0:
                    # Divergência: preço mais baixo, RSI mais alto
                    if (price_troughs.loc[idx] and price_troughs.loc[prev_idx] and
                        idx < prev_idx and rsi_current[0] > rsi_previous[0]):
                        divergence.loc[idx] = True
        
        return divergence
    
    def _detect_bearish_divergence(self, price_peaks: pd.Series, 
                                  rsi_peaks: pd.Series, window: int = 20) -> pd.Series:
        """Detecta divergência de baixa"""
        divergence = pd.Series(False, index=price_peaks.index)
        
        price_peak_indices = price_peaks[price_peaks].index
        rsi_peak_indices = rsi_peaks[rsi_peaks].index
        
        for i, idx in enumerate(price_peak_indices[1:], 1):
            # Procura por peak anterior dentro da janela
            prev_peaks = price_peak_indices[price_peak_indices < idx]
            prev_peaks = prev_peaks[prev_peaks >= idx - window]
            
            if len(prev_peaks) > 0:
                prev_idx = prev_peaks[-1]
                
                # Verifica se há peaks correspondentes no RSI
                rsi_current = rsi_peak_indices[abs(rsi_peak_indices - idx) <= 2]
                rsi_previous = rsi_peak_indices[abs(rsi_peak_indices - prev_idx) <= 2]
                
                if len(rsi_current) > 0 and len(rsi_previous) > 0:
                    # Divergência: preço mais alto, RSI mais baixo
                    if (price_peaks.loc[idx] and price_peaks.loc[prev_idx] and
                        idx > prev_idx and rsi_current[0] < rsi_previous[0]):
                        divergence.loc[idx] = True
        
        return divergence

def create_target_variable(df: pd.DataFrame, atr_multiplier: float = 1.0) -> pd.DataFrame:
    """
    Cria a variável alvo baseada na abordagem de Potencial de Tendência.
    
    Args:
        df: DataFrame com dados OHLCV e ATR
        atr_multiplier: Multiplicador do ATR para definir stop loss
        
    Returns:
        DataFrame com a coluna 'target' adicionada
    """
    logger.info("Criando variável alvo")
    
    result_df = df.copy()
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    atr = df['atr'].values
    
    targets = []
    
    for i in range(len(df)):
        if i >= len(df) - 1:  # Último ponto não pode ter target
            targets.append(0)
            continue
            
        current_close = close[i]
        current_atr = atr[i]
        
        # Define stop loss baseado no ATR
        stop_loss_distance = atr_multiplier * current_atr
        
        # Preços de stop para long e short
        long_stop = current_close - stop_loss_distance
        short_stop = current_close + stop_loss_distance
        
        # Targets de lucro (3R)
        long_target = current_close + (3 * stop_loss_distance)
        short_target = current_close - (3 * stop_loss_distance)
        
        # Analisa preços futuros para determinar se algum target é atingido
        future_highs = high[i+1:min(i+50, len(high))]  # Próximos 50 períodos
        future_lows = low[i+1:min(i+50, len(low))]
        
        target = 0  # Neutro por padrão
        
        # Verifica se long target é atingido antes do stop
        for j, (future_high, future_low) in enumerate(zip(future_highs, future_lows)):
            # Para posição long
            if future_low <= long_stop:  # Stop atingido primeiro
                break
            if future_high >= long_target:  # Target atingido
                target = 1  # Sinal de compra
                break
        
        # Se não encontrou sinal de compra, verifica short
        if target == 0:
            for j, (future_high, future_low) in enumerate(zip(future_highs, future_lows)):
                # Para posição short
                if future_high >= short_stop:  # Stop atingido primeiro
                    break
                if future_low <= short_target:  # Target atingido
                    target = -1  # Sinal de venda
                    break
        
        targets.append(target)
    
    result_df['target'] = targets
    
    # Log da distribuição dos targets
    target_counts = pd.Series(targets).value_counts()
    logger.info(f"Distribuição dos targets: {target_counts.to_dict()}")
    
    return result_df

