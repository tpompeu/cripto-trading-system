"""
Sistema de Trading Quantitativo - Strategy Evaluator
Versão: 2.01
Descrição: Avaliação de estratégias e cálculo de métricas de performance
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
import boto3
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

logger = logging.getLogger(__name__)

class StrategyEvaluator:
    """
    Classe responsável pela avaliação de estratégias de trading.
    Calcula métricas de performance, risco e robustez das estratégias.
    """
    
    def __init__(self, config: Dict):
        """
        Inicializa o StrategyEvaluator.
        
        Args:
            config: Configurações do sistema
        """
        self.config = config
        self.s3_client = boto3.client('s3')
        self.bucket_name = config.get('data_bucket_name')
        
        # Configurações de avaliação
        self.eval_config = config.get('evaluation_config', {
            'risk_free_rate': 0.02,  # Taxa livre de risco anual
            'trading_days_per_year': 252,
            'min_trades_for_evaluation': 30,
            'benchmark_return': 0.0  # Retorno do benchmark (buy and hold)
        })
        
    def evaluate_strategy(self, trades: List[Dict], 
                         market_data: pd.DataFrame,
                         strategy_params: Dict) -> Dict:
        """
        Avalia uma estratégia completa.
        
        Args:
            trades: Lista de trades executados
            market_data: Dados de mercado
            strategy_params: Parâmetros da estratégia
            
        Returns:
            Dicionário com avaliação completa
        """
        logger.info("Iniciando avaliação da estratégia")
        
        if len(trades) < self.eval_config['min_trades_for_evaluation']:
            logger.warning(f"Poucos trades para avaliação: {len(trades)}")
            return self._create_insufficient_data_result(trades)
        
        # Calcula métricas básicas
        basic_metrics = self._calculate_basic_metrics(trades)
        
        # Calcula métricas de risco
        risk_metrics = self._calculate_risk_metrics(trades)
        
        # Calcula métricas de performance
        performance_metrics = self._calculate_performance_metrics(trades, market_data)
        
        # Calcula métricas de robustez
        robustness_metrics = self._calculate_robustness_metrics(trades)
        
        # Análise temporal
        temporal_analysis = self._analyze_temporal_performance(trades, market_data)
        
        # Análise de drawdown
        drawdown_analysis = self._analyze_drawdowns(trades)
        
        # Score geral da estratégia
        strategy_score = self._calculate_strategy_score(
            basic_metrics, risk_metrics, performance_metrics, robustness_metrics
        )
        
        evaluation_result = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'strategy_params': strategy_params,
            'basic_metrics': basic_metrics,
            'risk_metrics': risk_metrics,
            'performance_metrics': performance_metrics,
            'robustness_metrics': robustness_metrics,
            'temporal_analysis': temporal_analysis,
            'drawdown_analysis': drawdown_analysis,
            'strategy_score': strategy_score,
            'recommendation': self._generate_recommendation(strategy_score, basic_metrics)
        }
        
        logger.info(f"Avaliação concluída. Score: {strategy_score:.2f}")
        return evaluation_result
    
    def _calculate_basic_metrics(self, trades: List[Dict]) -> Dict:
        """Calcula métricas básicas da estratégia"""
        logger.debug("Calculando métricas básicas")
        
        if not trades:
            return self._empty_metrics()
        
        # Extrai PnLs
        pnls = [trade.get('pnl_r', 0) for trade in trades]
        winning_trades = [pnl for pnl in pnls if pnl > 0]
        losing_trades = [pnl for pnl in pnls if pnl < 0]
        
        # Métricas básicas
        total_trades = len(trades)
        winning_count = len(winning_trades)
        losing_count = len(losing_trades)
        
        metrics = {
            'total_trades': total_trades,
            'winning_trades': winning_count,
            'losing_trades': losing_count,
            'win_rate': winning_count / total_trades if total_trades > 0 else 0,
            'loss_rate': losing_count / total_trades if total_trades > 0 else 0,
            'total_return': sum(pnls),
            'average_return': np.mean(pnls),
            'average_winner': np.mean(winning_trades) if winning_trades else 0,
            'average_loser': np.mean(losing_trades) if losing_trades else 0,
            'largest_winner': max(winning_trades) if winning_trades else 0,
            'largest_loser': min(losing_trades) if losing_trades else 0,
            'expectancy': np.mean(pnls),
            'profit_factor': self._calculate_profit_factor(pnls),
            'recovery_factor': self._calculate_recovery_factor(pnls),
            'payoff_ratio': self._calculate_payoff_ratio(winning_trades, losing_trades)
        }
        
        return metrics
    
    def _calculate_risk_metrics(self, trades: List[Dict]) -> Dict:
        """Calcula métricas de risco"""
        logger.debug("Calculando métricas de risco")
        
        if not trades:
            return self._empty_risk_metrics()
        
        pnls = [trade.get('pnl_r', 0) for trade in trades]
        cumulative_pnl = np.cumsum(pnls)
        
        # Volatilidade
        volatility = np.std(pnls)
        annualized_volatility = volatility * np.sqrt(self.eval_config['trading_days_per_year'])
        
        # Drawdown
        max_drawdown = self._calculate_max_drawdown(cumulative_pnl)
        avg_drawdown = self._calculate_average_drawdown(cumulative_pnl)
        
        # Value at Risk (VaR)
        var_95 = np.percentile(pnls, 5)  # VaR 95%
        var_99 = np.percentile(pnls, 1)  # VaR 99%
        
        # Conditional VaR (Expected Shortfall)
        cvar_95 = np.mean([pnl for pnl in pnls if pnl <= var_95])
        cvar_99 = np.mean([pnl for pnl in pnls if pnl <= var_99])
        
        # Downside deviation
        downside_returns = [pnl for pnl in pnls if pnl < 0]
        downside_deviation = np.std(downside_returns) if downside_returns else 0
        
        metrics = {
            'volatility': volatility,
            'annualized_volatility': annualized_volatility,
            'max_drawdown': max_drawdown,
            'average_drawdown': avg_drawdown,
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'cvar_99': cvar_99,
            'downside_deviation': downside_deviation,
            'upside_deviation': np.std([pnl for pnl in pnls if pnl > 0]),
            'skewness': stats.skew(pnls),
            'kurtosis': stats.kurtosis(pnls),
            'tail_ratio': self._calculate_tail_ratio(pnls)
        }
        
        return metrics
    
    def _calculate_performance_metrics(self, trades: List[Dict], 
                                     market_data: pd.DataFrame) -> Dict:
        """Calcula métricas de performance"""
        logger.debug("Calculando métricas de performance")
        
        if not trades:
            return self._empty_performance_metrics()
        
        pnls = [trade.get('pnl_r', 0) for trade in trades]
        
        # Retorno médio e total
        avg_return = np.mean(pnls)
        total_return = sum(pnls)
        
        # Sharpe Ratio
        excess_return = avg_return - (self.eval_config['risk_free_rate'] / self.eval_config['trading_days_per_year'])
        sharpe_ratio = excess_return / np.std(pnls) if np.std(pnls) > 0 else 0
        
        # Sortino Ratio
        downside_returns = [pnl for pnl in pnls if pnl < avg_return]
        downside_std = np.std(downside_returns) if downside_returns else 0.001
        sortino_ratio = excess_return / downside_std
        
        # Calmar Ratio
        max_dd = abs(self._calculate_max_drawdown(np.cumsum(pnls)))
        calmar_ratio = (total_return / len(pnls) * self.eval_config['trading_days_per_year']) / max_dd if max_dd > 0 else 0
        
        # Sterling Ratio
        avg_dd = abs(self._calculate_average_drawdown(np.cumsum(pnls)))
        sterling_ratio = (total_return / len(pnls) * self.eval_config['trading_days_per_year']) / avg_dd if avg_dd > 0 else 0
        
        # Information Ratio (vs benchmark)
        benchmark_return = self.eval_config['benchmark_return'] / self.eval_config['trading_days_per_year']
        tracking_error = np.std([pnl - benchmark_return for pnl in pnls])
        information_ratio = (avg_return - benchmark_return) / tracking_error if tracking_error > 0 else 0
        
        # Omega Ratio
        omega_ratio = self._calculate_omega_ratio(pnls, threshold=0)
        
        metrics = {
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'sterling_ratio': sterling_ratio,
            'information_ratio': information_ratio,
            'omega_ratio': omega_ratio,
            'annualized_return': avg_return * self.eval_config['trading_days_per_year'],
            'total_return': total_return,
            'return_to_drawdown_ratio': total_return / abs(max_dd) if max_dd != 0 else float('inf'),
            'gain_to_pain_ratio': self._calculate_gain_to_pain_ratio(pnls)
        }
        
        return metrics
    
    def _calculate_robustness_metrics(self, trades: List[Dict]) -> Dict:
        """Calcula métricas de robustez"""
        logger.debug("Calculando métricas de robustez")
        
        if not trades:
            return self._empty_robustness_metrics()
        
        pnls = [trade.get('pnl_r', 0) for trade in trades]
        
        # Consistência de performance
        monthly_returns = self._group_returns_by_period(trades, 'month')
        quarterly_returns = self._group_returns_by_period(trades, 'quarter')
        
        # Percentage of positive periods
        positive_months = len([r for r in monthly_returns if r > 0]) / len(monthly_returns) if monthly_returns else 0
        positive_quarters = len([r for r in quarterly_returns if r > 0]) / len(quarterly_returns) if quarterly_returns else 0
        
        # Stability metrics
        stability_ratio = self._calculate_stability_ratio(pnls)
        consistency_score = self._calculate_consistency_score(monthly_returns)
        
        # Outlier analysis
        outlier_impact = self._calculate_outlier_impact(pnls)
        
        metrics = {
            'positive_months_ratio': positive_months,
            'positive_quarters_ratio': positive_quarters,
            'monthly_return_std': np.std(monthly_returns) if monthly_returns else 0,
            'quarterly_return_std': np.std(quarterly_returns) if quarterly_returns else 0,
            'stability_ratio': stability_ratio,
            'consistency_score': consistency_score,
            'outlier_impact': outlier_impact,
            'trade_frequency': len(trades) / max(1, self._calculate_trading_days(trades)),
            'average_holding_period': self._calculate_average_holding_period(trades)
        }
        
        return metrics
    
    def _analyze_temporal_performance(self, trades: List[Dict], 
                                    market_data: pd.DataFrame) -> Dict:
        """Analisa performance temporal"""
        logger.debug("Analisando performance temporal")
        
        if not trades:
            return {}
        
        # Agrupa trades por período
        monthly_analysis = self._analyze_period_performance(trades, 'month')
        quarterly_analysis = self._analyze_period_performance(trades, 'quarter')
        yearly_analysis = self._analyze_period_performance(trades, 'year')
        
        # Análise de tendências
        trend_analysis = self._analyze_performance_trends(trades)
        
        # Análise sazonal
        seasonal_analysis = self._analyze_seasonal_patterns(trades)
        
        return {
            'monthly_analysis': monthly_analysis,
            'quarterly_analysis': quarterly_analysis,
            'yearly_analysis': yearly_analysis,
            'trend_analysis': trend_analysis,
            'seasonal_analysis': seasonal_analysis
        }
    
    def _analyze_drawdowns(self, trades: List[Dict]) -> Dict:
        """Analisa drawdowns detalhadamente"""
        logger.debug("Analisando drawdowns")
        
        if not trades:
            return {}
        
        pnls = [trade.get('pnl_r', 0) for trade in trades]
        cumulative_pnl = np.cumsum(pnls)
        
        # Identifica períodos de drawdown
        drawdown_periods = self._identify_drawdown_periods(cumulative_pnl)
        
        if not drawdown_periods:
            return {
                'max_drawdown': 0,
                'average_drawdown': 0,
                'drawdown_periods': 0,
                'average_recovery_time': 0,
                'longest_drawdown_period': 0
            }
        
        # Analisa cada período de drawdown
        drawdown_magnitudes = [dd['magnitude'] for dd in drawdown_periods]
        drawdown_durations = [dd['duration'] for dd in drawdown_periods]
        recovery_times = [dd['recovery_time'] for dd in drawdown_periods if dd['recovery_time'] is not None]
        
        return {
            'max_drawdown': max(drawdown_magnitudes),
            'average_drawdown': np.mean(drawdown_magnitudes),
            'drawdown_periods': len(drawdown_periods),
            'average_drawdown_duration': np.mean(drawdown_durations),
            'longest_drawdown_period': max(drawdown_durations),
            'average_recovery_time': np.mean(recovery_times) if recovery_times else None,
            'longest_recovery_time': max(recovery_times) if recovery_times else None,
            'drawdown_details': drawdown_periods[:5]  # Top 5 drawdowns
        }
    
    def _calculate_strategy_score(self, basic_metrics: Dict, risk_metrics: Dict,
                                performance_metrics: Dict, robustness_metrics: Dict) -> float:
        """Calcula score geral da estratégia"""
        logger.debug("Calculando score da estratégia")
        
        # Pesos para cada categoria
        weights = {
            'profitability': 0.3,
            'risk_management': 0.25,
            'consistency': 0.25,
            'robustness': 0.2
        }
        
        # Score de lucratividade
        profitability_score = min(100, max(0, (
            basic_metrics.get('expectancy', 0) * 20 +
            basic_metrics.get('win_rate', 0) * 50 +
            basic_metrics.get('profit_factor', 0) * 10
        )))
        
        # Score de gestão de risco
        max_dd = abs(risk_metrics.get('max_drawdown', 0))
        risk_score = min(100, max(0, (
            100 - max_dd * 20 +  # Penaliza drawdown alto
            (1 / (1 + risk_metrics.get('volatility', 1))) * 30 +
            performance_metrics.get('sharpe_ratio', 0) * 20
        )))
        
        # Score de consistência
        consistency_score = min(100, max(0, (
            robustness_metrics.get('positive_months_ratio', 0) * 40 +
            robustness_metrics.get('consistency_score', 0) * 30 +
            (1 - robustness_metrics.get('outlier_impact', 1)) * 30
        )))
        
        # Score de robustez
        robustness_score = min(100, max(0, (
            robustness_metrics.get('stability_ratio', 0) * 40 +
            performance_metrics.get('sortino_ratio', 0) * 30 +
            basic_metrics.get('recovery_factor', 0) * 30
        )))
        
        # Score final ponderado
        final_score = (
            profitability_score * weights['profitability'] +
            risk_score * weights['risk_management'] +
            consistency_score * weights['consistency'] +
            robustness_score * weights['robustness']
        )
        
        return final_score
    
    def _generate_recommendation(self, strategy_score: float, basic_metrics: Dict) -> Dict:
        """Gera recomendação baseada na avaliação"""
        
        if strategy_score >= 80:
            recommendation = "EXCELLENT"
            action = "DEPLOY"
            confidence = "HIGH"
        elif strategy_score >= 65:
            recommendation = "GOOD"
            action = "DEPLOY_WITH_MONITORING"
            confidence = "MEDIUM"
        elif strategy_score >= 50:
            recommendation = "AVERAGE"
            action = "PAPER_TRADE_EXTENDED"
            confidence = "MEDIUM"
        elif strategy_score >= 35:
            recommendation = "POOR"
            action = "OPTIMIZE_PARAMETERS"
            confidence = "LOW"
        else:
            recommendation = "VERY_POOR"
            action = "REJECT"
            confidence = "HIGH"
        
        # Considerações específicas
        considerations = []
        
        if basic_metrics.get('expectancy', 0) <= 0:
            considerations.append("Expectância negativa - revisar estratégia")
        
        if basic_metrics.get('total_trades', 0) < 100:
            considerations.append("Poucos trades - aumentar período de teste")
        
        if basic_metrics.get('win_rate', 0) < 0.3:
            considerations.append("Taxa de acerto baixa - verificar sinais")
        
        return {
            'recommendation': recommendation,
            'action': action,
            'confidence': confidence,
            'score': strategy_score,
            'considerations': considerations
        }
    
    # Métodos auxiliares
    def _calculate_profit_factor(self, pnls: List[float]) -> float:
        """Calcula profit factor"""
        gross_profit = sum([pnl for pnl in pnls if pnl > 0])
        gross_loss = abs(sum([pnl for pnl in pnls if pnl < 0]))
        return gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    def _calculate_recovery_factor(self, pnls: List[float]) -> float:
        """Calcula recovery factor"""
        total_return = sum(pnls)
        max_dd = abs(self._calculate_max_drawdown(np.cumsum(pnls)))
        return total_return / max_dd if max_dd > 0 else float('inf')
    
    def _calculate_payoff_ratio(self, winning_trades: List[float], 
                              losing_trades: List[float]) -> float:
        """Calcula payoff ratio"""
        avg_win = np.mean(winning_trades) if winning_trades else 0
        avg_loss = abs(np.mean(losing_trades)) if losing_trades else 0.001
        return avg_win / avg_loss
    
    def _calculate_max_drawdown(self, cumulative_returns: np.ndarray) -> float:
        """Calcula máximo drawdown"""
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = cumulative_returns - peak
        return np.min(drawdown)
    
    def _calculate_average_drawdown(self, cumulative_returns: np.ndarray) -> float:
        """Calcula drawdown médio"""
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = cumulative_returns - peak
        negative_drawdowns = drawdown[drawdown < 0]
        return np.mean(negative_drawdowns) if len(negative_drawdowns) > 0 else 0
    
    def _calculate_tail_ratio(self, pnls: List[float]) -> float:
        """Calcula tail ratio"""
        p95 = np.percentile(pnls, 95)
        p5 = np.percentile(pnls, 5)
        return abs(p95 / p5) if p5 != 0 else float('inf')
    
    def _calculate_omega_ratio(self, pnls: List[float], threshold: float = 0) -> float:
        """Calcula Omega ratio"""
        gains = sum([max(0, pnl - threshold) for pnl in pnls])
        losses = sum([max(0, threshold - pnl) for pnl in pnls])
        return gains / losses if losses > 0 else float('inf')
    
    def _calculate_gain_to_pain_ratio(self, pnls: List[float]) -> float:
        """Calcula Gain-to-Pain ratio"""
        total_gain = sum(pnls)
        total_pain = sum([abs(pnl) for pnl in pnls if pnl < 0])
        return total_gain / total_pain if total_pain > 0 else float('inf')
    
    def _group_returns_by_period(self, trades: List[Dict], period: str) -> List[float]:
        """Agrupa retornos por período"""
        # Implementação simplificada - assumindo trades têm timestamp
        # Em implementação real, seria necessário agrupar por data
        pnls = [trade.get('pnl_r', 0) for trade in trades]
        
        if period == 'month':
            # Agrupa em grupos de ~20 trades (aproximadamente mensal)
            group_size = max(1, len(pnls) // max(1, len(pnls) // 20))
        elif period == 'quarter':
            # Agrupa em grupos de ~60 trades (aproximadamente trimestral)
            group_size = max(1, len(pnls) // max(1, len(pnls) // 60))
        else:  # year
            # Agrupa em grupos de ~250 trades (aproximadamente anual)
            group_size = max(1, len(pnls) // max(1, len(pnls) // 250))
        
        grouped_returns = []
        for i in range(0, len(pnls), group_size):
            group = pnls[i:i + group_size]
            grouped_returns.append(sum(group))
        
        return grouped_returns
    
    def _calculate_stability_ratio(self, pnls: List[float]) -> float:
        """Calcula ratio de estabilidade"""
        if len(pnls) < 10:
            return 0
        
        # Divide em janelas e calcula correlação entre elas
        mid_point = len(pnls) // 2
        first_half = pnls[:mid_point]
        second_half = pnls[mid_point:]
        
        if len(first_half) == 0 or len(second_half) == 0:
            return 0
        
        # Calcula médias móveis para suavizar
        window = max(5, len(first_half) // 10)
        first_ma = pd.Series(first_half).rolling(window).mean().dropna()
        second_ma = pd.Series(second_half).rolling(window).mean().dropna()
        
        if len(first_ma) == 0 or len(second_ma) == 0:
            return 0
        
        # Correlação entre as duas metades
        min_len = min(len(first_ma), len(second_ma))
        correlation = np.corrcoef(first_ma[:min_len], second_ma[:min_len])[0, 1]
        
        return max(0, correlation) if not np.isnan(correlation) else 0
    
    def _calculate_consistency_score(self, period_returns: List[float]) -> float:
        """Calcula score de consistência"""
        if not period_returns:
            return 0
        
        positive_periods = len([r for r in period_returns if r > 0])
        total_periods = len(period_returns)
        
        # Score baseado na proporção de períodos positivos
        consistency = positive_periods / total_periods
        
        # Penaliza alta volatilidade entre períodos
        volatility_penalty = min(1, np.std(period_returns) / (abs(np.mean(period_returns)) + 0.001))
        
        return max(0, consistency - volatility_penalty * 0.5) * 100
    
    def _calculate_outlier_impact(self, pnls: List[float]) -> float:
        """Calcula impacto de outliers"""
        if len(pnls) < 10:
            return 0
        
        # Remove top 5% e bottom 5%
        sorted_pnls = sorted(pnls)
        n = len(sorted_pnls)
        trimmed_pnls = sorted_pnls[int(n * 0.05):int(n * 0.95)]
        
        if not trimmed_pnls:
            return 1
        
        # Compara média com e sem outliers
        original_mean = np.mean(pnls)
        trimmed_mean = np.mean(trimmed_pnls)
        
        if original_mean == 0:
            return 0
        
        impact = abs(original_mean - trimmed_mean) / abs(original_mean)
        return min(1, impact)
    
    def _calculate_trading_days(self, trades: List[Dict]) -> int:
        """Calcula número de dias de trading"""
        # Implementação simplificada
        return max(1, len(trades) // 2)  # Assume ~2 trades por dia
    
    def _calculate_average_holding_period(self, trades: List[Dict]) -> float:
        """Calcula período médio de holding"""
        # Implementação simplificada - retorna 1 (intraday)
        return 1.0
    
    def _analyze_period_performance(self, trades: List[Dict], period: str) -> Dict:
        """Analisa performance por período"""
        period_returns = self._group_returns_by_period(trades, period)
        
        if not period_returns:
            return {}
        
        return {
            'total_periods': len(period_returns),
            'positive_periods': len([r for r in period_returns if r > 0]),
            'negative_periods': len([r for r in period_returns if r < 0]),
            'average_return': np.mean(period_returns),
            'best_period': max(period_returns),
            'worst_period': min(period_returns),
            'volatility': np.std(period_returns)
        }
    
    def _analyze_performance_trends(self, trades: List[Dict]) -> Dict:
        """Analisa tendências de performance"""
        pnls = [trade.get('pnl_r', 0) for trade in trades]
        
        if len(pnls) < 20:
            return {}
        
        # Calcula médias móveis
        window = max(10, len(pnls) // 10)
        moving_avg = pd.Series(pnls).rolling(window).mean()
        
        # Tendência linear
        x = np.arange(len(moving_avg))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, moving_avg.dropna())
        
        return {
            'trend_slope': slope,
            'trend_r_squared': r_value ** 2,
            'trend_p_value': p_value,
            'trend_direction': 'improving' if slope > 0 else 'declining',
            'trend_strength': abs(r_value)
        }
    
    def _analyze_seasonal_patterns(self, trades: List[Dict]) -> Dict:
        """Analisa padrões sazonais"""
        # Implementação simplificada
        return {
            'seasonal_analysis': 'Not implemented - requires timestamp data'
        }
    
    def _identify_drawdown_periods(self, cumulative_returns: np.ndarray) -> List[Dict]:
        """Identifica períodos de drawdown"""
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = cumulative_returns - peak
        
        drawdown_periods = []
        in_drawdown = False
        start_idx = 0
        peak_value = 0
        
        for i, dd in enumerate(drawdown):
            if dd < 0 and not in_drawdown:
                # Início do drawdown
                in_drawdown = True
                start_idx = i
                peak_value = peak[i]
            elif dd >= 0 and in_drawdown:
                # Fim do drawdown
                in_drawdown = False
                magnitude = min(drawdown[start_idx:i])
                duration = i - start_idx
                recovery_time = i - start_idx  # Simplificado
                
                drawdown_periods.append({
                    'start_index': start_idx,
                    'end_index': i,
                    'magnitude': magnitude,
                    'duration': duration,
                    'recovery_time': recovery_time,
                    'peak_value': peak_value
                })
        
        # Ordena por magnitude
        drawdown_periods.sort(key=lambda x: x['magnitude'])
        
        return drawdown_periods
    
    def _empty_metrics(self) -> Dict:
        """Retorna métricas vazias"""
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'expectancy': 0,
            'total_return': 0,
            'profit_factor': 0
        }
    
    def _empty_risk_metrics(self) -> Dict:
        """Retorna métricas de risco vazias"""
        return {
            'volatility': 0,
            'max_drawdown': 0,
            'var_95': 0,
            'sharpe_ratio': 0
        }
    
    def _empty_performance_metrics(self) -> Dict:
        """Retorna métricas de performance vazias"""
        return {
            'sharpe_ratio': 0,
            'sortino_ratio': 0,
            'calmar_ratio': 0,
            'total_return': 0
        }
    
    def _empty_robustness_metrics(self) -> Dict:
        """Retorna métricas de robustez vazias"""
        return {
            'stability_ratio': 0,
            'consistency_score': 0,
            'outlier_impact': 0
        }
    
    def _create_insufficient_data_result(self, trades: List[Dict]) -> Dict:
        """Cria resultado para dados insuficientes"""
        return {
            'evaluation_timestamp': datetime.now().isoformat(),
            'error': 'Insufficient data for evaluation',
            'total_trades': len(trades),
            'min_required_trades': self.eval_config['min_trades_for_evaluation'],
            'recommendation': {
                'recommendation': 'INSUFFICIENT_DATA',
                'action': 'COLLECT_MORE_DATA',
                'confidence': 'HIGH'
            }
        }

