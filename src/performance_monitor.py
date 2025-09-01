"""
Sistema de Trading Quantitativo - Performance Monitor
Versão: 2.01
Descrição: Monitor de performance das estratégias em tempo real
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
import boto3
import json
from enum import Enum
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class PerformanceStatus(Enum):
    """Status de performance da estratégia"""
    EXCELLENT = "EXCELLENT"
    GOOD = "GOOD"
    AVERAGE = "AVERAGE"
    POOR = "POOR"
    DEGRADED = "DEGRADED"

class PerformanceMonitor:
    """
    Classe responsável pelo monitoramento de performance das estratégias.
    Acompanha métricas em tempo real e detecta degradação de performance.
    """
    
    def __init__(self, config: Dict):
        """
        Inicializa o PerformanceMonitor.
        
        Args:
            config: Configurações do sistema
        """
        self.config = config
        self.s3_client = boto3.client('s3')
        self.dynamodb = boto3.resource('dynamodb')
        self.cloudwatch = boto3.client('cloudwatch')
        
        # Configurações de performance
        self.perf_config = config.get('performance_config', {
            'monitoring_window_days': 30,
            'benchmark_window_days': 90,
            'min_trades_for_analysis': 20,
            'expectancy_threshold': 0.1,
            'sharpe_threshold': 0.5,
            'max_drawdown_threshold': 0.15,
            'win_rate_threshold': 0.4,
            'degradation_threshold': 0.3,  # 30% degradação para alerta
            'update_frequency_minutes': 15
        })
        
        # Tabela DynamoDB para métricas de performance
        self.performance_table = config.get('performance_table', 'trading-system-performance')
        
    def monitor_strategy_performance(self, strategy_id: str,
                                   recent_trades: List[Dict],
                                   benchmark_trades: List[Dict]) -> Dict:
        """
        Monitora performance de uma estratégia.
        
        Args:
            strategy_id: ID da estratégia
            recent_trades: Trades recentes
            benchmark_trades: Trades de benchmark (período de referência)
            
        Returns:
            Dicionário com análise de performance
        """
        logger.info(f"Monitorando performance da estratégia: {strategy_id}")
        
        # Calcula métricas atuais
        current_metrics = self._calculate_current_metrics(recent_trades)
        
        # Calcula métricas de benchmark
        benchmark_metrics = self._calculate_benchmark_metrics(benchmark_trades)
        
        # Compara performance
        performance_comparison = self._compare_performance(current_metrics, benchmark_metrics)
        
        # Avalia status de performance
        performance_status = self._assess_performance_status(current_metrics, benchmark_metrics)
        
        # Detecta degradação
        degradation_analysis = self._detect_performance_degradation(
            current_metrics, benchmark_metrics
        )
        
        # Gera insights
        insights = self._generate_performance_insights(
            current_metrics, benchmark_metrics, performance_comparison
        )
        
        # Calcula tendências
        trend_analysis = self._analyze_performance_trends(recent_trades)
        
        # Atualiza métricas no CloudWatch
        self._update_cloudwatch_metrics(strategy_id, current_metrics)
        
        # Salva histórico
        self._save_performance_history(strategy_id, current_metrics)
        
        performance_report = {
            'strategy_id': strategy_id,
            'timestamp': datetime.now().isoformat(),
            'performance_status': performance_status.value,
            'current_metrics': current_metrics,
            'benchmark_metrics': benchmark_metrics,
            'performance_comparison': performance_comparison,
            'degradation_analysis': degradation_analysis,
            'trend_analysis': trend_analysis,
            'insights': insights
        }
        
        logger.info(f"Performance monitorada. Status: {performance_status.value}")
        return performance_report
    
    def _calculate_current_metrics(self, trades: List[Dict]) -> Dict:
        """Calcula métricas de performance atuais"""
        logger.debug("Calculando métricas atuais")
        
        if not trades:
            return self._empty_metrics()
        
        # Extrai PnLs
        pnls = [trade.get('pnl_r', 0) for trade in trades]
        
        # Métricas básicas
        total_trades = len(trades)
        winning_trades = len([pnl for pnl in pnls if pnl > 0])
        losing_trades = len([pnl for pnl in pnls if pnl < 0])
        
        # Retornos
        total_return = sum(pnls)
        avg_return = np.mean(pnls)
        
        # Win rate
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Expectância
        expectancy = avg_return
        
        # Profit factor
        gross_profit = sum([pnl for pnl in pnls if pnl > 0])
        gross_loss = abs(sum([pnl for pnl in pnls if pnl < 0]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Sharpe ratio
        sharpe_ratio = avg_return / np.std(pnls) if np.std(pnls) > 0 else 0
        
        # Drawdown
        cumulative_pnl = np.cumsum(pnls)
        peak = np.maximum.accumulate(cumulative_pnl)
        drawdown = cumulative_pnl - peak
        max_drawdown = np.min(drawdown)
        
        # Métricas de consistência
        positive_periods = self._calculate_positive_periods(pnls)
        volatility = np.std(pnls)
        
        # Métricas de tempo
        avg_trade_duration = self._calculate_avg_trade_duration(trades)
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_return': total_return,
            'avg_return': avg_return,
            'expectancy': expectancy,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'positive_periods': positive_periods,
            'avg_trade_duration': avg_trade_duration,
            'calculation_timestamp': datetime.now().isoformat()
        }
    
    def _calculate_benchmark_metrics(self, benchmark_trades: List[Dict]) -> Dict:
        """Calcula métricas de benchmark"""
        logger.debug("Calculando métricas de benchmark")
        
        if not benchmark_trades:
            return self._empty_metrics()
        
        return self._calculate_current_metrics(benchmark_trades)
    
    def _compare_performance(self, current: Dict, benchmark: Dict) -> Dict:
        """Compara performance atual com benchmark"""
        logger.debug("Comparando performance")
        
        if not benchmark or benchmark.get('total_trades', 0) == 0:
            return {'comparison_available': False}
        
        # Calcula diferenças relativas
        comparisons = {}
        
        metrics_to_compare = [
            'win_rate', 'expectancy', 'profit_factor', 'sharpe_ratio',
            'max_drawdown', 'volatility', 'avg_return'
        ]
        
        for metric in metrics_to_compare:
            current_value = current.get(metric, 0)
            benchmark_value = benchmark.get(metric, 0)
            
            if benchmark_value != 0:
                relative_change = (current_value - benchmark_value) / abs(benchmark_value)
                comparisons[f'{metric}_change'] = relative_change
                comparisons[f'{metric}_improved'] = relative_change > 0
            else:
                comparisons[f'{metric}_change'] = 0
                comparisons[f'{metric}_improved'] = current_value > 0
        
        # Score geral de melhoria
        improvement_score = sum([
            1 if comparisons.get(f'{metric}_improved', False) else 0
            for metric in metrics_to_compare
        ]) / len(metrics_to_compare)
        
        comparisons['overall_improvement_score'] = improvement_score
        comparisons['comparison_available'] = True
        
        return comparisons
    
    def _assess_performance_status(self, current: Dict, benchmark: Dict) -> PerformanceStatus:
        """Avalia status de performance"""
        logger.debug("Avaliando status de performance")
        
        # Score baseado em métricas absolutas
        absolute_score = 0
        
        # Expectância
        if current.get('expectancy', 0) > self.perf_config['expectancy_threshold']:
            absolute_score += 2
        elif current.get('expectancy', 0) > 0:
            absolute_score += 1
        
        # Sharpe ratio
        if current.get('sharpe_ratio', 0) > self.perf_config['sharpe_threshold']:
            absolute_score += 2
        elif current.get('sharpe_ratio', 0) > 0:
            absolute_score += 1
        
        # Win rate
        if current.get('win_rate', 0) > self.perf_config['win_rate_threshold']:
            absolute_score += 1
        
        # Drawdown (penaliza)
        if abs(current.get('max_drawdown', 0)) < self.perf_config['max_drawdown_threshold']:
            absolute_score += 1
        
        # Profit factor
        if current.get('profit_factor', 0) > 1.5:
            absolute_score += 2
        elif current.get('profit_factor', 0) > 1.0:
            absolute_score += 1
        
        # Score relativo (comparação com benchmark)
        relative_score = 0
        if benchmark and benchmark.get('total_trades', 0) > 0:
            comparison = self._compare_performance(current, benchmark)
            if comparison.get('comparison_available', False):
                improvement_score = comparison.get('overall_improvement_score', 0)
                if improvement_score > 0.7:
                    relative_score += 2
                elif improvement_score > 0.5:
                    relative_score += 1
                elif improvement_score < 0.3:
                    relative_score -= 2
        
        # Score final
        total_score = absolute_score + relative_score
        
        # Determina status
        if total_score >= 8:
            return PerformanceStatus.EXCELLENT
        elif total_score >= 6:
            return PerformanceStatus.GOOD
        elif total_score >= 4:
            return PerformanceStatus.AVERAGE
        elif total_score >= 2:
            return PerformanceStatus.POOR
        else:
            return PerformanceStatus.DEGRADED
    
    def _detect_performance_degradation(self, current: Dict, benchmark: Dict) -> Dict:
        """Detecta degradação de performance"""
        logger.debug("Detectando degradação de performance")
        
        if not benchmark or benchmark.get('total_trades', 0) == 0:
            return {'degradation_detected': False, 'reason': 'No benchmark available'}
        
        degradation_signals = []
        
        # Verifica expectância
        current_expectancy = current.get('expectancy', 0)
        benchmark_expectancy = benchmark.get('expectancy', 0)
        
        if benchmark_expectancy > 0:
            expectancy_degradation = (benchmark_expectancy - current_expectancy) / benchmark_expectancy
            if expectancy_degradation > self.perf_config['degradation_threshold']:
                degradation_signals.append({
                    'metric': 'expectancy',
                    'degradation': expectancy_degradation,
                    'current': current_expectancy,
                    'benchmark': benchmark_expectancy
                })
        
        # Verifica Sharpe ratio
        current_sharpe = current.get('sharpe_ratio', 0)
        benchmark_sharpe = benchmark.get('sharpe_ratio', 0)
        
        if benchmark_sharpe > 0:
            sharpe_degradation = (benchmark_sharpe - current_sharpe) / benchmark_sharpe
            if sharpe_degradation > self.perf_config['degradation_threshold']:
                degradation_signals.append({
                    'metric': 'sharpe_ratio',
                    'degradation': sharpe_degradation,
                    'current': current_sharpe,
                    'benchmark': benchmark_sharpe
                })
        
        # Verifica win rate
        current_wr = current.get('win_rate', 0)
        benchmark_wr = benchmark.get('win_rate', 0)
        
        if benchmark_wr > 0:
            wr_degradation = (benchmark_wr - current_wr) / benchmark_wr
            if wr_degradation > self.perf_config['degradation_threshold']:
                degradation_signals.append({
                    'metric': 'win_rate',
                    'degradation': wr_degradation,
                    'current': current_wr,
                    'benchmark': benchmark_wr
                })
        
        # Verifica drawdown (piora se aumenta)
        current_dd = abs(current.get('max_drawdown', 0))
        benchmark_dd = abs(benchmark.get('max_drawdown', 0))
        
        if benchmark_dd > 0:
            dd_increase = (current_dd - benchmark_dd) / benchmark_dd
            if dd_increase > self.perf_config['degradation_threshold']:
                degradation_signals.append({
                    'metric': 'max_drawdown',
                    'degradation': dd_increase,
                    'current': current_dd,
                    'benchmark': benchmark_dd
                })
        
        # Determina se há degradação significativa
        degradation_detected = len(degradation_signals) >= 2  # Pelo menos 2 métricas degradadas
        
        return {
            'degradation_detected': degradation_detected,
            'degradation_signals': degradation_signals,
            'degradation_count': len(degradation_signals),
            'severity': 'HIGH' if len(degradation_signals) >= 3 else 'MEDIUM' if degradation_detected else 'LOW'
        }
    
    def _generate_performance_insights(self, current: Dict, benchmark: Dict,
                                     comparison: Dict) -> Dict:
        """Gera insights sobre performance"""
        logger.debug("Gerando insights de performance")
        
        insights = {
            'key_findings': [],
            'recommendations': [],
            'strengths': [],
            'weaknesses': []
        }
        
        # Analisa pontos fortes
        if current.get('expectancy', 0) > 0.2:
            insights['strengths'].append("Expectância positiva forte")
        
        if current.get('sharpe_ratio', 0) > 1.0:
            insights['strengths'].append("Excelente relação risco-retorno")
        
        if current.get('win_rate', 0) > 0.6:
            insights['strengths'].append("Alta taxa de acerto")
        
        if current.get('profit_factor', 0) > 2.0:
            insights['strengths'].append("Profit factor excelente")
        
        # Analisa pontos fracos
        if current.get('expectancy', 0) < 0:
            insights['weaknesses'].append("Expectância negativa")
            insights['recommendations'].append("Revisar estratégia de entrada e saída")
        
        if current.get('win_rate', 0) < 0.4:
            insights['weaknesses'].append("Taxa de acerto baixa")
            insights['recommendations'].append("Melhorar precisão dos sinais")
        
        if abs(current.get('max_drawdown', 0)) > 0.2:
            insights['weaknesses'].append("Drawdown elevado")
            insights['recommendations'].append("Implementar melhor gestão de risco")
        
        if current.get('volatility', 0) > 0.05:
            insights['weaknesses'].append("Alta volatilidade dos retornos")
            insights['recommendations'].append("Suavizar estratégia ou reduzir posições")
        
        # Insights de comparação
        if comparison.get('comparison_available', False):
            improvement_score = comparison.get('overall_improvement_score', 0)
            
            if improvement_score > 0.7:
                insights['key_findings'].append("Performance melhorou significativamente vs benchmark")
            elif improvement_score < 0.3:
                insights['key_findings'].append("Performance degradou vs benchmark")
                insights['recommendations'].append("Investigar causas da degradação")
        
        # Recomendações específicas
        if current.get('total_trades', 0) < self.perf_config['min_trades_for_analysis']:
            insights['recommendations'].append("Coletar mais dados para análise robusta")
        
        return insights
    
    def _analyze_performance_trends(self, trades: List[Dict]) -> Dict:
        """Analisa tendências de performance"""
        logger.debug("Analisando tendências de performance")
        
        if len(trades) < 10:
            return {'trend_analysis_available': False, 'reason': 'Insufficient data'}
        
        # Ordena trades por timestamp (assumindo que existe)
        sorted_trades = sorted(trades, key=lambda x: x.get('timestamp', 0))
        
        # Calcula retornos cumulativos
        pnls = [trade.get('pnl_r', 0) for trade in sorted_trades]
        cumulative_returns = np.cumsum(pnls)
        
        # Analisa tendência linear
        x = np.arange(len(cumulative_returns))
        slope = np.polyfit(x, cumulative_returns, 1)[0]
        
        # Analisa volatilidade ao longo do tempo
        window_size = max(5, len(pnls) // 5)
        rolling_volatility = pd.Series(pnls).rolling(window_size).std()
        
        # Analisa win rate ao longo do tempo
        rolling_wins = pd.Series([1 if pnl > 0 else 0 for pnl in pnls]).rolling(window_size).mean()
        
        # Determina direção da tendência
        if slope > 0.01:
            trend_direction = "IMPROVING"
        elif slope < -0.01:
            trend_direction = "DECLINING"
        else:
            trend_direction = "STABLE"
        
        return {
            'trend_analysis_available': True,
            'trend_direction': trend_direction,
            'trend_slope': float(slope),
            'volatility_trend': 'INCREASING' if rolling_volatility.iloc[-1] > rolling_volatility.iloc[0] else 'DECREASING',
            'win_rate_trend': 'IMPROVING' if rolling_wins.iloc[-1] > rolling_wins.iloc[0] else 'DECLINING',
            'recent_performance': 'POSITIVE' if sum(pnls[-5:]) > 0 else 'NEGATIVE'
        }
    
    def _calculate_positive_periods(self, pnls: List[float]) -> float:
        """Calcula proporção de períodos positivos"""
        if not pnls:
            return 0
        
        # Agrupa em períodos (simplificado - grupos de 5 trades)
        period_size = max(1, len(pnls) // 10)
        periods = []
        
        for i in range(0, len(pnls), period_size):
            period_pnl = sum(pnls[i:i + period_size])
            periods.append(period_pnl)
        
        positive_periods = len([p for p in periods if p > 0])
        return positive_periods / len(periods) if periods else 0
    
    def _calculate_avg_trade_duration(self, trades: List[Dict]) -> float:
        """Calcula duração média dos trades"""
        # Simplificado - retorna 1 (assumindo trades intraday)
        return 1.0
    
    def _update_cloudwatch_metrics(self, strategy_id: str, metrics: Dict):
        """Atualiza métricas no CloudWatch"""
        try:
            namespace = 'TradingSystem/Performance'
            
            # Métricas principais
            cloudwatch_metrics = [
                ('Expectancy', metrics.get('expectancy', 0)),
                ('WinRate', metrics.get('win_rate', 0)),
                ('SharpeRatio', metrics.get('sharpe_ratio', 0)),
                ('MaxDrawdown', abs(metrics.get('max_drawdown', 0))),
                ('ProfitFactor', metrics.get('profit_factor', 0)),
                ('TotalTrades', metrics.get('total_trades', 0))
            ]
            
            for metric_name, value in cloudwatch_metrics:
                self.cloudwatch.put_metric_data(
                    Namespace=namespace,
                    MetricData=[
                        {
                            'MetricName': metric_name,
                            'Dimensions': [
                                {
                                    'Name': 'StrategyId',
                                    'Value': strategy_id
                                }
                            ],
                            'Value': float(value),
                            'Timestamp': datetime.now()
                        }
                    ]
                )
            
            logger.debug(f"Métricas CloudWatch atualizadas para {strategy_id}")
            
        except Exception as e:
            logger.warning(f"Erro ao atualizar CloudWatch: {str(e)}")
    
    def _save_performance_history(self, strategy_id: str, metrics: Dict):
        """Salva histórico de performance no DynamoDB"""
        try:
            table = self.dynamodb.Table(self.performance_table)
            
            # Cria item com timestamp
            item = {
                'strategy_id': strategy_id,
                'timestamp': datetime.now().isoformat(),
                'metrics': metrics,
                'ttl': int((datetime.now() + timedelta(days=365)).timestamp())  # TTL de 1 ano
            }
            
            table.put_item(Item=item)
            logger.debug(f"Histórico de performance salvo para {strategy_id}")
            
        except Exception as e:
            logger.warning(f"Erro ao salvar histórico: {str(e)}")
    
    def get_performance_history(self, strategy_id: str, 
                              days: int = 30) -> List[Dict]:
        """
        Obtém histórico de performance.
        
        Args:
            strategy_id: ID da estratégia
            days: Número de dias de histórico
            
        Returns:
            Lista com histórico de performance
        """
        try:
            table = self.dynamodb.Table(self.performance_table)
            
            # Data de início
            start_date = datetime.now() - timedelta(days=days)
            
            response = table.query(
                KeyConditionExpression='strategy_id = :sid AND #ts >= :start_date',
                ExpressionAttributeNames={'#ts': 'timestamp'},
                ExpressionAttributeValues={
                    ':sid': strategy_id,
                    ':start_date': start_date.isoformat()
                },
                ScanIndexForward=True
            )
            
            return response.get('Items', [])
            
        except Exception as e:
            logger.error(f"Erro ao obter histórico: {str(e)}")
            return []
    
    def generate_performance_report(self, strategy_id: str) -> Dict:
        """
        Gera relatório completo de performance.
        
        Args:
            strategy_id: ID da estratégia
            
        Returns:
            Relatório de performance
        """
        logger.info(f"Gerando relatório de performance para {strategy_id}")
        
        # Obtém histórico
        history = self.get_performance_history(strategy_id, days=90)
        
        if not history:
            return {
                'strategy_id': strategy_id,
                'error': 'No performance history available'
            }
        
        # Analisa evolução das métricas
        evolution_analysis = self._analyze_metrics_evolution(history)
        
        # Calcula estatísticas agregadas
        aggregate_stats = self._calculate_aggregate_statistics(history)
        
        # Gera gráficos de performance
        chart_paths = self._generate_performance_charts(strategy_id, history)
        
        return {
            'strategy_id': strategy_id,
            'report_timestamp': datetime.now().isoformat(),
            'history_period_days': 90,
            'total_data_points': len(history),
            'evolution_analysis': evolution_analysis,
            'aggregate_statistics': aggregate_stats,
            'chart_paths': chart_paths
        }
    
    def _analyze_metrics_evolution(self, history: List[Dict]) -> Dict:
        """Analisa evolução das métricas ao longo do tempo"""
        if len(history) < 2:
            return {}
        
        # Extrai métricas ao longo do tempo
        timestamps = [item['timestamp'] for item in history]
        expectancies = [item['metrics'].get('expectancy', 0) for item in history]
        sharpe_ratios = [item['metrics'].get('sharpe_ratio', 0) for item in history]
        win_rates = [item['metrics'].get('win_rate', 0) for item in history]
        
        # Calcula tendências
        def calculate_trend(values):
            if len(values) < 2:
                return 0
            x = np.arange(len(values))
            return np.polyfit(x, values, 1)[0]
        
        return {
            'expectancy_trend': calculate_trend(expectancies),
            'sharpe_trend': calculate_trend(sharpe_ratios),
            'win_rate_trend': calculate_trend(win_rates),
            'overall_trend': 'IMPROVING' if calculate_trend(expectancies) > 0 else 'DECLINING'
        }
    
    def _calculate_aggregate_statistics(self, history: List[Dict]) -> Dict:
        """Calcula estatísticas agregadas"""
        if not history:
            return {}
        
        # Extrai todas as métricas
        all_expectancies = [item['metrics'].get('expectancy', 0) for item in history]
        all_sharpe = [item['metrics'].get('sharpe_ratio', 0) for item in history]
        all_win_rates = [item['metrics'].get('win_rate', 0) for item in history]
        
        return {
            'expectancy_stats': {
                'mean': np.mean(all_expectancies),
                'std': np.std(all_expectancies),
                'min': np.min(all_expectancies),
                'max': np.max(all_expectancies)
            },
            'sharpe_stats': {
                'mean': np.mean(all_sharpe),
                'std': np.std(all_sharpe),
                'min': np.min(all_sharpe),
                'max': np.max(all_sharpe)
            },
            'win_rate_stats': {
                'mean': np.mean(all_win_rates),
                'std': np.std(all_win_rates),
                'min': np.min(all_win_rates),
                'max': np.max(all_win_rates)
            }
        }
    
    def _generate_performance_charts(self, strategy_id: str, 
                                   history: List[Dict]) -> List[str]:
        """Gera gráficos de performance"""
        if len(history) < 2:
            return []
        
        chart_paths = []
        
        try:
            # Prepara dados
            timestamps = pd.to_datetime([item['timestamp'] for item in history])
            expectancies = [item['metrics'].get('expectancy', 0) for item in history]
            sharpe_ratios = [item['metrics'].get('sharpe_ratio', 0) for item in history]
            win_rates = [item['metrics'].get('win_rate', 0) for item in history]
            
            # Gráfico 1: Evolução da expectância
            plt.figure(figsize=(12, 6))
            plt.plot(timestamps, expectancies, marker='o', linewidth=2)
            plt.title(f'Evolução da Expectância - {strategy_id}')
            plt.xlabel('Data')
            plt.ylabel('Expectância')
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            expectancy_path = f'/tmp/performance_expectancy_{strategy_id}.png'
            plt.savefig(expectancy_path, dpi=300, bbox_inches='tight')
            plt.close()
            chart_paths.append(expectancy_path)
            
            # Gráfico 2: Métricas combinadas
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            axes[0, 0].plot(timestamps, expectancies, 'b-', marker='o')
            axes[0, 0].set_title('Expectância')
            axes[0, 0].grid(True, alpha=0.3)
            
            axes[0, 1].plot(timestamps, sharpe_ratios, 'g-', marker='o')
            axes[0, 1].set_title('Sharpe Ratio')
            axes[0, 1].grid(True, alpha=0.3)
            
            axes[1, 0].plot(timestamps, win_rates, 'r-', marker='o')
            axes[1, 0].set_title('Win Rate')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Gráfico de correlação
            axes[1, 1].scatter(expectancies, sharpe_ratios, alpha=0.6)
            axes[1, 1].set_xlabel('Expectância')
            axes[1, 1].set_ylabel('Sharpe Ratio')
            axes[1, 1].set_title('Expectância vs Sharpe')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            combined_path = f'/tmp/performance_combined_{strategy_id}.png'
            plt.savefig(combined_path, dpi=300, bbox_inches='tight')
            plt.close()
            chart_paths.append(combined_path)
            
        except Exception as e:
            logger.warning(f"Erro ao gerar gráficos: {str(e)}")
        
        return chart_paths
    
    def _empty_metrics(self) -> Dict:
        """Retorna métricas vazias"""
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'total_return': 0,
            'avg_return': 0,
            'expectancy': 0,
            'profit_factor': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'volatility': 0,
            'positive_periods': 0,
            'avg_trade_duration': 0
        }

