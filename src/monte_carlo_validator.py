"""
Sistema de Trading Quantitativo - Monte Carlo Validator
Versão: 2.01
Descrição: Validação de robustez com simulação de Monte Carlo e sinais aleatórios
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime
import boto3
import json
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

class MonteCarloValidator:
    """
    Classe responsável pela validação de robustez usando Monte Carlo.
    Testa se a estratégia mantém expectância positiva mesmo com sinais aleatórios,
    validando o princípio de Van Tharp de que a lucratividade emerge do gerenciamento de risco.
    """
    
    def __init__(self, config: Dict):
        """
        Inicializa o MonteCarloValidator.
        
        Args:
            config: Configurações do sistema
        """
        self.config = config
        self.s3_client = boto3.client('s3')
        self.bucket_name = config.get('data_bucket_name')
        
        # Configurações do Monte Carlo
        self.mc_config = config.get('monte_carlo_config', {
            'n_simulations': 1000,
            'expectancy_threshold': 0.0,
            'confidence_level': 0.95,
            'min_trades': 50,
            'parallel_workers': 4
        })
        
    def validate_strategy_robustness(self, backtest_results: Dict, 
                                   market_data: pd.DataFrame,
                                   strategy_params: Dict) -> Dict:
        """
        Valida robustez da estratégia usando Monte Carlo com sinais aleatórios.
        
        Args:
            backtest_results: Resultados do backtest original
            market_data: Dados de mercado para simulação
            strategy_params: Parâmetros da estratégia (para gestão de risco)
            
        Returns:
            Dicionário com resultados da validação Monte Carlo
        """
        logger.info("Iniciando validação Monte Carlo com sinais aleatórios")
        
        # Extrai distribuição de sinais do backtest original
        original_signals = backtest_results.get('signals', [])
        signal_distribution = self._calculate_signal_distribution(original_signals)
        
        logger.info(f"Distribuição de sinais original: {signal_distribution}")
        
        # Prepara dados para simulação
        simulation_data = self._prepare_simulation_data(market_data, strategy_params)
        
        # Executa simulações Monte Carlo
        simulation_results = self._run_monte_carlo_simulations(
            simulation_data, signal_distribution, strategy_params
        )
        
        # Analisa resultados
        validation_results = self._analyze_monte_carlo_results(
            simulation_results, backtest_results
        )
        
        # Gera relatório
        validation_results['report'] = self._generate_monte_carlo_report(
            simulation_results, validation_results
        )
        
        logger.info(f"Validação Monte Carlo concluída. Aprovada: {validation_results['validation_passed']}")
        return validation_results
    
    def _calculate_signal_distribution(self, signals: List[int]) -> Dict:
        """
        Calcula distribuição dos sinais do backtest original.
        
        Args:
            signals: Lista de sinais (-1, 0, 1)
            
        Returns:
            Dicionário com probabilidades de cada sinal
        """
        if not signals:
            # Distribuição padrão se não há sinais
            return {'buy_prob': 0.33, 'sell_prob': 0.33, 'neutral_prob': 0.34}
        
        signals_array = np.array(signals)
        total_signals = len(signals_array)
        
        buy_count = np.sum(signals_array == 1)
        sell_count = np.sum(signals_array == -1)
        neutral_count = np.sum(signals_array == 0)
        
        return {
            'buy_prob': buy_count / total_signals,
            'sell_prob': sell_count / total_signals,
            'neutral_prob': neutral_count / total_signals,
            'total_signals': total_signals
        }
    
    def _prepare_simulation_data(self, market_data: pd.DataFrame, 
                               strategy_params: Dict) -> pd.DataFrame:
        """
        Prepara dados para simulação Monte Carlo.
        
        Args:
            market_data: Dados de mercado
            strategy_params: Parâmetros da estratégia
            
        Returns:
            DataFrame preparado para simulação
        """
        logger.debug("Preparando dados para simulação")
        
        simulation_data = market_data.copy()
        
        # Calcula ATR se não estiver presente
        if 'atr' not in simulation_data.columns:
            from feature_engineering import FeatureEngineer
            feature_engineer = FeatureEngineer(strategy_params)
            simulation_data = feature_engineer.generate_all_features(simulation_data)
        
        # Calcula retornos
        simulation_data['returns'] = simulation_data['close'].pct_change()
        
        # Calcula volatilidade para gestão de risco
        atr_multiplier = strategy_params.get('atr_multiplier', 1.0)
        simulation_data['stop_distance'] = simulation_data['atr'] * atr_multiplier
        simulation_data['stop_distance_pct'] = simulation_data['stop_distance'] / simulation_data['close']
        
        # Remove linhas com dados insuficientes
        simulation_data = simulation_data.dropna()
        
        logger.debug(f"Dados preparados: {len(simulation_data)} períodos")
        return simulation_data
    
    def _run_monte_carlo_simulations(self, simulation_data: pd.DataFrame,
                                   signal_distribution: Dict,
                                   strategy_params: Dict) -> List[Dict]:
        """
        Executa simulações Monte Carlo em paralelo.
        
        Args:
            simulation_data: Dados preparados
            signal_distribution: Distribuição de sinais
            strategy_params: Parâmetros da estratégia
            
        Returns:
            Lista com resultados de cada simulação
        """
        logger.info(f"Executando {self.mc_config['n_simulations']} simulações Monte Carlo")
        
        n_simulations = self.mc_config['n_simulations']
        n_workers = self.mc_config['parallel_workers']
        
        # Divide simulações entre workers
        simulations_per_worker = n_simulations // n_workers
        remaining_simulations = n_simulations % n_workers
        
        simulation_batches = []
        for i in range(n_workers):
            batch_size = simulations_per_worker + (1 if i < remaining_simulations else 0)
            simulation_batches.append(batch_size)
        
        # Executa simulações em paralelo
        all_results = []
        
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = []
            
            for batch_size in simulation_batches:
                future = executor.submit(
                    self._run_simulation_batch,
                    simulation_data, signal_distribution, strategy_params, batch_size
                )
                futures.append(future)
            
            # Coleta resultados
            for future in as_completed(futures):
                batch_results = future.result()
                all_results.extend(batch_results)
        
        logger.info(f"Simulações concluídas: {len(all_results)} resultados")
        return all_results
    
    def _run_simulation_batch(self, simulation_data: pd.DataFrame,
                            signal_distribution: Dict,
                            strategy_params: Dict,
                            batch_size: int) -> List[Dict]:
        """
        Executa um lote de simulações.
        
        Args:
            simulation_data: Dados para simulação
            signal_distribution: Distribuição de sinais
            strategy_params: Parâmetros da estratégia
            batch_size: Número de simulações no lote
            
        Returns:
            Lista com resultados do lote
        """
        batch_results = []
        
        for sim_id in range(batch_size):
            try:
                # Gera sinais aleatórios
                random_signals = self._generate_random_signals(
                    len(simulation_data), signal_distribution
                )
                
                # Simula trading com sinais aleatórios
                simulation_result = self._simulate_trading(
                    simulation_data, random_signals, strategy_params
                )
                
                simulation_result['simulation_id'] = sim_id
                batch_results.append(simulation_result)
                
            except Exception as e:
                logger.warning(f"Erro na simulação {sim_id}: {str(e)}")
                continue
        
        return batch_results
    
    def _generate_random_signals(self, n_periods: int, 
                               signal_distribution: Dict) -> np.ndarray:
        """
        Gera sinais aleatórios baseados na distribuição original.
        
        Args:
            n_periods: Número de períodos
            signal_distribution: Distribuição de probabilidades
            
        Returns:
            Array com sinais aleatórios
        """
        # Probabilidades para cada classe
        probs = [
            signal_distribution['sell_prob'],    # -1
            signal_distribution['neutral_prob'], #  0
            signal_distribution['buy_prob']      #  1
        ]
        
        # Gera sinais aleatórios
        random_signals = np.random.choice([-1, 0, 1], size=n_periods, p=probs)
        
        return random_signals
    
    def _simulate_trading(self, data: pd.DataFrame, signals: np.ndarray,
                         strategy_params: Dict) -> Dict:
        """
        Simula trading com sinais dados.
        
        Args:
            data: Dados de mercado
            signals: Sinais de trading
            strategy_params: Parâmetros da estratégia
            
        Returns:
            Dicionário com resultados da simulação
        """
        trades = []
        current_position = 0
        entry_price = 0
        entry_stop = 0
        
        for i, (idx, row) in enumerate(data.iterrows()):
            if i >= len(signals):
                break
                
            signal = signals[i]
            current_price = row['close']
            atr = row['atr']
            stop_distance = row['stop_distance']
            
            # Fecha posição existente se necessário
            if current_position != 0:
                exit_trade = False
                exit_reason = ""
                exit_price = current_price
                
                # Verifica stop loss
                if current_position == 1:  # Long position
                    if current_price <= entry_stop:
                        exit_trade = True
                        exit_reason = "stop_loss"
                        exit_price = entry_stop
                elif current_position == -1:  # Short position
                    if current_price >= entry_stop:
                        exit_trade = True
                        exit_reason = "stop_loss"
                        exit_price = entry_stop
                
                # Verifica mudança de sinal
                if signal != current_position and signal != 0:
                    exit_trade = True
                    exit_reason = "signal_change"
                
                # Fecha posição neutra
                if signal == 0 and current_position != 0:
                    exit_trade = True
                    exit_reason = "neutral_signal"
                
                # Executa saída se necessário
                if exit_trade:
                    pnl = self._calculate_trade_pnl(
                        entry_price, exit_price, current_position, stop_distance
                    )
                    
                    trades.append({
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'position': current_position,
                        'pnl_r': pnl,  # PnL em múltiplos de R
                        'exit_reason': exit_reason
                    })
                    
                    current_position = 0
            
            # Abre nova posição se sinal não é neutro
            if signal != 0 and current_position == 0:
                current_position = signal
                entry_price = current_price
                
                # Define stop loss
                if signal == 1:  # Long
                    entry_stop = entry_price - stop_distance
                else:  # Short
                    entry_stop = entry_price + stop_distance
        
        # Calcula métricas da simulação
        if trades:
            pnls = [trade['pnl_r'] for trade in trades]
            
            metrics = {
                'total_trades': len(trades),
                'winning_trades': len([pnl for pnl in pnls if pnl > 0]),
                'losing_trades': len([pnl for pnl in pnls if pnl < 0]),
                'win_rate': len([pnl for pnl in pnls if pnl > 0]) / len(pnls),
                'expectancy': np.mean(pnls),
                'total_return': np.sum(pnls),
                'max_drawdown': self._calculate_max_drawdown(pnls),
                'profit_factor': self._calculate_profit_factor(pnls),
                'sharpe_ratio': np.mean(pnls) / np.std(pnls) if np.std(pnls) > 0 else 0
            }
        else:
            metrics = {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'expectancy': 0,
                'total_return': 0,
                'max_drawdown': 0,
                'profit_factor': 0,
                'sharpe_ratio': 0
            }
        
        return {
            'trades': trades,
            'metrics': metrics
        }
    
    def _calculate_trade_pnl(self, entry_price: float, exit_price: float,
                           position: int, stop_distance: float) -> float:
        """
        Calcula PnL do trade em múltiplos de R.
        
        Args:
            entry_price: Preço de entrada
            exit_price: Preço de saída
            position: Posição (1 para long, -1 para short)
            stop_distance: Distância do stop em preço
            
        Returns:
            PnL em múltiplos de R
        """
        if position == 1:  # Long
            pnl_price = exit_price - entry_price
        else:  # Short
            pnl_price = entry_price - exit_price
        
        # Converte para múltiplos de R
        r_multiple = pnl_price / stop_distance if stop_distance > 0 else 0
        
        return r_multiple
    
    def _calculate_max_drawdown(self, pnls: List[float]) -> float:
        """Calcula máximo drawdown"""
        if not pnls:
            return 0
        
        cumulative = np.cumsum(pnls)
        peak = np.maximum.accumulate(cumulative)
        drawdown = cumulative - peak
        
        return np.min(drawdown)
    
    def _calculate_profit_factor(self, pnls: List[float]) -> float:
        """Calcula profit factor"""
        if not pnls:
            return 0
        
        gross_profit = sum([pnl for pnl in pnls if pnl > 0])
        gross_loss = abs(sum([pnl for pnl in pnls if pnl < 0]))
        
        return gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    def _analyze_monte_carlo_results(self, simulation_results: List[Dict],
                                   backtest_results: Dict) -> Dict:
        """
        Analisa resultados das simulações Monte Carlo.
        
        Args:
            simulation_results: Resultados das simulações
            backtest_results: Resultados do backtest original
            
        Returns:
            Dicionário com análise dos resultados
        """
        logger.info("Analisando resultados Monte Carlo")
        
        # Extrai métricas de todas as simulações
        expectancies = [sim['metrics']['expectancy'] for sim in simulation_results 
                       if sim['metrics']['total_trades'] >= self.mc_config['min_trades']]
        
        if not expectancies:
            logger.warning("Nenhuma simulação com trades suficientes")
            return {
                'validation_passed': False,
                'reason': 'Insufficient trades in simulations',
                'statistics': {}
            }
        
        # Calcula estatísticas
        expectancy_mean = np.mean(expectancies)
        expectancy_std = np.std(expectancies)
        positive_expectancy_ratio = np.mean(np.array(expectancies) > self.mc_config['expectancy_threshold'])
        
        # Teste t para significância estatística
        t_stat, p_value = stats.ttest_1samp(expectancies, self.mc_config['expectancy_threshold'])
        
        # Critérios de aprovação
        criteria_met = {
            'positive_expectancy_ratio': positive_expectancy_ratio >= 0.95,
            'mean_expectancy_positive': expectancy_mean > self.mc_config['expectancy_threshold'],
            'statistical_significance': p_value < (1 - self.mc_config['confidence_level'])
        }
        
        validation_passed = all(criteria_met.values())
        
        # Compara com backtest original
        original_expectancy = backtest_results.get('expectancy', 0)
        
        analysis_results = {
            'validation_passed': validation_passed,
            'criteria_met': criteria_met,
            'statistics': {
                'n_simulations': len(simulation_results),
                'n_valid_simulations': len(expectancies),
                'expectancy_mean': expectancy_mean,
                'expectancy_std': expectancy_std,
                'expectancy_min': np.min(expectancies),
                'expectancy_max': np.max(expectancies),
                'positive_expectancy_ratio': positive_expectancy_ratio,
                't_statistic': t_stat,
                'p_value': p_value,
                'confidence_interval': stats.t.interval(
                    self.mc_config['confidence_level'],
                    len(expectancies) - 1,
                    loc=expectancy_mean,
                    scale=expectancy_std / np.sqrt(len(expectancies))
                )
            },
            'comparison_with_original': {
                'original_expectancy': original_expectancy,
                'monte_carlo_mean': expectancy_mean,
                'difference': original_expectancy - expectancy_mean,
                'original_better_than_random': original_expectancy > expectancy_mean
            }
        }
        
        return analysis_results
    
    def _generate_monte_carlo_report(self, simulation_results: List[Dict],
                                   analysis_results: Dict) -> Dict:
        """
        Gera relatório detalhado da validação Monte Carlo.
        
        Args:
            simulation_results: Resultados das simulações
            analysis_results: Análise dos resultados
            
        Returns:
            Dicionário com relatório
        """
        logger.info("Gerando relatório Monte Carlo")
        
        # Extrai métricas para análise
        all_metrics = []
        for sim in simulation_results:
            if sim['metrics']['total_trades'] >= self.mc_config['min_trades']:
                all_metrics.append(sim['metrics'])
        
        if not all_metrics:
            return {'error': 'No valid simulations for report generation'}
        
        # Estatísticas agregadas
        metric_names = ['expectancy', 'win_rate', 'total_return', 'max_drawdown', 
                       'profit_factor', 'sharpe_ratio']
        
        aggregated_stats = {}
        for metric in metric_names:
            values = [m[metric] for m in all_metrics]
            aggregated_stats[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values),
                'percentile_5': np.percentile(values, 5),
                'percentile_95': np.percentile(values, 95)
            }
        
        # Distribuição de resultados
        expectancies = [m['expectancy'] for m in all_metrics]
        distribution_analysis = {
            'positive_expectancy_count': len([e for e in expectancies if e > 0]),
            'negative_expectancy_count': len([e for e in expectancies if e < 0]),
            'zero_expectancy_count': len([e for e in expectancies if e == 0]),
            'expectancy_histogram': np.histogram(expectancies, bins=20)[0].tolist()
        }
        
        report = {
            'summary': {
                'validation_passed': analysis_results['validation_passed'],
                'total_simulations': len(simulation_results),
                'valid_simulations': len(all_metrics),
                'positive_expectancy_ratio': analysis_results['statistics']['positive_expectancy_ratio']
            },
            'statistical_analysis': analysis_results['statistics'],
            'aggregated_metrics': aggregated_stats,
            'distribution_analysis': distribution_analysis,
            'criteria_evaluation': analysis_results['criteria_met'],
            'comparison_with_original': analysis_results['comparison_with_original']
        }
        
        return report
    
    def save_monte_carlo_results(self, validation_results: Dict,
                               symbol: str, timeframe: str) -> str:
        """
        Salva resultados da validação Monte Carlo no S3.
        
        Args:
            validation_results: Resultados da validação
            symbol: Par de trading
            timeframe: Timeframe
            
        Returns:
            Path do arquivo salvo
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"monte_carlo_validation_{symbol}_{timeframe}_{timestamp}.json"
        s3_key = f"monte_carlo_results/{filename}"
        
        try:
            # Converte para JSON
            results_json = json.dumps(validation_results, indent=2, default=str)
            
            # Upload para S3
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=results_json,
                ContentType='application/json'
            )
            
            s3_path = f"s3://{self.bucket_name}/{s3_key}"
            logger.info(f"Resultados Monte Carlo salvos em: {s3_path}")
            return s3_path
            
        except Exception as e:
            logger.error(f"Erro ao salvar resultados Monte Carlo: {str(e)}")
            raise
    
    def generate_monte_carlo_plots(self, simulation_results: List[Dict],
                                 analysis_results: Dict) -> List[str]:
        """
        Gera gráficos dos resultados Monte Carlo.
        
        Args:
            simulation_results: Resultados das simulações
            analysis_results: Análise dos resultados
            
        Returns:
            Lista de paths dos gráficos salvos
        """
        logger.info("Gerando gráficos Monte Carlo")
        
        # Extrai expectâncias válidas
        expectancies = [sim['metrics']['expectancy'] for sim in simulation_results 
                       if sim['metrics']['total_trades'] >= self.mc_config['min_trades']]
        
        if not expectancies:
            logger.warning("Nenhuma simulação válida para gráficos")
            return []
        
        plot_paths = []
        
        # Gráfico 1: Histograma de expectâncias
        plt.figure(figsize=(10, 6))
        plt.hist(expectancies, bins=30, alpha=0.7, edgecolor='black')
        plt.axvline(x=0, color='red', linestyle='--', label='Expectância Zero')
        plt.axvline(x=np.mean(expectancies), color='green', linestyle='-', label='Média')
        plt.xlabel('Expectância (R-múltiplos)')
        plt.ylabel('Frequência')
        plt.title('Distribuição de Expectâncias - Monte Carlo')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        hist_path = '/tmp/monte_carlo_expectancy_histogram.png'
        plt.savefig(hist_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_paths.append(hist_path)
        
        # Gráfico 2: Curvas de capital das simulações
        plt.figure(figsize=(12, 8))
        
        # Plota algumas curvas de capital (máximo 50 para não sobrecarregar)
        sample_size = min(50, len(simulation_results))
        sample_indices = np.random.choice(len(simulation_results), sample_size, replace=False)
        
        for i, idx in enumerate(sample_indices):
            sim = simulation_results[idx]
            if sim['metrics']['total_trades'] >= self.mc_config['min_trades']:
                pnls = [trade['pnl_r'] for trade in sim['trades']]
                cumulative_pnl = np.cumsum(pnls)
                plt.plot(cumulative_pnl, alpha=0.3, color='blue', linewidth=0.5)
        
        plt.xlabel('Número de Trades')
        plt.ylabel('PnL Cumulativo (R-múltiplos)')
        plt.title('Curvas de Capital - Simulações Monte Carlo')
        plt.grid(True, alpha=0.3)
        
        curves_path = '/tmp/monte_carlo_equity_curves.png'
        plt.savefig(curves_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_paths.append(curves_path)
        
        return plot_paths

