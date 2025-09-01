"""
Sistema de Trading Quantitativo - Risk Monitor
Versão: 2.01
Descrição: Monitor de risco e circuit breaker para proteção do capital
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
import boto3
import json
from enum import Enum

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    """Níveis de risco do sistema"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class CircuitBreakerStatus(Enum):
    """Status do circuit breaker"""
    ACTIVE = "ACTIVE"
    TRIGGERED = "TRIGGERED"
    DISABLED = "DISABLED"

class RiskMonitor:
    """
    Classe responsável pelo monitoramento de risco em tempo real.
    Implementa circuit breakers e alertas para proteção do capital.
    """
    
    def __init__(self, config: Dict):
        """
        Inicializa o RiskMonitor.
        
        Args:
            config: Configurações do sistema
        """
        self.config = config
        self.s3_client = boto3.client('s3')
        self.sns_client = boto3.client('sns')
        self.dynamodb = boto3.resource('dynamodb')
        
        # Configurações de risco
        self.risk_config = config.get('risk_config', {
            'max_daily_drawdown': 0.05,  # 5% drawdown diário máximo
            'max_weekly_drawdown': 0.10,  # 10% drawdown semanal máximo
            'max_monthly_drawdown': 0.15,  # 15% drawdown mensal máximo
            'max_consecutive_losses': 5,
            'max_position_size': 0.02,  # 2% do capital por posição
            'volatility_threshold': 0.03,  # 3% volatilidade diária
            'correlation_threshold': 0.8,  # Correlação máxima entre posições
            'circuit_breaker_cooldown': 3600,  # 1 hora em segundos
            'emergency_stop_threshold': 0.08  # 8% drawdown para parada emergencial
        })
        
        # Estado do circuit breaker
        self.circuit_breaker_status = CircuitBreakerStatus.ACTIVE
        self.last_trigger_time = None
        
        # Tabela DynamoDB para estado do risco
        self.risk_state_table = config.get('risk_state_table', 'trading-system-risk-state')
        
    def monitor_portfolio_risk(self, portfolio_data: Dict, 
                             market_data: pd.DataFrame) -> Dict:
        """
        Monitora risco do portfólio em tempo real.
        
        Args:
            portfolio_data: Dados atuais do portfólio
            market_data: Dados de mercado recentes
            
        Returns:
            Dicionário com análise de risco
        """
        logger.info("Iniciando monitoramento de risco do portfólio")
        
        # Calcula métricas de risco atuais
        risk_metrics = self._calculate_current_risk_metrics(portfolio_data, market_data)
        
        # Avalia nível de risco
        risk_level = self._assess_risk_level(risk_metrics)
        
        # Verifica triggers de circuit breaker
        circuit_breaker_triggers = self._check_circuit_breaker_triggers(risk_metrics)
        
        # Gera alertas se necessário
        alerts = self._generate_risk_alerts(risk_metrics, risk_level)
        
        # Recomendações de ação
        recommendations = self._generate_risk_recommendations(risk_metrics, risk_level)
        
        # Atualiza estado no DynamoDB
        self._update_risk_state(risk_metrics, risk_level)
        
        risk_report = {
            'timestamp': datetime.now().isoformat(),
            'risk_level': risk_level.value,
            'circuit_breaker_status': self.circuit_breaker_status.value,
            'risk_metrics': risk_metrics,
            'circuit_breaker_triggers': circuit_breaker_triggers,
            'alerts': alerts,
            'recommendations': recommendations
        }
        
        logger.info(f"Monitoramento concluído. Nível de risco: {risk_level.value}")
        return risk_report
    
    def _calculate_current_risk_metrics(self, portfolio_data: Dict, 
                                      market_data: pd.DataFrame) -> Dict:
        """Calcula métricas de risco atuais"""
        logger.debug("Calculando métricas de risco")
        
        current_positions = portfolio_data.get('positions', [])
        account_balance = portfolio_data.get('account_balance', 0)
        equity_curve = portfolio_data.get('equity_curve', [])
        
        # Métricas de drawdown
        drawdown_metrics = self._calculate_drawdown_metrics(equity_curve)
        
        # Métricas de posição
        position_metrics = self._calculate_position_metrics(current_positions, account_balance)
        
        # Métricas de volatilidade
        volatility_metrics = self._calculate_volatility_metrics(market_data, equity_curve)
        
        # Métricas de correlação
        correlation_metrics = self._calculate_correlation_metrics(current_positions, market_data)
        
        # Métricas de liquidez
        liquidity_metrics = self._calculate_liquidity_metrics(current_positions, market_data)
        
        # Métricas de concentração
        concentration_metrics = self._calculate_concentration_metrics(current_positions)
        
        return {
            'drawdown': drawdown_metrics,
            'position': position_metrics,
            'volatility': volatility_metrics,
            'correlation': correlation_metrics,
            'liquidity': liquidity_metrics,
            'concentration': concentration_metrics,
            'timestamp': datetime.now().isoformat()
        }
    
    def _calculate_drawdown_metrics(self, equity_curve: List[float]) -> Dict:
        """Calcula métricas de drawdown"""
        if len(equity_curve) < 2:
            return {
                'current_drawdown': 0,
                'daily_drawdown': 0,
                'weekly_drawdown': 0,
                'monthly_drawdown': 0,
                'max_drawdown': 0,
                'consecutive_losses': 0
            }
        
        equity_series = pd.Series(equity_curve)
        
        # Drawdown atual
        peak = equity_series.expanding().max()
        current_drawdown = (equity_series.iloc[-1] - peak.iloc[-1]) / peak.iloc[-1]
        
        # Drawdowns por período
        daily_drawdown = self._calculate_period_drawdown(equity_series, days=1)
        weekly_drawdown = self._calculate_period_drawdown(equity_series, days=7)
        monthly_drawdown = self._calculate_period_drawdown(equity_series, days=30)
        
        # Máximo drawdown
        drawdown_series = (equity_series - peak) / peak
        max_drawdown = drawdown_series.min()
        
        # Perdas consecutivas
        returns = equity_series.pct_change().dropna()
        consecutive_losses = self._count_consecutive_losses(returns)
        
        return {
            'current_drawdown': float(current_drawdown),
            'daily_drawdown': float(daily_drawdown),
            'weekly_drawdown': float(weekly_drawdown),
            'monthly_drawdown': float(monthly_drawdown),
            'max_drawdown': float(max_drawdown),
            'consecutive_losses': consecutive_losses
        }
    
    def _calculate_period_drawdown(self, equity_series: pd.Series, days: int) -> float:
        """Calcula drawdown para um período específico"""
        if len(equity_series) < days:
            return 0
        
        period_start = equity_series.iloc[-days]
        period_end = equity_series.iloc[-1]
        period_peak = equity_series.iloc[-days:].max()
        
        return (period_end - period_peak) / period_peak if period_peak > 0 else 0
    
    def _count_consecutive_losses(self, returns: pd.Series) -> int:
        """Conta perdas consecutivas"""
        if len(returns) == 0:
            return 0
        
        consecutive = 0
        max_consecutive = 0
        
        for ret in returns:
            if ret < 0:
                consecutive += 1
                max_consecutive = max(max_consecutive, consecutive)
            else:
                consecutive = 0
        
        return max_consecutive
    
    def _calculate_position_metrics(self, positions: List[Dict], 
                                  account_balance: float) -> Dict:
        """Calcula métricas de posição"""
        if not positions or account_balance <= 0:
            return {
                'total_exposure': 0,
                'max_position_size': 0,
                'position_count': 0,
                'leverage': 0,
                'margin_usage': 0
            }
        
        total_exposure = 0
        max_position_size = 0
        
        for position in positions:
            position_value = abs(position.get('notional_value', 0))
            total_exposure += position_value
            max_position_size = max(max_position_size, position_value)
        
        leverage = total_exposure / account_balance if account_balance > 0 else 0
        margin_usage = sum(pos.get('margin_used', 0) for pos in positions)
        
        return {
            'total_exposure': float(total_exposure),
            'max_position_size': float(max_position_size / account_balance) if account_balance > 0 else 0,
            'position_count': len(positions),
            'leverage': float(leverage),
            'margin_usage': float(margin_usage / account_balance) if account_balance > 0 else 0
        }
    
    def _calculate_volatility_metrics(self, market_data: pd.DataFrame,
                                    equity_curve: List[float]) -> Dict:
        """Calcula métricas de volatilidade"""
        volatility_metrics = {}
        
        # Volatilidade do mercado
        if 'close' in market_data.columns and len(market_data) > 1:
            returns = market_data['close'].pct_change().dropna()
            market_volatility = returns.std()
            volatility_metrics['market_volatility'] = float(market_volatility)
            
            # Volatilidade realizada (últimos 20 períodos)
            if len(returns) >= 20:
                realized_vol = returns.tail(20).std()
                volatility_metrics['realized_volatility'] = float(realized_vol)
        
        # Volatilidade do portfólio
        if len(equity_curve) > 1:
            portfolio_returns = pd.Series(equity_curve).pct_change().dropna()
            portfolio_volatility = portfolio_returns.std()
            volatility_metrics['portfolio_volatility'] = float(portfolio_volatility)
        
        return volatility_metrics
    
    def _calculate_correlation_metrics(self, positions: List[Dict],
                                     market_data: pd.DataFrame) -> Dict:
        """Calcula métricas de correlação"""
        if len(positions) < 2:
            return {'max_correlation': 0, 'avg_correlation': 0}
        
        # Simplificado - em implementação real, calcularia correlação entre ativos
        # Por agora, retorna valores padrão
        return {
            'max_correlation': 0.5,  # Placeholder
            'avg_correlation': 0.3   # Placeholder
        }
    
    def _calculate_liquidity_metrics(self, positions: List[Dict],
                                   market_data: pd.DataFrame) -> Dict:
        """Calcula métricas de liquidez"""
        if not positions:
            return {'liquidity_score': 1.0, 'illiquid_positions': 0}
        
        # Simplificado - em implementação real, analisaria volume e spread
        return {
            'liquidity_score': 0.8,  # Placeholder
            'illiquid_positions': 0   # Placeholder
        }
    
    def _calculate_concentration_metrics(self, positions: List[Dict]) -> Dict:
        """Calcula métricas de concentração"""
        if not positions:
            return {'concentration_ratio': 0, 'herfindahl_index': 0}
        
        # Calcula concentração por ativo
        position_values = [abs(pos.get('notional_value', 0)) for pos in positions]
        total_value = sum(position_values)
        
        if total_value == 0:
            return {'concentration_ratio': 0, 'herfindahl_index': 0}
        
        # Concentração das top 3 posições
        sorted_values = sorted(position_values, reverse=True)
        top_3_concentration = sum(sorted_values[:3]) / total_value
        
        # Índice Herfindahl
        weights = [value / total_value for value in position_values]
        herfindahl_index = sum(w ** 2 for w in weights)
        
        return {
            'concentration_ratio': float(top_3_concentration),
            'herfindahl_index': float(herfindahl_index)
        }
    
    def _assess_risk_level(self, risk_metrics: Dict) -> RiskLevel:
        """Avalia nível de risco baseado nas métricas"""
        logger.debug("Avaliando nível de risco")
        
        risk_score = 0
        
        # Avalia drawdown
        drawdown = risk_metrics['drawdown']
        if abs(drawdown['daily_drawdown']) > self.risk_config['max_daily_drawdown']:
            risk_score += 3
        elif abs(drawdown['daily_drawdown']) > self.risk_config['max_daily_drawdown'] * 0.7:
            risk_score += 2
        elif abs(drawdown['daily_drawdown']) > self.risk_config['max_daily_drawdown'] * 0.5:
            risk_score += 1
        
        # Avalia posições
        position = risk_metrics['position']
        if position['max_position_size'] > self.risk_config['max_position_size']:
            risk_score += 2
        if position['leverage'] > 3:
            risk_score += 2
        elif position['leverage'] > 2:
            risk_score += 1
        
        # Avalia volatilidade
        volatility = risk_metrics['volatility']
        portfolio_vol = volatility.get('portfolio_volatility', 0)
        if portfolio_vol > self.risk_config['volatility_threshold']:
            risk_score += 2
        elif portfolio_vol > self.risk_config['volatility_threshold'] * 0.7:
            risk_score += 1
        
        # Avalia perdas consecutivas
        if drawdown['consecutive_losses'] >= self.risk_config['max_consecutive_losses']:
            risk_score += 3
        elif drawdown['consecutive_losses'] >= self.risk_config['max_consecutive_losses'] * 0.8:
            risk_score += 2
        
        # Determina nível de risco
        if risk_score >= 8:
            return RiskLevel.CRITICAL
        elif risk_score >= 5:
            return RiskLevel.HIGH
        elif risk_score >= 3:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def _check_circuit_breaker_triggers(self, risk_metrics: Dict) -> List[Dict]:
        """Verifica triggers do circuit breaker"""
        logger.debug("Verificando triggers do circuit breaker")
        
        triggers = []
        
        # Trigger 1: Drawdown diário excessivo
        daily_dd = abs(risk_metrics['drawdown']['daily_drawdown'])
        if daily_dd >= self.risk_config['max_daily_drawdown']:
            triggers.append({
                'type': 'daily_drawdown',
                'severity': 'HIGH',
                'value': daily_dd,
                'threshold': self.risk_config['max_daily_drawdown'],
                'action': 'CLOSE_ALL_POSITIONS'
            })
        
        # Trigger 2: Parada emergencial
        current_dd = abs(risk_metrics['drawdown']['current_drawdown'])
        if current_dd >= self.risk_config['emergency_stop_threshold']:
            triggers.append({
                'type': 'emergency_stop',
                'severity': 'CRITICAL',
                'value': current_dd,
                'threshold': self.risk_config['emergency_stop_threshold'],
                'action': 'EMERGENCY_STOP'
            })
        
        # Trigger 3: Perdas consecutivas
        consecutive_losses = risk_metrics['drawdown']['consecutive_losses']
        if consecutive_losses >= self.risk_config['max_consecutive_losses']:
            triggers.append({
                'type': 'consecutive_losses',
                'severity': 'HIGH',
                'value': consecutive_losses,
                'threshold': self.risk_config['max_consecutive_losses'],
                'action': 'PAUSE_TRADING'
            })
        
        # Trigger 4: Volatilidade excessiva
        portfolio_vol = risk_metrics['volatility'].get('portfolio_volatility', 0)
        if portfolio_vol > self.risk_config['volatility_threshold'] * 2:
            triggers.append({
                'type': 'high_volatility',
                'severity': 'MEDIUM',
                'value': portfolio_vol,
                'threshold': self.risk_config['volatility_threshold'] * 2,
                'action': 'REDUCE_POSITION_SIZE'
            })
        
        # Executa ações se há triggers críticos
        if triggers:
            self._execute_circuit_breaker_actions(triggers)
        
        return triggers
    
    def _execute_circuit_breaker_actions(self, triggers: List[Dict]):
        """Executa ações do circuit breaker"""
        logger.warning(f"Executando ações do circuit breaker: {len(triggers)} triggers")
        
        # Verifica se circuit breaker está em cooldown
        if self._is_in_cooldown():
            logger.info("Circuit breaker em cooldown, pulando ações")
            return
        
        critical_triggers = [t for t in triggers if t['severity'] == 'CRITICAL']
        high_triggers = [t for t in triggers if t['severity'] == 'HIGH']
        
        if critical_triggers:
            self._trigger_emergency_stop()
        elif high_triggers:
            self._trigger_high_risk_actions(high_triggers)
        
        # Atualiza timestamp do último trigger
        self.last_trigger_time = datetime.now()
        self.circuit_breaker_status = CircuitBreakerStatus.TRIGGERED
    
    def _is_in_cooldown(self) -> bool:
        """Verifica se circuit breaker está em cooldown"""
        if self.last_trigger_time is None:
            return False
        
        cooldown_period = timedelta(seconds=self.risk_config['circuit_breaker_cooldown'])
        return datetime.now() - self.last_trigger_time < cooldown_period
    
    def _trigger_emergency_stop(self):
        """Executa parada emergencial"""
        logger.critical("PARADA EMERGENCIAL ATIVADA")
        
        # Envia alerta crítico
        self._send_critical_alert("EMERGENCY STOP TRIGGERED", 
                                "Sistema ativou parada emergencial devido a risco crítico")
        
        # Atualiza status
        self.circuit_breaker_status = CircuitBreakerStatus.TRIGGERED
        
        # Em implementação real, fecharia todas as posições
        logger.critical("Todas as posições devem ser fechadas imediatamente")
    
    def _trigger_high_risk_actions(self, triggers: List[Dict]):
        """Executa ações para risco alto"""
        logger.warning("Ações de risco alto ativadas")
        
        for trigger in triggers:
            action = trigger['action']
            
            if action == 'CLOSE_ALL_POSITIONS':
                logger.warning("Fechando todas as posições devido a drawdown excessivo")
                self._send_alert("HIGH RISK", f"Fechando posições: {trigger['type']}")
                
            elif action == 'PAUSE_TRADING':
                logger.warning("Pausando trading devido a perdas consecutivas")
                self._send_alert("TRADING PAUSED", f"Trading pausado: {trigger['type']}")
                
            elif action == 'REDUCE_POSITION_SIZE':
                logger.warning("Reduzindo tamanho das posições devido a alta volatilidade")
                self._send_alert("POSITION SIZE REDUCED", f"Posições reduzidas: {trigger['type']}")
    
    def _generate_risk_alerts(self, risk_metrics: Dict, risk_level: RiskLevel) -> List[Dict]:
        """Gera alertas baseados no risco"""
        alerts = []
        
        # Alerta de nível de risco
        if risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            alerts.append({
                'type': 'risk_level',
                'severity': risk_level.value,
                'message': f"Nível de risco elevado: {risk_level.value}",
                'timestamp': datetime.now().isoformat()
            })
        
        # Alertas específicos
        drawdown = risk_metrics['drawdown']
        
        if abs(drawdown['daily_drawdown']) > self.risk_config['max_daily_drawdown'] * 0.8:
            alerts.append({
                'type': 'drawdown_warning',
                'severity': 'HIGH',
                'message': f"Drawdown diário próximo do limite: {drawdown['daily_drawdown']:.2%}",
                'timestamp': datetime.now().isoformat()
            })
        
        if drawdown['consecutive_losses'] >= self.risk_config['max_consecutive_losses'] * 0.8:
            alerts.append({
                'type': 'consecutive_losses',
                'severity': 'MEDIUM',
                'message': f"Perdas consecutivas: {drawdown['consecutive_losses']}",
                'timestamp': datetime.now().isoformat()
            })
        
        return alerts
    
    def _generate_risk_recommendations(self, risk_metrics: Dict, 
                                     risk_level: RiskLevel) -> List[str]:
        """Gera recomendações baseadas no risco"""
        recommendations = []
        
        if risk_level == RiskLevel.CRITICAL:
            recommendations.extend([
                "Parar trading imediatamente",
                "Fechar todas as posições",
                "Revisar estratégia de gestão de risco",
                "Aguardar condições de mercado mais favoráveis"
            ])
        
        elif risk_level == RiskLevel.HIGH:
            recommendations.extend([
                "Reduzir tamanho das posições",
                "Aumentar stops de proteção",
                "Monitorar mercado mais frequentemente",
                "Considerar fechar posições mais arriscadas"
            ])
        
        elif risk_level == RiskLevel.MEDIUM:
            recommendations.extend([
                "Manter vigilância aumentada",
                "Revisar correlações entre posições",
                "Verificar condições de mercado",
                "Preparar plano de contingência"
            ])
        
        # Recomendações específicas
        position = risk_metrics['position']
        if position['max_position_size'] > self.risk_config['max_position_size'] * 0.8:
            recommendations.append("Reduzir tamanho máximo das posições")
        
        if position['leverage'] > 2:
            recommendations.append("Reduzir alavancagem")
        
        drawdown = risk_metrics['drawdown']
        if drawdown['consecutive_losses'] >= 3:
            recommendations.append("Revisar sinais de entrada")
        
        return recommendations
    
    def _update_risk_state(self, risk_metrics: Dict, risk_level: RiskLevel):
        """Atualiza estado do risco no DynamoDB"""
        try:
            table = self.dynamodb.Table(self.risk_state_table)
            
            risk_state = {
                'timestamp': datetime.now().isoformat(),
                'risk_level': risk_level.value,
                'circuit_breaker_status': self.circuit_breaker_status.value,
                'risk_metrics': risk_metrics,
                'last_update': datetime.now().isoformat()
            }
            
            table.put_item(
                Item={
                    'state_id': 'current_risk_state',
                    **risk_state
                }
            )
            
            logger.debug("Estado de risco atualizado no DynamoDB")
            
        except Exception as e:
            logger.error(f"Erro ao atualizar estado de risco: {str(e)}")
    
    def _send_alert(self, subject: str, message: str):
        """Envia alerta via SNS"""
        try:
            topic_arn = self.config.get('sns_topic_arn')
            if topic_arn:
                self.sns_client.publish(
                    TopicArn=topic_arn,
                    Subject=f"Trading System Alert: {subject}",
                    Message=message
                )
                logger.info(f"Alerta enviado: {subject}")
            else:
                logger.warning("SNS topic não configurado, alerta não enviado")
                
        except Exception as e:
            logger.error(f"Erro ao enviar alerta: {str(e)}")
    
    def _send_critical_alert(self, subject: str, message: str):
        """Envia alerta crítico"""
        logger.critical(f"ALERTA CRÍTICO: {subject} - {message}")
        self._send_alert(f"CRITICAL: {subject}", message)
    
    def reset_circuit_breaker(self) -> bool:
        """
        Reseta o circuit breaker manualmente.
        
        Returns:
            True se resetado com sucesso
        """
        logger.info("Resetando circuit breaker manualmente")
        
        self.circuit_breaker_status = CircuitBreakerStatus.ACTIVE
        self.last_trigger_time = None
        
        # Atualiza estado
        try:
            table = self.dynamodb.Table(self.risk_state_table)
            table.update_item(
                Key={'state_id': 'current_risk_state'},
                UpdateExpression='SET circuit_breaker_status = :status, last_update = :timestamp',
                ExpressionAttributeValues={
                    ':status': CircuitBreakerStatus.ACTIVE.value,
                    ':timestamp': datetime.now().isoformat()
                }
            )
            
            logger.info("Circuit breaker resetado com sucesso")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao resetar circuit breaker: {str(e)}")
            return False
    
    def get_risk_status(self) -> Dict:
        """
        Obtém status atual do risco.
        
        Returns:
            Dicionário com status do risco
        """
        try:
            table = self.dynamodb.Table(self.risk_state_table)
            response = table.get_item(Key={'state_id': 'current_risk_state'})
            
            if 'Item' in response:
                return response['Item']
            else:
                return {
                    'risk_level': RiskLevel.LOW.value,
                    'circuit_breaker_status': self.circuit_breaker_status.value,
                    'last_update': datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Erro ao obter status de risco: {str(e)}")
            return {
                'error': str(e),
                'circuit_breaker_status': self.circuit_breaker_status.value
            }

