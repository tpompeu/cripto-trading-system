"""
Sistema de Trading Quantitativo - Alert Manager
Versão: 2.01
Descrição: Gerenciamento de alertas e notificações do sistema
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
import boto3
import json
from enum import Enum
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Níveis de severidade dos alertas"""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class AlertChannel(Enum):
    """Canais de notificação"""
    EMAIL = "EMAIL"
    SNS = "SNS"
    SLACK = "SLACK"
    CLOUDWATCH = "CLOUDWATCH"

class AlertManager:
    """
    Classe responsável pelo gerenciamento de alertas e notificações.
    Centraliza o envio de alertas através de múltiplos canais.
    """
    
    def __init__(self, config: Dict):
        """
        Inicializa o AlertManager.
        
        Args:
            config: Configurações do sistema
        """
        self.config = config
        self.sns_client = boto3.client('sns')
        self.ses_client = boto3.client('ses')
        self.cloudwatch = boto3.client('cloudwatch')
        self.dynamodb = boto3.resource('dynamodb')
        
        # Configurações de alertas
        self.alert_config = config.get('alert_config', {
            'default_channels': [AlertChannel.EMAIL.value, AlertChannel.SNS.value],
            'rate_limit_minutes': 15,  # Limite de rate para evitar spam
            'escalation_threshold_minutes': 60,  # Tempo para escalação
            'max_retries': 3,
            'retry_delay_seconds': 30
        })
        
        # Configurações de canais
        self.channel_config = config.get('channel_config', {
            'email': {
                'from_address': config.get('alert_email_from'),
                'to_addresses': config.get('alert_email_to', []),
                'smtp_server': config.get('smtp_server'),
                'smtp_port': config.get('smtp_port', 587),
                'smtp_username': config.get('smtp_username'),
                'smtp_password': config.get('smtp_password')
            },
            'sns': {
                'topic_arn': config.get('sns_topic_arn')
            },
            'slack': {
                'webhook_url': config.get('slack_webhook_url'),
                'channel': config.get('slack_channel', '#trading-alerts')
            }
        })
        
        # Tabela DynamoDB para histórico de alertas
        self.alerts_table = config.get('alerts_table', 'trading-system-alerts')
        
        # Cache de alertas recentes para rate limiting
        self.recent_alerts = {}
        
    def send_alert(self, alert_type: str, message: str, 
                   severity: AlertSeverity = AlertSeverity.INFO,
                   channels: Optional[List[AlertChannel]] = None,
                   metadata: Optional[Dict] = None) -> bool:
        """
        Envia um alerta através dos canais especificados.
        
        Args:
            alert_type: Tipo do alerta
            message: Mensagem do alerta
            severity: Severidade do alerta
            channels: Canais para envio (usa padrão se None)
            metadata: Metadados adicionais
            
        Returns:
            True se enviado com sucesso
        """
        logger.info(f"Enviando alerta: {alert_type} - {severity.value}")
        
        # Verifica rate limiting
        if self._is_rate_limited(alert_type, severity):
            logger.warning(f"Alerta {alert_type} limitado por rate limiting")
            return False
        
        # Usa canais padrão se não especificado
        if channels is None:
            channels = [AlertChannel(ch) for ch in self.alert_config['default_channels']]
        
        # Cria objeto do alerta
        alert = {
            'alert_id': self._generate_alert_id(),
            'alert_type': alert_type,
            'message': message,
            'severity': severity.value,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {},
            'channels': [ch.value for ch in channels]
        }
        
        # Salva no histórico
        self._save_alert_history(alert)
        
        # Envia através dos canais
        success_count = 0
        for channel in channels:
            try:
                if self._send_to_channel(alert, channel):
                    success_count += 1
                else:
                    logger.warning(f"Falha ao enviar para canal {channel.value}")
            except Exception as e:
                logger.error(f"Erro ao enviar para {channel.value}: {str(e)}")
        
        # Atualiza cache de rate limiting
        self._update_rate_limit_cache(alert_type, severity)
        
        # Considera sucesso se pelo menos um canal funcionou
        success = success_count > 0
        
        if success:
            logger.info(f"Alerta enviado com sucesso para {success_count}/{len(channels)} canais")
        else:
            logger.error("Falha ao enviar alerta para todos os canais")
        
        return success
    
    def send_risk_alert(self, risk_level: str, risk_metrics: Dict, 
                       triggers: List[Dict]) -> bool:
        """
        Envia alerta específico de risco.
        
        Args:
            risk_level: Nível de risco
            risk_metrics: Métricas de risco
            triggers: Triggers que causaram o alerta
            
        Returns:
            True se enviado com sucesso
        """
        severity = self._determine_risk_severity(risk_level)
        
        # Constrói mensagem
        message = f"ALERTA DE RISCO - Nível: {risk_level}\n\n"
        
        # Adiciona métricas principais
        if 'drawdown' in risk_metrics:
            dd = risk_metrics['drawdown']
            message += f"Drawdown atual: {dd.get('current_drawdown', 0):.2%}\n"
            message += f"Drawdown diário: {dd.get('daily_drawdown', 0):.2%}\n"
        
        # Adiciona triggers
        if triggers:
            message += f"\nTriggers ativados ({len(triggers)}):\n"
            for trigger in triggers:
                message += f"- {trigger.get('type', 'Unknown')}: {trigger.get('action', 'N/A')}\n"
        
        return self.send_alert(
            alert_type='risk_alert',
            message=message,
            severity=severity,
            metadata={
                'risk_level': risk_level,
                'risk_metrics': risk_metrics,
                'triggers': triggers
            }
        )
    
    def send_performance_alert(self, strategy_id: str, performance_status: str,
                             degradation_analysis: Dict) -> bool:
        """
        Envia alerta de performance.
        
        Args:
            strategy_id: ID da estratégia
            performance_status: Status de performance
            degradation_analysis: Análise de degradação
            
        Returns:
            True se enviado com sucesso
        """
        severity = self._determine_performance_severity(performance_status)
        
        message = f"ALERTA DE PERFORMANCE - Estratégia: {strategy_id}\n"
        message += f"Status: {performance_status}\n\n"
        
        if degradation_analysis.get('degradation_detected', False):
            message += "DEGRADAÇÃO DETECTADA:\n"
            for signal in degradation_analysis.get('degradation_signals', []):
                metric = signal.get('metric', 'Unknown')
                degradation = signal.get('degradation', 0)
                message += f"- {metric}: {degradation:.1%} de degradação\n"
        
        return self.send_alert(
            alert_type='performance_alert',
            message=message,
            severity=severity,
            metadata={
                'strategy_id': strategy_id,
                'performance_status': performance_status,
                'degradation_analysis': degradation_analysis
            }
        )
    
    def send_system_alert(self, component: str, status: str, 
                         error_details: Optional[str] = None) -> bool:
        """
        Envia alerta de sistema.
        
        Args:
            component: Componente do sistema
            status: Status do componente
            error_details: Detalhes do erro (opcional)
            
        Returns:
            True se enviado com sucesso
        """
        severity = AlertSeverity.ERROR if status == 'ERROR' else AlertSeverity.WARNING
        
        message = f"ALERTA DE SISTEMA - Componente: {component}\n"
        message += f"Status: {status}\n"
        
        if error_details:
            message += f"\nDetalhes do erro:\n{error_details}"
        
        return self.send_alert(
            alert_type='system_alert',
            message=message,
            severity=severity,
            metadata={
                'component': component,
                'status': status,
                'error_details': error_details
            }
        )
    
    def send_trade_alert(self, trade_info: Dict, alert_reason: str) -> bool:
        """
        Envia alerta de trade.
        
        Args:
            trade_info: Informações do trade
            alert_reason: Razão do alerta
            
        Returns:
            True se enviado com sucesso
        """
        message = f"ALERTA DE TRADE - {alert_reason}\n\n"
        message += f"Símbolo: {trade_info.get('symbol', 'N/A')}\n"
        message += f"Tipo: {trade_info.get('side', 'N/A')}\n"
        message += f"Quantidade: {trade_info.get('quantity', 'N/A')}\n"
        message += f"Preço: {trade_info.get('price', 'N/A')}\n"
        
        if 'pnl' in trade_info:
            message += f"PnL: {trade_info['pnl']:.2f}R\n"
        
        return self.send_alert(
            alert_type='trade_alert',
            message=message,
            severity=AlertSeverity.INFO,
            metadata={
                'trade_info': trade_info,
                'alert_reason': alert_reason
            }
        )
    
    def _send_to_channel(self, alert: Dict, channel: AlertChannel) -> bool:
        """Envia alerta para um canal específico"""
        try:
            if channel == AlertChannel.EMAIL:
                return self._send_email(alert)
            elif channel == AlertChannel.SNS:
                return self._send_sns(alert)
            elif channel == AlertChannel.SLACK:
                return self._send_slack(alert)
            elif channel == AlertChannel.CLOUDWATCH:
                return self._send_cloudwatch(alert)
            else:
                logger.warning(f"Canal não suportado: {channel.value}")
                return False
                
        except Exception as e:
            logger.error(f"Erro ao enviar para {channel.value}: {str(e)}")
            return False
    
    def _send_email(self, alert: Dict) -> bool:
        """Envia alerta por email"""
        email_config = self.channel_config.get('email', {})
        
        if not email_config.get('to_addresses'):
            logger.warning("Endereços de email não configurados")
            return False
        
        try:
            # Usa SES se disponível, senão SMTP
            if self._is_ses_configured():
                return self._send_ses_email(alert, email_config)
            else:
                return self._send_smtp_email(alert, email_config)
                
        except Exception as e:
            logger.error(f"Erro ao enviar email: {str(e)}")
            return False
    
    def _send_ses_email(self, alert: Dict, email_config: Dict) -> bool:
        """Envia email via AWS SES"""
        subject = f"[{alert['severity']}] Trading System Alert: {alert['alert_type']}"
        
        # Constrói corpo do email
        body = self._format_email_body(alert)
        
        try:
            response = self.ses_client.send_email(
                Source=email_config['from_address'],
                Destination={
                    'ToAddresses': email_config['to_addresses']
                },
                Message={
                    'Subject': {'Data': subject},
                    'Body': {
                        'Text': {'Data': body},
                        'Html': {'Data': self._format_html_email_body(alert)}
                    }
                }
            )
            
            logger.debug(f"Email enviado via SES: {response['MessageId']}")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao enviar via SES: {str(e)}")
            return False
    
    def _send_smtp_email(self, alert: Dict, email_config: Dict) -> bool:
        """Envia email via SMTP"""
        if not all([email_config.get('smtp_server'), 
                   email_config.get('smtp_username'),
                   email_config.get('smtp_password')]):
            logger.warning("Configuração SMTP incompleta")
            return False
        
        try:
            msg = MIMEMultipart()
            msg['From'] = email_config['from_address']
            msg['To'] = ', '.join(email_config['to_addresses'])
            msg['Subject'] = f"[{alert['severity']}] Trading System Alert: {alert['alert_type']}"
            
            body = self._format_email_body(alert)
            msg.attach(MIMEText(body, 'plain'))
            
            # Conecta ao servidor SMTP
            server = smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port'])
            server.starttls()
            server.login(email_config['smtp_username'], email_config['smtp_password'])
            
            # Envia email
            server.send_message(msg)
            server.quit()
            
            logger.debug("Email enviado via SMTP")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao enviar via SMTP: {str(e)}")
            return False
    
    def _send_sns(self, alert: Dict) -> bool:
        """Envia alerta via SNS"""
        sns_config = self.channel_config.get('sns', {})
        topic_arn = sns_config.get('topic_arn')
        
        if not topic_arn:
            logger.warning("SNS topic ARN não configurado")
            return False
        
        try:
            subject = f"[{alert['severity']}] {alert['alert_type']}"
            message = self._format_sns_message(alert)
            
            response = self.sns_client.publish(
                TopicArn=topic_arn,
                Subject=subject,
                Message=message
            )
            
            logger.debug(f"SNS enviado: {response['MessageId']}")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao enviar SNS: {str(e)}")
            return False
    
    def _send_slack(self, alert: Dict) -> bool:
        """Envia alerta via Slack"""
        slack_config = self.channel_config.get('slack', {})
        webhook_url = slack_config.get('webhook_url')
        
        if not webhook_url:
            logger.warning("Slack webhook URL não configurado")
            return False
        
        try:
            import requests
            
            # Formata mensagem para Slack
            slack_message = {
                'channel': slack_config.get('channel', '#trading-alerts'),
                'username': 'Trading System',
                'icon_emoji': self._get_slack_emoji(alert['severity']),
                'attachments': [{
                    'color': self._get_slack_color(alert['severity']),
                    'title': f"{alert['alert_type']} - {alert['severity']}",
                    'text': alert['message'],
                    'timestamp': int(datetime.fromisoformat(alert['timestamp']).timestamp())
                }]
            }
            
            response = requests.post(webhook_url, json=slack_message, timeout=10)
            response.raise_for_status()
            
            logger.debug("Slack enviado com sucesso")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao enviar Slack: {str(e)}")
            return False
    
    def _send_cloudwatch(self, alert: Dict) -> bool:
        """Envia alerta para CloudWatch"""
        try:
            # Cria métrica customizada no CloudWatch
            self.cloudwatch.put_metric_data(
                Namespace='TradingSystem/Alerts',
                MetricData=[
                    {
                        'MetricName': 'AlertCount',
                        'Dimensions': [
                            {
                                'Name': 'AlertType',
                                'Value': alert['alert_type']
                            },
                            {
                                'Name': 'Severity',
                                'Value': alert['severity']
                            }
                        ],
                        'Value': 1,
                        'Timestamp': datetime.fromisoformat(alert['timestamp'])
                    }
                ]
            )
            
            logger.debug("Métrica CloudWatch enviada")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao enviar CloudWatch: {str(e)}")
            return False
    
    def _is_rate_limited(self, alert_type: str, severity: AlertSeverity) -> bool:
        """Verifica se o alerta está limitado por rate limiting"""
        # Alertas críticos nunca são limitados
        if severity == AlertSeverity.CRITICAL:
            return False
        
        cache_key = f"{alert_type}_{severity.value}"
        rate_limit_minutes = self.alert_config['rate_limit_minutes']
        
        if cache_key in self.recent_alerts:
            last_sent = self.recent_alerts[cache_key]
            time_diff = datetime.now() - last_sent
            
            if time_diff < timedelta(minutes=rate_limit_minutes):
                return True
        
        return False
    
    def _update_rate_limit_cache(self, alert_type: str, severity: AlertSeverity):
        """Atualiza cache de rate limiting"""
        cache_key = f"{alert_type}_{severity.value}"
        self.recent_alerts[cache_key] = datetime.now()
        
        # Limpa entradas antigas do cache
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.recent_alerts = {
            k: v for k, v in self.recent_alerts.items() 
            if v > cutoff_time
        }
    
    def _save_alert_history(self, alert: Dict):
        """Salva alerta no histórico"""
        try:
            table = self.dynamodb.Table(self.alerts_table)
            
            # Adiciona TTL (30 dias)
            alert['ttl'] = int((datetime.now() + timedelta(days=30)).timestamp())
            
            table.put_item(Item=alert)
            logger.debug(f"Alerta salvo no histórico: {alert['alert_id']}")
            
        except Exception as e:
            logger.warning(f"Erro ao salvar histórico: {str(e)}")
    
    def _generate_alert_id(self) -> str:
        """Gera ID único para o alerta"""
        import uuid
        return str(uuid.uuid4())
    
    def _determine_risk_severity(self, risk_level: str) -> AlertSeverity:
        """Determina severidade baseada no nível de risco"""
        if risk_level == 'CRITICAL':
            return AlertSeverity.CRITICAL
        elif risk_level == 'HIGH':
            return AlertSeverity.ERROR
        elif risk_level == 'MEDIUM':
            return AlertSeverity.WARNING
        else:
            return AlertSeverity.INFO
    
    def _determine_performance_severity(self, performance_status: str) -> AlertSeverity:
        """Determina severidade baseada no status de performance"""
        if performance_status == 'DEGRADED':
            return AlertSeverity.ERROR
        elif performance_status == 'POOR':
            return AlertSeverity.WARNING
        else:
            return AlertSeverity.INFO
    
    def _is_ses_configured(self) -> bool:
        """Verifica se SES está configurado"""
        try:
            # Tenta listar identidades para verificar se SES está configurado
            self.ses_client.list_identities(MaxItems=1)
            return True
        except:
            return False
    
    def _format_email_body(self, alert: Dict) -> str:
        """Formata corpo do email"""
        body = f"ALERTA DO SISTEMA DE TRADING\n"
        body += f"{'=' * 50}\n\n"
        body += f"Tipo: {alert['alert_type']}\n"
        body += f"Severidade: {alert['severity']}\n"
        body += f"Timestamp: {alert['timestamp']}\n\n"
        body += f"Mensagem:\n{alert['message']}\n\n"
        
        if alert.get('metadata'):
            body += f"Metadados:\n"
            for key, value in alert['metadata'].items():
                body += f"- {key}: {value}\n"
        
        body += f"\n{'=' * 50}\n"
        body += f"Sistema de Trading Quantitativo v2.01"
        
        return body
    
    def _format_html_email_body(self, alert: Dict) -> str:
        """Formata corpo do email em HTML"""
        severity_color = {
            'INFO': '#17a2b8',
            'WARNING': '#ffc107',
            'ERROR': '#dc3545',
            'CRITICAL': '#6f42c1'
        }.get(alert['severity'], '#6c757d')
        
        html = f"""
        <html>
        <body style="font-family: Arial, sans-serif; margin: 20px;">
            <div style="border-left: 4px solid {severity_color}; padding-left: 20px;">
                <h2 style="color: {severity_color};">Alerta do Sistema de Trading</h2>
                <p><strong>Tipo:</strong> {alert['alert_type']}</p>
                <p><strong>Severidade:</strong> <span style="color: {severity_color};">{alert['severity']}</span></p>
                <p><strong>Timestamp:</strong> {alert['timestamp']}</p>
                <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 15px 0;">
                    <h4>Mensagem:</h4>
                    <p style="white-space: pre-line;">{alert['message']}</p>
                </div>
            </div>
            <hr>
            <p style="color: #6c757d; font-size: 12px;">Sistema de Trading Quantitativo v2.01</p>
        </body>
        </html>
        """
        return html
    
    def _format_sns_message(self, alert: Dict) -> str:
        """Formata mensagem para SNS"""
        message = f"[{alert['severity']}] {alert['alert_type']}\n\n"
        message += alert['message']
        
        if alert.get('metadata'):
            message += f"\n\nMetadados: {json.dumps(alert['metadata'], indent=2)}"
        
        return message
    
    def _get_slack_emoji(self, severity: str) -> str:
        """Retorna emoji para Slack baseado na severidade"""
        return {
            'INFO': ':information_source:',
            'WARNING': ':warning:',
            'ERROR': ':x:',
            'CRITICAL': ':rotating_light:'
        }.get(severity, ':question:')
    
    def _get_slack_color(self, severity: str) -> str:
        """Retorna cor para Slack baseado na severidade"""
        return {
            'INFO': 'good',
            'WARNING': 'warning',
            'ERROR': 'danger',
            'CRITICAL': '#6f42c1'
        }.get(severity, '#6c757d')
    
    def get_alert_history(self, hours: int = 24) -> List[Dict]:
        """
        Obtém histórico de alertas.
        
        Args:
            hours: Número de horas de histórico
            
        Returns:
            Lista de alertas
        """
        try:
            table = self.dynamodb.Table(self.alerts_table)
            
            # Data de início
            start_time = datetime.now() - timedelta(hours=hours)
            
            # Scan com filtro de timestamp (não é eficiente, mas funciona para volumes pequenos)
            response = table.scan(
                FilterExpression='#ts >= :start_time',
                ExpressionAttributeNames={'#ts': 'timestamp'},
                ExpressionAttributeValues={':start_time': start_time.isoformat()}
            )
            
            alerts = response.get('Items', [])
            
            # Ordena por timestamp
            alerts.sort(key=lambda x: x['timestamp'], reverse=True)
            
            return alerts
            
        except Exception as e:
            logger.error(f"Erro ao obter histórico: {str(e)}")
            return []
    
    def get_alert_statistics(self, days: int = 7) -> Dict:
        """
        Obtém estatísticas de alertas.
        
        Args:
            days: Número de dias para análise
            
        Returns:
            Estatísticas de alertas
        """
        alerts = self.get_alert_history(hours=days * 24)
        
        if not alerts:
            return {'total_alerts': 0}
        
        # Conta por severidade
        severity_counts = {}
        for alert in alerts:
            severity = alert.get('severity', 'UNKNOWN')
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # Conta por tipo
        type_counts = {}
        for alert in alerts:
            alert_type = alert.get('alert_type', 'unknown')
            type_counts[alert_type] = type_counts.get(alert_type, 0) + 1
        
        # Alertas por dia
        daily_counts = {}
        for alert in alerts:
            date = alert['timestamp'][:10]  # YYYY-MM-DD
            daily_counts[date] = daily_counts.get(date, 0) + 1
        
        return {
            'total_alerts': len(alerts),
            'period_days': days,
            'severity_breakdown': severity_counts,
            'type_breakdown': type_counts,
            'daily_counts': daily_counts,
            'avg_alerts_per_day': len(alerts) / days
        }

