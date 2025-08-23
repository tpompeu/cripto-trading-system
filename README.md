# Sistema de Trading Quantitativo para Criptomoedas

Sistema automatizado de trading algorítmico para criptomoedas com validação robusta via Walk-Forward Backtesting e gerenciamento de risco avançado.

## Características Principais

- Modelo de Machine Learning (LightGBM) para geração de sinais
- Validação robusta com Walk-Forward Backtesting
- Simulação de Monte Carlo com sinais aleatórios
- Gerenciamento de risco multinível com circuit breakers
- Pipeline CI/CD completo com AWS CodePipeline
- Operação com instâncias Spot para economia de custos

## Arquitetura

O sistema é implantado na AWS us-east-1 e utiliza os seguintes serviços:

- **Computação**: AWS Lambda, EC2 Spot Instances
- **Armazenamento**: S3, DynamoDB
- **Orquestração**: EventBridge, SQS
- **ML**: LightGBM com otimização de parâmetros
- **Monitoramento**: CloudWatch, Cost Explorer

## Instalação

Consulte o guia de instalação em [INSTALL_GUIDE.md](INSTALL_GUIDE.md) para instruções detalhadas.

## Configuração

O sistema é configurado através de dois mecanismos:

1. **Configuração estática** (`config.json`): Define a arquitetura da estratégia
2. **Configuração dinâmica** (Parameter Store): Parâmetros operacionais ajustáveis em tempo real

## Fluxo de Trabalho

1. **Download de dados**: Coleta diária de dados OHLCV da Binance
2. **Validação**: Walk-Forward Backtesting com otimização de parâmetros
3. **Simulação**: Teste de Monte Carlo com sinais aleatórios
4. **Incubação**: Paper trading para validação em tempo real
5. **Produção**: Operação com capital real e monitoramento contínuo
6. **Retreinamento**: Atualização periódica dos modelos com dados recentes

## Gerenciamento de Risco

O sistema implementa múltiplas camadas de proteção:

- Risco por operação baseado no framework de Van Tharp
- Circuit breakers para drawdown excessivo
- Position sizing dinâmico
- Break-even e trailing stops automáticos