#!/bin/bash
set -e

# Script de inicialização para instâncias EC2 Spot
# Sistema de Trading Quantitativo - Versão 2.01
# Data: 26 de agosto de 2025

# Configuração de logging
exec > >(tee /var/log/user-data.log|logger -t user-data -s 2>/dev/console) 2>&1
echo "=== INÍCIO DA CONFIGURAÇÃO DA INSTÂNCIA SPOT ==="
echo "Timestamp: $(date)"
echo "Instance ID: $(curl -s http://169.254.169.254/latest/meta-data/instance-id)"
echo "Instance Type: $(curl -s http://169.254.169.254/latest/meta-data/instance-type)"

# Função para logging com timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Função para tratamento de erros
handle_error() {
    log "ERRO: $1"
    log "Enviando notificação de falha..."
    
    # Envia notificação de erro (se configurado)
    if [ ! -z "$SNS_TOPIC_ARN" ]; then
        aws sns publish \
            --topic-arn "$SNS_TOPIC_ARN" \
            --message "Falha na inicialização da instância Spot: $1" \
            --subject "Erro - Instância Spot" \
            --region ap-southeast-1 || true
    fi
    
    log "Terminando instância devido ao erro..."
    shutdown -h +1
    exit 1
}

# Trap para capturar erros
trap 'handle_error "Script falhou na linha $LINENO"' ERR

log "Iniciando atualização do sistema..."

# Atualiza sistema operacional
yum update -y || handle_error "Falha na atualização do sistema"

log "Instalando pacotes essenciais..."

# Instala pacotes básicos
yum install -y \
    git \
    python3 \
    python3-pip \
    htop \
    tmux \
    wget \
    curl \
    unzip \
    gcc \
    gcc-c++ \
    make \
    openssl-devel \
    libffi-devel \
    python3-devel || handle_error "Falha na instalação de pacotes básicos"

log "Instalando AWS CLI v2..."

# Instala AWS CLI v2
cd /tmp
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" || handle_error "Falha no download do AWS CLI"
unzip awscliv2.zip || handle_error "Falha na descompactação do AWS CLI"
sudo ./aws/install || handle_error "Falha na instalação do AWS CLI"

# Verifica instalação do AWS CLI
aws --version || handle_error "AWS CLI não foi instalado corretamente"

log "Configurando Python e dependências..."

# Atualiza pip
python3 -m pip install --upgrade pip || handle_error "Falha na atualização do pip"

# Instala dependências Python essenciais
pip3 install \
    boto3 \
    pandas \
    numpy \
    scikit-learn \
    lightgbm \
    requests \
    python-dateutil \
    pytz \
    joblib \
    matplotlib \
    seaborn || handle_error "Falha na instalação de dependências Python"

# Instala TA-Lib (indicadores técnicos)
log "Instalando TA-Lib..."
cd /tmp
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz || handle_error "Falha no download do TA-Lib"
tar -xzf ta-lib-0.4.0-src.tar.gz || handle_error "Falha na descompactação do TA-Lib"
cd ta-lib/
./configure --prefix=/usr || handle_error "Falha na configuração do TA-Lib"
make || handle_error "Falha na compilação do TA-Lib"
make install || handle_error "Falha na instalação do TA-Lib"

# Instala wrapper Python do TA-Lib
pip3 install TA-Lib || handle_error "Falha na instalação do TA-Lib Python"

log "Criando estrutura de diretórios..."

# Cria diretório de trabalho
mkdir -p /opt/trading-system
mkdir -p /opt/trading-system/data
mkdir -p /opt/trading-system/models
mkdir -p /opt/trading-system/logs
mkdir -p /opt/trading-system/checkpoints

cd /opt/trading-system

# Define permissões
chown -R ec2-user:ec2-user /opt/trading-system

log "Baixando código da aplicação..."

# Baixa código do S3 (assumindo que foi carregado pelo pipeline)
if [ ! -z "$DATA_BUCKET" ]; then
    aws s3 cp s3://$DATA_BUCKET/code/model_validator.py ./ || handle_error "Falha no download do model_validator.py"
    aws s3 cp s3://$DATA_BUCKET/code/random_signal_simulator.py ./ || handle_error "Falha no download do random_signal_simulator.py"
    aws s3 cp s3://$DATA_BUCKET/code/config_service.py ./ || handle_error "Falha no download do config_service.py"
    aws s3 cp s3://$DATA_BUCKET/code/requirements.txt ./ || true  # Não crítico se não existir
else
    log "AVISO: DATA_BUCKET não definido, pulando download do código"
fi

log "Configurando variáveis de ambiente..."

# Configura variáveis de ambiente
cat > /opt/trading-system/.env << EOF
# Configurações da instância Spot
PROJECT_NAME=${PROJECT_NAME:-CryptoTradingSystem}
DATA_BUCKET=${DATA_BUCKET:-}
STRATEGY_ID=${STRATEGY_ID:-}
SYMBOL=${SYMBOL:-}
TIMEFRAME=${TIMEFRAME:-}
JOB_TYPE=${JOB_TYPE:-walk_forward_validation}

# Configurações AWS
AWS_DEFAULT_REGION=uap-southeast-1
AWS_REGION=ap-southeast-1

# Configurações de logging
LOG_LEVEL=INFO
PYTHONUNBUFFERED=1

# Configurações de performance
OMP_NUM_THREADS=4
NUMBA_NUM_THREADS=4
EOF

# Carrega variáveis de ambiente
source /opt/trading-system/.env

log "Configurando monitoramento..."

# Instala CloudWatch Agent (opcional)
if command -v amazon-cloudwatch-agent-ctl &> /dev/null; then
    log "CloudWatch Agent já instalado"
else
    log "Instalando CloudWatch Agent..."
    wget https://s3.amazonaws.com/amazoncloudwatch-agent/amazon_linux/amd64/latest/amazon-cloudwatch-agent.rpm
    rpm -U ./amazon-cloudwatch-agent.rpm || true
fi

log "Configurando script de execução principal..."

# Cria script principal de execução
cat > /opt/trading-system/run_validation.sh << 'EOF'
#!/bin/bash
set -e

# Carrega variáveis de ambiente
source /opt/trading-system/.env

# Função de logging
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a /opt/trading-system/logs/validation.log
}

log "=== INÍCIO DA EXECUÇÃO DE VALIDAÇÃO ==="
log "Strategy ID: $STRATEGY_ID"
log "Symbol: $SYMBOL"
log "Timeframe: $TIMEFRAME"
log "Job Type: $JOB_TYPE"

cd /opt/trading-system

# Verifica se os arquivos necessários existem
if [ ! -f "model_validator.py" ]; then
    log "ERRO: model_validator.py não encontrado"
    exit 1
fi

# Executa validação baseada no tipo de job
case "$JOB_TYPE" in
    "walk_forward_validation")
        log "Executando Walk-Forward Validation..."
        python3 model_validator.py 2>&1 | tee -a /opt/trading-system/logs/validation.log
        ;;
    "monte_carlo_simulation")
        log "Executando Monte Carlo Simulation..."
        python3 random_signal_simulator.py 2>&1 | tee -a /opt/trading-system/logs/validation.log
        ;;
    "model_retraining")
        log "Executando Model Retraining..."
        python3 model_retrainer.py 2>&1 | tee -a /opt/trading-system/logs/validation.log
        ;;
    *)
        log "ERRO: Tipo de job desconhecido: $JOB_TYPE"
        exit 1
        ;;
esac

log "=== VALIDAÇÃO CONCLUÍDA COM SUCESSO ==="

# Upload dos logs para S3
if [ ! -z "$DATA_BUCKET" ]; then
    log "Fazendo upload dos logs para S3..."
    aws s3 cp /opt/trading-system/logs/ s3://$DATA_BUCKET/logs/spot-instances/$(date +%Y-%m-%d)/ --recursive || true
fi

# Sinaliza conclusão
touch /opt/trading-system/VALIDATION_COMPLETE

log "Job finalizado. Instância será terminada em 5 minutos."
EOF

# Torna script executável
chmod +x /opt/trading-system/run_validation.sh
chown ec2-user:ec2-user /opt/trading-system/run_validation.sh

log "Configurando auto-terminação..."

# Cria script de auto-terminação
cat > /opt/trading-system/auto_terminate.sh << 'EOF'
#!/bin/bash

# Aguarda conclusão do job ou timeout
TIMEOUT=14400  # 4 horas
ELAPSED=0
INTERVAL=60    # Verifica a cada minuto

while [ $ELAPSED -lt $TIMEOUT ]; do
    if [ -f "/opt/trading-system/VALIDATION_COMPLETE" ]; then
        echo "Validação concluída. Terminando instância..."
        shutdown -h +2
        exit 0
    fi
    
    sleep $INTERVAL
    ELAPSED=$((ELAPSED + INTERVAL))
done

echo "Timeout atingido. Terminando instância..."
shutdown -h +1
EOF

chmod +x /opt/trading-system/auto_terminate.sh

log "Iniciando execução do job..."

# Executa job em background e inicia auto-terminação
nohup /opt/trading-system/run_validation.sh > /opt/trading-system/logs/nohup.log 2>&1 &
nohup /opt/trading-system/auto_terminate.sh > /opt/trading-system/logs/auto_terminate.log 2>&1 &

log "=== CONFIGURAÇÃO DA INSTÂNCIA SPOT CONCLUÍDA ==="
log "Job iniciado em background"
log "Auto-terminação configurada"
log "Logs disponíveis em /opt/trading-system/logs/"

# Envia notificação de sucesso na inicialização
if [ ! -z "$SNS_TOPIC_ARN" ]; then
    aws sns publish \
        --topic-arn "$SNS_TOPIC_ARN" \
        --message "Instância Spot inicializada com sucesso. Strategy: $STRATEGY_ID, Job: $JOB_TYPE" \
        --subject "Sucesso - Instância Spot Iniciada" \
        --region ap-southeast-1 || true
fi

log "Script user-data finalizado com sucesso"

