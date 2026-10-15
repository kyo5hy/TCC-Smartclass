#!/bin/bash

# setup_rpi.sh - Instalação Automatizada SmartClass RPi 4
echo "===================================================="
echo "Iniciando instalação do SmartClass no Raspberry Pi 4"
echo "===================================================="

# 1. Atualização do Sistema e Dependências Nativas
echo "[1/7] Atualizando sistema e dependências..."
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential cmake pkg-config \
    libjpeg-dev libpng-dev libtiff-dev \
    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
    libxvidcore-dev libx264-dev \
    libgtk-3-dev libatlas-base-dev gfortran \
    python3-dev python3-pip python3-venv \
    curl git

# 2. Aumento temporário de SWAP
echo "[2/7] Aumentando SWAP temporariamente..."
sudo dphys-swapfile swapoff
sudo sed -i 's/CONF_SWAPSIZE=.*/CONF_SWAPSIZE=2048/g' /etc/dphys-swapfile
sudo dphys-swapfile setup
sudo dphys-swapfile swapon

# 3. Configuração do Node.js e PM2
echo "[3/7] Instalando Node.js (LTS) e PM2..."
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install -y nodejs
sudo npm install -g pm2
pm2 startup

# 4. Configuração da API Node
echo "[4/7] Instalando dependências do backend Node.js..."
cd node_api
npm install
cd ..

# 5. Configuração do Ambiente Python
echo "[5/7] Configurando IA (Aguarde, a compilação do dlib pode demorar)..."
cd python_ia
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install numpy==1.26.4
pip install dlib
pip install -r requirements.txt
deactivate
cd ..

# 6. Restauração do SWAP original
echo "[6/7] Restaurando SWAP original..."
sudo dphys-swapfile swapoff
sudo sed -i 's/CONF_SWAPSIZE=.*/CONF_SWAPSIZE=100/g' /etc/dphys-swapfile
sudo dphys-swapfile setup
sudo dphys-swapfile swapon

# 7. Preparação da Estrutura de Execução
echo "[7/7] Preparando diretórios e variáveis de ambiente..."
mkdir -p logs
mkdir -p data/clips
mkdir -p data/faces
mkdir -p data/screenshots

if [ ! -f .env ]; then
    echo "Criando .env a partir do env.example..."
    cp env.example .env
fi

chmod +x setup_rpi.sh

echo "===================================================="
echo "✅ Instalação Concluída com Sucesso!"
echo "Lembre-se de editar o arquivo .env com suas senhas reais (nano .env)."
echo "Para iniciar o sistema: pm2 start ecosystem.config.js"
echo "===================================================="