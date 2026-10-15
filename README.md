Aqui está o seu `README.md` completo, reformatado utilizando **apenas Markdown puro** para as imagens (removendo tags HTML desnecessárias), com os **badges coloridos** de linguagens/bibliotecas no topo, a explicação detalhada do link **RTSP** e a **remoção completa de todas as citações numéricas** dos parágrafos, conforme solicitado.

Você pode copiar o bloco de código abaixo e colar diretamente no seu arquivo `README.md`:

```markdown
# 🎓 SmartClass

**Sistema Inteligente de Monitoramento de Frequência e Comportamento em Sala de Aula com Visão Computacional em Edge Computing**

<br>

<p align="center">
  <img src="https://img.shields.io/badge/STATUS-CONCLUÍDO-brightgreen?style=for-the-badge">
  <img src="https://img.shields.io/badge/NODE.JS-339933?style=for-the-badge&logo=nodedotjs&logoColor=white">
  <img src="https://img.shields.io/badge/EXPRESS.JS-000000?style=for-the-badge&logo=express&logoColor=white">
  <img src="https://img.shields.io/badge/PYTHON-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/FASTAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white">
  <img src="https://img.shields.io/badge/OPENCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white">
  <img src="https://img.shields.io/badge/DOCKER-2496ED?style=for-the-badge&logo=docker&logoColor=white">
  <img src="https://img.shields.io/badge/RASPBERRY%20PI-C51A4A?style=for-the-badge&logo=raspberrypi&logoColor=white">
  <img src="https://img.shields.io/badge/SQLITE-003B57?style=for-the-badge&logo=sqlite&logoColor=white">
  <img src="https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white">
  <img src="https://img.shields.io/badge/CSS3-1572B6?style=for-the-badge&logo=css3&logoColor=white">
  <img src="https://img.shields.io/badge/JAVASCRIPT-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black">
</p>

---

## 📖 Sobre o Projeto

O **SmartClass** é uma solução desenvolvida como Trabalho de Conclusão de Curso (TCC) em Tecnologia em Redes de Computadores no **Instituto Federal Catarinense (IFC) - Campus Brusque**. 

O sistema automatiza o registro de presença por reconhecimento facial, delimita cercas virtuais poligonais para localização dos alunos em suas bancadas e analisa em tempo real comportamentos como o uso indevido de smartphones, dispersão da atenção e indícios de interações físicas agressivas. 

Toda a arquitetura foi projetada para operar estritamente em **rede local (LAN)**, garantindo total conformidade com a **Lei Geral de Proteção de Dados (LGPD)** e preservando a privacidade institucional ao não reter vídeos contínuos. Além disso, o processamento de Inteligência Artificial foi migrado para o conceito de **Edge Computing**, rodando de forma autônoma em um **Raspberry Pi 4** com modelos otimizados em formato NCNN.

---

## ✨ Funcionalidades

- 👁️ **Chamada Automatizada:** Registro de presença por reconhecimento facial em tempo real.
- 🪑 **Cercas Virtuais (Polígonos):** Mapeamento dinâmico das bancadas da sala de aula via Canvas HTML5 para identificar se o aluno está no assento correto, circulando ou ausente.
- 📱 **Detecção de Comportamento:** Identificação de uso inadequado de celulares e análise de postura para detecção de dispersão e ações suspeitas (possíveis brigas).
- 📊 **Dashboards e Relatórios:** Gráficos interativos em Chart.js e exportação automática de diários de classe em formato Excel (`.xlsx`), colorizados conforme o engajamento.
- 📥 **Integração SIGAA:** Importação automatizada de turmas e dados curriculares diretamente de planilhas institucionais.
- 🔒 **Privacy by Design (LGPD):** Descarte automático de evidências pontuais de alertas após 24 horas e armazenamento apenas de vetores numéricos (encodings biométricos) e logs textuais.

---

## 🛠️ Tecnologias, Linguagens e Bibliotecas

### Backend & Web
- **Node.js** & **Express.js**: Gerenciamento de rotas, API REST, autenticação JWT e regras de negócio.
- **SQLite3**: Banco de dados relacional leve e portátil para metadados e logs.
- **SheetJS (xlsx)** & **ExcelJS**: Manipulação e importação de planilhas institucionais.

### Inteligência Artificial & Visão Computacional (Worker Python)
- **Python** & **FastAPI** / **Uvicorn**: Worker de processamento vetorial pesado assíncrono.
- **OpenCV**: Captura de fluxos de vídeo e manipulação de matrizes de imagem.
- **Dlib** & **face_recognition**: Extração de encodings faciais de 128 dimensões para identificação biométrica.
- **Ultralytics YOLOv8** (e modelos otimizados **NCNN** da Tencent): Para detecção de objetos (smartphones) e estimativa de pose (YOLOv8-pose).

### Frontend
- HTML5, CSS3 e JavaScript (Vanilla) com manipulação de **Canvas** para sobreposição telemétrica.
- **Chart.js** para renderização de gráficos institucionais.

---

## ⚙️ Hardware e Conectividade (Câmeras IP e RTSP)

- **Servidor de Borda / Processamento IA:** Raspberry Pi 4 (processamento autônomo em arquitetura ARM64 via contêineres Docker com motores NCNN).
- **Captação de Mídia:** Câmeras de segurança IP integradas via protocolo **RTSP** (Real-Time Streaming Protocol).

### Como configurar o link RTSP
Para conectar o SmartClass a uma câmera de segurança IP, o fluxo de vídeo é mapeado utilizando a URL padrão do protocolo RTSP. A estrutura básica utilizada nos arquivos de configuração do sistema é:

`rtsp://USUARIO:SENHA@IPCAMERA:554/cam/realmonitor?channel=1&subtype=1`

Onde:
- **USUARIO / SENHA**: Credenciais de acesso configuradas na sua câmera IP (ex: Hikvision, Intelbras).
- **IPCAMERA**: Endereço IP local da câmera na rede da instituição (ex: `192.168.1.100`).
- **554**: Porta padrão de comunicação RTSP.
- **channel=1&subtype=1**: Parâmetros de canal e resolução (geralmente o subfluxo `subtype=1` é utilizado para otimizar o desempenho do processamento de visão computacional na borda).

---

## 📸 Demonstração do Sistema

### Dashboard Principal & Visão Institucional
*Visão geral dos dados da instituição, assiduidade e métricas de engajamento.*

![Dashboard Principal](assets/dashboard_geral.png)

---

### Alertas para Validação (Celular e Ação Suspeita)
*Exemplo dos avisos visuais e sonoros gerados em tempo real quando o sistema detecta o uso de celulares ou interações físicas bruscas.*

![Alertas em Tempo Real](assets/alertas_validacao.png)

---

### Configuração de Sensibilidades e Bancadas
*Interface de mapeamento de cercas virtuais poligonais e ajuste de limiares de confiança.*

![Configuração de Sensibilidades](assets/config_sensibilidade.png)

---

## 📂 Estrutura do Projeto

```text
smartclass_bom/
├── data/
│   └── faces/              # Armazenamento seguro de referências biométricas
├── docker/
│   ├── Dockerfile.dev      # Configuração para ambiente de desenvolvimento
│   └── Dockerfile.rpi      # Configuração otimizada para Raspberry Pi (ARM64 / NCNN)
├── frontend/
│   ├── css/
│   │   └── style.css       # Estilos da interface web
│   ├── js/
│   │   └── app.js          # Lógica do frontend e renderização via Canvas
│   └── index.html          # Página principal do sistema
├── node_api/
│   ├── db.js               # Conexão e esquemas do SQLite
│   ├── package.json        # Dependências Node.js
│   └── server.js           # Servidor Express e rotas principais
├── python_ia/
│   ├── core/
│   │   ├── behavior_engine.py  # Heurísticas de comportamento (celular, briga, evasão)
│   │   ├── face_engine.py      # Motor de reconhecimento facial (Dlib)
│   │   └── pose_engine.py      # Estimativa de pose (YOLOv8-pose / NCNN)
│   ├── main.py             # Script principal do worker de IA
│   └── requirements.txt    # Dependências Python
├── docker-compose.yml      # Orquestração de microsserviços
├── INICIAR_SISTEMA.bat     # Atalho de inicialização rápida para Windows
└── ecosystem.config.js     # Configuração PM2 (opcional bare-metal)

```

---

## 🚀 Como Iniciar e Testar

### Pré-requisitos

* [Docker](https://www.docker.com/) e [Docker Compose](https://docs.docker.com/compose/) instalados na máquina ou em um Raspberry Pi.
* Node.js (caso deseje rodar scripts auxiliares localmente sem containers).

### Passos para Execução via Docker

1. **Clone o repositório:**
```bash
git clone [https://github.com/SEU-USUARIO/smartclass.git](https://github.com/SEU-USUARIO/smartclass.git)
cd smartclass/smartclass_bom

```


2. **Configure as variáveis de ambiente:**
Duplique o arquivo de exemplo de ambiente (`env.example`) renomeando-o para `.env` e ajuste as parâmetros conforme sua rede local, se necessário.
3. **Suba os containers utilizando o Docker Compose:**
```bash
docker-compose up --build

```


4. **Acesse a aplicação:**
Abra o seu navegador web e acesse:
```text
http://localhost:3000

```



### Executando Manualmente (Modo Desenvolvimento)

Se preferir rodar os microsserviços separadamente para testes de código:

* **Iniciar o Backend (Node.js):**
```bash
cd node_api
npm install
npm start

```


* **Iniciar o Worker de IA (Python):**
```bash
cd python_ia
pip install -r requirements.txt
python main.py

```



---

## 🧪 Testes e Validação

O sistema passou por três ciclos rigorosos de prototipagem iterativa, cobrindo:

* **Testes de Concorrência de Vídeo:** Desacoplamento do buffer de gravação para evitar *Race Conditions* no OpenCV.
* **Testes de Segurança (OWASP Top 10 / Auditoria Estática):** Correção de falhas de autenticação, controle de acesso e tratamento de rotas sensíveis (validados via `npm audit` e `pip-audit`).
* **Validação de Borda:** Estabilidade de inferência em CPU ARM sem aceleração dedicada por GPU.

---

## 🎓 Público-alvo & Contexto Acadêmico

* Gestores educacionais e coordenadores pedagógicos em busca de dados objetivos sobre assiduidade.
* Professores que necessitam de ferramentas automatizadas de chamada e relatórios comportamentais.
* Estudantes e pesquisadores das áreas de Redes de Computadores, Visão Computacional e IoT.

---

## 👨‍💻 Autor

**Caique de Jesus Cardim (Kyoshy)**

*Trabalho de Conclusão de Curso apresentado ao Instituto Federal Catarinense (IFC) - Campus Brusque como requisito parcial para obtenção do título de Tecnólogo em Redes de Computadores.*

* Orientador: Prof. Vinícius Barreto Klein

---

> ⭐ Se este projeto acadêmico foi útil para você, considere deixar uma estrela no repositório!

```

```