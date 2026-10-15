# main.py
import os

# Limita as threads do BLAS (dlib/numpy) ANTES de importar cv2/numpy/dlib, para não
# disputar núcleos com a inferência dos modelos.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import cv2
import re
import face_recognition
import numpy as np
import time
import os
import json
import sqlite3
import threading
import math
import queue
import asyncio
from multiprocessing import Process, Manager, Queue
from collections import deque
from datetime import datetime
from fastapi import FastAPI, Response, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from ultralytics import YOLO 
import uvicorn
from jose import jwt 
from dotenv import load_dotenv

from core.face_engine import FaceEngine
from core.pose_engine import PoseEngine
from core.behavior_engine import GestureTracker, PhoneConfirmer, LocalizadorBancada, CirculacaoTracker, selecionar_reconhecimentos

# ==========================================
# GESTÃO DE CAMINHOS ABSOLUTOS
# ==========================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ENV_PATH = os.path.join(BASE_DIR, '.env')
load_dotenv(ENV_PATH)

app = FastAPI(title="SmartClass - AI Video Worker")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"], max_age=86400)

DB_PATH = os.path.join(BASE_DIR, 'data', 'database.db')
DATA_FACES = os.path.join(BASE_DIR, 'data', 'faces')
DATA_ENCODINGS = os.path.join(BASE_DIR, 'data', 'encodings.pickle')
DATA_CLIPS = os.path.join(BASE_DIR, 'data', 'clips')
DATA_SCREENSHOTS = os.path.join(BASE_DIR, 'data', 'screenshots')

os.makedirs(DATA_CLIPS, exist_ok=True)
os.makedirs(DATA_SCREENSHOTS, exist_ok=True)

SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    print("❌ ERRO FATAL: SECRET_KEY não encontrada no arquivo .env")
    exit(1)
    
ALGORITHM = "HS256"

# Valores padrão de sensibilidade da IA — usados se o Master ainda não configurou nada.
# Mantidos em sincronia com CONFIG_IA_DEFAULTS em node_api/server.js.
CONFIG_IA_DEFAULTS = {
    "face_conf": "0.45",
    "phone_conf": "0.35",
    "tolerancia_facial": "0.48",
    "tempo_celular_seg": "4",
    "tempo_distracao_seg": "5",
    "confirmacoes_celular": "2",
    "max_distancia_px": "800",
    "acao_sensibilidade": "10"
}

# ==========================================
# RITMO DO LAÇO DE VÍDEO (processo principal)
# ==========================================
# Antes o laço de vídeo não tinha limite de taxa: cap.read() devolve o último frame mesmo
# que ele não seja novo, então o laço girava a toda a velocidade (o MESMO frame era copiado,
# desenhado, codificado em JPEG e serializado para o processo da IA dezenas de vezes por
# segundo), consumindo mais de 1 núcleo do Raspberry que a IA precisava.
INTERVALO_ENVIO_IA = 0.10        # s entre frames enviados ao processo da IA (ela processa ~1 por segundo)
INTERVALO_LEITURA_ESTADO = 0.10  # s entre leituras do shared_state (cada leitura é um IPC com o Manager)
INTERVALO_FRAME_CLIP = 0.10      # 10 fps = o mesmo fps usado pelo VideoWriter em save_clip()


def imgsz_do_modelo(pasta_ncnn, padrao=640):
    """Resolução de entrada do modelo NCNN, lida do metadata.yaml que o ultralytics grava na
    exportação. O NCNN tem o tamanho FIXO dentro do .param: se você reexportar o modelo com
    outra resolução (ex.: imgsz=(256, 320) = altura, largura), basta copiar a pasta nova e
    reconstruir a imagem — o código passa a usar o tamanho certo sozinho.
    Devolve int (quadrado) ou (altura, largura)."""
    try:
        import yaml
        with open(os.path.join(pasta_ncnn, 'metadata.yaml'), encoding='utf-8') as f:
            meta = yaml.safe_load(f) or {}
        v = meta.get('imgsz')
        if isinstance(v, (list, tuple)) and len(v) >= 1:
            v = [int(x) for x in v]
            return v[0] if (len(v) == 1 or v[0] == v[1]) else (v[0], v[1])
        if v:
            return int(v)
    except Exception as e:
        print(f"⚠️ [Modelo] Não consegui ler imgsz de {pasta_ncnn} ({e}); usando {padrao}.")
    return padrao


# ==========================================
# INICIALIZAÇÃO GLOBAL DOS MODELOS DE IA
# ==========================================
face_engine = FaceEngine(data_path=DATA_FACES, encodings_file=DATA_ENCODINGS)
_pose_ncnn_dir = os.path.join(BASE_DIR, 'python_ia', 'yolov8n-pose_ncnn_model')
pose_engine = PoseEngine(ncnn_path=_pose_ncnn_dir, imgsz=imgsz_do_modelo(_pose_ncnn_dir))

# Detector de Celular
# PRIORIDADE: modelo NCNN "nano" (o mesmo que já roda o reconhecimento facial
# perfeitamente rápido). Antes este projeto tentava usar 'yolov8s.pt' em PyTorch
# puro por ser "mais preciso" — só que em CPU/ARM fraco (Raspberry Pi e
# similares) uma única inferência dele pode levar SEGUNDOS, o que na prática
# reduz demais quantas vezes por segundo a câmera é checada, e faz o celular
# demorar muito pra ser flagrado mesmo estando parado na frente da câmera.
# NCNN é ordens de magnitude mais rápido nesse hardware — rodando muito mais
# vezes por segundo, a MESMA foto do celular acaba sendo testada dezenas de
# vezes a mais por segundo, o que na prática dá muito mais precisão e
# velocidade combinadas do que um modelo "melhor" só que muito mais lento.
phone_detector = None
phone_model_is_ncnn = False
phone_detector_path = os.path.join(BASE_DIR, 'python_ia', 'yolov8n_ncnn_model')
if os.path.exists(phone_detector_path):
    try:
        phone_detector = YOLO(phone_detector_path, task="detect")
        phone_model_is_ncnn = True
        print("✅ YOLO-Celular NCNN (Edge) carregado com sucesso!")
    except Exception as e:
        print(f"⚠️ Falha ao carregar NCNN de celular ({e}). Tentando PyTorch nano...")
if phone_detector is None:
    try:
        phone_detector = YOLO('yolov8n.pt')
        print("✅ YOLO-Celular PyTorch nano (fallback) carregado com sucesso!")
    except Exception as e:
        print(f"⚠️ Falha ao carregar yolov8n.pt ({e}). Tentando yolov8s.pt (mais lento)...")
        phone_detector = YOLO('yolov8s.pt')

IMGSZ_CELULAR = imgsz_do_modelo(phone_detector_path) if phone_model_is_ncnn else 640

# Detector de Rostos (Tratado como POSE)
face_detector_path = os.path.join(BASE_DIR, 'python_ia', 'yolov8n-face_ncnn_model') # Ajuste leve de path
IMGSZ_ROSTO = 640
try:
    if os.path.exists(face_detector_path):
        face_detector = YOLO(face_detector_path, task="pose") 
        IMGSZ_ROSTO = imgsz_do_modelo(face_detector_path)
        print("✅ YOLO-Face NCNN (Edge) carregado com sucesso!")
    else:
        face_detector = YOLO(os.path.join(BASE_DIR, 'python_ia', 'yolov8n-face.pt')) 
        print("✅ YOLO-Face PyTorch (Standard) carregado com sucesso!")
except Exception as e:
    print(f"⚠️ Aviso ao carregar face_detector: {e}")
    face_detector = YOLO('yolov8n.pt') 

# Warm-up: roda uma inferência "descartável" em cada modelo assim que o worker
# sobe, ainda ANTES de qualquer aula começar. A primeira chamada de cada modelo
# YOLO/face_recognition sempre é mais lenta (compilação interna/alocações);
# pagar esse custo aqui evita os ~10s de atraso no reconhecimento quando o
# professor efetivamente inicia a aula.
models_warmed = threading.Event()


def warmup_models():
    try:
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        face_detector(dummy, conf=0.40, imgsz=IMGSZ_ROSTO, verbose=False)
        dummy_rgb = cv2.cvtColor(dummy, cv2.COLOR_BGR2RGB)
        face_recognition.face_encodings(
            dummy_rgb, known_face_locations=[(80, 560, 400, 80)], num_jitters=1
        )
        face_recognition.face_locations(dummy_rgb)
        print("🔥 [Warm-up] Detector e encoder facial pré-aquecidos.")
    except Exception as e:
        print(f"⚠️ [Warm-up] Falha ao pré-aquecer reconhecimento facial: {e}")
    try:
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        phone_detector(dummy, classes=[67], conf=0.20, imgsz=IMGSZ_CELULAR, verbose=False)
        pose_engine.analyze_frame(dummy)
    except Exception as e:
        print(f"⚠️ [Warm-up] Falha ao pré-aquecer pose/celular (não crítico): {e}")
    finally:
        models_warmed.set()

threading.Thread(target=warmup_models, daemon=True).start()

def cleanup_old_clips():
    while True:
        try:
            now = time.time()
            for folder in [DATA_CLIPS, DATA_SCREENSHOTS]:
                for f in os.listdir(folder):
                    fpath = os.path.join(folder, f)
                    if os.path.isfile(fpath) and now - os.path.getmtime(fpath) > 86400:
                        os.remove(fpath)
        except: pass
        time.sleep(3600) 

threading.Thread(target=cleanup_old_clips, daemon=True).start()

def get_db():
    conn = sqlite3.connect(DB_PATH, timeout=20.0)
    conn.execute('PRAGMA journal_mode=WAL;')
    conn.row_factory = sqlite3.Row
    return conn

def garantir_colunas_live_status():
    """
    Migração idempotente, rodada pelo PRÓPRIO python_ia ao subir.
    Antes, essa coluna só era criada pelo node_api (db.js); se alguém
    reconstruísse (docker compose build) só o container do python_ia sem
    reconstruir o node_api também, a coluna nunca era criada e TODO ciclo
    da IA quebrava com "no column named bancada_atual" — o que também
    vazava conexões (ver face_loop) e podia derrubar o processo.
    Agora o python_ia garante essa coluna sozinho, sem depender do node_api.
    """
    try:
        conn = sqlite3.connect(DB_PATH, timeout=20.0)
        cur = conn.execute("PRAGMA table_info(live_status)")
        colunas = {row[1] for row in cur.fetchall()}
        if colunas and 'bancada_atual' not in colunas:
            conn.execute("ALTER TABLE live_status ADD COLUMN bancada_atual TEXT")
            conn.commit()
            print("🔧 Migração aplicada: coluna 'bancada_atual' adicionada em live_status.")
        conn.close()
    except Exception as e:
        print(f"⚠️ Não foi possível checar/migrar live_status (não crítico): {e}")

garantir_colunas_live_status()

@app.post("/api/reload_faces")
def reload_faces(background_tasks: BackgroundTasks):
    background_tasks.add_task(face_engine.generate_all_encodings)
    return {"status": "Recarregando em background..."}

class RTSPCamera:
    def __init__(self, src):
        self.src = src
        self.cap = None
        self.ret = False
        self.frame = None
        self.running = True
        self.lock = threading.Lock() 
        self.cond = threading.Condition(self.lock)
        self.frame_id = 0
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        if str(self.src).isdigit(): 
            self.cap = cv2.VideoCapture(int(self.src))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        else: 
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|stimeout;5000000"
            self.cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
            
        if self.cap and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            self.running = False
            return

        erros = 0
        while self.running:
            ret, frame = self.cap.read()
            if ret: 
                frame_redim = cv2.resize(frame, (640, 480))
                with self.cond:
                    self.ret = True
                    self.frame = frame_redim
                    self.frame_id += 1
                    self.cond.notify_all()
                erros = 0
            else:
                erros += 1
                if erros > 50: 
                    self.running = False
                    break
                time.sleep(0.05) 
            
        with self.cond:
            self.cond.notify_all()
        if self.cap:
            try: self.cap.release(); time.sleep(0.1)
            except: pass

    def read(self):
        with self.lock:
            if self.frame is None: return False, None
            return self.ret, self.frame.copy()

    def read_new(self, ultimo_id, timeout=0.5):
        """Espera um frame NOVO (diferente de 'ultimo_id') em vez de devolver o mesmo frame de novo.
        Devolve (ret, frame, id):
          (False, None, id) -> ainda não chegou nenhum frame (conectando / sem câmera)
          (True,  None, id) -> câmera ok, mas nenhum frame novo dentro do timeout
          (True, frame, id) -> frame novo"""
        with self.cond:
            if self.frame is None:
                if self.running:
                    self.cond.wait(timeout)
                if self.frame is None:
                    return False, None, ultimo_id
            elif self.frame_id == ultimo_id and self.running:
                self.cond.wait(timeout)
            if self.frame_id == ultimo_id:
                return True, None, ultimo_id
            return self.ret, self.frame.copy(), self.frame_id

    def release(self):
        self.running = False
        try:
            if self.thread and self.thread.is_alive(): self.thread.join(timeout=1.5)
        except BaseException: pass 

def save_clip(frames_list, filepath):
    if not frames_list: return
    out = None
    try:
        h, w = frames_list[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filepath, fourcc, 10.0, (w, h))
        for f in frames_list:
            out.write(f)
        print(f"🎬 Clipe Salvo com Sucesso: {filepath}")
    except Exception as e:
        print(f"❌ Erro ao tentar salvar vídeo: {e}")
    finally:
        if out is not None:
            out.release()

@app.get("/play_clip/{filename}")
async def play_clip(filename: str, token: str, request: Request):
    # Corrige controle de acesso quebrado: antes, QUALQUER pessoa na rede (sem login)
    # conseguia assistir ao clipe de um aluno só sabendo/adivinhando o nome do arquivo.
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if not payload.get("email"): return Response(status_code=401)
    except Exception:
        return Response(status_code=401)
    # Corrige Path Traversal: nome de arquivo só pode conter letras, números, _, - e .
    # (bloqueia "../../../etc/passwd" e afins).
    if not re.fullmatch(r"[a-zA-Z0-9_\-.]+\.mp4", filename):
        return Response(status_code=400)
    filepath = os.path.join(DATA_CLIPS, filename)

    async def generate():
        for _ in range(15):
            if os.path.exists(filepath): break
            await asyncio.sleep(0.2)

        if not os.path.exists(filepath):
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "VÍDEO PROCESSANDO...", (130, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,200,255), 2)
            ret, buffer = cv2.imencode('.jpg', blank)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            return

        cap = cv2.VideoCapture(filepath)
        # Trava de segurança: antes este laço rodava o vídeo em loop PARA SEMPRE,
        # sem nunca verificar se quem estava assistindo ainda estava conectado.
        # Se o navegador fechasse a aba/trocasse de tela no meio de uma aula, o
        # stream ficava "vivo" no servidor, ocupando uma conexão/thread
        # indefinidamente — com vários alertas isso ia acumulando até travar o
        # servidor inteiro (inclusive a aula ao vivo, que passava a exigir F5).
        # Agora ele para sozinho ao detectar desconexão, ou no máximo após
        # repetir o vídeo algumas vezes / 5 minutos, o que vier primeiro.
        MAX_REPETICOES = 3
        MAX_SEGUNDOS = 300
        inicio = time.time()
        repeticoes = 0
        try:
            while cap.isOpened():
                if await request.is_disconnected():
                    break
                if (time.time() - inicio) > MAX_SEGUNDOS:
                    break

                ret, frame = cap.read()
                if not ret:
                    repeticoes += 1
                    if repeticoes >= MAX_REPETICOES:
                        break
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                    if not ret: break
                    await asyncio.sleep(1.0)

                ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
                try:
                    yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                except Exception:
                    break  # cliente desconectou no meio do envio — encerra sem barulho
                await asyncio.sleep(0.08)
        finally:
            cap.release()

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")


def gerar_frames_mapeamento(sala_id: int):
    conn = get_db(); cursor = conn.cursor()
    cursor.execute('SELECT camera_url FROM salas WHERE id=?', (sala_id,))
    row = cursor.fetchone(); conn.close()
    if not row: return
    
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(blank_frame, "CONECTANDO CAMERA...", (140, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    ret, buffer = cv2.imencode('.jpg', blank_frame)
    yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap = RTSPCamera(row['camera_url'])
    erros_tentativas = 0
    try:
        while cap.running or erros_tentativas < 50:
            ret, frame = cap.read()
            if not ret or frame is None: 
                erros_tentativas += 1
                time.sleep(0.1)
                continue
            erros_tentativas = 0
            ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.05) 
    except Exception: pass 
    finally: cap.release()

# =====================================================================================
# WORKER DA IA ISOLADO EM UM PROCESSO (FUGINDO DO GIL)
# =====================================================================================
def ia_process_worker(shared_state, frame_queue, clip_queue, aula_dict, zonas, alunos_dict, assentos_aula, db_path, data_screenshots_path, ia_config):
    os.makedirs(data_screenshots_path, exist_ok=True)

    # O PyTorch (pré/pós-processamento e NMS do ultralytics) abre 1 thread por núcleo POR CHAMADA
    # e as 3 threads de inferência chamam ao mesmo tempo (origem do aviso "NMS time limit
    # exceeded"). Os tensores aqui são minúsculos; 1 thread basta.
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass
    print(f"📐 [IA] Resolução dos modelos: rosto={IMGSZ_ROSTO} pose={pose_engine.imgsz} celular={IMGSZ_CELULAR}")

    # Sensibilidade configurável pelo Administrador Master (Gerenciar Usuários > Config. IA)
    FACE_CONF = float(ia_config.get("face_conf", 0.60))
    PHONE_CONF = float(ia_config.get("phone_conf", 0.60))
    TOLERANCIA_FACIAL = float(ia_config.get("tolerancia_facial", 0.48))
    # Nº de detecções (dentro de uma janela curta) para confirmar um celular antes
    # de gerar o alerta. Permite usar uma confiança de detecção mais alta (menos
    # falso positivo tipo "capa reconhecida como celular") sem perder velocidade:
    # como a checagem roda muitas vezes por segundo, exigir 2 acertos numa janela
    # de ~1,2s custa só uma fração de segundo a mais, não fica lento na prática.
    CONFIRMACOES_CELULAR = max(1, int(float(ia_config.get("confirmacoes_celular", 2))))
    MAX_DIST_CELULAR_PX = float(ia_config.get("max_distancia_px", 800))
    TEMPO_EVASAO = max(1, int(aula_dict.get('tempo_evasao_min') or 15)) * 60
    COOLDOWN_ALERTA_CELULAR = float(ia_config.get("tempo_celular_seg", 4))
    TEMPO_DISTRACAO = float(ia_config.get("tempo_distracao_seg", 5))
    print(f"⚙️ [IA] Sensibilidade efetiva: rosto={FACE_CONF} celular={PHONE_CONF} tolerancia_facial={TOLERANCIA_FACIAL} confirmacoes_celular={CONFIRMACOES_CELULAR}")

    def get_local_db():
        conn = sqlite3.connect(db_path, timeout=20.0)
        conn.execute('PRAGMA journal_mode=WAL;')
        # Em WAL, synchronous=NORMAL não corrompe o banco (só pode perder o último commit numa
        # queda de energia) e elimina o fsync em cada commit — num SD de Raspberry isso custa
        # dezenas de ms por gravação de status.
        conn.execute('PRAGMA synchronous=NORMAL;')
        conn.row_factory = sqlite3.Row
        return conn

    def save_screenshot_robust(filename, frame):
        try:
            full_path = os.path.join(data_screenshots_path, filename)
            cv2.imwrite(full_path, frame)
            print(f"📸 Screenshot salvo com sucesso: {full_path}")
        except Exception as e:
            print(f"❌ Falha ao salvar screenshot ({filename}): {e}")

    def still_running():
        try:
            return shared_state.get("running", True)
        except Exception:
            return False

    # =====================================================================
    # ARQUITETURA EM 2 THREADS — a peça-chave para o reconhecimento
    # INSTANTÂNEO (tipo controlador de acesso facial):
    #
    # Antes, TUDO rodava em sequência num único laço: postura (ação
    # suspeita) + celular + rosto + reconhecimento — um atrás do outro, a
    # cada ciclo. Em hardware real, e principalmente com câmera IP, isso
    # fazia o "tempo real" virar vários segundos de atraso.
    #
    # Agora: uma thread ALIMENTADORA sempre guarda o frame mais recente;
    # a thread de ROSTO roda o mais rápido que o hardware permitir, só
    # cuidando de reconhecer e atualizar status; e a thread de
    # COMPORTAMENTO cuida de celular/ação suspeita num ritmo próprio, sem
    # jamais travar a thread de rosto. As duas rodam ao mesmo tempo de
    # verdade (o GIL do Python é liberado durante as chamadas pesadas do
    # OpenCV/YOLO/dlib, então isso é paralelismo real, não só troca de
    # contexto).
    # =====================================================================
    frame_lock = threading.Lock()
    latest_frame_holder = {'frame': None, 'ver': 0}

    def get_latest_frame_if_new(ultima_versao):
        """Devolve (frame, versao) só se chegou um frame NOVO desde 'ultima_versao'.
        Antes, os 3 laços reprocessavam o mesmo frame várias vezes quando a câmera
        era mais lenta que a IA — CPU gasta à toa, roubada do rosto e do celular."""
        with frame_lock:
            f = latest_frame_holder['frame']
            v = latest_frame_holder['ver']
            if f is None or v == ultima_versao:
                return None, ultima_versao
            return f.copy(), v

    def frame_receiver():
        while still_running():
            try:
                f = frame_queue.get(timeout=0.5)
                with frame_lock:
                    latest_frame_holder['frame'] = f
                    latest_frame_holder['ver'] += 1
            except queue.Empty:
                continue
            except Exception:
                break

    behavior_lock = threading.Lock()
    behavior_state = {'phone_boxes': [], 'looking_down_boxes': []}
    reconhecimento_inicial_resolvido = threading.Event()
    inicio_prioridade_reconhecimento = time.monotonic()
    LIMITE_PRIORIDADE_RECONHECIMENTO_S = 10.0

    def aguardar_primeiro_reconhecimento():
        if reconhecimento_inicial_resolvido.is_set():
            return False
        restante = LIMITE_PRIORIDADE_RECONHECIMENTO_S - (
            time.monotonic() - inicio_prioridade_reconhecimento
        )
        if restante <= 0:
            reconhecimento_inicial_resolvido.set()
            return False
        reconhecimento_inicial_resolvido.wait(timeout=min(restante, 0.05))
        return not reconhecimento_inicial_resolvido.is_set()

    # ---------------------------------------------------------------
    # Prioriza os ciclos de rosto/status e celular. A análise de pose não pode
    # inserir pausas adicionais nesses fluxos críticos; os tempos são registrados
    # no log "🩻 [Pose]" a cada 5s para revelar limites do hardware.
    tempos_ms = {'pose': 0.0, 'rosto': 0.0, 'rosto_ciclo': 0.0, 'celular': 0.0}
    fases_ms = {'pose': None, 'rosto': None, 'celular': None}   # (pré, inferência, pós) em ms

    def inferir(nome, fn):
        t_ini = time.time()
        out = fn()
        dur = (time.time() - t_ini) * 1000.0
        tempos_ms[nome] = dur if tempos_ms[nome] == 0.0 else (0.8 * tempos_ms[nome] + 0.2 * dur)
        # Diagnóstico: separa quanto do tempo é a rede (inferência) e quanto é pré/pós-processamento.
        try:
            res = out[1] if isinstance(out, tuple) else out
            sp = res[0].speed
            amostra = (float(sp.get('preprocess', 0.0)), float(sp.get('inference', 0.0)), float(sp.get('postprocess', 0.0)))
            ant = fases_ms.get(nome)
            fases_ms[nome] = amostra if ant is None else tuple(0.8 * a + 0.2 * b for a, b in zip(ant, amostra))
        except Exception:
            pass
        return out

    def _fmt_fases(nome):
        f = fases_ms.get(nome)
        return "-" if f is None else f"{f[0]:.0f}/{f[1]:.0f}/{f[2]:.0f}"

    # ---------------------------------------------------------------
    # THREAD DE CELULAR: como o rosto, processa cada frame novo sem pausa fixa;
    # a frequência efetiva depende do tempo de inferência do hardware.
    # ---------------------------------------------------------------
    def phone_loop():
        last_alert_celular = 0
        # Timestamps dos últimos frames em que um celular "plausível" apareceu —
        # janela deslizante usada para confirmar antes de disparar o alerta.
        # Confirmação POR OBJETO: as N detecções do painel ("Confirmações p/ Celular") precisam cair
        # no mesmo lugar. Antes, 2 falsos positivos em pontos diferentes da sala já confirmavam.
        # Detecção "forte" (confiança alta) já confirma na hora, sem esperar a 2ª
        # detecção: celular nítido na frente da câmera não precisa de votação.
        # Detecção fraca (perto do limite) continua exigindo CONFIRMACOES_CELULAR.
        PHONE_CONF_FORTE = max(0.55, PHONE_CONF)
        confirmador = PhoneConfirmer(PHONE_CONF_FORTE, CONFIRMACOES_CELULAR, 2.5)
        # Quanto tempo o celular continua "visível" para o status do aluno depois
        # que a rede perde ele por 1-2 frames (evita piscar USANDO CELULAR/FOCADO).
        PHONE_HOLD = 1.0
        ultimas_boxes, ultimo_phone_visto = [], 0.0
        ultima_versao = -1

        # O celular visto de FRENTE (tela virada para a câmera, escura/refletindo luz) é bem
        # pior para o modelo do que visto de costas. Quando a passada normal não acha nada,
        # tentamos UMA variante da imagem por ciclo (rodízio) — nos testes, gama escura e
        # CLAHE recuperaram detecções do celular de frente sem criar falso positivo. A
        # confiança mínima continua sendo a do painel (PHONE_CONF): só muda a imagem, não o limiar.
        def _gama(img, g):
            lut = np.array([((i / 255.0) ** g) * 255 for i in range(256)], dtype=np.uint8)
            return cv2.LUT(img, lut)

        _clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        def _clahe_bgr(img):
            l_, a_, b_ = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2LAB))
            return cv2.cvtColor(cv2.merge([_clahe.apply(l_), a_, b_]), cv2.COLOR_LAB2BGR)

        variantes_celular = [lambda im: _gama(im, 1.4), _clahe_bgr, lambda im: _gama(im, 0.7)]
        idx_variante = 0
        ultima_variante_t = 0.0
        INTERVALO_VARIANTE_S = 4.0   # limita inferências extras quando o modelo é lento no hardware

        def _detectar_celular(img, frame_w, frame_h, frame_area):
            phone_boxes = []
            try:
                results_obj = inferir('celular', lambda: phone_detector(img, classes=[67], conf=PHONE_CONF, imgsz=IMGSZ_CELULAR, verbose=False))
                for r_obj in results_obj:
                    for box in r_obj.boxes:
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        bw, bh = max(1, x2 - x1), max(1, y2 - y1)
                        razao_lados = (bw / bh) if bh > bw else (bh / bw)
                        area_relativa = (bw * bh) / frame_area
                        if razao_lados > 0.95 or area_relativa > 0.60:
                            continue
                        phone_boxes.append({'box': (x1, y1, x2, y2), 'claimed': False, 'conf': float(box.conf[0])})
            except Exception:
                pass
            return phone_boxes

        while still_running():
            if aguardar_primeiro_reconhecimento():
                continue
            frame, ultima_versao = get_latest_frame_if_new(ultima_versao)
            if frame is None:
                time.sleep(0.01)
                continue
            try:
                agora = time.time()
                frame_h, frame_w = frame.shape[:2]
                frame_area = max(1, frame_w * frame_h)
                phone_boxes = _detectar_celular(frame, frame_w, frame_h, frame_area)
                if not phone_boxes and (agora - ultima_variante_t) >= INTERVALO_VARIANTE_S:
                    phone_boxes = _detectar_celular(variantes_celular[idx_variante % len(variantes_celular)](frame), frame_w, frame_h, frame_area)
                    idx_variante += 1
                    ultima_variante_t = time.time()
                if phone_boxes:
                    ultimas_boxes, ultimo_phone_visto = phone_boxes, agora
                    boxes_publicar = phone_boxes
                elif ultimas_boxes and (agora - ultimo_phone_visto) < PHONE_HOLD:
                    boxes_publicar = ultimas_boxes
                else:
                    ultimas_boxes = []
                    boxes_publicar = []

                with behavior_lock:
                    behavior_state['phone_boxes'] = boxes_publicar

                celular_confirmado = confirmador.update(agora, phone_boxes)

                # ALERTA DE CELULAR: dispara para QUALQUER celular CONFIRMADO na
                # sala, sem precisar vincular a um aluno. O professor valida DEPOIS
                # quem estava usando, revendo o vídeo. A confirmação por janela evita
                # que 1 frame ruim (falso positivo isolado) já gere um alerta.
                if celular_confirmado and (agora - last_alert_celular > COOLDOWN_ALERTA_CELULAR):
                    last_alert_celular = agora
                    conn = get_local_db()
                    try:
                        v_filename = f"celular_{int(agora)}.mp4"
                        img_filename = f"celular_img_{int(agora)}.jpg"
                        save_screenshot_robust(img_filename, frame)
                        conn.execute("INSERT INTO alertas (aula_id, mensagem, data_hora, video_file, image_file) VALUES (?, ?, ?, ?, ?)", (aula_dict['id'], "📱 CELULAR DETECTADO na sala — revise o vídeo para identificar o(a) aluno(a).", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), v_filename, img_filename))
                        conn.commit()
                        clip_queue.put(v_filename)
                    finally:
                        conn.close()  # SEMPRE fecha, mesmo se o INSERT falhar — evita vazamento de conexões
            except Exception as e:
                print(f"❌ [Thread Celular] Erro: {e}")
                time.sleep(0.5)

            time.sleep(0.005)

    # ---------------------------------------------------------------
    # THREAD DE AÇÃO SUSPEITA (postura): roda sem pausa fixa, com o modelo de
    # pose NCNN. Detecta "braço levantado" + "movimento rápido do pulso" (balançar,
    # acenar forte, arremessar) próximos no tempo.
    #
    # O que impedia de funcionar (corrigido aqui):
    #  1) Pulsos borrados (braço em movimento) vinham com confiança < 0.5 e o
    #     ultralytics os zerava -> sem pulso, sem velocidade, sem alerta. Agora o
    #     pulso é aceito a partir de 0.20 (ver também core/pose_engine.py).
    #  2) Todas as janelas de tempo eram FIXAS (1,0s / 1,5s) e pensadas para uma
    #     pose rodando a vários FPS. No Raspberry, com 3 modelos disputando a CPU,
    #     a pose roda a ~1-2 FPS: o evento "braço levantado" de um ciclo já tinha
    #     expirado quando chegava o "movimento rápido" do ciclo seguinte, e
    #     confirmação em múltiplos ciclos era impossível dentro de uma janela curta.
    #     Agora o gesto é validado pela sequência temporal dos pulsos.
    #  3) A velocidade em px/s ficava baixa quando o intervalo entre dois ciclos
    #     era grande (um aceno vai e volta entre as duas amostras). Agora um
    #     deslocamento grande entre amostras distantes também conta.
    #  4) O aviso "AÇÃO SUSPEITA" na tela durava um único ciclo (piscava).
    #  5) Erros na leitura da pose eram engolidos em silêncio (except: pass).
    #     Agora aparecem no log, junto com um resumo a cada 5s do que a pose vê.
    # ---------------------------------------------------------------
    def suspicious_loop():
        last_alert_suspeita = 0
        # O rastreio de pessoas/pulsos e a classificação do gesto agora vivem em core/behavior_engine.py (GestureTracker).
        # Amostras e limiares do gesto são relativos à largura dos ombros de cada pessoa (independe da distância da câmera).
        MIN_CONF_PULSO = 0.20        # pulsos borrados/pequenos precisam sobreviver ao filtro de confiança
        MIN_CONF_TRONCO = 0.50       # nariz/ombros: mantém o critério usado pelo "distraído"
        sensibilidade_acao = max(1, min(10, int(float(ia_config.get("acao_sensibilidade", 10)))))
        gesto_tracker = GestureTracker(MIN_CONF_PULSO, MIN_CONF_TRONCO, sensibilidade_acao)
        proxima_leitura_sensibilidade = 0.0
        motivo_suspeita = ''
        ciclo_medio = 0.5            # segundos entre ciclos da pose (média móvel)
        ultimo_ciclo_t = None
        ultima_versao = -1
        suspeita_ate = 0.0
        ultimo_erro_log = 0.0
        diag = {'t': time.time(), 'ciclos': 0, 'pessoas': 0, 'pulsos': 0, 'levantado': 0, 'rapido': 0, 'pedidos': 0, 'vmax': 0.0}

        def _ponto(pk, idx, min_conf):
            """(x, y) do keypoint se confiável, senão (0.0, 0.0)."""
            x, y = float(pk[idx][0]), float(pk[idx][1])
            c = float(pk[idx][2]) if len(pk[idx]) > 2 else 1.0
            if c >= min_conf and x > 0 and y > 0:
                return (x, y)
            return (0.0, 0.0)

        while still_running():
            if aguardar_primeiro_reconhecimento():
                continue
            frame, ultima_versao = get_latest_frame_if_new(ultima_versao)
            if frame is None:
                time.sleep(0.01)
                continue
            try:
                agora = time.time()
                if ultimo_ciclo_t is not None:
                    ciclo_medio = 0.8 * ciclo_medio + 0.2 * min(agora - ultimo_ciclo_t, 3.0)
                ultimo_ciclo_t = agora
                JANELA_COMBINACAO = max(1.0, 2.5 * ciclo_medio)
                if agora >= proxima_leitura_sensibilidade:
                    proxima_leitura_sensibilidade = agora + 2.0
                    conn_cfg = None
                    try:
                        conn_cfg = sqlite3.connect(db_path, timeout=2.0)
                        conn_cfg.row_factory = sqlite3.Row
                        row_cfg = conn_cfg.execute(
                            "SELECT valor FROM configuracoes_ia WHERE chave = 'acao_sensibilidade'"
                        ).fetchone()
                        if row_cfg is not None:
                            novo_nivel = int(float(row_cfg['valor']))
                            if 1 <= novo_nivel <= 10 and novo_nivel != gesto_tracker.sensibilidade:
                                gesto_tracker.set_sensitivity(novo_nivel)
                                print(f"⚙️ [Ação Suspeita] Sensibilidade atualizada pelo painel: {novo_nivel}/10")
                    except (TypeError, ValueError, sqlite3.Error) as e_cfg:
                        if agora - ultimo_erro_log > 5:
                            ultimo_erro_log = agora
                            print(f"⚠️ [Ação Suspeita] Não foi possível atualizar a sensibilidade salva: {e_cfg}")
                    finally:
                        if conn_cfg is not None:
                            conn_cfg.close()
                is_suspeita_now = False

                _, pose_results = inferir('pose', lambda: pose_engine.analyze_frame(frame))
                looking_down_boxes = []
                persons = []
                try:
                    for r_pose in pose_results:
                        kps = getattr(r_pose, 'keypoints', None)
                        if kps is None or getattr(kps, 'data', None) is None:
                            continue
                        kd = kps.data
                        kd = kd.cpu().numpy() if hasattr(kd, 'cpu') else np.asarray(kd)
                        for pk in kd:
                            if len(pk) <= 10:
                                continue
                            nose = _ponto(pk, 0, MIN_CONF_TRONCO)
                            l_shoulder = _ponto(pk, 5, MIN_CONF_TRONCO); r_shoulder = _ponto(pk, 6, MIN_CONF_TRONCO)
                            persons.append({
                                'wrists': (tuple(float(v) for v in pk[9][:3]), tuple(float(v) for v in pk[10][:3])),
                                'elbows': (tuple(float(v) for v in pk[7][:3]), tuple(float(v) for v in pk[8][:3])),
                                'shoulders': (tuple(float(v) for v in pk[5][:3]), tuple(float(v) for v in pk[6][:3])),
                            })
                            if nose[1] > 0 and l_shoulder[1] > 0 and r_shoulder[1] > 0:
                                shoulder_avg_y = (l_shoulder[1] + r_shoulder[1]) / 2
                                if nose[1] > (shoulder_avg_y + 10):
                                    looking_down_boxes.append((int(nose[0]), int(nose[1])))
                except Exception as e_pose:
                    if agora - ultimo_erro_log > 5:
                        ultimo_erro_log = agora
                        print(f"⚠️ [Thread Ação Suspeita] Falha ao ler keypoints da pose: {e_pose}")

                # Classificação temporal do gesto (ver core/behavior_engine.py): uma subida rápida
                # precisa ser seguida por outro movimento rápido; levantar e manter parado não basta.
                suspeito_ciclo, d_gesto = gesto_tracker.update(agora, persons, max(2.5, 4.0 * ciclo_medio), ciclo_medio)
                diag['pulsos'] += d_gesto['pulsos']; diag['levantado'] += d_gesto['levantado']
                diag['rapido'] += d_gesto['golpes']; diag['pedidos'] += d_gesto['pedidos']
                vmax_ciclo = d_gesto['vmax']
                if suspeito_ciclo:
                    motivo_suspeita = d_gesto['motivo']
                    is_suspeita_now = True

                if is_suspeita_now and (agora - last_alert_suspeita > 8):
                    last_alert_suspeita = agora
                    print(f"🙌 [Ação Suspeita] DETECTADA ({motivo_suspeita}) — gerando alerta e clipe.")
                    conn = get_local_db()
                    try:
                        v_filename = f"suspeita_{int(agora)}.mp4"
                        img_filename = f"suspeita_img_{int(agora)}.jpg"
                        save_screenshot_robust(img_filename, frame)
                        conn.execute("INSERT INTO alertas (aula_id, mensagem, data_hora, video_file, image_file) VALUES (?, ?, ?, ?, ?)", (aula_dict['id'], f"🙌 AÇÃO SUSPEITA: Movimento brusco de braço detectado na sala ({motivo_suspeita}) — revise o vídeo.", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), v_filename, img_filename))
                        conn.commit()
                        clip_queue.put(v_filename)
                    finally:
                        conn.close()  # SEMPRE fecha, mesmo se o INSERT falhar — evita vazamento de conexões

                with behavior_lock:
                    behavior_state['looking_down_boxes'] = looking_down_boxes

                # Mantém o aviso na tela por 3s (antes durava um único ciclo e piscava).
                if is_suspeita_now:
                    suspeita_ate = agora + 3.0
                shared_state["fighting"] = agora < suspeita_ate

                # Resumo de diagnóstico a cada 5s: mostra o que a pose está enxergando.
                diag['ciclos'] += 1
                diag['pessoas'] += len(persons)
                diag['vmax'] = max(diag['vmax'], vmax_ciclo)
                if agora - diag['t'] >= 5.0:
                    dur = agora - diag['t']
                    print(f"🩻 [Pose] {diag['ciclos'] / dur:.1f} ciclos/s | pessoas/ciclo={diag['pessoas'] / max(1, diag['ciclos']):.1f} | pulsos visíveis={diag['pulsos']} | braço levantado={diag['levantado']} | golpes bruscos={diag['rapido']} | pedidos de fala={diag['pedidos']} | vel. máx={diag['vmax']:.1f} LO/s | ms/inferência: pose={tempos_ms['pose']:.0f} rosto={tempos_ms['rosto']:.0f} celular={tempos_ms['celular']:.0f} | ciclo rosto={tempos_ms['rosto_ciclo']:.0f}ms | janela={JANELA_COMBINACAO:.1f}s | fases pré/inf/pós ms: pose={_fmt_fases('pose')} rosto={_fmt_fases('rosto')} celular={_fmt_fases('celular')}")
                    diag.update({'t': agora, 'ciclos': 0, 'pessoas': 0, 'pulsos': 0, 'levantado': 0, 'rapido': 0, 'pedidos': 0, 'vmax': 0.0})
            except Exception as e:
                print(f"❌ [Thread Ação Suspeita] Erro: {e}")
                time.sleep(0.5)

            time.sleep(0.005)

    # ---------------------------------------------------------------
    # THREAD DE ROSTO — o "coração" do sistema. Roda sem pausa fixa,
    # processando de novo assim que um novo frame chega. É aqui que mora
    # o requisito de reconhecer e atualizar o status instantaneamente.
    # ---------------------------------------------------------------
    def face_loop():
        active_students = {}
        aluno_ultimo_visto = {}
        aluno_alertado_sumico = set()
        distracao_timers = {}
        aluno_alertado_distracao = set()
        last_distracao_time = {}

        # -----------------------------------------------------------------
        # RASTREIO DE ROSTOS (a mudança que acelera tudo)
        #
        # Antes: a CADA ciclo, para CADA rosto, rodava o encoding do dlib
        # (a parte mais pesada, ~0.2-0.5s por rosto num Raspberry Pi) e só então
        # atualizava status/bancada. Resultado: o status só mudava depois de
        # um ciclo lento inteiro — e, com nenhum rosto na tela, ainda rodava o
        # detector HOG do dlib em TODO ciclo.
        #
        # Agora: a detecção (YOLO, rápida) roda todo ciclo e cada rosto vira um
        # "track" que acompanha a pessoa de um frame pro outro. O reconhecimento
        # (dlib, lento) roda só quando o track é NOVO ou de tempos em tempos para
        # reconferir — a identidade fica em cache no track. Assim posição/bancada
        # e status atualizam na velocidade da detecção, não do reconhecimento.
        # -----------------------------------------------------------------
        tracks = {}
        next_track_id = 0
        MIN_TRACK_GRACE_S = 1.5
        MAX_TRACK_GRACE_S = 15.0
        MIN_FACE_PX = 40              # rosto menor que isso não rende encoding confiável
        INTERVALO_RETRY_NOVO = 0.15   # s entre tentativas p/ rosto ainda sem identidade
        INTERVALO_RETRY_DESCONHECIDO = 1.0
        INTERVALO_FALLBACK_HOG = 1.5  # s mínimos entre execuções do HOG (só se o YOLO não achar nada)
        INTERVALO_REVERIFICA_HANDOFF = 15.0 - 0.3  # = INTERVALO_REVERIFICA do behavior_engine menos 0,3s
        HOLD_DISTRACAO = 2.0          # s que "DISTRAÍDO" permanece após levantar a cabeça (era 5)
        HOLD_NAO_VISIVEL_S = 0.7      # s que um rosto que sumiu do frame ainda conta como "presente" (antes: toda a tolerância do track, 1,5s a 15s)
        HANDOFF_IDENTIDADE_S = 2.5    # s: rosto novo herda a identidade de um track conhecido que sumiu há pouco (troca de bancada)
        ultimo_fallback_hog = 0.0
        reconhecimento_jobs = queue.Queue(maxsize=4)
        reconhecimento_resultados = queue.Queue()
        reconhecimentos_pendentes = set()
        ultimo_erro_reconhecimento = 0.0
        reconhecimento_inicial_logado = False
        localizador_bancada = LocalizadorBancada(zonas)  # histerese: rosto na borda não pisca entre bancadas
        circulacao_tracker = CirculacaoTracker()
        ultimo_gravado = {}           # matricula -> (status, bancada, t) p/ não regravar sem mudança
        ultimo_erro_detector_rosto = 0.0
        ultima_versao = -1

        def recognition_worker():
            while still_running():
                try:
                    tid, rgb_frame, box, center, size = reconhecimento_jobs.get(timeout=0.5)
                except queue.Empty:
                    continue
                try:
                    inicio_encoding = time.perf_counter()
                    encodings = face_recognition.face_encodings(rgb_frame, [box])
                    encoding_ms = (time.perf_counter() - inicio_encoding) * 1000.0
                    detalhes = (
                        face_engine.find_best_match_details(encodings[0], tolerance=TOLERANCIA_FACIAL)
                        if encodings else None
                    )
                    reconhecimento_resultados.put((tid, detalhes, center, size, encoding_ms, None))
                except Exception as e:
                    reconhecimento_resultados.put((tid, None, center, size, 0.0, str(e)))
                finally:
                    reconhecimento_jobs.task_done()

        threading.Thread(target=recognition_worker, daemon=True, name="face-recognition").start()

        def aplicar_resultado_reconhecimento(item, agora):
            nonlocal ultimo_erro_reconhecimento, reconhecimento_inicial_logado
            tid, detalhes, origem, tamanho_origem, encoding_ms, erro = item
            reconhecimentos_pendentes.discard(tid)
            tr = tracks.get(tid)
            if erro:
                reconhecimento_inicial_resolvido.set()
                if agora - ultimo_erro_reconhecimento >= 5.0:
                    ultimo_erro_reconhecimento = agora
                    print(f"⚠️ [FaceEngine] Falha ao codificar rosto: {erro}")
                return
            if tr is None or detalhes is None:
                reconhecimento_inicial_resolvido.set()
                return

            deslocamento = math.hypot(
                tr['center'][0] - origem[0], tr['center'][1] - origem[1]
            )
            if deslocamento > max(3.0 * float(tamanho_origem), 150.0):
                return

            resultado, _, _, _ = detalhes
            tr['tentativas'] = (tr.get('tentativas', 0) + 1) if resultado == "Desconhecido" else 0
            if resultado == tr['matricula']:
                tr['fails'] = 0
            elif resultado != "Desconhecido":
                for otid, otr in tracks.items():
                    if otid != tid and otr['matricula'] == resultado:
                        otr['matricula'], otr['last_rec'], otr['fails'] = None, 0.0, 0
                tr['matricula'], tr['fails'] = resultado, 0
                if not reconhecimento_inicial_logado:
                    reconhecimento_inicial_logado = True
                    duracao_inicial = time.time() - tr.get('created_at', agora)
                    print(
                        f"⏱️ [Reconhecimento Inicial] identidade em {duracao_inicial:.2f}s "
                        f"(encoder={encoding_ms:.0f}ms, detector={tempos_ms['rosto']:.0f}ms)."
                    )
            elif tr['matricula'] is None:
                tr['matricula'] = "Desconhecido"
            else:
                tr['fails'] += 1
                if tr['fails'] >= 3:
                    tr['matricula'], tr['fails'] = "Desconhecido", 0

            if resultado != "Desconhecido":
                reconhecimento_inicial_resolvido.set()

        while still_running():
            frame, ultima_versao = get_latest_frame_if_new(ultima_versao)
            if frame is None:
                time.sleep(0.01)
                continue
            inicio_ciclo_rosto = time.perf_counter()
            conn = None
            try:
                agora = time.time()
                conn = get_local_db()
                live_status_dict = {}
                live_bancada_dict = {}

                with behavior_lock:
                    # cópia dos dicts: o 'claimed_by' é por ciclo e não pode vazar entre ciclos
                    phone_boxes = [dict(p) for p in behavior_state['phone_boxes']]
                    for p in phone_boxes:
                        p.pop('claimed_by', None)
                    looking_down_boxes = list(behavior_state['looking_down_boxes'])

                frame_h, frame_w = frame.shape[:2]

                # 1) DETECÇÃO (rápida) — todo ciclo
                face_boxes = []
                try:
                    res_faces = inferir('rosto', lambda: face_detector(frame, conf=FACE_CONF, imgsz=IMGSZ_ROSTO, verbose=False))
                    for r_face in res_faces:
                        for box in r_face.boxes:
                            fx1, fy1, fx2, fy2 = box.xyxy[0]
                            l_, t_ = max(0, int(fx1)), max(0, int(fy1))
                            r_, b_ = min(frame_w, int(fx2)), min(frame_h, int(fy2))
                            if r_ > l_ and b_ > t_:
                                face_boxes.append((t_, r_, b_, l_))
                except Exception as e:
                    if agora - ultimo_erro_detector_rosto >= 5.0:
                        ultimo_erro_detector_rosto = agora
                        print(f"⚠️ [Thread Rosto] Falha na detecção facial: {e}")

                # Fallback HOG: só se o YOLO não achou NADA, no máximo a cada 1,5s e em
                # meia resolução. Antes rodava em resolução cheia em TODO ciclo sem rosto
                # (ex.: aluno saiu da câmera) — exatamente quando mais precisava ser rápido.
                if not face_boxes and (agora - ultimo_fallback_hog) >= INTERVALO_FALLBACK_HOG:
                    ultimo_fallback_hog = agora
                    try:
                        small_rgb = cv2.cvtColor(cv2.resize(frame, (0, 0), fx=0.5, fy=0.5), cv2.COLOR_BGR2RGB)
                        for (t_, r_, b_, l_) in face_recognition.face_locations(small_rgb):
                            face_boxes.append((t_ * 2, r_ * 2, b_ * 2, l_ * 2))
                    except Exception:
                        pass

                # 2) RASTREIO — casa cada detecção com o track mais próximo
                deteccoes = []
                for (t_, r_, b_, l_) in face_boxes:
                    deteccoes.append({'box': (t_, r_, b_, l_), 'center': ((l_ + r_) // 2, (t_ + b_) // 2), 'size': max(r_ - l_, b_ - t_)})
                pares = []
                for di, d in enumerate(deteccoes):
                    for tid, tr in tracks.items():
                        dist = math.hypot(d['center'][0] - tr['center'][0], d['center'][1] - tr['center'][1])
                        # Quem está andando para outra bancada se desloca mais que 1 rosto entre
                        # dois frames; o raio cresce com o tempo sem ver o track (até 3x).
                        dt_sem_ver = min(max(agora - tr['last_seen'], 0.0), 0.6)
                        vx, vy = tr.get('vel', (0.0, 0.0))
                        # Posição prevista: continua andando na mesma direção. Assim o track
                        # ACOMPANHA o aluno que se mexe rápido em vez de perdê-lo (e perder o nome).
                        dist_prev = math.hypot(d['center'][0] - (tr['center'][0] + vx * dt_sem_ver),
                                               d['center'][1] - (tr['center'][1] + vy * dt_sem_ver))
                        dist = min(dist, dist_prev)
                        raio = max(d['size'], tr['size']) * (1.0 + 2.0 * min(max(agora - tr['last_seen'], 0.0), 1.0))
                        if dist <= raio:
                            pares.append((dist, di, tid))
                pares.sort()
                det_usada, trk_usada, visiveis = set(), set(), set()
                for dist, di, tid in pares:
                    if di in det_usada or tid in trk_usada:
                        continue
                    det_usada.add(di); trk_usada.add(tid); visiveis.add(tid)
                    d, tr = deteccoes[di], tracks[tid]
                    dt_obs = agora - tr['last_seen']
                    if dt_obs > 0.02:
                        nvx = (d['center'][0] - tr['center'][0]) / dt_obs
                        nvy = (d['center'][1] - tr['center'][1]) / dt_obs
                        ovx, ovy = tr.get('vel', (0.0, 0.0))
                        tr['vel'] = (0.5 * ovx + 0.5 * nvx, 0.5 * ovy + 0.5 * nvy)
                    tr['box'], tr['center'], tr['size'], tr['last_seen'] = d['box'], d['center'], d['size'], agora
                for di, d in enumerate(deteccoes):
                    if di in det_usada:
                        continue
                    novo = {'box': d['box'], 'center': d['center'], 'size': d['size'], 'last_seen': agora,
                            'matricula': None, 'last_rec': 0.0, 'fails': 0, 'created_at': agora}
                    # HANDOFF DE IDENTIDADE: se alguém já identificado sumiu do frame há instantes
                    # perto daqui (trocou de bancada / virou o rosto), o rosto novo herda o nome
                    # NA HORA — sem esperar o encoding do dlib — e é reconferido logo em seguida.
                    # Antes, o status só acompanhava depois de reconhecer de novo (~0,3-1s+).
                    melhor_antigo, melhor_dist = None, None
                    for otid, otr in tracks.items():
                        if otid in trk_usada or otid in visiveis:
                            continue
                        if otr['matricula'] in (None, "Desconhecido"):
                            continue
                        if (agora - otr['last_seen']) > HANDOFF_IDENTIDADE_S:
                            continue
                        dist_o = math.hypot(d['center'][0] - otr['center'][0], d['center'][1] - otr['center'][1])
                        if dist_o <= 10.0 * max(d['size'], otr['size']) and (melhor_dist is None or dist_o < melhor_dist):
                            melhor_antigo, melhor_dist = otid, dist_o
                    if melhor_antigo is not None:
                        novo['matricula'] = tracks[melhor_antigo]['matricula']
                        novo['last_rec'] = agora - INTERVALO_REVERIFICA_HANDOFF  # reconfere em ~0,3s
                        novo['banc'] = tracks[melhor_antigo].get('banc', {})
                        del tracks[melhor_antigo]
                    tracks[next_track_id] = novo
                    visiveis.add(next_track_id)
                    next_track_id += 1
                track_grace = min(
                    MAX_TRACK_GRACE_S,
                    max(MIN_TRACK_GRACE_S, tempos_ms['rosto_ciclo'] / 1000.0 + 0.75),
                )
                for tid in [k for k, tr in tracks.items() if (agora - tr['last_seen']) > track_grace]:
                    del tracks[tid]

                # Apply any completed encodings before starting another expensive
                # detector pass; otherwise the first name waits through two inferences.
                while True:
                    try:
                        item = reconhecimento_resultados.get_nowait()
                    except queue.Empty:
                        break
                    aplicar_resultado_reconhecimento(item, time.time())

                # O encoding dlib é executado em segundo plano para não segurar atualização
                # de bancada/status dos alunos que já têm identidade.
                candidatos = [
                    tid for tid in selecionar_reconhecimentos(tracks, visiveis, agora, MIN_FACE_PX)
                    if tid not in reconhecimentos_pendentes
                ]
                if candidatos:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    for tid in candidatos:
                        tr = tracks[tid]
                        try:
                            reconhecimento_jobs.put_nowait(
                                (tid, rgb_frame, tr['box'], tr['center'], tr['size'])
                            )
                        except queue.Full:
                            break
                        tr['last_rec'] = agora
                        reconhecimentos_pendentes.add(tid)

                # At startup, wait briefly for the first descriptor so its result
                # reaches the live overlay without waiting for another face inference.
                if candidatos and not reconhecimento_inicial_resolvido.is_set():
                    try:
                        item = reconhecimento_resultados.get(timeout=1.0)
                    except queue.Empty:
                        pass
                    else:
                        aplicar_resultado_reconhecimento(item, time.time())

                # Tolerância de rosto que sumiu: ~1,5 ciclo (cobre UM frame sem detecção) limitada a
                # [0,3s ; HOLD_NAO_VISIVEL_S]. Com ciclo rápido o status acompanha mais depressa.
                hold_nao_visivel = max(0.3, min(HOLD_NAO_VISIVEL_S, 1.5 * tempos_ms['rosto_ciclo'] / 1000.0))
                detected_students_info = []
                for tid, tr in tracks.items():
                    detected_students_info.append({'matricula': tr['matricula'] or "Desconhecido", 'box': tr['box'],
                                                   'center': tr['center'], 'visivel': tid in visiveis,
                                                   'size': tr['size'], 'estado_banc': tr.setdefault('banc', {}),
                                                   # "ativo" = visível agora ou sumiu há menos de HOLD_NAO_VISIVEL_S.
                                                   # Track antigo (sumiu há mais que isso) NÃO define mais status/bancada:
                                                   # antes ele segurava "FOCADO na bancada antiga" por 1,5s a 15s.
                                                   'ativo': (tid in visiveis) or ((agora - tr['last_seen']) <= hold_nao_visivel)})

                for p in phone_boxes:
                    px1, py1, px2, py2 = p['box']
                    pcx, pcy = (px1 + px2) // 2, (py1 + py2) // 2
                    min_dist = float('inf'); closest_mat = None
                    for st in detected_students_info:
                        dist = math.hypot(pcx - st['center'][0], pcy - st['center'][1])
                        if dist < min_dist:
                            min_dist = dist; closest_mat = st['matricula']
                    if closest_mat and min_dist < MAX_DIST_CELULAR_PX:
                        p['claimed_by'] = closest_mat; p['claimed'] = True

                current_faces = []
                for st in detected_students_info:
                    if not st['ativo']:
                        continue
                    matricula_aluno = st['matricula']
                    top, right, bottom, left = st['box']
                    cx, cy = st['center']

                    bancada_atual = localizador_bancada.localizar(st['estado_banc'], cx, cy, st['size'], agora)

                    if matricula_aluno != "Desconhecido" and matricula_aluno in alunos_dict:
                        live_bancada_dict[matricula_aluno] = bancada_atual
                        aluno_ultimo_visto[matricula_aluno] = agora
                        circulando = circulacao_tracker.update(
                            matricula_aluno, bancada_atual == "Área Livre", cx, cy, st['size'], agora
                        )
                        if matricula_aluno in aluno_alertado_sumico: aluno_alertado_sumico.remove(matricula_aluno)
                        if matricula_aluno not in active_students or (agora - active_students[matricula_aluno] > 10):
                            conn.execute("INSERT INTO presenca (aluno_id, aula_id, bancada_atual, data_hora, tipo) VALUES (?, ?, ?, ?, 'PRESENTE')", (matricula_aluno, aula_dict['id'], bancada_atual, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                            conn.commit()
                            active_students[matricula_aluno] = agora

                    is_holding_phone = any(p.get('claimed_by') == matricula_aluno for p in phone_boxes)
                    is_head_down = any(math.hypot(nx - cx, ny - cy) < 150 for nx, ny in looking_down_boxes)

                    if matricula_aluno != "Desconhecido" and matricula_aluno in alunos_dict:
                        if is_holding_phone:
                            live_status_dict[matricula_aluno] = "USANDO CELULAR"
                            if matricula_aluno in distracao_timers: del distracao_timers[matricula_aluno]
                            if matricula_aluno in aluno_alertado_distracao: aluno_alertado_distracao.remove(matricula_aluno)
                        elif bancada_atual == "Área Livre":
                            live_status_dict[matricula_aluno] = (
                                "CIRCULANDO (Fora da bancada)" if circulando else "FORA DA BANCADA"
                            )
                            if matricula_aluno in distracao_timers: del distracao_timers[matricula_aluno]
                            if matricula_aluno in aluno_alertado_distracao: aluno_alertado_distracao.remove(matricula_aluno)
                        elif is_head_down:
                            live_status_dict[matricula_aluno] = "DISTRAÍDO"
                            last_distracao_time[matricula_aluno] = agora
                            if matricula_aluno not in distracao_timers: distracao_timers[matricula_aluno] = agora
                            elif agora - distracao_timers[matricula_aluno] > TEMPO_DISTRACAO:
                                if matricula_aluno not in aluno_alertado_distracao:
                                    conn.execute("INSERT INTO alertas (aula_id, mensagem, data_hora) VALUES (?, ?, ?)", (aula_dict['id'], f"😴 DISPERSÃO: Aluno(a) {alunos_dict[matricula_aluno]} desatento na aula.", datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                                    conn.commit()
                                    aluno_alertado_distracao.add(matricula_aluno)
                        else:
                            bancada_certa = assentos_aula.get(matricula_aluno)
                            fora_do_lugar = bool(bancada_certa and bancada_atual != "Área Livre" and bancada_atual != bancada_certa)
                            # O "resto" de distração (debounce) não pode mais esconder uma troca de
                            # bancada: antes, quem levantava a cabeça e mudava de lugar ficava até 5s
                            # como DISTRAÍDO antes de aparecer BANCADA ERRADA.
                            if not fora_do_lugar and (agora - last_distracao_time.get(matricula_aluno, 0)) < HOLD_DISTRACAO:
                                is_head_down = True; live_status_dict[matricula_aluno] = "DISTRAÍDO"
                            else:
                                if fora_do_lugar:
                                    live_status_dict[matricula_aluno] = f"BANCADA ERRADA (está em {bancada_atual})"
                                else:
                                    live_status_dict[matricula_aluno] = "FOCADO"
                                if matricula_aluno in distracao_timers: del distracao_timers[matricula_aluno]
                                if matricula_aluno in aluno_alertado_distracao: aluno_alertado_distracao.remove(matricula_aluno)

                    cor = (0, 0, 255) if is_holding_phone else ((0, 200, 255) if is_head_down else (66, 193, 122))
                    if bancada_atual == "Área Livre" or (assentos_aula.get(matricula_aluno) and bancada_atual != assentos_aula.get(matricula_aluno)): cor = (0, 200, 255)

                    nome_exibicao = alunos_dict.get(matricula_aluno, "Desconhecido")
                    if nome_exibicao != "Desconhecido" and st.get('visivel', True):
                        status_tela = "Celular" if is_holding_phone else ("Distraido" if is_head_down else "Focado")
                        current_faces.append((top, right, bottom, left, nome_exibicao, bancada_atual, cor, status_tela))

                # Atualiza a tela (quadrados/nomes) ANTES de qualquer escrita no banco:
                # assim, mesmo que a gravação do status abaixo falhe por algum motivo,
                # o vídeo ao vivo continua sendo desenhado normalmente — antes, um erro
                # de banco aqui fazia o reconhecimento "sumir" da tela inteira.
                shared_state["faces"] = current_faces

                for mat, ult_visto in list(aluno_ultimo_visto.items()):
                    if agora - ult_visto > TEMPO_EVASAO:
                        if mat not in aluno_alertado_sumico:
                            conn.execute("INSERT INTO alertas (aula_id, mensagem, data_hora) VALUES (?, ?, ?)", (aula_dict['id'], f"🏃 EVASÃO: Aluno(a) {alunos_dict.get(mat, mat)} sumiu da câmera há {TEMPO_EVASAO // 60} minutos.", datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                            conn.commit()
                            aluno_alertado_sumico.add(mat)

                # Dupla verificação pela bancada: aluno com assento definido que
                # NÃO foi detectado neste ciclo não pode continuar "FOCADO"
                # indefinidamente — vira Disperso (dentro do tempo de evasão) ou
                # Evasão (fora do tempo).
                detectados_agora = {st['matricula'] for st in detected_students_info if st['ativo'] and st['matricula'] != "Desconhecido"}
                for mat_assento in set(assentos_aula.keys()) | set(aluno_ultimo_visto.keys()):
                    if mat_assento in alunos_dict and mat_assento not in detectados_agora:
                        ult_visto = aluno_ultimo_visto.get(mat_assento)
                        if ult_visto is None:
                            continue
                        if agora - ult_visto > TEMPO_EVASAO:
                            live_status_dict[mat_assento] = "EVASÃO"
                            live_bancada_dict[mat_assento] = "Área Livre"
                        else:
                            live_status_dict[mat_assento] = "DISPERSO (Fora da bancada)"
                            live_bancada_dict[mat_assento] = "Área Livre"

                if len(live_status_dict) > 0:
                    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    gravou = False
                    for mat, st in live_status_dict.items():
                        banc = live_bancada_dict.get(mat, "Área Livre")
                        ant = ultimo_gravado.get(mat)
                        if ant and ant[0] == st and ant[1] == banc and (agora - ant[2]) < 2.0:
                            continue  # nada mudou: não precisa regravar a cada ciclo
                        conn.execute("INSERT OR REPLACE INTO live_status (aluno_matricula, aula_id, status_atencao, atualizado_em, bancada_atual) VALUES (?, ?, ?, ?, ?)", (mat, aula_dict['id'], st, now_str, banc))
                        ultimo_gravado[mat] = (st, banc, agora)
                        gravou = True
                    if gravou:
                        conn.commit()
            except Exception as e:
                print(f"❌ [Thread Rosto] Erro: {e}")
                time.sleep(0.3)
            finally:
                # SEMPRE fecha a conexão, mesmo se algo acima falhar — antes, uma
                # exceção (ex: coluna faltando no banco) pulava o conn.close() e
                # vazava UMA conexão SQLite por ciclo, várias vezes por segundo,
                # até esgotar os recursos do processo e derrubá-lo (crash).
                try:
                    if conn is not None:
                        conn.close()
                except Exception:
                    pass
                tempos_ms['rosto_ciclo'] = (time.perf_counter() - inicio_ciclo_rosto) * 1000.0

            time.sleep(0.005)  # cede a CPU por um instante só; roda de novo assim que possível

    t_frame = threading.Thread(target=frame_receiver, daemon=True)
    t_phone = threading.Thread(target=phone_loop, daemon=True)
    t_suspeita = threading.Thread(target=suspicious_loop, daemon=True)
    t_frame.start()
    t_phone.start()
    t_suspeita.start()

    face_loop()  # roda na thread principal deste processo — mantém o processo vivo

@app.get("/map_feed/{sala_id}")
def map_feed(sala_id: int, token: str):
    # Corrige controle de acesso quebrado: token era opcional (default None) — bastava
    # NÃO enviar o parâmetro "token" na URL para pular a validação por completo.
    try:
        jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except Exception:
        return Response(status_code=401)
    return StreamingResponse(gerar_frames_mapeamento(sala_id), media_type="multipart/x-mixed-replace; boundary=frame")


_video_feed_sessions = {}
_video_feed_sessions_lock = threading.Lock()


class VideoFeedSession:
    def __init__(self, usuario_id):
        self.usuario_id = usuario_id
        self.stop_event = threading.Event()
        self.condition = threading.Condition()
        self.sequence = 0
        self.latest_chunk = None
        self.clients = 0
        self.closed = False
        self.thread = threading.Thread(target=self._produce, daemon=True)
        self.thread.start()

    def _produce(self):
        try:
            for chunk in gerar_frames_camera(self.usuario_id, self.stop_event):
                with self.condition:
                    self.latest_chunk = chunk
                    self.sequence += 1
                    self.condition.notify_all()
                if self.stop_event.is_set():
                    break
        except Exception as e:
            print(f"❌ [Video Feed] Erro na sessão do usuário {self.usuario_id}: {e}")
        finally:
            with self.condition:
                self.closed = True
                self.condition.notify_all()
            with _video_feed_sessions_lock:
                if _video_feed_sessions.get(self.usuario_id) is self:
                    del _video_feed_sessions[self.usuario_id]

    def add_client(self):
        with self.condition:
            self.clients += 1

    def remove_client(self):
        with self.condition:
            self.clients = max(0, self.clients - 1)
            if self.clients == 0:
                self.stop_event.set()
                self.condition.notify_all()

    def stream(self):
        seen_sequence = -1
        try:
            while True:
                with self.condition:
                    while seen_sequence == self.sequence and not self.closed:
                        self.condition.wait(timeout=0.5)
                    if seen_sequence == self.sequence and self.closed:
                        return
                    chunk = self.latest_chunk
                    seen_sequence = self.sequence
                if chunk is not None:
                    yield chunk
        finally:
            self.remove_client()


def obter_sessao_video(usuario_id):
    with _video_feed_sessions_lock:
        session = _video_feed_sessions.get(usuario_id)
        if session is None or session.stop_event.is_set() or session.closed:
            session = VideoFeedSession(usuario_id)
            _video_feed_sessions[usuario_id] = session
        session.add_client()
        return session


def gerar_frames_camera(usuario_id: int, stop_event=None):
    conn = get_db(); cursor = conn.cursor()
    cursor.execute('SELECT s.camera_url, a.materia_id, a.sala_id, a.id, m.tempo_evasao_min FROM aulas a JOIN salas s ON a.sala_id = s.id JOIN materias m ON a.materia_id = m.id WHERE a.status = "EM_ANDAMENTO" AND a.usuario_id = ? ORDER BY a.id DESC LIMIT 1', (usuario_id,))
    aula = cursor.fetchone()
    if not aula: conn.close(); return

    # Configurações de sensibilidade da IA definidas pelo Administrador Master
    cursor.execute('SELECT chave, valor FROM configuracoes_ia')
    ia_config = dict(CONFIG_IA_DEFAULTS)
    for row in cursor.fetchall():
        ia_config[row['chave']] = row['valor']

    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(blank_frame, "INICIANDO IA...", (190, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,200,255), 2)
    ret, buffer = cv2.imencode('.jpg', blank_frame)
    yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap = RTSPCamera(aula['camera_url'])
    
    cursor.execute('SELECT nome_bancada, coordenadas_json FROM zonas_sala WHERE sala_id = ?', (aula['sala_id'],))
    zonas = [{"nome": r['nome_bancada'], "pontos": json.loads(r['coordenadas_json'])} for r in cursor.fetchall()]
    cursor.execute('SELECT matricula, nome FROM alunos WHERE usuario_id = ?', (usuario_id,))
    alunos_dict = {row['matricula']: row['nome'] for row in cursor.fetchall()}
    cursor.execute('SELECT aluno_matricula, bancada_nome FROM assentos WHERE materia_id = ? AND sala_id = ?', (aula['materia_id'], aula['sala_id']))
    assentos_aula = {row['aluno_matricula']: row['bancada_nome'] for row in cursor.fetchall()}
    conn.close()

    # Do not fork the live camera worker while first-use model warm-up is still
    # consuming CPU; it otherwise competes with the first real face recognition.
    models_warmed.wait()
    manager = Manager()
    shared_state = manager.dict({
        "faces": [],
        "fighting": False,
        "running": True
    })
    
    clip_queue = Queue()
    frame_queue = Queue(maxsize=1) 

    ia_process = Process(target=ia_process_worker, args=(
        shared_state, frame_queue, clip_queue,
        dict(aula), zonas, alunos_dict, assentos_aula, DB_PATH, DATA_SCREENSHOTS, ia_config
    ))
    ia_process.daemon = True
    ia_process.start()

    loop_count = 0
    erros_tentativas = 0
    erros_loop_consecutivos = 0
    frame_buffer = deque(maxlen=20) 
    active_recordings = []
    ultimo_frame_id = -1
    ultimo_envio_ia = 0.0
    ultima_leitura_estado = 0.0
    ultimo_frame_clip = 0.0
    last_known_faces = []
    is_fighting_now_global = False

    try:
        while (stop_event is None or not stop_event.is_set()) and (cap.running or erros_tentativas < 50):
            try:
                # Espera um frame NOVO da câmera (antes reprocessava o mesmo frame a toda velocidade).
                ret, frame, ultimo_frame_id = cap.read_new(ultimo_frame_id, 0.5)
                if not ret:
                    erros_tentativas += 1
                    time.sleep(0.05)
                    continue
                if frame is None:
                    if not cap.running:
                        time.sleep(0.05)
                    continue

                erros_tentativas = 0
                erros_loop_consecutivos = 0
                agora_loop = time.monotonic()

                # A IA processa ~1 frame por segundo: não adianta serializar 1 MB a cada frame da câmera.
                if (agora_loop - ultimo_envio_ia) >= INTERVALO_ENVIO_IA:
                    try:
                        if frame_queue.full():
                            frame_queue.get_nowait() 
                        frame_queue.put_nowait(frame.copy())
                        ultimo_envio_ia = agora_loop
                    except:
                        pass

                # Cada leitura do shared_state é uma chamada IPC ao processo Manager.
                if (agora_loop - ultima_leitura_estado) >= INTERVALO_LEITURA_ESTADO:
                    ultima_leitura_estado = agora_loop
                    last_known_faces = list(shared_state.get("faces", []))
                    is_fighting_now_global = shared_state.get("fighting", False)

                while not clip_queue.empty():
                    try:
                        v_filename = clip_queue.get_nowait()
                        active_recordings.append({'frames': list(frame_buffer), 'remaining': 30, 'filename': v_filename})
                    except:
                        break

                for (top, right, bottom, left, nome_exibicao, bancada_atual, cor, status_tela) in last_known_faces:
                    cv2.rectangle(frame, (left, top), (right, bottom), cor, 2)
                    cv2.putText(frame, f"{nome_exibicao} ({status_tela})" if status_tela != "Focado" else nome_exibicao, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor, 2)

                for z in zonas:
                    pts = np.array(z['pontos'], np.int32)
                    cv2.polylines(frame, [pts], True, (255, 200, 0), 2) 
                    cv2.putText(frame, z['nome'], tuple(z['pontos'][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,200,0), 2)

                if is_fighting_now_global:
                    cv2.putText(frame, "ACAO SUSPEITA!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 140, 255), 4)

                # Os clipes são gravados a 10 fps (save_clip), então só guardamos 1 frame a cada 0,1s:
                # os 20 frames de antes + 30 de depois do alerta viram 5s reais de vídeo.
                if (agora_loop - ultimo_frame_clip) >= INTERVALO_FRAME_CLIP:
                    ultimo_frame_clip = agora_loop
                    frame_desenhado = frame.copy()
                    frame_buffer.append(frame_desenhado)

                    for rec in active_recordings[:]:
                        rec['frames'].append(frame_desenhado)
                        rec['remaining'] -= 1
                        if rec['remaining'] <= 0:
                            filepath = os.path.join(DATA_CLIPS, rec['filename'])
                            threading.Thread(target=save_clip, args=(list(rec['frames']), filepath)).start()
                            active_recordings.remove(rec)

                ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 55])
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                
                loop_count += 1
                if loop_count >= 60:
                    loop_count = 0
                    c_chk = get_db(); cur_chk = c_chk.cursor()
                    cur_chk.execute('SELECT status FROM aulas WHERE id=?', (aula['id'],))
                    chk_row = cur_chk.fetchone(); c_chk.close()
                    if not chk_row or chk_row['status'] != 'EM_ANDAMENTO': break

            except Exception as loop_error:
                # Antes: catch-all que só imprimia e reentrava no laço IMEDIATAMENTE,
                # sem nenhuma pausa nem limite. Se o erro fosse persistente (ex: o
                # cliente derrubou a conexão e cada tentativa de "yield" falhava de
                # novo), isso virava um laço infinito consumindo 100% de uma CPU só
                # imprimindo log — travando a aula inteira (só resolvia com F5) e
                # podendo até atrasar outras partes do sistema.
                # Agora: se o erro persistir muitas vezes seguidas, encerra o stream
                # de forma limpa (libera câmera, para a IA da aula) em vez de girar
                # para sempre; e sempre espera um pouco antes de tentar de novo.
                erros_loop_consecutivos += 1
                print(f"Ignorando falha isolada no frame principal ({erros_loop_consecutivos}/30): {loop_error}")
                if erros_loop_consecutivos >= 30:
                    print("⚠️ Muitos erros seguidos no /video_feed — encerrando este stream (o cliente provavelmente desconectou).")
                    break
                time.sleep(0.2)
                continue 
                
    except Exception as fatal_e:
        print(f"Erro fatal na câmera: {fatal_e}") 
    finally:
        try:
            shared_state["running"] = False
        except Exception:
            pass  # O processo do Manager já pode ter sido encerrado (cliente desconectou de repente)
        try:
            ia_process.terminate()
            ia_process.join(timeout=1.0)
        except Exception: pass
        cap.release()
        manager.shutdown()

@app.get("/video_feed")
def video_feed(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email = payload.get("email")
        conn = get_db(); cursor = conn.cursor()
        cursor.execute("SELECT id FROM usuarios WHERE email = ?", (email,))
        user = cursor.fetchone(); conn.close()
        
        if not user: return Response(status_code=401)
        session = obter_sessao_video(user['id'])
        return StreamingResponse(session.stream(), media_type="multipart/x-mixed-replace; boundary=frame")
    except: 
        return Response(status_code=401)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)