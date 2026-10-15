import os
import torch
from ultralytics import YOLO

# Registra a classe PoseModel como segura globalmente (PyTorch 2.6+)
try:
    from ultralytics.nn.tasks import PoseModel
    torch.serialization.add_safe_globals([PoseModel])
except Exception:
    pass


def _manter_keypoints_de_baixa_confianca():
    """
    CORREÇÃO DA 'AÇÃO SUSPEITA' QUE NUNCA DISPARAVA.

    Em várias versões do ultralytics, a classe Keypoints ZERA as coordenadas
    (x=0, y=0) de todo ponto com confiança < 0.5 assim que o resultado é criado.
    Só que o pulso de um braço balançando rápido fica borrado na câmera, e a
    rede quase sempre dá confiança < 0.5 pra ele — então justamente durante o
    gesto o pulso "sumia" (virava 0,0), a velocidade nunca era calculada e o
    alerta jamais disparava. Aqui desligamos essa máscara: o resultado passa a
    trazer as coordenadas reais + a confiança, e quem decide o limite é o
    main.py (que usa um limite mais baixo só para os pulsos).
    """
    try:
        from ultralytics.engine.results import Keypoints, BaseTensor

        def _init(self, keypoints, orig_shape):
            if keypoints.ndim == 2:
                keypoints = keypoints[None, :]
            BaseTensor.__init__(self, keypoints, orig_shape)
            self.has_visible = self.data.shape[-1] == 3

        Keypoints.__init__ = _init
        print("✅ Keypoints: máscara de confiança < 0.5 desativada (pulsos borrados agora são mantidos).")
    except Exception as e:
        print(f"⚠️ Não foi possível ajustar Keypoints do ultralytics ({e}) — seguindo com o comportamento padrão.")


_manter_keypoints_de_baixa_confianca()


class PoseEngine:
    def __init__(self, model_path="yolov8n-pose.pt", ncnn_path=None, imgsz=640):
        # PRIORIDADE: modelo NCNN (muito mais rápido em CPU/ARM — mesma técnica já
        # usada no detector facial). O modelo PyTorch puro (.pt) só entra como
        # fallback se o NCNN não estiver disponível, pois em hardware fraco ele é
        # lento o bastante para atrasar a detecção de "Ação Suspeita" por vários
        # segundos, mesmo rodando numa thread própria.
        self.is_ncnn = False
        self.imgsz = 640   # resolução de entrada; só muda se o modelo NCNN foi exportado com outro tamanho
        if ncnn_path and os.path.exists(ncnn_path):
            try:
                self.model = YOLO(ncnn_path, task="pose")
                self.is_ncnn = True
                self.imgsz = imgsz
                print("✅ YOLO-Pose NCNN (Edge) carregado com sucesso!")
                return
            except Exception as e:
                print(f"⚠️ Falha ao carregar Pose NCNN ({e}). Usando modelo PyTorch padrão.")
        self.model = YOLO(model_path, task="pose")
        print("✅ YOLO-Pose PyTorch (Standard) carregado com sucesso!")

    def analyze_frame(self, frame):
        # conf=0.20: pessoa com o braço em movimento fica "borrada" e a rede dá
        # confiança menor para a pessoa inteira; com o padrão (0.25) ela some do frame.
        results = self.model(frame, conf=0.20, imgsz=self.imgsz, verbose=False, device="cpu")
        alerts = []
        return alerts, results
