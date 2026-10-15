# core/behavior_engine.py
"""
Lógica de comportamento do SmartClass SEM dependência de modelos de IA
(só math/numpy/cv2). O main.py chama estas classes; os modelos continuam lá.
Separar isso permite testar a regra com trajetórias sintéticas, sem câmera.

Nada aqui substitui os parâmetros do painel do perfil master:
  - confiança facial / tolerância de reconhecimento  -> continuam no main.py / face_engine.py
  - confiança do celular, confirmações, cooldown, distância -> passados por parâmetro
  - ações suspeitas -> sequência e temporização avaliadas pelo GestureTracker
As constantes abaixo são as mesmas "calibrações internas" que já existiam no código
(velocidade mínima, margem de braço levantado etc.), agora em unidades que não dependem
da distância da câmera.
"""
import math
from collections import deque

import cv2
import numpy as np

AREA_LIVRE = "Área Livre"

# =====================================================================================
# 1) AÇÃO SUSPEITA — movimento brusco do pulso x pedido de fala
# Unidades: LO = largura entre os ombros da própria pessoa (normaliza a distância da
# câmera: 1 LO ≈ 0,4 m) e segundos.
#
# Referências físicas usadas na calibração (braço levantando ~2,2 LO de amplitude):
#   pedido de fala gradual (≥ 1 s) ........ ~2,5 LO/s  -> NÃO é brusco
#   levantar/abaixar em ≤ 0,5 s ............ ~6 LO/s   -> brusco
#   balançar forte a 3 Hz .................. ~8 LO/s   -> brusco
# =====================================================================================
AMPLITUDE_MIN_LO = 0.55          # deslocamento mínimo (pico a pico) para contar como "golpe"
VELOCIDADE_BRUSCA_LO_S = 3.5     # velocidade média 10–90% do golpe (LO/s)
DESCANSO_MAX_S = 0.6             # parado no extremo por mais que isso = mão mantida/mexendo, desfaz o par
NIVEL_MAO_ACIMA_OMBRO_LO = 0.0   # pulso na altura dos ombros ou acima ("mão levantada de verdade")
NIVEL_BRACO_ELEVADO_LO = 0.45    # pulso até 0,45 LO abaixo do ombro (equivale à margem antiga de 60 px)
ALCANCE_MAX_LO = 2.6             # pulso mais longe que isso do ombro é erro do keypoint
JANELA_GESTO_MIN_S = 2.5         # histórico do pulso analisado
RECENCIA_S = 1.0                 # o último golpe precisa ser recente para o alerta estar "ativo"
GAP_MIN_S = 0.8                  # pulso sumido por mais que isso quebra o histórico
PEDIDO_FALA_MIN_S = 1.5          # mão parada e acima do ombro por este tempo = pedido de fala
ESCALA_MIN_PX = 20.0             # ombros mais juntos que isso: pessoa longe/de perfil, sem referência
TTL_PESSOA_S = 2.0               # pessoa sem aparecer por este tempo perde o histórico

# --- Modo de AMOSTRAGEM LENTA (pose a ~1 ciclo/s ou menos, como em hardware fraco com 3 modelos
# disputando a CPU). Nesse ritmo não existe "velocidade": entre duas amostras o braço pode ter feito
# o gesto inteiro. Só dá para ver POSIÇÕES discretas, então a regra vira "o pulso salta de um extremo
# a outro em amostras seguidas, sempre com o braço elevado e sem parar":
#   levantar, segurar e abaixar (pedido de fala) mostra amostras PARADAS no topo -> não alerta.
DT_AMOSTRAGEM_LENTA_S = 0.3     # abaixo de ~3 FPS a velocidade deixa de ser confiável
SALTOS_LENTA_MIN = 2             # saltos seguidos em sentidos opostos (ex.: alto-baixo-alto)
SALTOS_LENTA_MIN_MUITO_LENTO = 2 # mantém sensibilidade mesmo quando a pose demora mais de 1,2 s
DT_MUITO_LENTO_S = 1.2
PARADO_LENTA_LO = 0.35           # duas amostras seguidas a menos que isso = mão parada (desfaz a sequência)
VELOCIDADE_SEQUENCIA_LO_S = 3.5
INTERVALO_GOLPE_APOS_SUBIDA_S = 0.9


def _limiares_gesto(sensibilidade):
    nivel = max(1, min(10, int(sensibilidade)))
    return {
        'amplitude_lenta': 0.95 - 0.055 * (nivel - 1),
        'saltos_lentos': 3 if nivel <= 4 else 2,
        'subida': 0.75 - 0.035 * (nivel - 1),
        'golpe': 0.65 - 0.035 * (nivel - 1),
        'desvio_direcao': 0.40 - 0.025 * (nivel - 1),
        'extensao_frente': 0.50 - 0.035 * (nivel - 1),
        'amplitude_densa': 0.65 - 0.025 * (nivel - 1),
    }


def _detectar_subida_e_golpe(ts, xs, ys, ciclo_s, extensoes=None, limiares=None):
    """Detecta uma subida rápida seguida por outro movimento rápido e distinto."""
    limiares = limiares or _limiares_gesto(10)
    intervalo_max = min(4.0, max(INTERVALO_GOLPE_APOS_SUBIDA_S, 1.5 * ciclo_s))
    dt_med = (ts[-1] - ts[0]) / max(len(ts) - 1, 1)
    for i in range(len(ts) - 2):
        subida_dt = ts[i + 1] - ts[i]
        subida_dx = xs[i + 1] - xs[i]
        subida_dy = ys[i + 1] - ys[i]
        subida_amp = math.hypot(subida_dx, subida_dy)
        subida_vel_min = VELOCIDADE_SEQUENCIA_LO_S if dt_med <= DT_AMOSTRAGEM_LENTA_S else 0.0
        if subida_dt <= 0 or subida_dt > intervalo_max \
                or ys[i] < 0.0 or ys[i + 1] > NIVEL_BRACO_ELEVADO_LO \
                or subida_dy > -limiares['subida'] \
                or subida_amp / subida_dt < subida_vel_min - 1e-9:
            continue

        for j in range(i + 1, len(ts) - 1):
            intervalo = ts[j] - ts[i + 1]
            if intervalo > intervalo_max:
                break
            if math.hypot(xs[j] - xs[i + 1], ys[j] - ys[i + 1]) > 0.45:
                continue

            golpe_dt = ts[j + 1] - ts[j]
            golpe_dx = xs[j + 1] - xs[j]
            golpe_dy = ys[j + 1] - ys[j]
            golpe_amp = math.hypot(golpe_dx, golpe_dy)
            golpe_vel_min = VELOCIDADE_SEQUENCIA_LO_S if dt_med <= DT_AMOSTRAGEM_LENTA_S else 0.0
            if golpe_dt <= 0:
                continue

            produto = subida_dx * golpe_dx + subida_dy * golpe_dy
            alinhamento = produto / max(subida_amp * golpe_amp, 1e-6)
            golpe_lateral = (
                golpe_amp >= limiares['golpe'] - 1e-9
                and abs(golpe_dx) >= limiares['golpe'] - 1e-9
                and abs(golpe_dx) >= 0.50 * abs(golpe_dy)
                and golpe_dy <= 0.55
                and alinhamento <= 1.0 - limiares['desvio_direcao']
                and golpe_amp / golpe_dt >= golpe_vel_min - 1e-9
            )
            golpe_frente = (
                extensoes is not None
                and extensoes[i + 1] is not None
                and extensoes[j + 1] is not None
                and extensoes[j + 1] - extensoes[i + 1] >= limiares['extensao_frente']
                and golpe_dy <= 0.25
            )
            if golpe_lateral or golpe_frente:
                return ts[j + 1]
    return None


def _pontos_viragem(v, limiar):
    """Zigzag: índices dos extremos alternados (pico/vale) cuja oscilação é >= limiar.
    O último índice devolvido é o extremo EM ANDAMENTO (permite alertar durante o golpe)."""
    n = len(v)
    if n < 2:
        return []
    hi = lo = 0
    direcao = 0
    cur = 0
    pts = []
    for i in range(1, n):
        if direcao == 0:
            if v[i] > v[hi]:
                hi = i
            if v[i] < v[lo]:
                lo = i
            if v[hi] - v[lo] >= limiar:
                if hi > lo:
                    pts.append(lo); direcao = 1; cur = hi
                else:
                    pts.append(hi); direcao = -1; cur = lo
        elif direcao == 1:
            if v[i] >= v[cur]:
                cur = i
            elif v[cur] - v[i] >= limiar:
                pts.append(cur); direcao = -1; cur = i
        else:
            if v[i] <= v[cur]:
                cur = i
            elif v[i] - v[cur] >= limiar:
                pts.append(cur); direcao = 1; cur = i
    if direcao != 0:
        pts.append(cur)
    return pts


def _cruzamento(ts, v, i0, i1, nivel, sentido):
    """Instante (interpolado) em que v cruza 'nivel' no sentido dado, procurando em (i0, i1]."""
    for i in range(i0 + 1, i1 + 1):
        if (v[i] - nivel) * sentido >= 0:
            a, b = v[i - 1], v[i]
            if b == a:
                return ts[i]
            f = min(1.0, max(0.0, (nivel - a) / (b - a)))
            return ts[i - 1] + f * (ts[i] - ts[i - 1])
    return ts[i1]


def _golpes(ts, v, pts, vel_limite=VELOCIDADE_BRUSCA_LO_S, amp_min=AMPLITUDE_MIN_LO):
    """Cada trecho entre dois extremos vira um 'golpe' com amplitude e velocidade 10–90%."""
    golpes = []
    for p, q in zip(pts, pts[1:]):
        amp = abs(v[q] - v[p])
        if amp <= 0:
            continue
        sentido = 1 if v[q] > v[p] else -1
        t_ini = _cruzamento(ts, v, p, q, v[p] + sentido * 0.1 * amp, sentido)
        t_fim = _cruzamento(ts, v, p, q, v[p] + sentido * 0.9 * amp, sentido)
        dur = max(t_fim - t_ini, 1e-3)
        vel = 0.8 * amp / dur
        golpes.append({'p': p, 'q': q, 'amp': amp, 'vel': vel, 't_ini': t_ini, 't_fim': t_fim,
                       'brusco': (amp >= amp_min and vel >= vel_limite)})
    return golpes


def _sustentado(v, k, amp_ref):
    """Com amostragem densa, um extremo real tem vizinhos 'a caminho'; um ponto isolado que
    some no quadro seguinte é erro de keypoint (pulso borrado que pula de lugar)."""
    viz = [j for j in (k - 1, k + 1) if 0 <= j < len(v)]
    return any(abs(v[j] - v[k]) <= 0.8 * amp_ref for j in viz)


def _classificar_lento(ts, xs, ys, agora, dt_med, res, limiares):
    """Classificação com poucas amostras por segundo (ver DT_AMOSTRAGEM_LENTA_S)."""
    res['vmax'] = max([math.hypot(xs[i+1]-xs[i], ys[i+1]-ys[i]) / max(ts[i+1]-ts[i], 1e-3) for i in range(len(ts)-1)] or [0.0])
    # pedido de fala: as 2 últimas amostras acima do ombro e praticamente no mesmo lugar
    if len(ts) >= 2 and ys[-1] <= NIVEL_MAO_ACIMA_OMBRO_LO and ys[-2] <= NIVEL_MAO_ACIMA_OMBRO_LO \
            and math.hypot(xs[-1]-xs[-2], ys[-1]-ys[-2]) <= PARADO_LENTA_LO:
        res['pedido_fala'] = True
    if (agora - ts[-1]) > max(RECENCIA_S, 1.5 * dt_med):
        return res
    saltos = []
    for i in range(len(ts) - 1, 0, -1):
        d = math.hypot(xs[i]-xs[i-1], ys[i]-ys[i-1])
        if d < limiares['amplitude_lenta']:
            break
        saltos.append((i, d))
    saltos.reverse()
    if len(saltos) < (
        max(SALTOS_LENTA_MIN_MUITO_LENTO if dt_med > DT_MUITO_LENTO_S else SALTOS_LENTA_MIN,
            limiares['saltos_lentos'])
    ):
        return res
    idxs = [saltos[0][0] - 1] + [i for i, _ in saltos]
    if min(ys[i] for i in idxs) > NIVEL_BRACO_ELEVADO_LO:
        return res
    dirs = [(xs[i]-xs[i-1], ys[i]-ys[i-1]) for i, _ in saltos]
    if max(abs(dx) for dx, _ in dirs) < 0.1:
        return res
    if not all((a[0]*b[0] + a[1]*b[1]) < 0 for a, b in zip(dirs, dirs[1:])):
        return res
    res['cadeia'] = len(saltos)
    res['suspeito'] = True
    res['motivo'] = 'braço indo e voltando bruscamente (amostragem lenta)'
    return res


def classificar_pulso(amostras, agora, ciclo_s=0.1, sensibilidade=10):
    """
    amostras: [(t, x_rel, y_rel, conf)] do MESMO pulso (coordenadas em LO relativas ao centro dos ombros;
              y negativo = acima do ombro).
    Regras (evidência mínima para "suspeito"):
      * golpe brusco = oscilação >= AMPLITUDE_MIN_LO feita a >= VELOCIDADE_BRUSCA_LO_S;
      * golpes seguidos só formam sequência se o pulso NÃO ficou parado no extremo por mais de
        DESCANSO_MAX_S (mão mantida levantada = pedido de fala, não é arremesso/briga);
      * "sobe e desce brusco": 2 golpes verticais em sequência com o topo na altura do ombro ou acima;
      * "balanço brusco": 3 golpes em sequência (qualquer eixo) com o braço elevado.
    Levantar a mão devagar, manter e abaixar devagar NUNCA gera golpe brusco.
    """
    res = {'suspeito': False, 'motivo': '', 'cadeia': 0, 'vmax': 0.0, 'pedido_fala': False}
    limiares = _limiares_gesto(sensibilidade)
    n = len(amostras)
    if n < 3:
        return res
    ts = [a[0] for a in amostras]
    xs = [a[1] for a in amostras]
    ys = [a[2] for a in amostras]
    extensoes = [a[4] if len(a) > 4 else None for a in amostras]
    evento_arremesso = _detectar_subida_e_golpe(ts, xs, ys, ciclo_s, extensoes, limiares)
    if evento_arremesso is not None:
        res['suspeito'] = True
        res['motivo'] = 'subida rápida seguida de golpe'
        res['evento_t'] = evento_arremesso
        res['cadeia'] = 2
    dt_med = (ts[-1] - ts[0]) / (n - 1)
    if dt_med > DT_AMOSTRAGEM_LENTA_S:
        lento = _classificar_lento(ts, xs, ys, agora, dt_med, res, limiares)
        if evento_arremesso is not None:
            lento['suspeito'] = True
            lento['motivo'] = res['motivo']
            lento['evento_t'] = evento_arremesso
        return lento

    # --- pedido de fala (informativo/diagnóstico): mão parada, acima do ombro, por alguns segundos
    if ts[0] <= agora - PEDIDO_FALA_MIN_S + max(0.25, ciclo_s):
        rec = [(x, y) for t, x, y in zip(ts, xs, ys) if t >= agora - PEDIDO_FALA_MIN_S]
        if len(rec) >= 3:
            rx = [r[0] for r in rec]; ry = [r[1] for r in rec]
            if max(ry) <= NIVEL_MAO_ACIMA_OMBRO_LO and (max(rx) - min(rx)) <= 0.6 and (max(ry) - min(ry)) <= 0.6:
                res['pedido_fala'] = True

    # Pose lenta (poucos FPS): cada golpe cobre 2+ amostras e a interpolação suaviza a velocidade
    # (golpe real de ~7 LO/s mede ~3,7–4); o limiar acompanha. Com amostragem MUITO esparsa um
    # "sobe e desce" de 1 amostra é indistinguível de erro de keypoint: exige 3 golpes em vez de 2.
    esparsa = dt_med > 0.15
    vel_limite = VELOCIDADE_BRUSCA_LO_S * (0.8 if esparsa else 1.0)
    cadeia_min_vertical = 3 if dt_med > 0.25 else 2
    for eixo, v in (('vertical', ys), ('horizontal', xs)):
        golpes = _golpes(
            ts, v, _pontos_viragem(v, limiares['amplitude_densa']),
            vel_limite, limiares['amplitude_densa']
        )
        if not golpes:
            continue
        res['vmax'] = max(res['vmax'], max(g['vel'] for g in golpes))
        cadeia = []
        for g in reversed(golpes):
            if not g['brusco']:
                break
            if cadeia and (cadeia[-1]['t_ini'] - g['t_fim']) > DESCANSO_MAX_S:
                break
            cadeia.append(g)
        cadeia.reverse()
        if not cadeia or (agora - cadeia[-1]['t_fim']) > RECENCIA_S:
            continue
        if dt_med <= 0.25:  # até ~4 FPS ainda dá para exigir amostras "a caminho" do extremo
            ok = all(_sustentado(v, a['q'], min(a['amp'], b['amp'])) for a, b in zip(cadeia, cadeia[1:]))
            ok = ok and _sustentado(v, cadeia[-1]['q'], cadeia[-1]['amp'])
            if not ok:
                continue
        topo = min(ys[i] for i in [cadeia[0]['p']] + [g['q'] for g in cadeia])
        res['cadeia'] = max(res['cadeia'], len(cadeia))
        if len(cadeia) >= 3 and topo <= NIVEL_BRACO_ELEVADO_LO:
            res['suspeito'] = True; res['motivo'] = 'balanço brusco do braço'
        elif len(cadeia) >= cadeia_min_vertical and eixo == 'vertical' and topo <= NIVEL_MAO_ACIMA_OMBRO_LO:
            res['suspeito'] = True; res['motivo'] = 'levantar e abaixar a mão rapidamente'
    return res


class GestureTracker:
    """Acompanha cada pessoa (casamento 1-para-1 por posição dos ombros) e o histórico de
    cada pulso, e diz se há ação suspeita neste ciclo."""

    def __init__(self, min_conf_pulso=0.30, min_conf_tronco=0.50, sensibilidade=10):
        self.min_conf_pulso = min_conf_pulso
        self.min_conf_tronco = min_conf_tronco
        self.sensibilidade = max(1, min(10, int(sensibilidade)))
        self.pessoas = []

    def set_sensitivity(self, sensibilidade):
        nivel = max(1, min(10, int(sensibilidade)))
        if nivel != self.sensibilidade:
            self.sensibilidade = nivel
            self.pessoas.clear()

    def update(self, agora, pessoas, janela_s=JANELA_GESTO_MIN_S, ciclo_s=0.5):
        """pessoas: pulso, cotovelo e ombros por lado (pixels, confiança crua)."""
        diag = {'pulsos': 0, 'levantado': 0, 'golpes': 0, 'pedidos': 0, 'vmax': 0.0, 'motivo': ''}
        dets = []
        for p in pessoas:
            ls, rs = p['shoulders']
            if min(ls[2], rs[2]) < self.min_conf_tronco or min(ls[0], ls[1], rs[0], rs[1]) <= 0:
                continue  # sem os dois ombros não há referência confiável de escala/altura
            largura = math.hypot(ls[0] - rs[0], ls[1] - rs[1])
            if largura < ESCALA_MIN_PX:
                continue
            dets.append({
                'o': ((ls[0] + rs[0]) / 2.0, (ls[1] + rs[1]) / 2.0),
                'largura': largura,
                'wrists': p['wrists'],
                'elbows': p.get('elbows', ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))),
            })

        pares = []
        for di, d in enumerate(dets):
            for ti, tr in enumerate(self.pessoas):
                dist = math.hypot(d['o'][0] - tr['o'][0], d['o'][1] - tr['o'][1])
                if dist <= 1.2 * max(d['largura'], tr['escala']):
                    pares.append((dist, di, ti))
        pares.sort()
        det_usada, trk_usada = set(), set()
        for dist, di, ti in pares:
            if di in det_usada or ti in trk_usada:
                continue
            det_usada.add(di); trk_usada.add(ti)
            tr, d = self.pessoas[ti], dets[di]
            tr['o'], tr['visto'] = d['o'], agora
            tr['escala'] = 0.7 * tr['escala'] + 0.3 * d['largura']
            d['trk'] = tr
        for di, d in enumerate(dets):
            if di in det_usada:
                continue
            tr = {'o': d['o'], 'escala': d['largura'], 'visto': agora, 'hist': [deque(), deque()],
                  'ultimo_evento': [0.0, 0.0]}
            self.pessoas.append(tr)
            d['trk'] = tr
        self.pessoas = [t for t in self.pessoas if (agora - t['visto']) <= max(TTL_PESSOA_S, janela_s)]

        gap_max = max(GAP_MIN_S, 2.5 * ciclo_s)
        suspeito = False
        for d in dets:
            tr = d['trk']
            for idx in (0, 1):
                hist = tr['hist'][idx]
                x, y, c = d['wrists'][idx]
                if c >= self.min_conf_pulso and x > 0 and y > 0:
                    rx = (x - tr['o'][0]) / tr['escala']
                    ry = (y - tr['o'][1]) / tr['escala']
                    if math.hypot(rx, ry) <= ALCANCE_MAX_LO:
                        ex, ey, ec = d['elbows'][idx]
                        extensao = (
                            math.hypot(x - ex, y - ey) / tr['escala']
                            if ec >= self.min_conf_pulso and ex > 0 and ey > 0
                            else None
                        )
                        if hist and (agora - hist[-1][0]) > gap_max:
                            hist.clear()
                        hist.append((agora, rx, ry, c, extensao))
                        diag['pulsos'] += 1
                        if ry <= NIVEL_BRACO_ELEVADO_LO:
                            diag['levantado'] += 1
                while hist and (agora - hist[0][0]) > janela_s:
                    hist.popleft()
                r = classificar_pulso(list(hist), agora, ciclo_s, self.sensibilidade)
                diag['vmax'] = max(diag['vmax'], r['vmax'])
                diag['golpes'] += 1 if r['cadeia'] else 0
                diag['pedidos'] += 1 if r['pedido_fala'] else 0
                if r['suspeito']:
                    evento_t = r.get('evento_t')
                    if evento_t is None or evento_t > tr['ultimo_evento'][idx]:
                        suspeito = True
                        diag['motivo'] = r['motivo']
                        if evento_t is not None:
                            tr['ultimo_evento'][idx] = evento_t
        return suspeito, diag


# =====================================================================================
# 2) CELULAR — confirmação por N detecções DO MESMO OBJETO
# =====================================================================================
class PhoneConfirmer:
    """
    Antes: N detecções na janela contavam em QUALQUER lugar da sala (dois falsos positivos em
    pontos diferentes já "confirmavam"). Agora as detecções são agrupadas por posição (mesmo
    objeto) e só um grupo com N detecções confirma. Os parâmetros do painel continuam mandando:
      conf_forte    -> detecção nítida confirma na hora (como antes: max(0.55, confiança do painel))
      confirmacoes  -> "Confirmações p/ Celular" (1–5)
    """

    def __init__(self, conf_forte, confirmacoes, janela_s=2.5):
        self.conf_forte = conf_forte
        self.confirmacoes = max(1, int(confirmacoes))
        self.janela_s = janela_s
        self.tracks = []

    def update(self, agora, deteccoes, janela_s=None):
        """deteccoes: [{'box': (x1,y1,x2,y2), 'conf': float}] já filtradas pela confiança do painel."""
        janela = janela_s or self.janela_s
        forte = False
        usados = set()
        for d in sorted(deteccoes, key=lambda d: -d.get('conf', 0.0)):
            x1, y1, x2, y2 = d['box']
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            tam = max(x2 - x1, y2 - y1, 1)
            melhor, melhor_dist = None, None
            for i, tr in enumerate(self.tracks):
                if i in usados:
                    continue
                dist = math.hypot(cx - tr['c'][0], cy - tr['c'][1])
                if dist <= max(2.0 * max(tam, tr['tam']), 60) and (melhor_dist is None or dist < melhor_dist):
                    melhor, melhor_dist = i, dist
            if melhor is None:
                self.tracks.append({'c': (cx, cy), 'tam': tam, 'hits': deque([agora]), 'ult': agora})
                usados.add(len(self.tracks) - 1)
            else:
                tr = self.tracks[melhor]
                tr['c'], tr['tam'], tr['ult'] = (cx, cy), tam, agora
                tr['hits'].append(agora)
                usados.add(melhor)
            if d.get('conf', 0.0) >= self.conf_forte:
                forte = True
        self.tracks = [t for t in self.tracks if (agora - t['ult']) <= janela]
        for t in self.tracks:
            while t['hits'] and (agora - t['hits'][0]) > janela:
                t['hits'].popleft()
        confirmado = forte or any(len(t['hits']) >= self.confirmacoes for t in self.tracks)
        return confirmado


# =====================================================================================
# 3) BANCADA — correta / outra / fora, sem piscar na borda
# =====================================================================================
MARGEM_BANCADA_REL = 0.25
# Tolerância ao sair da bancada: só absorve UM frame ruim (ou, no máximo, 0,4s).
# Antes era 2,0s + 2 amostras — o aluno saía da bancada e o status ficava preso
# na bancada antiga por 2s ou mais.
TEMPO_GRACA_FORA_BANCADA_S = 0.4
# Se o rosto está claramente fora (mais de 1 rosto de distância da borda), não há dúvida:
# sai da bancada NO MESMO FRAME, sem esperar a segunda amostra.
SAIDA_IMEDIATA_REL = 1.0
TEMPO_FORA_PARA_CIRCULAR_S = 0.3
DESLOCAMENTO_CIRCULANDO_REL = 0.75
GAP_MAX_CIRCULANDO_S = 8.0


class LocalizadorBancada:
    """Escolhe a zona mais próxima pela distância assinada e mantém o track na transição."""

    def __init__(self, zonas):
        self.zonas = [(z['nome'], np.array(z['pontos'], np.int32)) for z in zonas]

    def localizar(self, estado, cx, cy, tamanho, agora):
        if not self.zonas:
            return AREA_LIVRE
        deps = {nome: cv2.pointPolygonTest(poly, (float(cx), float(cy)), True) for nome, poly in self.zonas}
        melhor_nome = max(deps, key=lambda k: deps[k])
        melhor_d = deps[melhor_nome]
        margem = MARGEM_BANCADA_REL * max(float(tamanho), 1.0)
        atual = estado.get('nome')

        def fixar(nome):
            estado['nome'] = nome
            if nome != AREA_LIVRE:
                estado.pop('fora_desde', None)
                estado.pop('fora_amostras', None)
            return nome

        if atual is None:
            if melhor_d >= -margem:
                return fixar(melhor_nome)
            estado['fora_desde'] = agora
            estado['fora_amostras'] = 1
            return fixar(AREA_LIVRE)
        d_atual = deps.get(atual) if atual != AREA_LIVRE else None
        if melhor_d >= -margem:
            empate = max(2.0, 0.05 * max(float(tamanho), 1.0))
            if d_atual is not None and d_atual >= -margem and d_atual >= melhor_d - empate:
                return fixar(atual)
            return fixar(melhor_nome)
        fora_desde = estado.setdefault('fora_desde', agora)
        estado['fora_amostras'] = estado.get('fora_amostras', 0) + 1
        claramente_fora = melhor_d <= -SAIDA_IMEDIATA_REL * max(float(tamanho), 1.0)
        if atual != AREA_LIVRE and not claramente_fora and (
            estado['fora_amostras'] < 2
            and agora - fora_desde < TEMPO_GRACA_FORA_BANCADA_S
        ):
            return atual
        return fixar(AREA_LIVRE)


class CirculacaoTracker:
    """Distingue alguém circulando de uma detecção momentânea fora das zonas."""

    def __init__(self):
        self.tracks = {}

    def update(self, matricula, fora_da_bancada, cx, cy, tamanho, agora):
        if not fora_da_bancada:
            self.tracks.pop(matricula, None)
            return False

        estado = self.tracks.get(matricula)
        if estado is None or agora - estado['visto'] > GAP_MAX_CIRCULANDO_S:
            estado = {
                'desde': agora,
                'visto': agora,
                'origem': (cx, cy),
                'centro_anterior': (cx, cy),
                'tamanho': max(float(tamanho), 1.0),
                'amostras_em_movimento': 0,
            }
            self.tracks[matricula] = estado
            return False

        estado['visto'] = agora
        tempo_fora = agora - estado['desde']
        deslocamento = math.hypot(cx - estado['origem'][0], cy - estado['origem'][1])
        escala_rosto = max((estado['tamanho'] + float(tamanho)) / 2.0, 1.0)
        passo = math.hypot(cx - estado['centro_anterior'][0], cy - estado['centro_anterior'][1])
        if passo >= 0.15 * escala_rosto:
            estado['amostras_em_movimento'] += 1
        estado['centro_anterior'] = (cx, cy)
        return tempo_fora >= TEMPO_FORA_PARA_CIRCULAR_S \
            and estado['amostras_em_movimento'] >= 2 \
            and deslocamento >= DESLOCAMENTO_CIRCULANDO_REL * escala_rosto


# =====================================================================================
# 4) RECONHECIMENTO FACIAL — quem reconhecer neste ciclo
# =====================================================================================
INTERVALO_RETRY_NOVO = 0.15           # s entre tentativas p/ rosto ainda sem identidade
INTERVALO_RETRY_DESCONHECIDO = 1.0    # s (após as tentativas rápidas)
INTERVALO_RETRY_DESCONHECIDO_RAPIDO = 0.3  # s nas primeiras tentativas (1ª leitura ruim: ângulo/borrão)
TENTATIVAS_RAPIDAS = 3
INTERVALO_REVERIFICA = 15.0           # s; evita repetir o encoding a cada ciclo lento da câmera
MAX_RECONHECIMENTOS_NOVOS_CICLO = 3   # rostos novos/desconhecidos por ciclo (antes: 1)


def selecionar_reconhecimentos(tracks, visiveis, agora, min_face_px):
    """Devolve os ids de track a reconhecer neste ciclo. Rostos novos/desconhecidos têm prioridade e
    podem ser vários por ciclo (começo da aula: sala inteira aparece de uma vez); a reconferência de
    quem já foi identificado só roda quando não há ninguém esperando identificação."""
    candidatos = []
    for tid in visiveis:
        tr = tracks[tid]
        if tr['size'] < min_face_px:
            continue
        idade = agora - tr['last_rec']
        mat = tr['matricula']
        if mat is None and idade >= INTERVALO_RETRY_NOVO:
            candidatos.append((0, -tr['size'], tid))
        elif mat == "Desconhecido":
            intervalo = INTERVALO_RETRY_DESCONHECIDO_RAPIDO if tr.get('tentativas', 0) < TENTATIVAS_RAPIDAS else INTERVALO_RETRY_DESCONHECIDO
            if idade >= intervalo:
                candidatos.append((1, -tr['size'], tid))
        elif mat not in (None, "Desconhecido") and idade >= INTERVALO_REVERIFICA:
            candidatos.append((2, -tr['size'], tid))
    candidatos.sort()
    prioritarios = [c[2] for c in candidatos if c[0] < 2][:MAX_RECONHECIMENTOS_NOVOS_CICLO]
    if prioritarios:
        return prioritarios
    return [candidatos[0][2]] if candidatos else []
