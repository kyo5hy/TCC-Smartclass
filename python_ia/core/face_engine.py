# core/face_engine.py
import face_recognition
import os
import re
import pickle
import cv2
import numpy as np

class FaceEngine:
    def __init__(self, data_path="data/faces", encodings_file="data/encodings.pickle"):
        self.data_path = data_path
        self.encodings_file = encodings_file
        self.known_encodings = []
        self.known_names = []
        os.makedirs(self.data_path, exist_ok=True)
        self.load_encodings()

    def load_encodings(self):
        if os.path.exists(self.encodings_file):
            try:
                with open(self.encodings_file, "rb") as f:
                    data = pickle.load(f)
                    self.known_encodings = data.get("encodings", [])
                    self.known_names = data.get("names", [])
            except Exception:
                self.generate_all_encodings()
        else:
            self.generate_all_encodings()

    def _resize_image_for_edge(self, image, max_size=1000):
        """ Redimensiona imagens gigantes para evitar que o Raspberry Pi fique sem RAM """
        h, w = image.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        return image

    def find_best_match(self, encoding, tolerance=0.48):
        return self.find_best_match_details(encoding, tolerance)[0]

    def find_best_match_details(self, encoding, tolerance=0.48):
        """
        Retorna identidade, distância, margem para outra identidade e se a
        correspondência é forte o bastante para dispensar confirmação temporal.

        Bug corrigido: antes usávamos face_recognition.compare_faces() e pegávamos
        matches.index(True) — ou seja, o PRIMEIRO rosto cadastrado que caísse dentro
        da tolerância "vencia", mesmo que outro cadastro fosse uma correspondência
        muito melhor (menor distância real). Isso é o que causava duas pessoas
        diferentes (inclusive de sexos diferentes) serem reconhecidas como a mesma
        matrícula. Agora comparamos a distância contra TODOS os encodings conhecidos
        e escolhemos o menor — e só aceitamos se, além de estar dentro da tolerância,
        houver uma margem clara para o segundo melhor candidato (evita ambiguidade
        entre dois rostos parecidos).
        """
        if not self.known_encodings or not self.known_names:
            return "Desconhecido", float("inf"), 0.0, False

        distances = face_recognition.face_distance(self.known_encodings, encoding)
        usable_count = min(len(distances), len(self.known_names))
        if usable_count == 0 or not np.isfinite(distances[:usable_count]).all():
            return "Desconhecido", float("inf"), 0.0, False

        distances = distances[:usable_count]
        best_idx = int(np.argmin(distances))
        best_dist = float(distances[best_idx])
        if best_dist > tolerance:
            return "Desconhecido", best_dist, 0.0, False

        # Verifica ambiguidade: se existir outro cadastro de OUTRA pessoa quase tão
        # próximo quanto o melhor candidato, é mais seguro marcar como Desconhecido
        # do que arriscar confundir duas pessoas parecidas.
        best_name = self.known_names[best_idx]
        outros = [d for i, d in enumerate(distances) if self.known_names[i] != best_name]
        margem = float("inf")
        if outros:
            margem = float(min(outros) - best_dist)
            if margem < 0.03:
                return "Desconhecido", best_dist, margem, False

        forte = best_dist <= tolerance - 0.04 and margem >= 0.06
        return best_name, best_dist, margem, forte

    def generate_all_encodings(self):
        temp_encodings = []
        temp_names = []
        
        print("⚙️ [FaceEngine] A processar rostos para a Base de Dados da IA...")
        
        for file_name in os.listdir(self.data_path):
            if file_name.lower().endswith((".jpg", ".png", ".jpeg")):
                try:
                    image_path = os.path.join(self.data_path, file_name)
                    
                    # Usa o OpenCV para ler e redimensionar a imagem de forma segura
                    cv_img = cv2.imread(image_path)
                    if cv_img is None:
                        continue
                        
                    cv_img = self._resize_image_for_edge(cv_img)
                    rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                    
                    # Localiza o rosto na imagem redimensionada (modelo 'hog' é mais rápido em CPU)
                    face_locations = face_recognition.face_locations(rgb_img, model="hog")
                    
                    if face_locations:
                        # Extrai a assinatura 128-d (encoding). num_jitters=3 faz o
                        # face_recognition gerar pequenas variações aleatórias da
                        # imagem e tirar a média dos encodings — deixa a assinatura
                        # mais robusta a variações de ângulo/luz sem custo nenhum em
                        # tempo real (isso roda só uma vez, ao cadastrar a foto).
                        encodings = face_recognition.face_encodings(rgb_img, face_locations, num_jitters=3)
                        if encodings:
                            # Suporta MÚLTIPLAS fotos por aluno: "12345.jpg" (a principal)
                            # e opcionalmente "12345_2.jpg", "12345_3.jpg"... — todas
                            # contam como referências da MESMA matrícula, melhorando
                            # bastante a precisão (ângulos/luz diferentes) sem exigir
                            # nenhuma mudança na tela de cadastro.
                            matricula_base = re.sub(r'_\d+$', '', os.path.splitext(file_name)[0])
                            temp_encodings.append(encodings[0])
                            temp_names.append(matricula_base)
                    else:
                        print(f"⚠️ [AVISO] Nenhum rosto detetado na imagem ({file_name}). Tente uma foto mais clara.")
                        
                except Exception as e:
                    print(f"⚠️ [ERRO] Falha ao processar ({file_name}): {e}")
        
        self.known_encodings = temp_encodings
        self.known_names = temp_names
        
        try:
            with open(self.encodings_file, "wb") as f:
                pickle.dump({"encodings": self.known_encodings, "names": self.known_names}, f)
            print(f"✅ [FaceEngine] {len(self.known_names)} referência(s) faciais carregadas e gravadas com sucesso!")
        except Exception as e:
            print(f"❌ [ERRO] Falha ao gravar encodings.pickle: {e}")