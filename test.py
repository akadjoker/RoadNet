import cv2
import time
import numpy as np
import torch
from torch import nn
from fastSmall import JetsonRoadNet  
from torchvision import transforms


def estimate_angle_from_mask(mask, img_size=(128, 128)):
    """
    Recebe uma máscara binária segmentada e devolve um ângulo normalizado entre -1 e 1.
    """
    # Morphology para suavizar
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Só a metade inferior
    height = img_size[1]
    lower_half = mask_clean[int(height/2):, :]

    # Detectar linhas com Hough
    lines = cv2.HoughLinesP(lower_half, 1, np.pi/180, threshold=20, minLineLength=10, maxLineGap=10)

    if lines is None:
        return 0.0  # Não encontrou linhas, manter direção

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1)  # Radianos
        angles.append(np.degrees(angle))

    # Média dos ângulos encontrados
    mean_angle = np.mean(angles)

    # Converter de graus para valor normalizado (-1 a 1)
    # Supondo que 0 graus = em frente, positivo para direita
    max_angle = 45  # Considera no máximo 45º para cada lado
    normalized_angle = np.clip(mean_angle / max_angle, -1, 1)

    return normalized_angle


# ==== CONFIGURAÇÕES ====
MODEL_PATH = 'road_model/best_model.pth'
IMG_SIZE = 256
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


USE_CSI = True  
USE_MORPH=True

# ==== PREPARAR MODELO ====
model = JetsonRoadNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# ==== TRANSFORMAÇÃO ====
def preprocess(frame):
    img_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)
    return img_tensor

 
if USE_CSI:
    cam_set = "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=640, height=480, framerate=30/1 ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink"
    cap = cv2.VideoCapture(cam_set, cv2.CAP_GSTREAMER)
else:
    cap = cv2.VideoCapture("../estrada.mp4")

if not cap.isOpened():
    print("Erro ao abrir câmara.")
    exit()


fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

roi_height = height // 3  # terço inferior da imagem
roi_top = height - roi_height

 
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    original = frame.copy()
    h, w = frame.shape[:2]

    # Pré-processamento
    tensor = preprocess(frame)

    # Inferência
    with torch.no_grad():
        pred = model(tensor)
    pred_np = pred.squeeze().cpu().numpy()
    pred = pred.cpu().squeeze().numpy()
    pred = (pred > 0.3).astype(np.uint8) * 255
    pred = cv2.resize(pred, (w, h))
    #pred_resized = cv2.resize(pred, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)



    # === Morphology ===
    if USE_MORPH:
        pred = cv2.morphologyEx(pred, cv2.MORPH_CLOSE, kernel)
        pred = cv2.morphologyEx(pred, cv2.MORPH_OPEN, kernel)
        pred = cv2.dilate(pred, kernel, iterations=1)



    # Criar overlay verde
    mask_color = np.zeros_like(original)
    mask_color[:, :, 1] = pred
    overlay = cv2.addWeighted(original, 0.5, mask_color, 0.9, 0)
    
    roi_mask = pred[roi_top:height, :]
    edges = cv2.Canny(roi_mask, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=100)
    
 
    left_lines = []
    right_lines = []
    center_x = width // 2
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1:  
                continue
                
            slope = (y2 - y1) / (x2 - x1)
            
            # Filtrar linhas horizontais
            if abs(slope) < 0.3:
                continue
                
            # Classificar como linha esquerda ou direita
            if slope < 0:  # Linha esquerda (negativa na imagem)
                left_lines.append(line[0])
            else:  # Linha direita (positiva na imagem)
                right_lines.append(line[0])
        
        # Desenhar linhas detectadas no frame
        if len(left_lines) > 0:
            for x1, y1, x2, y2 in left_lines:
                cv2.line(overlay, (x1, y1 + roi_top), (x2, y2 + roi_top), (255, 0, 0), 2)
                
        if len(right_lines) > 0:
            for x1, y1, x2, y2 in right_lines:
                cv2.line(overlay, (x1, y1 + roi_top), (x2, y2 + roi_top), (0, 0, 255), 2)
        
        # Calcular pontos médios para cada lado
        left_x = 0
        right_x = width
        
        if len(left_lines) > 0:
            left_points = np.array(left_lines)
            left_x = np.mean(left_points[:, [0, 2]])
            
        if len(right_lines) > 0:
            right_points = np.array(right_lines)
            right_x = np.mean(right_points[:, [0, 2]])
        
        # Calcular o ponto central desejado (entre as duas linhas)
        target_position = (left_x + right_x) / 2
        
        # Calcular o erro de posição (diferença entre o centro do carro e a posição alvo)
        error = center_x - target_position
        
        # Calcular o ângulo de direção (positivo: virar à direita, negativo: virar à esquerda)
        # O fator de escala pode ser ajustado baseado na sensibilidade desejada
        max_angle = 30  # ângulo máximo de esterçamento em graus
        steering_angle = np.clip(error / (width/4) * max_angle, -max_angle, max_angle)
        
        # Desenhar uma indicação do ângulo de direção
        direction_text = f"angle: {steering_angle:.1f}°"
        cv2.putText(overlay, direction_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Desenhar uma linha indicando a direção
        start_point = (center_x, height)
        end_point = (int(center_x - steering_angle * 5), height - 150)  # Multiplicador para visualização
        cv2.line(overlay, start_point, end_point, (0, 255, 255), 3)
        
        # Desenhar indicadores de posição
        cv2.circle(overlay, (int(target_position), height - 50), 10, (255, 255, 0), -1)  # Posição alvo
        cv2.circle(overlay, (center_x, height - 50), 10, (0, 0, 255), -1)  # Centro do carro
    

    # ==== FPS ====
    current_time = time.time()
    fps = 1.0 / (current_time - prev_time)
    prev_time = current_time
    cv2.putText(overlay, f"FPS: {fps:.1f} ", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Road Detection', overlay)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

