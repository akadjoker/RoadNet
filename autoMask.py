import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from UNet import UNet  

# ======= CONFIGURAÇÕES =======

IMAGE_FOLDER = "real" 
MASK_OUTPUT_FOLDER = "masks" 
MODEL_PATH = "road_model/big_model.pth"
IMG_SIZE = 256

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(MASK_OUTPUT_FOLDER, exist_ok=True)

# ======= CARREGAR MODELO =======

model = UNet()
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# ======= PROCESSAR IMAGENS =======

image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

for img_name in tqdm(image_files, desc="Segmentando imagens"):
    img_path = os.path.join(IMAGE_FOLDER, img_name)
    img = cv2.imread(img_path)
    orig_size = (img.shape[1], img.shape[0])  # (width, height)

    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = model(img_tensor)
        pred_mask = (pred.squeeze().cpu().numpy() > 0.2).astype(np.uint8) * 255

    # Redimensionar a máscara para o tamanho original da imagem
    pred_mask_resized = cv2.resize(pred_mask, orig_size)

    output_path = os.path.join(MASK_OUTPUT_FOLDER, os.path.splitext(img_name)[0] + "_mask.png")
    cv2.imwrite(output_path, pred_mask_resized)

print(f"Segmentação concluída. Máscaras salvas em: {MASK_OUTPUT_FOLDER}")

