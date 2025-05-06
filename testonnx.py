import cv2
import time
import numpy as np
import numpy as np
import onnxruntime as ort



# ==== CONFIGURAÇÕES ====
MODEL_PATH = 'road_model/small_model.onnx'
IMG_SIZE = 256
 

USE_CSI = True  

providers = [
    ('TensorrtExecutionProvider', {}),
    'CUDAExecutionProvider',
    'CPUExecutionProvider'
]
session = ort.InferenceSession(MODEL_PATH, providers=providers)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name



 
if USE_CSI:
    cam_set = "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=640, height=480, framerate=30/1 ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink"
    cap = cv2.VideoCapture(cam_set, cv2.CAP_GSTREAMER)
else:
    cap = cv2.VideoCapture("../estrada.mp4")

if not cap.isOpened():
    print("Erro ao abrir câmara.")
    exit()



width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

 
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    orig_size = (frame.shape[1], frame.shape[0])

    # ----- Preprocessar -----
    frame_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    input_tensor = img_norm.transpose(2, 0, 1).reshape(1, 3, IMG_SIZE, IMG_SIZE)

    # ----- Inferência -----
    pred = session.run([output_name], {input_name: input_tensor})[0]
    mask = (pred[0, 0] > 0.3).astype(np.uint8) * 255

    # ----- Pós-processamento -----
    mask_resized = cv2.resize(mask, orig_size)
    mask_color = np.zeros_like(frame)
    mask_color[:, :, 2] = mask_resized  # Vermelho

    overlay = cv2.addWeighted(frame, 0.5, mask_color, 1.0, 0)
      

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

