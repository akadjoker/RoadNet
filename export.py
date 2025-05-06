import torch
from fastSmall import JetsonRoadNet  

IMG_SIZE = 256
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = JetsonRoadNet()
model.load_state_dict(torch.load("road_model/best_model.pth"))
model = model.to(DEVICE)
model.eval()

dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)

torch.onnx.export(
    model,
    dummy_input,
    "road_model/small_model.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=11
)

print("Modelo exportado para unet_model.onnx com sucesso.")

