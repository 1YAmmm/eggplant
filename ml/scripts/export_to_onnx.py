  
# Initialize Model with Pretrained Weights
import timm
import torch

# from ml.scripts.train_v2 import DEVICE, MODEL_NAME, NUM_CLASSES
DATA_DIR = "data/dataset"  # Adjust to your dataset path
BATCH_SIZE = 32
NUM_CLASSES = 6  # Adjust to your exact number of diseases/pests
EPOCHS = 15
LEARNING_RATE = 1e-3
MODEL_NAME = "mobilenetv3_large_100"  # or "efficientnet_lite0"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=NUM_CLASSES).to(DEVICE)
    

model.load_state_dict(torch.load("eggplant_model_best.pth"))
model.eval()
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, "eggplant_model.onnx", opset_version=12)