import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
from ultralytics import YOLO
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageDraw, ImageFont
import io
import numpy as np
import cv2
import base64

# ==============================
# 1. FastAPI Initialization
# ==============================

app = FastAPI()

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================
# 2. Pest Remedies (Common for All Models)
# ==============================

pest_remedies = {
    "rsc": "Apply neem oil or insecticidal soap to control this pest.",
    "looper": "Use Bacillus thuringiensis (Bt) as a natural pesticide.",
    "rsm": "Maintain high humidity and use acaricides to manage mites.",
    "thrips": "Introduce beneficial insects like lacewings or use blue sticky traps.",
    "jassid": "Spray with imidacloprid or other systemic insecticides.",
    "tmb": "Remove affected leaves and apply copper fungicides.",
    "healthy": "No action needed; the plant is healthy!"
}

# ==============================
# 3. CNN Model for Classification
# ==============================

class CustomCNN(nn.Module):
    def __init__(self, num_classes=6):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 32 * 32, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

cnn_model = CustomCNN(num_classes=6)
cnn_model.load_state_dict(torch.load("model.pt", map_location=torch.device("cpu")))
cnn_model.eval()

cnn_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

cnn_class_names = ["TMB", "Thrips", "Jassid", "Looper", "RSM", "RSC"]

@app.post("/predict-cnn/")
async def predict_cnn(file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        image = cnn_transform(image).unsqueeze(0)

        with torch.no_grad():
            output = cnn_model(image)
            predicted_class = torch.argmax(output, dim=1).item()

        return {"class": cnn_class_names[predicted_class], 
                "confidence": torch.softmax(output, dim=1)[0][predicted_class].item(), 
                "remedy": pest_remedies[cnn_class_names[predicted_class].lower()]}
    
    except Exception as e:
        return {"error": str(e)}

# ==============================
# 4. YOLO Model for Object Detection
# ==============================

yolo_model = YOLO("best.pt")

yolo_class_names = ["rsc", "looper", "thrips", "jassid", "rsm", "tmb", "healthy"]

@app.post("/predict-yolo/")
async def predict_yolo(file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        image_cv = np.array(image)
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

        results = yolo_model(image)

        predictions = []
        for result in results:
            for box in result.boxes:
                label = yolo_class_names[int(box.cls)]
                predictions.append({"label": label, "confidence": float(box.conf), "remedy": pest_remedies[label]})

        return {"predictions": predictions}
    
    except Exception as e:
        return {"error": str(e)}

# ==============================
# 5. Mask R-CNN Model for Instance Segmentation
# ==============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mrcnn_model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
mrcnn_model.to(device)
mrcnn_model.eval()

mrcnn_class_names = {1: "rsc", 2: "looper", 3: "rsm", 4: "thrips", 5: "jassid", 6: "tmb", 7: "healthy"}

@app.post("/predict-mrcnn/")
async def predict_mrcnn(file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        input_tensor = F.to_tensor(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = mrcnn_model(input_tensor)

        pred_scores = outputs[0]['scores'].cpu().numpy()
        pred_labels = outputs[0]['labels'].cpu().numpy()

        if len(pred_scores) == 0 or max(pred_scores) < 0.35:
            return {"message": "No high-confidence predictions"}

        highest_pred = np.argmax(pred_scores)
        class_name = mrcnn_class_names[pred_labels[highest_pred]]

        return {"prediction": class_name, "confidence": float(pred_scores[highest_pred]), "remedy": pest_remedies[class_name]}
    
    except Exception as e:
        return {"error": str(e)}

# ==============================
# 6. Run FastAPI
# ==============================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
