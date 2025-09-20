# predict_api.py
import torch
from torch.nn import functional as F
from fastapi import FastAPI, UploadFile, Form, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import BertTokenizer
from PIL import Image
from torchvision import transforms
import io

from model import CivicIssueModel

app = FastAPI()

# ------------------------
# CORS (open for all origins)
# ------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # ✅ allow all origins
    allow_credentials=True,
    allow_methods=["*"],      # ✅ allow all HTTP methods
    allow_headers=["*"],      # ✅ allow all headers
    expose_headers=["Content-Type", "Authorization"]
)

# ------------------------
# 1. Setup
# ------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Image preprocessing
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load trained model
model = CivicIssueModel(category_classes=5, priority_classes=4).to(device)
model.load_state_dict(torch.load("models/multi_modal_model.pth", map_location=device))
model.eval()

# ------------------------
# 2. API Route
# ------------------------
@app.post("/predict")
async def predict(
    text: str = Form(None),
    description: str = Form(None),
    lat: float = Form(None),
    latitude: float = Form(None),
    lon: float = Form(None),
    longitude: float = Form(None),
    file: UploadFile = File(None),
):
    try:
        desc = description or text
        if not desc:
            return JSONResponse({"error": "description/text is required"}, status_code=400)

        # Tokenize text
        inputs = tokenizer(
            desc, return_tensors="pt", padding="max_length",
            truncation=True, max_length=32
        )
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        # Check for image
        if file is None:
            return JSONResponse({"error": "Image file required"}, status_code=400)
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img = image_transform(img).unsqueeze(0).to(device)

        # Handle coordinates
        lat_val = lat if lat is not None else latitude
        lon_val = lon if lon is not None else longitude
        if lat_val is None or lon_val is None:
            lat_val, lon_val = 0.0, 0.0
        metadata = torch.tensor([[float(lat_val), float(lon_val)]], dtype=torch.float).to(device)

        # Model prediction
        with torch.no_grad():
            category_logits, priority_logits = model(input_ids, attention_mask, img, metadata)
            category_pred = torch.argmax(F.softmax(category_logits, dim=1), dim=1).item()
            priority_pred = torch.argmax(F.softmax(priority_logits, dim=1), dim=1).item()

        return {
            "category_prediction": int(category_pred),
            "priority_prediction": int(priority_pred)
        }

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

