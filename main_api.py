import predict_api 
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import torch
import os
from model import MultiModalClassifier, TextEncoder, ImageEncoder, MetadataEncoder
from transformers import BertTokenizer
from torchvision import transforms
from PIL import Image
import pandas as pd

# -------------------------------
# Initialize FastAPI app
# -------------------------------
app = FastAPI(title="MultiModal Prediction API", version="1.0")
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# 🔥 Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # for testing, allow all
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# Paths & constants
# -------------------------------
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "multi_modal_model_clean.pth")
DATA_DIR = "data/user_reports"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# -------------------------------
# Load model and tokenizer
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate model components
text_encoder = TextEncoder(out_dim=128)
image_encoder = ImageEncoder(out_dim=128)
meta_encoder = MetadataEncoder(input_dim=2, out_dim=32)
model = MultiModalClassifier(
    text_dim=128, image_dim=128, meta_dim=32,
    category_classes=7, priority_classes=4
)

# Load trained weights safely
if os.path.exists(MODEL_PATH):
    try:
        state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()
        print("✅ Model loaded successfully with strict=False")
        print("⚠️ Missing keys:", missing)
        print("⚠️ Unexpected keys:", unexpected)
    except Exception as e:
        print(f"⚠️ Failed to load model weights: {e}")
else:
    print(f"⚠️ Model file not found at {MODEL_PATH}")

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Image preprocessing
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -------------------------------
# Health & root endpoints
# -------------------------------
@app.get("/")
async def root():
    return {"message": "✅ API is running. Use /predict to get predictions."}

@app.get("/health")
async def health():
    return {"status": "healthy", "device": str(device)}

# -------------------------------
# Prediction endpoint
# -------------------------------
@app.post("/predict")
async def predict_issue(
    description: str = Form(...),
    latitude: float = Form(...),
    longitude: float = Form(...),
    file: UploadFile = File(...)
):
    try:
        # 1. Process text
        text_inputs = tokenizer(description, return_tensors="pt", padding=True, truncation=True)
        text_emb = text_encoder(text_inputs["input_ids"], text_inputs["attention_mask"])

        # 2. Process image
        img = Image.open(file.file).convert("RGB")
        img = image_transform(img).unsqueeze(0)
        image_emb = image_encoder(img)

        # 3. Process metadata
        meta = torch.tensor([[latitude, longitude]], dtype=torch.float32)
        meta_emb = meta_encoder(meta)

        # 4. Run model
        with torch.no_grad():
            category_logits, priority_logits = model(text_emb, image_emb, meta_emb)

        category_pred = torch.argmax(category_logits, dim=1).item()
        priority_pred = torch.argmax(priority_logits, dim=1).item()

        # 5. Save request + predictions
        df = pd.DataFrame([{
            "description": description,
            "latitude": latitude,
            "longitude": longitude,
            "category_pred": category_pred,
            "priority_pred": priority_pred
        }])
        df.to_csv(os.path.join(DATA_DIR, "predictions.csv"),
                  mode="a", header=not os.path.exists(os.path.join(DATA_DIR, "predictions.csv")),
                  index=False)

        return JSONResponse({
            "category_prediction": category_pred,
            "priority_prediction": priority_pred
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)



