from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import os
import pandas as pd
from PIL import Image
from transformers import BertTokenizer
from torchvision import transforms
from model import MultiModalClassifier, TextEncoder, ImageEncoder, MetadataEncoder

# ✅ Kaggle API
from kaggle.api.kaggle_api_extended import KaggleApi

app = FastAPI(title="MultiModal Prediction API")

# ✅ Enable CORS so frontend (Netlify) can connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or restrict to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# Download model from Kaggle
# -------------------------------
MODEL_PATH = "models/multi_modal_model.pth"

if not os.path.exists(MODEL_PATH):
    os.makedirs("models", exist_ok=True)
    print("📥 Downloading model from Kaggle...")

    api = KaggleApi()
    api.authenticate()  # Needs kaggle.json in ~/.kaggle

    # Replace with your dataset name
    api.dataset_download_file(
        "gaurishmalhotra/multimodel",
        file_name="multi_modal_model.pth",
        path="models"
    )

    # Kaggle downloads as .zip → unzip
    import zipfile
    zip_path = MODEL_PATH + ".zip"
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall("models")
        os.remove(zip_path)

# -------------------------------
# Load model
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

text_encoder = TextEncoder(out_dim=128)
image_encoder = ImageEncoder(out_dim=128)
meta_encoder = MetadataEncoder(input_dim=2, out_dim=32)

model = MultiModalClassifier(
    text_dim=128, image_dim=128, meta_dim=32,
    category_classes=7, priority_classes=4
)

# ✅ Torch 2.6 fix
state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=False)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

print("✅ Model loaded successfully!")

# -------------------------------
# Tokenizer + Image transforms
# -------------------------------
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -------------------------------
# Predict Endpoint
# -------------------------------
@app.post("/predict")
async def predict_issue(
    description: str = Form(...),
    latitude: float = Form(...),
    longitude: float = Form(...),
    file: UploadFile = File(...)
):
    try:
        # 1. Text
        text_inputs = tokenizer(description, return_tensors="pt", padding=True, truncation=True)
        text_emb = text_encoder(text_inputs["input_ids"], text_inputs["attention_mask"])

        # 2. Image
        img = Image.open(file.file).convert("RGB")
        img = image_transform(img).unsqueeze(0)
        image_emb = image_encoder(img)

        # 3. Metadata
        meta = torch.tensor([[latitude, longitude]], dtype=torch.float32)
        meta_emb = meta_encoder(meta)

        # 4. Model Prediction
        with torch.no_grad():
            category_logits, priority_logits = model(text_emb, image_emb, meta_emb)

        category_pred = torch.argmax(category_logits, dim=1).item()
        priority_pred = torch.argmax(priority_logits, dim=1).item()

        # Save record
        df = pd.DataFrame([{
            "description": description,
            "latitude": latitude,
            "longitude": longitude,
            "category_pred": category_pred,
            "priority_pred": priority_pred
        }])
        os.makedirs("data/user_reports", exist_ok=True)
        df.to_csv("data/user_reports/predictions.csv", mode="a",
                  header=not os.path.exists("data/user_reports/predictions.csv"),
                  index=False)

        return JSONResponse({
            "category_prediction": category_pred,
            "priority_prediction": priority_pred
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
