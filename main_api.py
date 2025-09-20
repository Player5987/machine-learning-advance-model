from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import kaggle
import os
import torch
from transformers import BertTokenizer
from torchvision import transforms
from PIL import Image
import pandas as pd

# import your model classes
from model import MultiModalClassifier, TextEncoder, ImageEncoder, MetadataEncoder

app = FastAPI()

# -------------------------------
# Kaggle dataset + model setup
# -------------------------------
DATA_DIR = "data"
MODEL_DIR = "models"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

DATASET = "gaurishmalhotra/dataofimagesandtext"   # your images + csv data
MODELSET = "gaurishmalhotra/multimodel"           # your trained model (.pth file)

# Download dataset if not already present
if not os.listdir(DATA_DIR):
    kaggle.api.dataset_download_files(DATASET, path=DATA_DIR, unzip=True)
    print("✅ Dataset downloaded from Kaggle")

# Download model if not already present
if not os.listdir(MODEL_DIR):
    kaggle.api.dataset_download_files(MODELSET, path=MODEL_DIR, unzip=True)
    print("✅ Model downloaded from Kaggle")

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

# Adjust file name if different inside your Kaggle dataset
MODEL_PATH = os.path.join(MODEL_DIR, "multi_modal_model_clean.pth")

if os.path.exists(MODEL_PATH):
    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    print("✅ Model loaded with strict=False")
    print("⚠️ Missing keys:", missing)
    print("⚠️ Unexpected keys:", unexpected)
else:
    print(f"⚠️ Model file not found at {MODEL_PATH}")

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -------------------------------
# Endpoints
# -------------------------------
@app.get("/dataset-info")
async def dataset_info():
    if not os.listdir(DATA_DIR):
        return JSONResponse({"error": "Dataset not loaded"}, status_code=500)
    files = os.listdir(DATA_DIR)
    return {"message": "Dataset downloaded", "files": files}


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

        # 4. Inference
        with torch.no_grad():
            category_logits, priority_logits = model(text_emb, image_emb, meta_emb)

        category_pred = torch.argmax(category_logits, dim=1).item()
        priority_pred = torch.argmax(priority_logits, dim=1).item()

        # Save result
        os.makedirs("data/user_reports", exist_ok=True)
        df = pd.DataFrame([{
            "description": description,
            "latitude": latitude,
            "longitude": longitude,
            "category_pred": category_pred,
            "priority_pred": priority_pred
        }])
        df.to_csv("data/user_reports/predictions.csv",
                  mode="a",
                  header=not os.path.exists("data/user_reports/predictions.csv"),
                  index=False)

        return {"category_prediction": category_pred, "priority_prediction": priority_pred}

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

