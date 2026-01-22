import torch
import torchvision.transforms as transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile
import io

from models.tea_model import TeaNet

# initialize the FastAPI app
app = FastAPI()

# model config
MODEL_PATH = "saved_models/tea_grading_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# class labels
GRADE_LABELS = ["OP", "OP1", "OPA"]
QUALITY_LABELS = [str(i) for i in range(1, 11)]

# load trained model
model = TeaNet(num_grades=3, num_qualities=10)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# prediction endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # preprocess
        image = transform(image).unsqueeze(0).to(DEVICE)

        # run through the model
        with torch.no_grad():
            grade_out, quality_out = model(image)

            # get prediction indices
            grade_pred = torch.argmax(grade_out, dim=1).item()
            quality_pred = torch.argmax(quality_out, dim=1).item()

        # convert indices to labels
        grade_label = GRADE_LABELS[grade_pred]
        quality_label = QUALITY_LABELS[quality_pred]

        # success response
        return {
            "success": True,
            "response": {
                "status": 200,
                "data": {
                    "grade": grade_label,
                    "quality": quality_label
                }
            }
        }

    except Exception as e:
        # error response
        return {
            "success": False,
            "response": {
                "status": 401,
                "data": {
                    "message": str(e)
                }
            }
        }