import torch
import torchvision.transforms as transforms
from PIL import Image
import sys
import os
import torch.nn.functional as F

from models.tea_model import TeaNet

# model config
MODEL_PATH = "saved_models/tea_grading_model_v3.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# class labels
GRADE_LABELS = ["OP", "OP1", "OPA", "NOT_TEA"]
QUALITY_LABELS = [str(i) for i in range(1, 11)]

# load trained model
print("loading model...")

model = TeaNet(num_grades=4, num_qualities=10)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

print("model loaded successfully")
print(f"using device: {DEVICE}")

# image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# prediction function
def predict_image(image_path):
    if not os.path.exists(image_path):
        print(f"\nimage not found: {image_path}")
        return

    # read and preprocess image
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)

    # run through the model
    with torch.no_grad():
        grade_out, quality_out = model(image)

        # apply softmax to get probabilities
        grade_probs = F.softmax(grade_out, dim=1)

        # get prediction indices
        grade_pred = torch.argmax(grade_probs, dim=1).item()
        quality_pred = torch.argmax(quality_out, dim=1).item()

        grade_conf = grade_probs[0][grade_pred].item()

    # convert indices to labels
    grade_label = GRADE_LABELS[grade_pred]

    # print results
    print("\nprediction result:")
    print(f"Image: {image_path}")
    print(f"grade: {grade_label}")
    print(f"grade confidence: {grade_conf:.4f}")

    # check for invalid images
    if grade_label == "NOT_TEA":
        print("This image is not a valid tea sample!")
        return

    quality_label = int(QUALITY_LABELS[quality_pred])
    print(f"quality: {quality_label}")

# run from terminal
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_path>")
        print("Example: python predict.py sample.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    predict_image(image_path)
