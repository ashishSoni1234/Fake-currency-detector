from flask import Flask, render_template, request
import torch
from torchvision import models, transforms
from PIL import Image

app = Flask(__name__)  # ✅ FIXED

# Load model architecture
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 2)
model.load_state_dict(torch.load("model/currency_classifier_resnet18.pth", map_location=device))
model = model.to(device)
model.eval()

# Transformation
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

class_names = ['FAKE 500', 'REAL 500']

def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        _, pred = torch.max(outputs, 1)
        return class_names[pred.item()]

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        if "file" not in request.files:
            return "No file uploaded."
        file = request.files["file"]
        if file.filename == "":
            return "No selected file."
        filepath = "static/" + file.filename
        file.save(filepath)
        prediction = predict_image(filepath)
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":  # ✅ FIXED
    app.run(debug=True)
