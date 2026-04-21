import streamlit as st
from PIL import Image
import os
import torch
from torch import nn
from torchvision import models, transforms
from ultralytics import YOLO

# ResNet tells us the overall damage category (e.g. front crushed, rear breakage)
# YOLOv8 goes further and draws boxes around the exact damaged parts
# Both run on the same uploaded image — no need to upload twice

RESNET_CLASS_NAMES = [
    'Front Breakage', 'Front Crushed', 'Front Normal',
    'Rear Breakage', 'Rear Crushed', 'Rear Normal'
]

# We only fine-tune layer4 and the final FC — freezing everything else keeps training fast
class CarClassifierResNet(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.model = models.resnet50(weights="DEFAULT")
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.layer4.parameters():
            param.requires_grad = True
        self.model.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.model.fc.in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)

@st.cache_resource
def load_resnet_model():
    # build the path relative to this file so it works on any machine
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "model", "saved_model_Car_Damage_Detection.pth")
    if not os.path.exists(model_path):
        return None, f"ResNet model not found at: {model_path}"
    try:
        model = CarClassifierResNet()
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        model.eval()
        return model, None
    except Exception as e:
        return None, str(e)

def classify_image(model, image_path):
    image = Image.open(image_path).convert("RGB")
    # same normalization that was used during training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(tensor)
        _, predicted = torch.max(output, 1)
    return RESNET_CLASS_NAMES[predicted.item()]


YOLO_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best.onnx")
YOLO_CLASS_NAMES = [
    'Front-Windscreen-Damage', 'Headlight-Damage', 'Major-Rear-Bumper-Dent',
    'Rear-windscreen-Damage', 'RunningBoard-Dent', 'Sidemirror-Damage',
    'Signlight-Damage', 'Taillight-Damage', 'bonnet-dent', 'doorouter-dent',
    'doorouter-scratch', 'fender-dent', 'front-bumper-dent',
    'front-bumper-scratch', 'medium-Bodypanel-Dent', 'paint-chip',
    'paint-trace', 'pillar-dent', 'quaterpanel-dent', 'rear-bumper-dent',
    'rear-bumper-scratch', 'roof-dent'
]
# keeping confidence low (0.10) so we don't miss subtle damage
CONFIDENCE_THRESHOLD = 0.10
IOU_THRESHOLD = 0.45

@st.cache_resource
def load_yolo_model():
    if not os.path.exists(YOLO_MODEL_PATH):
        return None, f"YOLO model not found at: {YOLO_MODEL_PATH}"
    try:
        model = YOLO(YOLO_MODEL_PATH)
        return model, None
    except Exception as e:
        return None, str(e)


st.set_page_config(page_title="Vehicle Damage Detection", page_icon="🚗", layout="wide")
st.title("Vehicle Damage Detection")
st.write("Upload an image or use your webcam to detect damage.")

input_method = st.radio("Choose input method:", ("Upload Image", "Use Webcam"))

image_source = None
# save to a temp file so ResNet can read it from disk
temp_path = "/tmp/temp_combined.jpg"

if input_method == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image_source = Image.open(uploaded_file).convert("RGB")
elif input_method == "Use Webcam":
    camera_image = st.camera_input("Take a picture")
    if camera_image is not None:
        image_source = Image.open(camera_image).convert("RGB")

if image_source is not None:
    image_source.save(temp_path)

    # run YOLOv8 first — its annotated image is the main visual output
    yolo, yolo_err = load_yolo_model()
    if yolo_err:
        st.error(f"Could not load YOLO model: {yolo_err}")
    else:
        with st.spinner("Running damage detection..."):
            results = yolo.predict(
                source=image_source,
                imgsz=640,
                conf=CONFIDENCE_THRESHOLD,
                iou=IOU_THRESHOLD,
                save=False,
                show=False,
                verbose=False,
                device="cpu",
            )

        if results:
            # plot() returns BGR, flip to RGB for Streamlit
            annotated_pil = Image.fromarray(results[0].plot()[..., ::-1])
            st.image(annotated_pil, caption="Detected Damage (YOLOv8)", use_column_width=True)

            detections_found = any(len(r.boxes) > 0 for r in results)
            if not detections_found:
                st.info("No specific damage regions detected.")

    # ResNet result sits right below the image as a quick summary label
    resnet, resnet_err = load_resnet_model()
    if resnet_err:
        st.error(f"Could not load ResNet model: {resnet_err}")
    else:
        with st.spinner("Classifying damage type..."):
            label = classify_image(resnet, temp_path)
        st.success(f"Damage Type (ResNet50): **{label}**")
