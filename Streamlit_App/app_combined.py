import io
import streamlit as st
from PIL import Image, ImageDraw
import os
import numpy as np
import onnxruntime as ort
import torch
from torch import nn
from torchvision import models, transforms

RESNET_CLASS_NAMES = [
    'Front Breakage', 'Front Crushed', 'Front Normal',
    'Rear Breakage', 'Rear Crushed', 'Rear Normal'
]

YOLO_CLASS_NAMES = [
    'Front-Windscreen-Damage', 'Headlight-Damage', 'Major-Rear-Bumper-Dent',
    'Rear-windscreen-Damage', 'RunningBoard-Dent', 'Sidemirror-Damage',
    'Signlight-Damage', 'Taillight-Damage', 'bonnet-dent', 'doorouter-dent',
    'doorouter-scratch', 'fender-dent', 'front-bumper-dent',
    'front-bumper-scratch', 'medium-Bodypanel-Dent', 'paint-chip',
    'paint-trace', 'pillar-dent', 'quaterpanel-dent', 'rear-bumper-dent',
    'rear-bumper-scratch', 'roof-dent'
]

CONFIDENCE_THRESHOLD = 0.10
IOU_THRESHOLD = 0.45
YOLO_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best.onnx")


# --- ResNet model ---

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

def classify_image(model, image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor = transform(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        output = model(tensor)
        _, predicted = torch.max(output, 1)
    return RESNET_CLASS_NAMES[predicted.item()]


# --- YOLO ONNX inference (no ultralytics / opencv required) ---

@st.cache_resource
def load_yolo_session():
    if not os.path.exists(YOLO_MODEL_PATH):
        return None, f"YOLO model not found at: {YOLO_MODEL_PATH}"
    try:
        session = ort.InferenceSession(YOLO_MODEL_PATH, providers=["CPUExecutionProvider"])
        return session, None
    except Exception as e:
        return None, str(e)

def _letterbox(image, size=640):
    orig_w, orig_h = image.size
    scale = min(size / orig_w, size / orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    resized = image.resize((new_w, new_h), Image.BILINEAR)
    padded = Image.new("RGB", (size, size), (114, 114, 114))
    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2
    padded.paste(resized, (pad_x, pad_y))
    arr = np.array(padded, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)[np.newaxis]  # BCHW
    return arr, scale, pad_x, pad_y

def _nms(boxes, scores, iou_threshold):
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[1:][iou <= iou_threshold]
    return keep

def run_yolo(session, image):
    orig_w, orig_h = image.size
    inp, scale, pad_x, pad_y = _letterbox(image)

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: inp})

    # YOLOv8 ONNX output: [1, 4+num_classes, 8400]
    pred = outputs[0][0].T  # [8400, 4+num_classes]
    boxes_xywh = pred[:, :4]
    class_scores = pred[:, 4:]

    class_ids = np.argmax(class_scores, axis=1)
    confs = class_scores[np.arange(len(class_scores)), class_ids]

    mask = confs >= CONFIDENCE_THRESHOLD
    boxes_xywh, confs, class_ids = boxes_xywh[mask], confs[mask], class_ids[mask]

    if len(boxes_xywh) == 0:
        return image.copy(), []

    # cx,cy,w,h → x1,y1,x2,y2 in original image coordinates
    x1 = (boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2 - pad_x) / scale
    y1 = (boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2 - pad_y) / scale
    x2 = (boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2 - pad_x) / scale
    y2 = (boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2 - pad_y) / scale
    x1 = np.clip(x1, 0, orig_w)
    y1 = np.clip(y1, 0, orig_h)
    x2 = np.clip(x2, 0, orig_w)
    y2 = np.clip(y2, 0, orig_h)
    boxes = np.stack([x1, y1, x2, y2], axis=1)

    keep = _nms(boxes, confs, IOU_THRESHOLD)
    boxes, confs, class_ids = boxes[keep], confs[keep], class_ids[keep]

    # Draw boxes with PIL
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    for (bx1, by1, bx2, by2), conf, cls_id in zip(boxes, confs, class_ids):
        label = f"{YOLO_CLASS_NAMES[int(cls_id)]} {conf:.2f}"
        draw.rectangle([bx1, by1, bx2, by2], outline="red", width=3)
        draw.rectangle([bx1, by1, bx1 + len(label) * 7, by1 + 16], fill="red")
        draw.text((bx1 + 2, by1), label, fill="white")

    detected = [YOLO_CLASS_NAMES[int(c)] for c in class_ids]
    return annotated, detected


# --- Streamlit UI ---

st.set_page_config(page_title="Vehicle Damage Detection", page_icon="🚗", layout="wide")
st.title("Vehicle Damage Detection")
st.write("Upload an image or use your webcam to detect damage.")

input_method = st.radio("Choose input method:", ("Upload Image", "Use Webcam"))

image_source = None

if input_method == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Read bytes explicitly so the file pointer is always at the start
        image_bytes = uploaded_file.read()
        image_source = Image.open(io.BytesIO(image_bytes)).convert("RGB")
elif input_method == "Use Webcam":
    camera_image = st.camera_input("Take a picture")
    if camera_image is not None:
        image_bytes = camera_image.read()
        image_source = Image.open(io.BytesIO(image_bytes)).convert("RGB")

if image_source is not None:
    st.image(image_source, caption="Uploaded Image", use_container_width=True)

    yolo_session, yolo_err = load_yolo_session()
    if yolo_err:
        st.error(f"Could not load YOLO model: {yolo_err}")
    else:
        try:
            with st.spinner("Running damage detection..."):
                annotated, detected = run_yolo(yolo_session, image_source)
            st.image(annotated, caption="Detected Damage (YOLOv8)", use_container_width=True)
            if not detected:
                st.info("No specific damage regions detected.")
        except Exception as e:
            st.error(f"YOLO inference failed: {e}")

    resnet, resnet_err = load_resnet_model()
    if resnet_err:
        st.error(f"Could not load ResNet model: {resnet_err}")
    else:
        try:
            with st.spinner("Classifying damage type..."):
                label = classify_image(resnet, image_source)
            st.success(f"Damage Type (ResNet50): **{label}**")
        except Exception as e:
            st.error(f"ResNet inference failed: {e}")
