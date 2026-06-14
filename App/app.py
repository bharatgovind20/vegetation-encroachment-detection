import segmentation_models_pytorch as smp
from albumentations.pytorch import ToTensorV2
import albumentations as A
from PIL import Image
from pathlib import Path
import streamlit as st
import numpy as np
import torch
import cv2
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


# ----------------------------
# Page Configuration
# ----------------------------

st.set_page_config(
    page_title="Vegetation Encroachment Detection",
    page_icon="🌿",
    layout="wide"
)

st.title("🌿 Vegetation Encroachment Detection Near Power Lines")
st.markdown(
    """
    This dashboard uses a trained **Attention U-Net** model to detect vegetation and power-line/corridor regions 
    from UAV/RGB imagery and estimate **2D proximity-based encroachment risk**.
    
    **Class Mapping:**  
    - Black = Background  
    - Red = Power line / Utility corridor  
    - Green = Vegetation  
    """
)


# ----------------------------
# Paths and Constants
# ----------------------------

MODEL_PATH = Path(
    r"C:\BITS AI ML\Project final\Vegetation_Encroachment_Project\models\attention_unet_vepl_dice_focal_best.pth")

IMAGE_SIZE = 256
GSD_FACTOR = 0.03  # assumed meters per pixel


# ----------------------------
# Model Loading
# ----------------------------

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=3,
        decoder_attention_type="scse"
    )

    try:
        state_dict = torch.load(
            MODEL_PATH,
            map_location=device,
            weights_only=True
        )
    except TypeError:
        state_dict = torch.load(
            MODEL_PATH,
            map_location=device
        )

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model, device


# ----------------------------
# Image Transform
# ----------------------------

transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(),
    ToTensorV2()
])


# ----------------------------
# Risk Assessment Logic
# ----------------------------

def calculate_risk_final(pred_mask, gsd_factor=0.03, min_corridor_percent=0.5):
    """
    Final 2D proximity-based vegetation encroachment risk assessment.

    Class mapping:
    0 = Background
    1 = Power line / Utility Corridor
    2 = Vegetation
    """

    background_mask = (pred_mask == 0).astype(np.uint8)
    corridor_mask = (pred_mask == 1).astype(np.uint8)
    vegetation_mask = (pred_mask == 2).astype(np.uint8)

    vegetation_pixels = int(vegetation_mask.sum())
    corridor_pixels = int(corridor_mask.sum())
    background_pixels = int(background_mask.sum())

    total_pixels = pred_mask.shape[0] * pred_mask.shape[1]

    vegetation_coverage_percent = vegetation_pixels / total_pixels * 100
    corridor_coverage_percent = corridor_pixels / total_pixels * 100
    background_coverage_percent = background_pixels / total_pixels * 100

    if vegetation_pixels == 0:
        return {
            "risk_level": "No Vegetation Detected",
            "min_distance_pixels": None,
            "risk_distance_pixels": None,
            "distance_meters": None,
            "vegetation_coverage_percent": vegetation_coverage_percent,
            "corridor_coverage_percent": corridor_coverage_percent,
            "background_coverage_percent": background_coverage_percent,
            "critical_zone_percent": 0,
            "high_zone_percent": 0,
            "medium_zone_percent": 0,
            "reliability_flag": "Warning: No vegetation detected. Manual review recommended."
        }

    if corridor_pixels == 0 or corridor_coverage_percent < min_corridor_percent:
        return {
            "risk_level": "Manual Review - Corridor Not Reliably Detected",
            "min_distance_pixels": None,
            "risk_distance_pixels": None,
            "distance_meters": None,
            "vegetation_coverage_percent": vegetation_coverage_percent,
            "corridor_coverage_percent": corridor_coverage_percent,
            "background_coverage_percent": background_coverage_percent,
            "critical_zone_percent": 0,
            "high_zone_percent": 0,
            "medium_zone_percent": 0,
            "reliability_flag": "Warning: Power-line/corridor prediction too low. Risk result not reliable."
        }

    if corridor_coverage_percent > 30:
        reliability_flag = "Warning: Power-line/corridor prediction unusually high. Manual review recommended."
    else:
        reliability_flag = "Prediction coverage normal."

    corridor_inverse = 1 - corridor_mask

    distance_map = cv2.distanceTransform(
        corridor_inverse,
        cv2.DIST_L2,
        5
    )

    vegetation_distances = distance_map[vegetation_mask == 1]

    min_distance = float(np.min(vegetation_distances))
    risk_distance = float(np.percentile(vegetation_distances, 5))
    distance_meters = risk_distance * gsd_factor

    critical_zone_percent = float(np.mean(vegetation_distances <= 5) * 100)
    high_zone_percent = float(np.mean(vegetation_distances <= 15) * 100)
    medium_zone_percent = float(np.mean(vegetation_distances <= 30) * 100)

    if risk_distance <= 5:
        risk_level = "Critical"
    elif risk_distance <= 15:
        risk_level = "High"
    elif risk_distance <= 30:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    return {
        "risk_level": risk_level,
        "min_distance_pixels": min_distance,
        "risk_distance_pixels": risk_distance,
        "distance_meters": distance_meters,
        "vegetation_coverage_percent": vegetation_coverage_percent,
        "corridor_coverage_percent": corridor_coverage_percent,
        "background_coverage_percent": background_coverage_percent,
        "critical_zone_percent": critical_zone_percent,
        "high_zone_percent": high_zone_percent,
        "medium_zone_percent": medium_zone_percent,
        "reliability_flag": reliability_flag
    }


# ----------------------------
# Visualization Helpers
# ----------------------------

def colorize_mask(mask):
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    color_mask[mask == 0] = [0, 0, 0]        # Background - black
    color_mask[mask == 1] = [255, 0, 0]      # Power line/corridor - red
    color_mask[mask == 2] = [0, 255, 0]      # Vegetation - green

    return color_mask


def create_overlay(image, mask, alpha=0.45):
    image = image.copy()

    if image.max() <= 1:
        image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)

    color_mask = np.zeros_like(image)

    color_mask[mask == 1] = [255, 0, 0]
    color_mask[mask == 2] = [0, 255, 0]

    overlay = cv2.addWeighted(image, 1 - alpha, color_mask, alpha, 0)

    return overlay


def get_recommendation(risk_level):
    if risk_level == "Critical":
        return "Urgent inspection and vegetation trimming recommended."
    elif risk_level == "High":
        return "Schedule field inspection soon. Vegetation is close to the corridor."
    elif risk_level == "Medium":
        return "Monitor the area and plan preventive maintenance."
    elif risk_level == "Low":
        return "No immediate action required. Continue routine monitoring."
    elif "Manual Review" in risk_level:
        return "Manual review required because corridor detection is not reliable."
    else:
        return "Review prediction output before taking action."


# ----------------------------
# Prediction Function
# ----------------------------

def predict_image(uploaded_image, model, device):
    image = Image.open(uploaded_image).convert("RGB")
    image_np = np.array(image)

    original_resized = cv2.resize(image_np, (IMAGE_SIZE, IMAGE_SIZE))

    augmented = transform(image=image_np)
    input_tensor = augmented["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.argmax(output, dim=1)

    pred_mask = pred[0].cpu().numpy()

    color_mask = colorize_mask(pred_mask)
    overlay = create_overlay(original_resized, pred_mask)
    risk_result = calculate_risk_final(pred_mask, gsd_factor=GSD_FACTOR)

    return original_resized, color_mask, overlay, risk_result


# ----------------------------
# Main App
# ----------------------------

if not MODEL_PATH.exists():
    st.error(f"Model file not found at: {MODEL_PATH}")
    st.stop()

model, device = load_model()

st.sidebar.header("Model Information")
st.sidebar.write("Model: Attention U-Net")
st.sidebar.write("Encoder: ResNet34")
st.sidebar.write("Loss used during training: Dice + Focal Loss")
st.sidebar.write("Input size: 256 × 256")
st.sidebar.write(f"Device: {device}")

st.sidebar.header("Risk Thresholds")
st.sidebar.write("Critical: ≤ 5 px")
st.sidebar.write("High: 6–15 px")
st.sidebar.write("Medium: 16–30 px")
st.sidebar.write("Low: > 30 px")

st.sidebar.warning(
    "This system provides 2D projected risk screening. "
    "DSM/LiDAR is required for true 3D clearance validation."
)

uploaded_file = st.file_uploader(
    "Upload a UAV/RGB image",
    type=["jpg", "jpeg", "png", "tif", "tiff"]
)

if uploaded_file is not None:
    with st.spinner("Running segmentation and risk assessment..."):
        original, mask, overlay, risk_result = predict_image(
            uploaded_file,
            model,
            device
        )

    st.subheader("Prediction Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(original, caption="Original Image", use_container_width=True)

    with col2:
        st.image(mask, caption="Predicted Mask", use_container_width=True)

    with col3:
        st.image(overlay, caption="Overlay", use_container_width=True)

    st.subheader("Risk Assessment")

    risk_level = risk_result["risk_level"]

    if risk_level == "Critical":
        st.error(f"Risk Level: {risk_level}")
    elif risk_level == "High":
        st.warning(f"Risk Level: {risk_level}")
    elif risk_level == "Medium":
        st.info(f"Risk Level: {risk_level}")
    elif risk_level == "Low":
        st.success(f"Risk Level: {risk_level}")
    else:
        st.warning(f"Risk Level: {risk_level}")

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

    with metric_col1:
        value = risk_result["risk_distance_pixels"]
        st.metric(
            "Risk Distance",
            "N/A" if value is None else f"{value:.2f} px"
        )

    with metric_col2:
        value = risk_result["distance_meters"]
        st.metric(
            "Approx. Distance",
            "N/A" if value is None else f"{value:.3f} m"
        )

    with metric_col3:
        st.metric(
            "Vegetation Coverage",
            f"{risk_result['vegetation_coverage_percent']:.2f}%"
        )

    with metric_col4:
        st.metric(
            "Corridor Coverage",
            f"{risk_result['corridor_coverage_percent']:.2f}%"
        )

    st.subheader("Detailed Metrics")

    st.table({
        "Metric": [
            "Background Coverage",
            "Vegetation Coverage",
            "Power-line/Corridor Coverage",
            "Critical Zone Vegetation",
            "High Zone Vegetation",
            "Medium Zone Vegetation",
            "Reliability Flag",
            "Recommendation"
        ],
        "Value": [
            f"{risk_result['background_coverage_percent']:.2f}%",
            f"{risk_result['vegetation_coverage_percent']:.2f}%",
            f"{risk_result['corridor_coverage_percent']:.2f}%",
            f"{risk_result['critical_zone_percent']:.2f}%",
            f"{risk_result['high_zone_percent']:.2f}%",
            f"{risk_result['medium_zone_percent']:.2f}%",
            risk_result["reliability_flag"],
            get_recommendation(risk_level)
        ]
    })

    st.subheader("Important Note")
    st.info(
        "The current system estimates 2D projected vegetation-to-corridor proximity. "
        "It cannot determine whether a power line is physically above vegetation. "
        "DSM/LiDAR integration is required for height-aware clearance validation."
    )

else:
    st.info("Upload an image to run vegetation encroachment detection.")
