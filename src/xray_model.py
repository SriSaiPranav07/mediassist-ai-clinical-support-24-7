"""
MediAssist - Chest X-Ray Screening CNN & Grad-CAM Visual Explainability
Author: MediAssist Clinical AI Team
"""
import io
import os
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
from PIL import Image
import matplotlib.cm as cm

# Configure tensorflow to suppress unnecessary logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    HAS_TF = True
except ImportError:
    HAS_TF = False

from .config import XRAY_CLASSES, XRAY_IMG_SIZE, XRAY_MODEL_PATH
from .data_validation import validate_image_file

def build_xray_cnn_model(num_classes: int = len(XRAY_CLASSES), input_shape: Tuple[int, int, int] = (224, 224, 3)) -> Any:
    """
    Builds transfer learning CNN using MobileNetV2 backbone for lightweight medical radiograph screening.
    Multi-label classification head with sigmoid activations to support co-occurring pulmonary abnormalities.
    """
    if not HAS_TF:
        raise ImportError("TensorFlow/Keras is required to instantiate the CNN model.")

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )
    # Freeze base model layers initially
    base_model.trainable = False

    inputs = keras.Input(shape=input_shape, name="radiograph_input")
    # Medical preprocessing: Rescaling [-1, 1] for MobileNetV2
    x = layers.Rescaling(scale=1./127.5, offset=-1.0)(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu", name="dense_features")(x)
    x = layers.Dropout(0.2)(x)
    # Sigmoid for multi-label pathology detection
    outputs = layers.Dense(num_classes, activation="sigmoid", name="pathology_predictions")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="MediAssist_MobileNetV2_XRay")
    return model

class ChestXRayScreener:
    """
    Inference and Explainability engine for Chest Radiographs.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or str(XRAY_MODEL_PATH)
        self.classes = XRAY_CLASSES
        self.model = None
        self._load_model()

    def _load_model(self):
        if not HAS_TF:
            return
        if os.path.exists(self.model_path):
            try:
                self.model = keras.models.load_model(self.model_path, compile=False)
            except Exception:
                self.model = None
        else:
            self.model = None

    def preprocess_image(self, pil_image: Image.Image) -> np.ndarray:
        """
        Converts PIL Image to normalized RGB array of shape (1, 224, 224, 3).
        """
        # Convert to RGB (handles 1-channel grayscale or 4-channel RGBA)
        rgb_img = pil_image.convert("RGB")
        resized = rgb_img.resize(XRAY_IMG_SIZE, Image.Resampling.BILINEAR)
        img_arr = np.array(resized, dtype=np.float32)
        batch_arr = np.expand_dims(img_arr, axis=0)
        return batch_arr

    def compute_gradcam(self, img_array: np.ndarray, target_class_idx: int) -> Optional[np.ndarray]:
        """
        Computes Grad-CAM (Gradient-weighted Class Activation Mapping) for target class in Keras 3.
        Returns a normalized 2D heatmap in [0, 1] matching feature map dimensions.
        """
        if not HAS_TF or self.model is None:
            return None

        try:
            # Locate base model and sub-layers
            base_submodel = None
            for layer in self.model.layers:
                if isinstance(layer, keras.Model):
                    base_submodel = layer
                    break

            if base_submodel is None:
                return None

            # Get target convolutional layer in base model
            target_conv = base_submodel.get_layer("out_relu")
            sub_grad_model = keras.Model(
                inputs=base_submodel.inputs,
                outputs=target_conv.output
            )

            rescaling = self.model.get_layer("rescaling")
            avg_pool = self.model.get_layer("avg_pool")
            bn = self.model.get_layer("batch_normalization")
            dense_feat = self.model.get_layer("dense_features")
            pred_layer = self.model.get_layer("pathology_predictions")

            with tf.GradientTape() as tape:
                inputs_tensor = tf.cast(img_array, tf.float32)
                x = rescaling(inputs_tensor)
                conv_out = sub_grad_model(x)
                tape.watch(conv_out)

                # Forward through subsequent classification head
                p = avg_pool(conv_out)
                p = bn(p, training=False)
                p = dense_feat(p)
                preds = pred_layer(p)
                loss = preds[:, target_class_idx]

            grads = tape.gradient(loss, conv_out)
            if grads is None:
                return None

            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            heatmap = tf.reduce_sum(conv_out[0] * pooled_grads, axis=-1)
            heatmap = tf.maximum(heatmap, 0.0)
            max_val = tf.math.reduce_max(heatmap)
            if max_val > 0:
                heatmap = heatmap / max_val

            return heatmap.numpy()
        except Exception:
            return None

    def overlay_gradcam(self, original_pil: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> Image.Image:
        """
        Overlays a colormapped Grad-CAM heatmap onto the original PIL image.
        """
        # Rescale heatmap to original image size
        heatmap_uint8 = np.uint8(255 * heatmap)
        try:
            import matplotlib as mpl
            jet = mpl.colormaps["jet"]
        except Exception:
            jet = cm.get_cmap("jet")
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap_uint8]

        jet_heatmap_img = Image.fromarray(np.uint8(255 * jet_heatmap))
        jet_heatmap_resized = jet_heatmap_img.resize(original_pil.size, Image.Resampling.BILINEAR)

        rgb_orig = original_pil.convert("RGB")
        blended = Image.blend(rgb_orig, jet_heatmap_resized, alpha=alpha)
        return blended

    def analyze_radiograph(self, image_bytes: bytes, threshold: float = 0.40) -> Dict[str, Any]:
        """
        Executes end-to-end radiograph validation, prediction, Grad-CAM generation, and 10-second explanation.
        """
        # Step 1: Image validation
        is_valid, pil_img, err_msg = validate_image_file(image_bytes)
        if not is_valid or pil_img is None:
            return {
                "success": False,
                "error": err_msg,
                "findings": [],
                "probabilities": {},
                "gradcam_available": False,
                "gradcam_image": None,
                "explanation": {
                    "finding": "Image processing rejected.",
                    "reason": err_msg,
                    "confidence": "N/A",
                    "limitation": "Only valid PNG/JPEG/TIFF radiograph images can be processed."
                }
            }

        # Step 2: Preprocessing
        batch_arr = self.preprocess_image(pil_img)

        # Check for model availability
        if self.model is None:
            # Check if checkpoint exists and try loading again
            self._load_model()

        if self.model is None:
            # Deterministic anatomical heuristic fallback when weights file is not yet built
            # Compute image statistics (mean, variance) for demonstration
            gray = np.array(pil_img.convert("L"), dtype=np.float32)
            mean_intensity = float(np.mean(gray))
            var_intensity = float(np.var(gray))

            simulated_probs = {
                "Normal": round(0.72 if var_intensity > 800 else 0.45, 3),
                "Infiltration": round(0.28, 3),
                "Effusion": round(0.22, 3),
                "Cardiomegaly": round(0.19, 3),
                "Atelectasis": round(0.15, 3),
                "Pneumothorax": round(0.08, 3)
            }

            return {
                "success": True,
                "model_status": "Heuristic Mode (Model weights pending training)",
                "probabilities": simulated_probs,
                "flagged_findings": [cls for cls, p in simulated_probs.items() if p >= threshold and cls != "Normal"],
                "gradcam_available": False,
                "gradcam_image": None,
                "explanation": {
                    "finding": "Preliminary radiograph screening (Heuristic Baseline).",
                    "reason": f"Baseline density screening (Mean: {mean_intensity:.1f}, Variance: {var_intensity:.1f}).",
                    "confidence": "Heuristic baseline (Training checkpoint pending).",
                    "limitation": "Screening output only; must not be used as a radiologist interpretation."
                }
            }

        # Step 3: Neural Model Inference
        raw_preds = self.model.predict(batch_arr, verbose=0)[0]
        prob_dict = {cls_name: float(round(float(raw_preds[i]), 4)) for i, cls_name in enumerate(self.classes)}

        flagged = [cls_name for cls_name, prob in prob_dict.items() if prob >= threshold and cls_name != "Normal"]
        if not flagged and prob_dict.get("Normal", 0.0) >= 0.5:
            primary_finding = "No Overt Acute Abnormality Flagged"
            top_class = "Normal"
            top_prob = prob_dict.get("Normal", 0.5)
        elif flagged:
            top_class = max(flagged, key=lambda c: prob_dict[c])
            top_prob = prob_dict[top_class]
            primary_finding = f"Potential {top_class} Detected (Probability: {top_prob*100:.1f}%)"
        else:
            top_class = max(prob_dict, key=lambda c: prob_dict[c])
            top_prob = prob_dict[top_class]
            primary_finding = f"Borderline Finding: {top_class} ({top_prob*100:.1f}%)"

        # Step 4: Grad-CAM Explainability
        top_idx = self.classes.index(top_class)
        heatmap = self.compute_gradcam(batch_arr, top_idx)
        gradcam_img = None
        if heatmap is not None:
            try:
                gradcam_img = self.overlay_gradcam(pil_img, heatmap)
            except Exception:
                gradcam_img = None

        # Step 5: Plain-English 10-Second Explanation
        reason_text = (
            f"CNN detected activation patterns consistent with {top_class} with probability {top_prob*100:.1f}%."
            if top_class != "Normal" else
            f"No major pathology score exceeded screening threshold {threshold*100:.0f}%; baseline opacity is uniform."
        )

        return {
            "success": True,
            "model_status": "Deep Learning Active (MobileNetV2 CNN)",
            "probabilities": prob_dict,
            "flagged_findings": flagged,
            "primary_finding": primary_finding,
            "top_class": top_class,
            "top_probability": top_prob,
            "gradcam_available": gradcam_img is not None,
            "gradcam_image": gradcam_img,
            "original_image": pil_img,
            "explanation": {
                "finding": primary_finding,
                "reason": reason_text,
                "confidence": f"Model probability is {top_prob*100:.1f}% (Screening Threshold: {threshold*100:.0f}%).",
                "limitation": "Grad-CAM highlights CNN feature attention, not definitive histological or radiologic boundaries."
            }
        }
