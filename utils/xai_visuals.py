"""
MediAssist - Explainable AI (XAI) Visualizations
LIME Token Attribution & Grad-CAM Anatomical Saliency
Author: Zelvora Industry Labs / MediAssist Team
"""
import io
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def generate_lime_text_explanation(
    text: str,
    classifier_model: Any,
    vectorizer: Any,
    num_features: int = 8
) -> Dict[str, Any]:
    """
    Computes Local Interpretable Model-agnostic Explanations (LIME) for clinical text.
    Identifies which specific words/tokens drove the triage classification.
    """
    if not text or classifier_model is None or vectorizer is None:
        return {"tokens": [], "weights": [], "plot": None}
    
    try:
        from lime.lime_text import LimeTextExplainer
        
        def predict_proba_fn(texts: List[str]):
            feats = vectorizer.transform(texts)
            return classifier_model.predict_proba(feats)
        
        class_names = list(classifier_model.classes_)
        explainer = LimeTextExplainer(
            class_names=class_names,
            bow=True,
            split_expression=r"\W+",
            random_state=42
        )
        
        # Explain instance
        exp = explainer.explain_instance(
            text,
            predict_proba_fn,
            num_features=num_features,
            labels=[0, 1, 2] if len(class_names) >= 3 else [0]
        )
        
        available_labels = exp.available_labels()
        target_label = available_labels[0] if available_labels else 0
        raw_weights = exp.as_list(label=target_label)
        
        tokens = [item[0] for item in raw_weights]
        weights = [round(float(item[1]), 4) for item in raw_weights]
        predicted_class = class_names[target_label] if target_label < len(class_names) else "Predicted Class"
        
        # Build clean Plotly horizontal bar chart
        df_lime = [{"Token": t, "Importance": w} for t, w in zip(tokens, weights)]
        # Sort by absolute weight
        df_lime = sorted(df_lime, key=lambda x: abs(x["Importance"]), reverse=False)
        
        tokens_sorted = [d["Token"] for d in df_lime]
        weights_sorted = [d["Importance"] for d in df_lime]
        colors = ["#ef4444" if w > 0 else "#3b82f6" for w in weights_sorted]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=weights_sorted,
            y=tokens_sorted,
            orientation="h",
            marker=dict(color=colors),
            text=[f"{w:+.3f}" for w in weights_sorted],
            textposition="auto"
        ))
        
        fig.update_layout(
            title=f"LIME Token Attributions for '{predicted_class}'",
            xaxis_title="Weight (Influence toward category)",
            yaxis_title="Clinical Token",
            height=280,
            margin=dict(l=10, r=10, t=35, b=10),
            template="plotly_white"
        )
        
        return {
            "tokens": tokens,
            "weights": weights,
            "predicted_class": predicted_class,
            "plot": fig
        }
    except Exception as e:
        # Fallback keyword attribution using TF-IDF weights directly
        try:
            feats = vectorizer.transform([text])
            feature_names = np.array(vectorizer.get_feature_names_out())
            scores = feats.toarray()[0]
            top_indices = np.argsort(scores)[-num_features:][::-1]
            
            fallback_tokens = [feature_names[i] for i in top_indices if scores[i] > 0]
            fallback_weights = [round(float(scores[i]), 4) for i in top_indices if scores[i] > 0]
            
            fig = px.bar(
                x=fallback_weights,
                y=fallback_tokens,
                orientation="h",
                labels={"x": "TF-IDF Weight", "y": "Token"},
                title="Symptom Feature Importance (TF-IDF Attribution)"
            )
            fig.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10))
            return {
                "tokens": fallback_tokens,
                "weights": fallback_weights,
                "predicted_class": "Active Presentation",
                "plot": fig
            }
        except Exception:
            return {"tokens": [], "weights": [], "plot": None}


def generate_gradcam_overlay(
    image_bytes: bytes,
    keras_model: Any,
    class_index: int = 0
) -> Tuple[Optional[Image.Image], Optional[Image.Image], Optional[Image.Image]]:
    """
    Computes a Grad-CAM activation heatmap for a chest radiograph and generates:
    1. Original Radiograph (PIL)
    2. Jet Color Heatmap (PIL)
    3. Superimposed Blend (PIL)
    """
    if not image_bytes or keras_model is None:
        return None, None, None
    
    try:
        import tensorflow as tf
        from matplotlib import cm
        
        orig_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        target_size = (224, 224)
        resized_img = orig_img.resize(target_size)
        
        img_array = np.array(resized_img, dtype=np.float32) / 255.0
        img_tensor = np.expand_dims(img_array, axis=0)
        
        # Locate the final 4D convolutional layer
        conv_layer = None
        for layer in reversed(keras_model.layers):
            if len(layer.output_shape) == 4:
                conv_layer = layer
                break
        
        if conv_layer is None:
            return orig_img, None, orig_img
        
        # Construct gradient model
        grad_model = tf.keras.models.Model(
            inputs=[keras_model.inputs],
            outputs=[conv_layer.output, keras_model.output]
        )
        
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_tensor)
            loss = predictions[:, class_index]
        
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # ReLU and normalize
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        heatmap_np = heatmap.numpy()
        
        # Resize heatmap to match original image dimensions
        heatmap_pil = Image.fromarray(np.uint8(255 * heatmap_np)).resize(orig_img.size, Image.BILINEAR)
        heatmap_resized = np.array(heatmap_pil)
        
        # Apply colormap (jet)
        jet = cm.get_cmap("jet")
        jet_colors = jet(heatmap_resized / 255.0)[:, :, :3]
        jet_heatmap_pil = Image.fromarray(np.uint8(255 * jet_colors))
        
        # Superimpose: 65% original + 35% heatmap
        orig_np = np.array(orig_img, dtype=np.float32)
        blended = np.uint8(orig_np * 0.65 + np.array(jet_heatmap_pil, dtype=np.float32) * 0.35)
        blended_pil = Image.fromarray(blended)
        
        return orig_img, jet_heatmap_pil, blended_pil
    except Exception:
        # Graceful fallback: return original image
        try:
            fallback = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            return fallback, None, fallback
        except Exception:
            return None, None, None


def render_xai_comparison_columns(
    orig_img: Optional[Image.Image],
    blended_img: Optional[Image.Image],
    probabilities: Dict[str, float],
    threshold: float = 0.40
) -> None:
    """
    Renders a side-by-side Explainable AI comparison in Streamlit:
    Column 1: Original Radiograph
    Column 2: Grad-CAM Attention Heatmap (Superimposed)
    Column 3: Multi-Label Probability Bar Chart with Threshold Guideline
    """
    col1, col2, col3 = st.columns([1, 1, 1.2])
    
    with col1:
        st.markdown("**1. Original Radiograph**")
        if orig_img is not None:
            st.image(orig_img, use_container_width=True, caption="Source Chest X-Ray (PA/AP)")
        else:
            st.info("No radiograph image loaded.")
            
    with col2:
        st.markdown("**2. Grad-CAM Anatomical Saliency**")
        if blended_img is not None:
            st.image(blended_img, use_container_width=True, caption="Attention Heatmap Overlay (Red = High Saliency)")
        else:
            st.caption("Grad-CAM overlay requires active neural network weights.")
            
    with col3:
        st.markdown("**3. Pathology Probabilities**")
        if probabilities:
            df_prob = [{"Pathology": k, "Confidence": v * 100.0} for k, v in probabilities.items()]
            df_prob = sorted(df_prob, key=lambda x: x["Confidence"], reverse=True)
            
            fig = px.bar(
                df_prob,
                x="Confidence",
                y="Pathology",
                orientation="h",
                text=[f"{item['Confidence']:.1f}%" for item in df_prob],
                color="Confidence",
                color_continuous_scale="Blues",
                range_x=[0, 100]
            )
            # Add vertical line for confidence threshold
            fig.add_vline(
                x=threshold * 100.0,
                line_dash="dash",
                line_color="#ef4444",
                annotation_text=f"Threshold ({threshold*100:.0f}%)",
                annotation_position="top right"
            )
            fig.update_layout(
                height=260,
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis_title="Confidence (%)",
                yaxis_title="",
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("No probability predictions available.")
