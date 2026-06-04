from __future__ import annotations

import streamlit as st

from PIL import Image

from predict_test_run import build_transform, load_model, predict_pil_image


st.set_page_config(
    page_title="MRI Manipulation Detector",
    page_icon="🧠",
    layout="wide",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
            .stApp {
                background: radial-gradient(circle at top left, #1f2937 0%, #0f172a 45%, #020617 100%);
                color: #e5e7eb;
            }
            .hero {
                padding: 2.2rem 2rem 1.4rem 2rem;
                border-radius: 24px;
                background: rgba(15, 23, 42, 0.72);
                border: 1px solid rgba(148, 163, 184, 0.2);
                box-shadow: 0 18px 60px rgba(0, 0, 0, 0.35);
                margin-bottom: 1.5rem;
            }
            .hero h1 {
                margin: 0;
                font-size: 2.6rem;
                line-height: 1.05;
                color: #f8fafc;
            }
            .hero p {
                margin-top: 0.75rem;
                max-width: 70ch;
                color: #cbd5e1;
                font-size: 1.02rem;
            }
            .result-card {
                padding: 1.25rem 1.3rem;
                border-radius: 18px;
                background: rgba(15, 23, 42, 0.82);
                border: 1px solid rgba(148, 163, 184, 0.18);
                margin-top: 1rem;
            }
            .label {
                font-size: 0.9rem;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                color: #94a3b8;
                margin-bottom: 0.35rem;
            }
            .verdict {
                font-size: 1.8rem;
                font-weight: 800;
                margin: 0;
            }
            .small-note {
                color: #94a3b8;
                font-size: 0.92rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def get_model_and_transform():
    device = st.session_state.get("device")
    if device is None:
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        st.session_state["device"] = device

    model = load_model(device)
    transform = build_transform()
    return device, model, transform


def main() -> None:
    inject_styles()

    st.markdown(
        """
        <div class="hero">
            <h1>MRI Manipulation Detector</h1>
            <p>
                Upload an MRI scan and the existing deepfake model will estimate whether the image is
                real or manipulated. The app uses the trained checkpoint from this project and shows
                a simple confidence score for the result.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    device, model, transform = get_model_and_transform()

    left, right = st.columns([1, 1.1], gap="large")

    with left:
        st.subheader("Upload MRI scan")
        uploaded_file = st.file_uploader(
            "Choose a PNG, JPG, JPEG, BMP, TIFF, or WEBP image.",
            type=["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"],
        )
        analyze = st.button("Analyze image", type="primary", use_container_width=True, disabled=uploaded_file is None)
        st.caption("The model returns the predicted class and confidence for the uploaded scan.")

    with right:
        st.subheader("Prediction")
        if uploaded_file is None:
            st.info("Upload an MRI image to see the prediction here.")
        elif analyze:
            image = Image.open(uploaded_file).convert("RGB")
            label, confidence = predict_pil_image(model, image, device, transform)

            verdict = "FAKE / MANIPULATED" if label == "manipulated" else "REAL"
            color = "#fb7185" if label == "manipulated" else "#4ade80"

            st.image(image, caption=uploaded_file.name, use_container_width=True)
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="label">Verdict</div>
                    <p class="verdict" style="color: {color};">{verdict}</p>
                    <div class="small-note">Confidence: {confidence * 100:.1f}%</div>
                    <div class="small-note">Device: {device}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif uploaded_file is not None:
            st.warning("Click Analyze image to run the model.")


if __name__ == "__main__":
    main()