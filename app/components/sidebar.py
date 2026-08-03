"""Sidebar component for Streamlit UI managing system configuration and controls."""

import streamlit as st
import psutil
import os
from src.models.model_manager import ModelManager
from src.utils.logger import get_memory_usage_mb


def render_sidebar():
    """Renders sidebar controls and returns configuration options."""
    st.sidebar.markdown("## ⚙️ Hybrid Router Configuration")
    
    # Task Override Selection
    task_override_option = st.sidebar.selectbox(
        "Routing Mode",
        options=[
            "Auto-Detect (Hybrid Router)",
            "Summarization",
            "Sentiment Analysis",
            "Question Answering",
            "Text Generation",
            "Named Entity Recognition",
            "Translation"
        ],
        help="Select Auto-Detect to let rule-based heuristics and zero-shot intent classifier route the request."
    )
    
    # Map selection string to task key
    task_override_map = {
        "Auto-Detect (Hybrid Router)": None,
        "Summarization": "summarization",
        "Sentiment Analysis": "sentiment",
        "Question Answering": "question_answering",
        "Text Generation": "text_generation",
        "Named Entity Recognition": "named_entity_recognition",
        "Translation": "translation"
    }
    selected_override = task_override_map[task_override_option]

    # Confidence Threshold Slider
    confidence_threshold = st.sidebar.slider(
        "Zero-Shot Confidence Threshold",
        min_value=0.20,
        max_value=0.90,
        value=0.55,
        step=0.05,
        help="Minimum required score for zero-shot classifier before triggering fallback model."
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎛️ Pipeline Hyperparameters")
    
    max_tokens = st.sidebar.slider(
        "Max Token Output Length",
        min_value=30,
        max_value=512,
        value=150,
        step=10
    )
    
    temperature = st.sidebar.slider(
        "Generation Temperature",
        min_value=0.1,
        max_value=1.5,
        value=0.7,
        step=0.1,
        help="Higher values increase output randomness for creative text generation."
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💻 Hardware & Memory Monitor")
    ram_mb = get_memory_usage_mb()
    st.sidebar.metric("Process RSS Memory", f"{ram_mb} MB")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Metrics & Evaluation")
    enable_eval = st.sidebar.checkbox(
        "Enable Reference Evaluation (ROUGE / BERTScore)",
        value=False,
        help="Allows inputting a ground-truth reference text to calculate ROUGE and BERTScore quality metrics."
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🧹 Cache Management")
    if st.sidebar.button("Clear Model Cache & Memory", use_container_width=True):
        manager = ModelManager()
        manager.clear_cache()
        st.sidebar.success("Model memory cache cleared!")

    return {
        "task_override": selected_override,
        "confidence_threshold": confidence_threshold,
        "max_length": max_tokens,
        "temperature": temperature,
        "enable_eval": enable_eval
    }
