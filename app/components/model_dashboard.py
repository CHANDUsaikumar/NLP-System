"""Dashboard component visualizing model metrics, routing decisions, strategy, and memory usage."""

import streamlit as st
import plotly.graph_objects as go
from src.utils.validators import PipelineResponsePayload

STRATEGY_BADGES = {
    "Rule-Based Heuristic": "⚡ Rule-Based Heuristic",
    "Zero-Shot Classification": "🎯 Zero-Shot Classifier",
    "Confidence Fallback": "🛡️ Confidence Fallback",
    "Manual Override": "⚙️ Manual Override"
}


def render_routing_dashboard(response: PipelineResponsePayload):
    """Renders visual cards, strategy badges, explainable rationales, memory, and output text."""
    st.markdown("### 🔀 Router & Inference Analytics")

    # Row 1: Key Metric Cards
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown(
            f"""
            <div class="metric-container">
                <div class="metric-title">Detected Task</div>
                <div class="metric-value">{response.task.replace('_', ' ').title()}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-container">
                <div class="metric-title">Selected Model</div>
                <div class="metric-value" style="font-size: 1.05rem; line-height: 2rem;">{response.selected_model.split('/')[-1]}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        conf_pct = round(response.confidence_score * 100, 1)
        st.markdown(
            f"""
            <div class="metric-container">
                <div class="metric-title">Confidence</div>
                <div class="metric-value">{conf_pct}%</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:
        st.markdown(
            f"""
            <div class="metric-container">
                <div class="metric-title">Routing Strategy</div>
                <div class="metric-value" style="font-size: 0.95rem;">{STRATEGY_BADGES.get(response.routing_strategy, response.routing_strategy)}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col5:
        st.markdown(
            f"""
            <div class="metric-container">
                <div class="metric-title">Total Latency</div>
                <div class="metric-value">{response.total_latency_ms} ms</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 2: Secondary System Metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("⏱️ Routing Latency", f"{response.routing_latency_ms} ms")
    with m2:
        st.metric("⚡ Inference Latency", f"{response.inference_latency_ms} ms")
    with m3:
        st.metric("🚀 Throughput", f"{response.token_throughput} t/s")
    with m4:
        st.metric("💾 RAM Usage", f"{response.memory_usage_mb} MB")

    # Routing Rationale Callout
    st.info(
        f"💡 **Explainable Routing Rationale**:\n\n{response.routing_reason}\n\n"
        f"*(Confidence Score: `{round(response.confidence_score * 100, 1)}%` | "
        f"Strategy: `{response.routing_strategy}` | Device: `{response.device_used.upper()}`)*"
    )

    # Preprocessed Features Expander
    if response.preprocessed_features:
        with st.expander("🔍 View Pre-processing Feature Extraction Details"):
            feats = response.preprocessed_features
            c1, c2, c3, c4 = st.columns(4)
            c1.write(f"**Word Count:** {feats.get('word_count', 0)}")
            c2.write(f"**Sentence Count:** {feats.get('sentence_count', 0)}")
            c3.write(f"**Document Size:** {feats.get('document_size', 'N/A')}")
            c4.write(f"**Question Mark:** {'Yes' if feats.get('has_question_mark') else 'No'}")

            if feats.get("interrogative_word"):
                st.write(f"**Detected Interrogative Word:** `{feats.get('interrogative_word')}`")
            if feats.get("imperative_verb"):
                st.write(f"**Detected Imperative Verb:** `{feats.get('imperative_verb')}`")

    # Response Output Display Box
    st.markdown("### 📝 Pipeline Output")
    st.markdown(f'<div class="output-box">{response.output_text}</div>', unsafe_allow_html=True)
