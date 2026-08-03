"""Evaluation component rendering quantitative metrics (ROUGE, BERTScore), router accuracy, and confusion matrix."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List
from src.evaluation.router_evaluator import RouterEvaluationReport


def render_evaluation_metrics(metrics: Dict[str, Any]):
    """Renders ROUGE, BLEU, and BERTScore evaluation card metrics for single responses."""
    if not metrics:
        return

    st.markdown("### 📊 Quantitative Downstream Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("ROUGE-1 (F1)", f"{metrics.get('rouge1_f1', 0.0):.4f}")
    with col2:
        st.metric("ROUGE-2 (F1)", f"{metrics.get('rouge2_f1', 0.0):.4f}")
    with col3:
        st.metric("ROUGE-L (F1)", f"{metrics.get('rougeL_f1', 0.0):.4f}")
    with col4:
        bert_score = metrics.get('bert_score_f1', 0.0)
        st.metric("BERTScore (F1)", f"{bert_score:.4f}")


def render_router_evaluation_report(report: RouterEvaluationReport):
    """Renders comprehensive Router Evaluation metrics, confusion matrix, and classification report."""
    st.markdown("## 🎯 Router Evaluation & Accuracy Benchmark Report")

    # Key Summary Cards
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Routing Accuracy", f"{report.accuracy * 100:.1f}%", delta=f"{report.correct_predictions}/{report.total_samples} Passed")
    c2.metric("Macro F1-Score", f"{report.f1_macro:.4f}")
    c3.metric("Macro Precision", f"{report.precision_macro:.4f}")
    c4.metric("Macro Recall", f"{report.recall_macro:.4f}")
    c5.metric("Avg Routing Latency", f"{report.average_latency_ms} ms")

    st.markdown("<br>", unsafe_allow_html=True)

    # Strategy Distribution & Confusion Matrix Row
    col_strat, col_cm = st.columns([1, 1])

    with col_strat:
        st.markdown("### 📊 Strategy Distribution")
        strat_df = pd.DataFrame([
            {"Strategy": k, "Count": v} for k, v in report.strategy_distribution.items()
        ])
        if not strat_df.empty:
            fig_pie = px.pie(
                strat_df,
                names="Strategy",
                values="Count",
                title="Routing Strategy Breakdown",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    with col_cm:
        st.markdown("### 🧩 Confusion Matrix Heatmap")
        tasks = list(report.confusion_matrix.keys())
        cm_data = [[report.confusion_matrix[row].get(col, 0) for col in tasks] for row in tasks]

        fig_cm = px.imshow(
            cm_data,
            x=tasks,
            y=tasks,
            labels=dict(x="Predicted Task", y="True Target Task", color="Count"),
            text_auto=True,
            color_continuous_scale="Blues",
            title="Router Task Confusion Matrix"
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    # Per-Class Classification Report Table
    st.markdown("### 📋 Per-Class Classification Report")
    per_class_list = []
    for task, m in report.per_class_metrics.items():
        per_class_list.append({
            "Task Category": task.replace("_", " ").title(),
            "Support (Samples)": m["support"],
            "Precision": f"{m['precision']:.4f}",
            "Recall": f"{m['recall']:.4f}",
            "F1-Score": f"{m['f1_score']:.4f}"
        })
    df_per_class = pd.DataFrame(per_class_list)
    st.dataframe(df_per_class, use_container_width=True)

    # Detailed Test Case Log Table
    with st.expander("🔍 View All Evaluation Test Case Log Details"):
        df_details = pd.DataFrame(report.detailed_results)
        st.dataframe(df_details, use_container_width=True)


def render_benchmark_suite(benchmark_results: List[Dict[str, Any]]):
    """Renders tabular and chart summaries for system benchmarking."""
    if not benchmark_results:
        return

    st.markdown("### 🚀 System Benchmark Suite Results")
    df = pd.DataFrame(benchmark_results)

    # Metric summary table
    st.dataframe(df, use_container_width=True)

    # Latency vs Model plot
    fig = px.bar(
        df,
        x="id",
        y="latency_ms",
        color="actual_task",
        hover_data=["model_used", "tokens_per_sec", "confidence"],
        title="Latency (ms) by Task & Model Candidate",
        labels={"id": "Benchmark Case #", "latency_ms": "Latency (ms)", "actual_task": "Routed Task"}
    )
    st.plotly_chart(fig, use_container_width=True)
