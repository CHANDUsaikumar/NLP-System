"""Streamlit Application Entry Point for Adaptive NLP Multi-Model System."""

import os
from pathlib import Path
import streamlit as st

# Set page config FIRST before any other streamlit commands
st.set_page_config(
    page_title="Adaptive NLP Multi-Model System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

import sys
# Append root workspace directory to sys.path for clean imports
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.router.dynamic_router import DynamicRouter
from src.router.decision_engine import DecisionEngine
from src.evaluation.router_evaluator import RouterEvaluator
from src.evaluation.benchmark import SystemBenchmark
from src.utils.validators import UserRequestPayload
from src.utils.exceptions import InputValidationError, RoutingError, InferenceError
from app.components.sidebar import render_sidebar
from app.components.model_dashboard import render_routing_dashboard
from app.components.evaluation_view import render_evaluation_metrics, render_router_evaluation_report, render_benchmark_suite


def load_css():
    """Injects custom CSS styling."""
    css_path = ROOT_DIR / "app" / "assets" / "style.css"
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


@st.cache_resource
def get_router_instance() -> DynamicRouter:
    """Caches router instance across Streamlit reruns."""
    return DynamicRouter()


def main():
    load_css()

    st.markdown(
        """
        <div style="text-align: center; padding-bottom: 20px;">
            <h1 style="background: linear-gradient(90deg, #6366f1, #a855f7, #ec4899); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 2.8rem; font-weight: 800;">
                🤖 Adaptive NLP Multi-Model Router
            </h1>
            <p style="color: #94a3b8; font-size: 1.1rem; max-width: 850px; margin: 0 auto;">
                Production-quality hybrid routing system combining rule-based heuristics with zero-shot transformer intent classification, explainable rationales, and real-time metrics evaluation.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    sidebar_opts = render_sidebar()
    router = get_router_instance()
    # Update threshold on decision engine if adjusted in UI
    router.decision_engine.confidence_threshold = sidebar_opts["confidence_threshold"]

    tab1, tab2, tab3, tab4 = st.tabs([
        "🚀 Live Interactive Router",
        "🎯 Router Evaluation & Confusion Matrix",
        "📊 System Benchmarks",
        "📖 Architecture & Explainability"
    ])

    with tab1:
        st.markdown("### 📥 Input Prompt & Document Upload")
        
        # File Upload Widget (.txt)
        uploaded_file = st.file_uploader("Upload a text document (.txt):", type=["txt"], help="Upload a .txt document to automatically populate the input text area.")
        
        default_prompt = "What is the capital of France and what is its primary historical significance?"
        
        if uploaded_file is not None:
            try:
                uploaded_content = uploaded_file.read().decode("utf-8")
                st.session_state["prompt_input"] = uploaded_content
                st.success(f"Successfully loaded file '{uploaded_file.name}' ({len(uploaded_content)} characters).")
            except Exception as e:
                st.error(f"Failed to read file content: {e}")

        if "prompt_input" not in st.session_state:
            st.session_state["prompt_input"] = default_prompt

        # Large Text Area Box
        prompt_input = st.text_area(
            "Enter text prompt or document content:",
            value=st.session_state["prompt_input"],
            height=160,
            placeholder="Type a Q&A question, summarization request, product review, creative story prompt, NER request, or translation task..."
        )

        reference_text = None
        if sidebar_opts["enable_eval"]:
            reference_text = st.text_area(
                "Enter ground-truth reference text (for ROUGE / BLEU / BERTScore evaluation):",
                height=90,
                placeholder="Optional ground truth reference text..."
            )

        # Action Buttons Row: Execute and Clear
        col_exec, col_clear = st.columns([4, 1])
        with col_exec:
            execute_btn = st.button("⚡ Execute Adaptive Pipeline", type="primary", use_container_width=True)
        with col_clear:
            clear_btn = st.button("🧹 Clear Input", type="secondary", use_container_width=True)

        if clear_btn:
            st.session_state["prompt_input"] = ""
            st.session_state.pop("pipeline_response", None)
            st.rerun()

        if execute_btn:
            if not prompt_input.strip():
                st.warning("Please enter a valid non-empty prompt or upload a text file.")
            else:
                with st.spinner("Classifying intent via hybrid router and executing model pipeline..."):
                    try:
                        payload = UserRequestPayload(
                            prompt=prompt_input,
                            task_override=sidebar_opts["task_override"],
                            reference_text=reference_text,
                            max_length=sidebar_opts["max_length"],
                            temperature=sidebar_opts["temperature"]
                        )
                        
                        response = router.process_request(payload)
                        st.session_state["pipeline_response"] = response

                    except InputValidationError as e:
                        st.error(f"❌ Input Validation Error: {e}")
                    except RoutingError as e:
                        st.error(f"❌ Routing Error: {e}")
                    except InferenceError as e:
                        st.error(f"❌ Inference Error: {e}")
                    except Exception as e:
                        st.error(f"❌ Unexpected Error: {str(e)}")

        # Render Response Output & Metrics if available
        if "pipeline_response" in st.session_state:
            res = st.session_state["pipeline_response"]
            render_routing_dashboard(res)

            if res.eval_metrics:
                st.markdown("---")
                render_evaluation_metrics(res.eval_metrics)

    with tab2:
        st.markdown("### 🎯 Hybrid Router Evaluation Engine")
        st.write("Evaluates the router performance against a ground-truth labeled benchmark dataset (`assets/evaluation_dataset.json`).")

        if st.button("▶️ Run Router Evaluation Benchmark", type="primary"):
            with st.spinner("Evaluating router on labeled benchmark dataset..."):
                evaluator = RouterEvaluator(decision_engine=router.decision_engine)
                report = evaluator.evaluate()
                st.session_state["router_eval_report"] = report
                st.success(f"Evaluation complete! Overall Accuracy: {report.accuracy * 100:.1f}%")

        if "router_eval_report" in st.session_state:
            render_router_evaluation_report(st.session_state["router_eval_report"])

    with tab3:
        st.markdown("### 🚀 Run Standard System Benchmark")
        st.write("Executes end-to-end model inference over standard benchmark prompts to measure latency, throughput, and memory usage.")

        if st.button("▶️ Launch System Benchmark Suite", type="secondary"):
            with st.spinner("Running benchmark suite over candidate models..."):
                benchmark = SystemBenchmark(router=router)
                results = benchmark.run_suite()
                st.session_state["benchmark_results"] = results
                st.success("Benchmark suite completed!")

        if "benchmark_results" in st.session_state:
            render_benchmark_suite(st.session_state["benchmark_results"])

    with tab4:
        st.markdown("### 🏛️ System Architecture Overview")
        st.markdown(
            """
            #### 1. Hybrid 5-Stage Routing Architecture
            - **Stage 1 (Pre-processing)**: Cleans text and extracts syntactic indicators (char/word count, question marks, interrogative starters, imperative verbs, translation phrases, NER directives, document size).
            - **Stage 2 (Rule-Based Heuristics)**: Deterministic high-precision regex keyword and rule matching.
            - **Stage 3 (Zero-Shot Classifier)**: Semantic NLI transformer intent classification (`valhalla/distilbart-mnli-12-3`).
            - **Stage 4 (Decision Engine & Fallback)**: Verifies confidence against configurable threshold ($\text{score} \ge 0.55$). Falls back to General Instruction Q&A model if below threshold.
            - **Stage 5 (Explainability Generator)**: Generates human-readable rationales explaining exact match reasons.

            #### 2. Task Category & Model Pipeline Registry
            | Task Category | Primary Model Candidate | Fallback Model Candidate |
            | :--- | :--- | :--- |
            | **Summarization** | `sshleifer/distilbart-cnn-12-6` | `t5-small` |
            | **Sentiment Analysis** | `cardiffnlp/twitter-roberta-base-sentiment-latest` | `distilbert-base-uncased-finetuned-sst-2-english` |
            | **Question Answering** | `google/flan-t5-base` | `google/flan-t5-small` |
            | **Text Generation** | `gpt2-medium` | `gpt2` |
            | **NER** | `dslim/bert-base-NER` | `elastic/distilbert-base-uncased-finetuned-conll03-english` |
            | **Translation** | `Helsinki-NLP/opus-mt-en-fr` | `Helsinki-NLP/opus-mt-en-es` |

            #### 3. Enterprise Software Architecture
            - **SOLID Principles**: Decoupled single-responsibility modules (`preprocessing`, `heuristic_router`, `zero_shot_router`, `decision_engine`, `rationale_generator`).
            - **Singleton Model Manager**: Lazily instantiates and caches models in warm memory.
            - **Pydantic Data Contracts**: Input payload validation and output schema guarantees.
            - **Telemetry & Resource Monitoring**: Structured JSONL request logs and RSS memory usage tracking.
            """
        )


if __name__ == "__main__":
    main()
