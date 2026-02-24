import streamlit as st
import json
import pandas as pd
from schema import BenchmarkConfig
from runner import serve_then_bench
from datetime import datetime
import os
import io
from huggingface_hub import list_models
from transformers import AutoTokenizer
from contextlib import redirect_stdout
import time
from helper import tail_file, read_new_logs
import threading
from resolver import CapabilityResolver



st.set_page_config(page_title="vLLM Benchmark Tool", layout="wide")
# ===============================
# MAIN APP TABS
# ===============================
run_tab, history_tab = st.tabs(["Benchmark", "History"])


st.markdown(
    """
    <style>
    /* ================= SIDEBAR ================= */

    [data-testid="stSidebar"] * {
        font-size: 19px !important;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        font-size: 25px !important;
        font-weight: 600;
    }

    /* ================= SUMMARY METRICS (DEFAULT = SMALL) ================= */

    [data-testid="stMetric"] {
        padding: 6px 10px !important;
    }

    [data-testid="stMetricLabel"] {
        font-size: 11px !important;
    }

    [data-testid="stMetricValue"] {
        font-size: 18px !important;
        font-weight: 600;
    }

    /* ================= RESULTS METRICS (INSIDE TABS = LARGE) ================= */

    [data-testid="stTabs"] [data-testid="stMetric"] {
        padding: 12px 16px !important;
    }

    [data-testid="stTabs"] [data-testid="stMetricLabel"] {
        font-size: 26px !important;
    }

    [data-testid="stTabs"] [data-testid="stMetricValue"] {
        font-size: 26px !important;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    /* ================= INPUT CONTROLS ================= */

    /* Selectbox / Multiselect / TextInput / NumberInput */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    div[data-baseweb="textarea"] > div {
        min-height: 42px !important;
        font-size: 15px !important;
    }

    /* Selected value text */
    div[data-baseweb="select"] span {
        font-size: 15px !important;
    }

    /* Number input arrows */
    div[data-baseweb="input"] button {
        height: 18px !important;
    }

    /* Slider label */
    div[data-testid="stSlider"] label {
        font-size: 15px !important;
    }

    /* Slider track height */
    div[data-testid="stSlider"] > div {
        padding-top: 8px !important;
    }

    /* Radio buttons */
    div[data-testid="stRadio"] label {
        font-size: 15px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


#helpers
@st.cache_data(ttl=3600)
def fetch_hf_models(limit: int = 200):
    """
    Fetch public text-generation & chat models from Hugging Face.
    Cached to avoid repeated network calls.
    """
    try:
        models = list_models(
            filter=["text-generation"],
            sort="downloads",
            direction=-1,
            limit=limit,
        )
        return [m.modelId for m in models]
    except Exception as e:
        return []


@st.cache_data(ttl=3600)
def detect_model_type(model_id: str) -> str:
    try:
        write_server_log("[LOADING TOK] Tokenizer being loaded to check model type")
        tok = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True
        )
        write_server_log("[TOK LOADED] Successfully Loaded Tokenizer")

        if hasattr(tok, "chat_template") and tok.chat_template:
            return "chat"

    except Exception:
        pass

    return "base"


# Model configurations
MODEL_CONFIGS = {
    "gpt2": {
        "max_context": 1024,
        "params": "125M",
        "description": "Small GPT-2 model for testing"
    },
    "NousResearch/Hermes-3-Llama-3.1-8B": {
        "max_context": 131072,
        "params": "8B",
        "description": "Hermes 3 Llama 3.1 8B model"
    }
}

PRESET_CONFIGS = {
    "Quick Test": {
        "num_iters_latency": 5,
        "num_prompts_throughput": 5,
        "batch_size": 1,
        "input_len": 32,
        "output_len": 32,
    },
    "Standard": {
        "num_iters_latency": 10,
        "num_prompts_throughput": 10,
        "batch_size": 1,
        "input_len": 128,
        "output_len": 128,
    },
    "Heavy Load": {
        "num_iters_latency": 30,
        "num_prompts_throughput": 50,
        "batch_size": 4,
        "input_len": 512,
        "output_len": 256,
    }
}

with run_tab:
    st.title("vLLM Benchmark Tool")
    st.write("Run online serving benchmarks on various LLM models")

    # Sidebar for configuration
    st.sidebar.header("Benchmark Configuration")

    # Benchmark mode selector (Offline vs Online)
    benchmark_mode = st.sidebar.radio(
        "Benchmark Mode",
        ("Online (Real-time Serving)"),
        help="Online: Benchmark against a running server"
    )

    #Basic Available Model Configs
    st.sidebar.markdown("-----")
    st.sidebar.header("Model Serving Parameters")
    # Benchmark type selector
    st.sidebar.info("All model params are automatically fetched and set. Over-riding these may enable unexpected behaviour.")

    #st.sidebar.info(f"Benchmark Type: Online serving benchmark")



    # Fetch HF models
    hf_models = fetch_hf_models()

    # Pre-configured models first, HF models after
    model_choices = (
        list(MODEL_CONFIGS.keys())
        + [m for m in hf_models if m not in MODEL_CONFIGS]
    )

    model_name = st.sidebar.selectbox(
        "Model/Repo Name",
        model_choices,
        help="""
    Select a pre-configured model or any Hugging Face text-generation model.
    [Browse on Hugging Face](https://huggingface.co/)
    """
    )


    resolver = CapabilityResolver()
    capabilities = resolver.resolve(model_name)



    # Show model info
    model_info = MODEL_CONFIGS.get(
        model_name,
        {
            "max_context": 8192,
            "params": "Unknown",
            "description": "Hugging Face model (auto-detected)"
        }
    )
    model_type = detect_model_type(model_name)

    if model_type == "chat":
        dataset_name = "sharegpt"
        dataset_path = "ShareGPT_V3_unfiltered_cleaned_split.json"
        endpoint = "/v1/chat/completions"
    else:
        dataset_name = "random"
        dataset_path = None
        endpoint = "/v1/completions"

    with st.sidebar:
        st.info(f"Model Info\nContext: {model_info['max_context']:,} tokens\nSize: {model_info['params']}\n{model_info['description']}")

    col1, col2 = st.sidebar.columns(2)
    with col1:
        dtype = st.selectbox(
            "Data Type",
            capabilities.supported_dtypes,
            index=capabilities.supported_dtypes.index(
        capabilities.default_dtype
    ) if capabilities.default_dtype in capabilities.supported_dtypes else 0,
            help="""
            Model precision
            [Learn more at Official Docs](https://docs.vllm.ai/en/latest/cli/serve/#-dtype)
            """
        )

    with col2:
        gpu_id = st.number_input(
            "GPU ID",
            value=7,
            min_value=0,
            help="GPU device ID to use"
        )

    # Auto-set max_model_len based on model
    default_max_len = capabilities.max_model_len


    col1, col2 = st.sidebar.columns(2)
    with col1:
        max_model_len = st.number_input(
            "Max Model Length",
            value=default_max_len,
            min_value=512,
            help=f"Maximum sequence length (model supports up to {default_max_len:,})"
        )

    with col2:
        quantization = st.selectbox(
        "Quantization",
        capabilities.supported_quantizations,
        index=0,
        help=f"Available quantizations are fetched from Hugging Face model card"
        )

    gpu_memory_util = st.sidebar.slider(
        "GPU Memory Utilization",
        min_value=0.1,
        max_value=1.0,
        value=0.85,
        step=0.05,
        help="Target GPU memory utilization rate"
    )


    st.sidebar.markdown("---")
    st.sidebar.subheader("Model Benchmarking Params")


        # Online benchmarking uses ShareGPT dataset

    st.sidebar.write(f"**Dataset**: {dataset_name}")
    if model_type == "chat":
        st.sidebar.info("Chat models require the conversational style dataset like Sharegpt")
        
        
        # Number of prompts for online benchmark
    num_prompts = st.sidebar.number_input(
            "Total Number of Prompts(Requests)",
            value=10,
            min_value=1,
            help=""" Number of requests sent throughout the run
            [Num-prompts](https://docs.vllm.ai/en/latest/cli/bench/serve/#-num-prompts)
            """
    )

    num_concurrency = st.sidebar.number_input(
            "Maximum Concurrency",
            value=10,
            min_value=1,
            help="""Number of requests processsed concurrently in the run
            [Max Concurrency](https://docs.vllm.ai/en/latest/cli/bench/serve/#-max-concurrency)
            """
    )

    col1, col2 = st.sidebar.columns(2)
    with col1:
        input_len = st.number_input(
                "Input Length",
                value=32,
                min_value=16,
                help="""Number of input tokens
                (Input-Len)[https://docs.vllm.ai/en/latest/cli/bench/serve/#-input-len]"""
        )    

    with col2:
        output_len = st.number_input(
                "Output Length",
                value=32,
                min_value=16,
                help="""Number of output tokens
                [Output-Len](https://docs.vllm.ai/en/latest/cli/bench/serve/#-output-len)
                """
        )   
        
    # Validation warning
    total_tokens = input_len + output_len
    if total_tokens > max_model_len:
        st.sidebar.warning(f"Total tokens ({total_tokens}) exceeds max_model_len ({max_model_len})")

    st.sidebar.markdown("---")

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Your Configuration Summary")
        
        # Display config in boxes/metric cards
        # Row 1: Mode and Type
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Mode", "Online")
        with c2:
            st.metric("Type", "Real-time serving")
        
        # Row 2: Model and GPU
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Model", model_name.split("/")[-1])
        with c2:
            st.metric("GPU ID", gpu_id)
        
            # Online config
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Dataset", dataset_name)
        with c2:
            st.metric("Prompts", num_prompts)
            # Offline config
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Data Type", dtype)
        with c2:
            st.metric("Quantization", quantization if quantization else "None")
            
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Max Model Len", f"{max_model_len:,}")
        with c2:
            st.metric("GPU Memory Util", f"{gpu_memory_util:.0%}")
            
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Input Length", f"{input_len} tok")
        with c2:
            st.metric("Output Length", f"{output_len} tok")
            

    with col2:
        st.subheader("Action")
        
            
        run_button = st.button("Run Online Benchmark",
            width='stretch',
            type="primary",
            disabled=st.session_state.get("benchmark_running", False)
        )
    st.markdown("---")

    # Run benchmark when button clicked
    if run_button:
        # Double-click protection: check if a benchmark is already running
        if "benchmark_running" not in st.session_state:
            st.session_state.benchmark_running = False
        
        if st.session_state.benchmark_running:
            st.error(" A benchmark is already running. Please wait for it to complete.")
        else:
            st.session_state.benchmark_running = True
            
            # Create a placeholder for status messages that will be replaced
            status_placeholder = st.empty()
            status_placeholder.info("Benchmarking Started... This may take several minutes. Refer progress bar below")
            
            # Create tabs for results display
            tab1, tab2, tab3, tab4 = st.tabs([
                "Results", 
                "Configuration", 
                "Stdout", 
                "Debug",
            ])
            
            try:
                # Create benchmark config
                cfg = BenchmarkConfig(
                    #benchmark_mode=benchmark_mode,
                    #benchmark_type=config_benchmark_type,
                    model_name=model_name,
                    dtype=dtype,
                    max_model_len=max_model_len,
                    input_len=input_len,
                    output_len=output_len,
                    num_prompts=num_prompts,
                    gpu_memory_util=gpu_memory_util,
                    quantization=quantization,
                    max_concurrency=num_concurrency,
                    dataset_name= dataset_name,
                    dataset_path= dataset_path,
                    endpoint = endpoint

                )

                benchmark_mode = "Online (Real-time Serving)"
            
                # --- Progress UI ---
                progress_bar = st.progress(0)
                progress_text = st.empty()
                live_log_placeholder = st.empty()
                log_buffer = ""


                result_container = {}

                def run_benchmark():
                    result_container["result"] = serve_then_bench(
                        cfg, gpu_id=gpu_id, host="127.0.0.1", port=8000
                    )

                # Capture starting file position ONCE
                log_file = "server.log"
                cursor = os.path.getsize(log_file) if os.path.exists(log_file) else 0

                thread = threading.Thread(target=run_benchmark)
                thread.start()

                progress_value = 0.0
                target_progress = 0.0
                current_stage = "Initializing..."

                while thread.is_alive():
                    try:
                        logs, cursor = read_new_logs(log_file, cursor)

                        if logs:
                            log_buffer += logs
                            live_log_placeholder.code(log_buffer[-5000:], language="text")

                        # -----------------------------
                        # Stage Detection
                        # -----------------------------
                        if "[START SERVER]" in logs:
                            target_progress = max(target_progress, 0.15)
                            current_stage = "Starting vLLM server..."

                        if "vLLM server is ready" in logs:
                            target_progress = max(target_progress, 0.35)
                            current_stage = "Server ready. Preparing benchmark..."

                        if "[SERVE_THEN_BENCH]" in logs:
                            target_progress = max(target_progress, 0.65)
                            current_stage = "Running benchmark..."

                        if "[PARSE_METRICS]" in logs:
                            target_progress = max(target_progress, 0.85)
                            current_stage = "Parsing metrics..."

                        if "Terminating server" in logs:
                            target_progress = 1.0
                            current_stage = "Finalizing..."

                        # -----------------------------
                        # 🔥 FAKE MICRO-PROGRESS (35 → 65)
                        # -----------------------------
                        if (
                            current_stage == "Running benchmark..."
                            and progress_value < 0.65
                            and target_progress < 0.65
                        ):
                            progress_value += 0.005  # micro movement
                            progress_value = min(progress_value, 0.64)

                        # -----------------------------
                        # Smooth easing toward target
                        # -----------------------------
                        if progress_value < target_progress:
                            progress_value += (target_progress - progress_value) * 0.12
                            progress_value = min(progress_value, target_progress)

                        progress_bar.progress(progress_value)
                        progress_text.info(
                            f"{current_stage} ({int(progress_value * 100)}%)"
                        )

                        if not thread.is_alive() and target_progress < 1.0:
                            progress_text.error("Not Completed")
                            current_stage = "Benchmark finished"
                            target_progress = 1.0


                    except Exception as e:
                        progress_text.error(f"Progress error: {str(e)}")
                        break

                    time.sleep(0.2)

                thread.join()

                try:
                    logs, cursor = read_new_logs(log_file, cursor)
                    if logs:
                        log_buffer += logs
                        live_log_placeholder.code(log_buffer[-5000:], language="text")
                except Exception:
                    pass

                result = result_container.get("result", {})

                progress_bar.progress(1.0)
                progress_text.empty()




                
                # Check for errors
                if "error" in result:
                    error_msg = result.get("error_message", result.get("error", "Unknown error"))
                    status_placeholder.error(f"Benchmark failed: {error_msg}")
                elif result.get("returncode") != 0:
                    status_placeholder.error(f"Benchmark failed with return code {result["returncode"]}See server stdout or logs.txt for details.")
                else:
                    status_placeholder.success("Benchmark completed successfully!")
                
                # Display results in tabs (accessible for all branches)
                with tab1:
                    if "error" in result or result.get("returncode") != 0:
                        st.error("Benchmark failed. Check other tabs for details.")
                    else:
                        st.subheader("Benchmark Metrics")
                        metrics = result.get("metrics", {})
                        
                        if metrics:
                            if benchmark_mode == "Online (Real-time Serving)":
                                # Display online metrics in a more organized way
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.write("**Request Metrics**")
                                    if "successful_requests" in metrics:
                                        st.metric("Successful Requests", int(metrics["successful_requests"]))
                                    if "benchmark_duration_sec" in metrics:
                                        st.metric("Duration (s)", f"{metrics['benchmark_duration_sec']:.2f}")
                                    if "request_throughput" in metrics:
                                        st.metric("Request Throughput (req/s)", f"{metrics['request_throughput']:.2f}")
                                
                                with col2:
                                    st.write("**Token Metrics**")
                                    if "total_input_tokens" in metrics:
                                        st.metric("Total Input Tokens", int(metrics["total_input_tokens"]))
                                    if "total_generated_tokens" in metrics:
                                        st.metric("Total Generated Tokens", int(metrics["total_generated_tokens"]))
                                    if "total_token_throughput" in metrics:
                                        st.metric("Total Token Throughput (tok/s)", f"{metrics['total_token_throughput']:.2f}")
                                
                                st.divider()
                                
                                # Latency metrics (TTFT, TPOT, ITL)
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.write("**Time to First Token (TTFT) - ms**")
                                    #if "mean_ttft_ms" in metrics:
                                        #st.metric("Mean", f"{metrics['mean_ttft_ms']:.2f}")
                                    if "median_ttft_ms" in metrics:
                                        st.metric("Median", f"{metrics['median_ttft_ms']:.2f}")
                                    #if "p99_ttft_ms" in metrics:
                                        #st.metric("P99", f"{metrics['p99_ttft_ms']:.2f}")
                                
                                with col2:
                                    st.write("**Time Per Output Token (TPOT) - ms**")
                                    #if "mean_tpot_ms" in metrics:
                                        #st.metric("Mean", f"{metrics['mean_tpot_ms']:.2f}")
                                    if "median_tpot_ms" in metrics:
                                        st.metric("Median", f"{metrics['median_tpot_ms']:.2f}")
                                    #if "p99_tpot_ms" in metrics:
                                        #st.metric("P99", f"{metrics['p99_tpot_ms']:.2f}")
                                
                                with col3:
                                    st.write("**Inter-token Latency (ITL) - ms**")
                                    #if "mean_itl_ms" in metrics:
                                        #st.metric("Mean", f"{metrics['mean_itl_ms']:.2f}")
                                    if "median_itl_ms" in metrics:
                                        st.metric("Median", f"{metrics['median_itl_ms']:.2f}")
                                    #if "p99_itl_ms" in metrics:
                                        #st.metric("P99", f"{metrics['p99_itl_ms']:.2f}")
                            else:
                                # Display offline metrics
                                metric_cols = st.columns(len(metrics) if len(metrics) < 4 else 4)
                                for idx, (key, value) in enumerate(metrics.items()):
                                    with metric_cols[idx % len(metric_cols)]:
                                        # Format the key nicely
                                        display_key = key.replace("_", " ").title()
                                        
                                        if isinstance(value, float):
                                            st.metric(display_key, f"{value:.4f}")
                                        else:
                                            st.metric(display_key, value)
                                
                                # Display all metrics in a table
                                st.subheader("Detailed Metrics Table")
                                metrics_df = {
                                    "Metric": [k.replace("_", " ").title() for k in metrics.keys()],
                                    "Value": list(metrics.values())
                                }
                                st.dataframe(metrics_df, width='stretch')
                        else:
                            st.warning("No metrics were extracted. Check the logs for details.")
                        
                        # Display runtime
                        st.divider()
                        runtime = result.get("runtime_sec", 0)
                        st.metric("Total Runtime", f"{runtime:.2f} seconds")
                
                with tab2:
                    st.subheader("Benchmark Configuration")
                    config = result.get("config", {})
                    st.json(config)
                
                with tab3:
                    st.subheader("Raw Stdout Output")
                    stdout = result.get("stdout", "")
                    if stdout:
                        st.code(stdout[-3000:], language="text")
                    else:
                        st.write("No stdout captured")
                
                with tab4:
                    st.subheader("Debug Information")
                    metrics = result.get("metrics", {})
                    st.write(f"**Metrics Found**: {len(metrics)}")
                    st.write(f"**Metrics Keys**: {list(metrics.keys())}")
                    st.write("**Full Metrics Dictionary**:")
                    st.json(metrics if metrics else {"message": "No metrics extracted"})
                    
                    st.divider()
                    st.write("**Command Executed**:")
                    from cli_builder import build_cli
                    cmd = build_cli(cfg)
                    st.code(" ".join(cmd), language="bash")
                                        
                
                # Export results button
                st.divider()
                st.subheader("Export Results")
                
                col1, col2 = st.columns(2)
                with col1:
                    json_str = json.dumps(result, indent=2)
                    st.download_button(
                        label="Download as JSON",
                        data=json_str,
                        file_name=f"benchmark_{model_name.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                
                with col2:
                    # Create CSV for metrics
                    if result.get("metrics"):
                        csv_data = "Metric,Value\n"
                        for k, v in result.get("metrics", {}).items():
                            csv_data += f"{k},{v}\n"
                        
                        st.download_button(
                            label="Download Metrics as CSV",
                            data=csv_data,
                            file_name=f"metrics_{model_name.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
        
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.error("Stack trace:")
                st.code(str(e), language="text")
                import traceback
                st.code(traceback.format_exc(), language="text")
            finally:
                # Reset benchmark running flag
                st.session_state.benchmark_running = False

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray; font-size: 12px;'>
        vLLM Benchmark Tool | Powered by Streamlit
        </div>
        """,
        unsafe_allow_html=True
    )

# ============================================================
# RUN HISTORY TAB (JSONL VIEWER)
# ============================================================

with history_tab:

    st.header("Benchmark Run History")

    history_file = "runs_history.jsonl"

    if not os.path.exists(history_file):
        st.info("No benchmark history found yet.")
        st.stop()

    try:
        df = pd.read_json(history_file, lines=True)
    except Exception as e:
        st.error(f"Failed to read history file: {str(e)}")
        st.stop()

    if df.empty:
        st.info("History file exists but contains no runs.")
        st.stop()

    # Sort latest first
    df = df.sort_values("timestamp", ascending=False)

    st.success(f"Total Runs Recorded: {len(df)}")

    # ------------------------
    # Optional Filters
    # ------------------------
    col1, col2 = st.columns(2)

    with col1:
        models = st.multiselect(
            "Filter by Model",
            options=sorted(df["model"].dropna().unique()),
            default=sorted(df["model"].dropna().unique())
        )

    with col2:
        modes = st.multiselect(
            "Filter by Mode",
            options=sorted(df["benchmark_mode"].dropna().unique()),
            default=sorted(df["benchmark_mode"].dropna().unique())
        )

    filtered_df = df[
        (df["model"].isin(models)) &
        (df["benchmark_mode"].isin(modes))
    ]

    st.write(f"Showing {len(filtered_df)} runs")

    st.dataframe(filtered_df, width='stretch')
    # ------------------------
    # Clear History Button
    # ------------------------
    st.divider()

    if st.button("🗑 Clear History"):
        open(history_file, "w").close()
        st.success("History cleared.")
        st.rerun()
