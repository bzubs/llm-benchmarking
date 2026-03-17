import streamlit as st
import json
import pandas as pd
import os
import io
import threading
import time
from datetime import datetime
from contextlib import redirect_stdout
from helper import tail_file, read_new_logs, fetch_hf_models, detect_model_type
from resolver import CapabilityResolver
from schema import BenchmarkConfig
from runner import serve_then_bench
from configs import MODEL_CONFIGS, PRESET_CONFIGS
from login import show_auth_screen
from typing import cast, Literal

#CONFIG FOR FILES
log_file = "server.log"
history_file = "runs_history.jsonl"


st.set_page_config(page_title="vLLM Benchmark", layout="wide")


#session vars
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "username" not in st.session_state:
    st.session_state.username = None

# Block app if not logged in
if not st.session_state.authenticated:
    show_auth_screen()
    st.stop()

# Main App Tabs
benchmark_tab, history_tab = st.tabs(["Benchmark", "History"])

#CSS markdown for custom size of fonts and UI elements
st.markdown(
    """
    <style>
    /* SIDEBAR */

    [data-testid="stSidebar"] * {
        font-size: 19px !important;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        font-size: 25px !important;
        font-weight: 600;
    }

    /* SUMMARY METRICS */

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

    /*RESULTS METRICS */

    [data-testid="stTabs"] [data-testid="stMetric"] {
        padding: 12px 16px !important;
    }

    [data-testid="stTabs"] [data-testid="stMetricLabel"] {
        font-size: 26px !important;
    }

    [data-testid="stTabs"] [data-testid="stMetricValue"] {
        font-size: 30px !important;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    /* INPUT CONTROLS */

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

with benchmark_tab:
    st.title("vLLM Benchmark")
    st.write("Run online serving benchmarks on various LLM models")

    if "last_result" not in st.session_state:
        st.session_state.last_result = None

    if "benchmark_running" not in st.session_state:
        st.session_state.benchmark_running = False

    if "is_unvalid" not in st.session_state:
        st.session_state.is_unvalid = False


    #sidebar user card
    with st.sidebar:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(
                f"""
                <div style="
                    display:flex;
                    align-items:center;
                    gap:6px;
                    padding:4px 6px;
                    border-radius:6px;
                    background: rgba(255,255,255,0.04);
                    border: 1px solid rgba(255,255,255,0.08);
                ">
                    <div style="
                        width:22px; height:22px;
                        border-radius:50%;
                        background: linear-gradient(135deg, #6366f1, #8b5cf6);
                        display:flex; align-items:center; justify-content:center;
                        font-size:10px; font-weight:700; color:white; flex-shrink:0;
                    ">
                        {str(st.session_state.username)[0].upper()}
                    </div>
                    <div style="font-size:11px; font-weight:600; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">
                        {st.session_state.username}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with col2:
            if st.button("↩ Logout", key="logout", help="Logout"):
                st.session_state.authenticated = False
                st.session_state.username = None
                st.rerun()

    st.sidebar.markdown("---")

    # Sidebar for configuration
    st.sidebar.markdown(
    """
    <h2 style="
        font-size: 30px;
        font-weight: 800;
        color: #FFFFFF;
        margin-bottom: 5px;
    ">
    Benchmark Configuration
    </h2>
    """,
    unsafe_allow_html=True
    )

    #preset = st.sidebar.radio(
       # "Select Preset Config",
       # (list(PRESET_CONFIGS.keys()) + ["Custom"]),
       # help="Select a test from some pre-defined configs"
    #)

    #st.sidebar.markdown("-----")
    st.sidebar.subheader("Model Serving Parameters")
    st.sidebar.info("All model params are automatically fetched and set. Over-riding these may enable unexpected behaviour.")

    # Fetch HF models
    hf_models = fetch_hf_models()

    # Pre-configured models + HF models 
    model_choices = (
        list(MODEL_CONFIGS.keys()) +
        [m for m in hf_models]
    )

    model_name = st.sidebar.selectbox(
        "Model/Repo Name",
        model_choices,
        help="""
    Select a pre-configured model or any Hugging Face text-generation model.
    [Browse on Hugging Face](https://huggingface.co/)
    """
    )

    #Custom implemented logic for resolving model params based on model_name
    resolver = CapabilityResolver()
    capabilities = resolver.resolve(model_name)


    # Detect model type chat/base
    model_type = detect_model_type(model_name)

    #use dataset random for base model or sharegpt for chat
    if model_type == "chat":
        dataset_name = "sharegpt"
        dataset_path = "ShareGPT_V3_unfiltered_cleaned_split.json"
        endpoint = "/v1/chat/completions"
    else:
        dataset_name = "random"
        dataset_path = None
        endpoint = "/v1/completions"

    
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

    # Auto-set max_model_len based on detection
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
        help=f"""Available quantizations are fetched from Hugging Face model card
        [Learn more at Official Docs](https://docs.vllm.ai/en/latest/cli/serve/#-quantization-q)
        """
        )

    gpu_memory_util = st.sidebar.slider(
        "GPU Memory Utilization",
        min_value=0.1,
        max_value=1.0,
        value=0.85,
        step=0.05,
        help="Target GPU memory utilization rate"
    )

    with st.sidebar:
        st.subheader("Model Info")
        st.markdown(f"Detected Model Type: {model_type}")
        st.markdown(f"Dataset used: {dataset_name}")
        if model_type == "chat":
            st.info("Chat models require the conversational style dataset like Sharegpt")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Model Benchmarking Parameters")
  
    # Number of prompts for online benchmark
    num_prompts = st.sidebar.number_input(
            "Total Number of Prompts(Requests)",
            value=10,
            min_value=1,
            help=""" Number of requests sent throughout the run
            [Num requests](https://docs.vllm.ai/en/latest/cli/bench/serve/#-num-prompts)
            """
    )

    #num_concurrency suggests no of concurrent requests sent during a run

    num_concurrency = st.sidebar.number_input(
            "Maximum Concurrency",
            value=1,
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
                [Input-Len](https://docs.vllm.ai/en/latest/cli/bench/serve/#-input-len)"""
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
        
        # Row 1: Model and GPU
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Model", model_name.split("/")[-1])
        with c2:
            st.metric("GPU ID", gpu_id)
        
        # Dataset Config
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Dataset", dataset_name)
        with c2:
            st.metric("Prompts", num_prompts)
    
        #dtype config
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Data Type", dtype)
        with c2:
            st.metric("Quantization", quantization)
            
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
        
            
        run_button = st.button("Run Benchmark",
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
                    username= str(st.session_state.username),
                    model_name=model_name,
                    dtype=cast(Literal["auto", "float16", "float32", "bfloat16", "fp16"]), dtype),
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

                        
                        # Stage Detection
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

                     
                        if (
                            current_stage == "Running benchmark..."
                            and progress_value < 0.65
                            and target_progress < 0.65
                        ):
                            progress_value += 0.005  # micro movement
                            progress_value = min(progress_value, 0.64)

                       
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
                progress_bar.empty()
                progress_text.empty()

                # Check for errors
                if result.get("error"):
                    error_msg = result.get("error", "Unknown error")
                    status_placeholder.error(f"Benchmark failed: {error_msg}")

                elif result.get("returncode") != 0:
                    status_placeholder.error(
                        f"""Benchmark failed with return code {result.get('returncode')}.
                        Error: {result.get('stderr', 'Check logs.txt for details.')}"""
                        )

                else:
                    status_placeholder.success("Benchmark completed successfully!")
                # Display results in tabs (accessible for all branches)
                with tab1:
                    if result.get("returncode") != 0:
                        st.error("Benchmark failed. Check other tabs for details.")
                    else:
                        st.subheader("Benchmark Metrics")
                        metrics = result.get("metrics", {})
                        
                        if metrics:
                            # Display metrics in a more organized way
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

                            st.divider()
                            st.subheader("GPU Metrics")

                            col1, col2 = st.columns(2)

                            with col1:
                                if "avg_gpu_util_percent" in metrics:
                                    st.metric("Avg GPU Util (%)", f"{metrics['avg_gpu_util_percent']:.2f}")
                                if "avg_gpu_mem_mb" in metrics:
                                    st.metric("Avg GPU Memory (MB)", f"{metrics['avg_gpu_mem_mb']:.2f}")

                            with col2:
                                if "peak_gpu_util_percent" in metrics:
                                    st.metric("Peak GPU Util (%)", f"{metrics['peak_gpu_util_percent']:.2f}")
                                if "peak_gpu_mem_mb" in metrics:
                                    st.metric("Peak GPU Memory (MB)", f"{metrics['peak_gpu_mem_mb']:.2f}")

                            
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
        vLLM Benchmark Tool | ©️2026
        </div>
        """,
        unsafe_allow_html=True
    )

# RUN HISTORY TAB
# ============================================================
with history_tab:

    st.header("Benchmark Run History")
    st.caption(f"Showing runs for user: {st.session_state.username}")


    if not os.path.exists(history_file):
        st.info("No benchmark history found yet.")
        st.stop()

    try:
        df = pd.read_json(history_file, lines=True)
        if "username" in df.columns:
            df = df[df["username"] == st.session_state.username]
    except Exception as e:
        st.error(f"Failed to read history file: {str(e)}")
        st.stop()

    if df.empty:
        st.info("You have no runs yet. Click on Run Benchmark to start your first one")
        st.stop()

    # ------------------------
    # Cleanup Columns
    # ------------------------
    df = df.drop(columns=["benchmark_type", "benchmark_mode"], errors="ignore")

    # Convert timestamp safely
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Sort latest first
    df = df.sort_values("timestamp", ascending=False)

    st.success(f"Total Runs Recorded: {len(df)}")

    st.divider()

    # ------------------------
    # Filters Section
    # ------------------------
    st.subheader("Filters")

    col1, col2 = st.columns(2)

    with col1:
        model_options = sorted(df["model"].dropna().unique().tolist())
        model_options.insert(0, "All Models")

        selected_model = st.selectbox(
            "Filter by Model",
            options=model_options,
            index=0
        )

    # Throughput filter
    with col2:
        min_throughput = st.number_input(
            "Min Throughput (req/sec)",
            min_value=0.0,
            value=0.0,
            step=1.0
        )

    col3, col4 = st.columns(2)

    # Date filter
    with col3:
        if "timestamp" in df.columns:
            date_range = st.date_input(
                "Filter by Date Range",
                value=(
                    df["timestamp"].min().date(),
                    df["timestamp"].max().date()
                )
            )
        else:
            date_range = None

    # GPU Util filter
    with col4:
        gpu_threshold = st.slider(
            "Min Avg GPU Utilization (%)",
            0, 100, 0
        )

    # Apply Filters
    filtered_df = df.copy()

    # Model filter
    if selected_model != "All Models":
        filtered_df = filtered_df[
            filtered_df["model"] == selected_model
        ]

    # Throughput filter
    if "throughput" in filtered_df.columns:
        filtered_df = filtered_df[
            filtered_df["throughput"].fillna(0) >= min_throughput
        ]

    # GPU Util filter
    if "avg_gpu_util_percent" in filtered_df.columns:
        filtered_df = filtered_df[
            filtered_df["avg_gpu_util_percent"].fillna(0) >= gpu_threshold
        ]

    # Date filter
    if date_range and isinstance(date_range, tuple):
        start_date, end_date = date_range
        if "timestamp" in filtered_df.columns:
            filtered_df = filtered_df[
                (filtered_df["timestamp"].dt.date >= start_date) &
                (filtered_df["timestamp"].dt.date <= end_date)
            ]

    st.divider()

    
    # Display Results
    st.write(f"### Showing {len(filtered_df)} runs")

    st.dataframe(
        filtered_df,
        width='stretch',
        hide_index=True
    )

    st.divider()

    # Clear History Button
    if st.button("🗑️ Clear History"):
        open(history_file, "w").close()
        st.success("History cleared.")
        st.rerun()
