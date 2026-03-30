import streamlit as st
import pandas as pd
import os
import time
from datetime import datetime
from helper import fetch_hf_models, detect_model_type
from resolver import CapabilityResolver
from schema import BenchmarkConfig, BenchTaskResponse, DType, Quant, GPUType
from configs import MODEL_CONFIGS, PRESET_CONFIGS
from login import show_auth_screen
from dotenv import load_dotenv
import requests

load_dotenv()

# CONFIG FOR FILES
history_file = os.getenv("HISTORY_FILE", "runs_history.jsonl")

# CONFIG FOR BACKEND
backend_port = os.getenv("BACKEND_PORT")
BACKEND_URL = f"http://127.0.0.1:{backend_port}"

# CONFIG FOR GPUS
gpu_ids_env = os.getenv("GPU_IDS", "")
gpu_ids_env = gpu_ids_env.strip("[]")
gpu_id_list = [int(x.strip()) for x in gpu_ids_env.split(",") if x.strip()]
n_gpus_available = len(gpu_id_list)

st.set_page_config(page_title="vLLM Benchmark", layout="wide")

# session vars
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

# CSS markdown for custom size of fonts and UI elements
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
    unsafe_allow_html=True,
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
    unsafe_allow_html=True,
)

with benchmark_tab:
    st.title("vLLM Benchmark")
    st.write("Run online serving benchmarks on various LLM models")

    if "last_result" not in st.session_state:
        st.session_state.last_result = None

    if "benchmark_running" not in st.session_state:
        st.session_state.benchmark_running = False

    if "benchmark_requested" not in st.session_state:
        st.session_state.benchmark_requested = False

    if "is_unvalid" not in st.session_state:
        st.session_state.is_unvalid = False

    is_valid = True

    # sidebar user card
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
                unsafe_allow_html=True,
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
        unsafe_allow_html=True,
    )

    st.sidebar.subheader("GPU Info")
    st.sidebar.markdown(f"Available count of GPUs: {n_gpus_available}")
    st.sidebar.markdown("-----")

    st.sidebar.subheader("Model Serving Parameters")
    st.sidebar.info(
        "All model params are automatically fetched and set. Over-riding these may enable unexpected behaviour."
    )

    # Fetch HF models
    hf_models = fetch_hf_models()

    # Pre-configured models + HF models
    model_choices = list(MODEL_CONFIGS.keys()) + [m for m in hf_models]

    model_name = st.sidebar.selectbox(
        "Model/Repo Name",
        model_choices,
        help="""
    Select a pre-configured model or any Hugging Face text-generation model.
    [Browse on Hugging Face](https://huggingface.co/)
    """,
    )

    # Custom implemented logic for resolving model params based on model_name
    resolver = CapabilityResolver()
    capabilities = resolver.resolve(model_name)

    # Detect model type chat/base
    model_type = detect_model_type(model_name)

    # use dataset random for base model and set proper endpoint
    dataset_name = "random"
    dataset_path = None
    endpoint = "/v1/completions"

    col1, col2 = st.sidebar.columns(2)
    with col1:
        dtype = st.selectbox(
            "Data Type",
            capabilities.supported_dtypes,
            index=(
                capabilities.supported_dtypes.index(capabilities.default_dtype)
                if capabilities.default_dtype in capabilities.supported_dtypes
                else 0
            ),
            help="""
            Model precision
            [Learn more at Official Docs](https://docs.vllm.ai/en/latest/cli/serve/#-dtype)
            """,
        )

    with col2:
        n_gpus_required = st.number_input(
            "Number of GPUs",
            value=1,
            min_value=1,
            max_value=2,
            help="Number of GPUs to Run Benchmark on",
        )

    # Auto-set max_model_len based on detection
    default_max_len = capabilities.max_model_len

    col1, col2 = st.sidebar.columns(2)
    with col1:
        max_model_len = st.number_input(
            "Max Model Length",
            value=default_max_len,
            min_value=512,
            help=f"Maximum sequence length (model supports up to {default_max_len:,})",
        )

    with col2:
        quantization = st.selectbox(
            "Quantization",
            capabilities.supported_quantizations,
            index=0,
            help=f"""Available quantizations are fetched from Hugging Face model card
        [Learn more at Official Docs](https://docs.vllm.ai/en/latest/cli/serve/#-quantization-q)
        """,
        )

    col1, col2 = st.sidebar.columns(2)
    with col1:
        tp_size = st.number_input(
            "Tensor Parallel Size",
            value=1,
            min_value=1,
            help=f"""Tensor Parllel Size
            [Learn More at Official Docs](https://docs.vllm.ai/en/latest/cli/serve/#-tensor-parallel-size-tp)""",
        )

    with col2:
        dp_size = st.number_input(
            "Data Parallel Size",
            value=1,
            min_value=1,
            help=f"""Data Parallel Size
        [Learn more at Official Docs](https://docs.vllm.ai/en/latest/cli/serve/#-data-parallel-size-dp)
        """,
        )

    # validations for early exit
    total_required = tp_size * dp_size

    if total_required > n_gpus_available:
        st.sidebar.error(
            f"Invalid config: {total_required} exceeds available GPUs ({n_gpus_available})"
        )
        is_valid = False

    if total_required < n_gpus_required:
        st.sidebar.warning(
            f"You are under-utilizing GPUs. {total_required} required are less than {n_gpus_required} provisioned GPUs. Decrement Required Number of GPUs"
        )

    if n_gpus_required > n_gpus_available:
        st.sidebar.error(
            f"Requested GPUs ({n_gpus_required}) > available ({n_gpus_available})"
        )
        is_valid = False

    with st.sidebar:
        gpu_type = st.selectbox(
            "GPU Type",
            ["H100", "L4", "L40s"],
            index=0,
            help=f"""Available types of GPUs to run the benchmark on
            """,
        )

    gpu_memory_util = st.sidebar.slider(
        "GPU Memory Utilization",
        min_value=0.1,
        max_value=1.0,
        value=0.85,
        step=0.05,
        help="Target GPU memory utilization rate",
    )

    with st.sidebar:
        st.subheader("Model Info")
        st.markdown(f"Detected Model Type: {model_type}")
        st.markdown(f"Dataset used: {dataset_name}")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Model Benchmarking Parameters")

    # Number of prompts for online benchmark
    num_prompts = st.sidebar.number_input(
        "Total Number of Prompts(Requests)",
        value=10,
        min_value=1,
        help=""" Number of requests sent throughout the run
            [Num requests](https://docs.vllm.ai/en/latest/cli/bench/serve/#-num-prompts)
            """,
    )

    # num_concurrency suggests no of concurrent requests sent during a run
    num_concurrency = st.sidebar.number_input(
        "Maximum Concurrency",
        value=1,
        min_value=1,
        help="""Number of requests processsed concurrently in the run
            [Max Concurrency](https://docs.vllm.ai/en/latest/cli/bench/serve/#-max-concurrency)
            """,
    )

    col1, col2 = st.sidebar.columns(2)
    with col1:
        input_len = st.number_input(
            "Input Length",
            value=32,
            min_value=16,
            help="""Number of input tokens
                [Input-Len](https://docs.vllm.ai/en/latest/cli/bench/serve/#-input-len)""",
        )

    with col2:
        output_len = st.number_input(
            "Output Length",
            value=32,
            min_value=16,
            help="""Number of output tokens
                [Output-Len](https://docs.vllm.ai/en/latest/cli/bench/serve/#-output-len)
                """,
        )

    use_unbounded = st.sidebar.checkbox(
        "Enable Default Request Rate",
        value=True,
        help="If enabled, request rate is set to default value and handled by vLLM",
    )

    # Validations warning to help early exit
    total_tokens = input_len + output_len

    if total_tokens > max_model_len:
        st.sidebar.error(
            f"Total tokens ({total_tokens}) exceed max_model_len ({max_model_len})"
        )
        is_valid = False

    if not use_unbounded:
        request_rate = st.sidebar.number_input(
            "Request Rate (req/sec)",
            value=1,
            min_value=1,
            help="Number of requests per second",
        )
    else:
        request_rate = None

    st.sidebar.markdown("---")

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Your Configuration Summary")

        # Row 1: Model and request rate
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Model", model_name.split("/")[-1])
        with c2:
            st.metric("Request Rate", request_rate if request_rate else "default")

        # Dataset Config
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Dataset", dataset_name)
        with c2:
            st.metric("Prompts", num_prompts)

        # dtype and quant config
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
            st.metric("GPU Type", f"{gpu_type}")
        with c2:
            st.metric("GPUs Required", f"{n_gpus_required}")

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Tensor Parallel Size", f"{tp_size}")
        with c2:
            st.metric("Data Parallel Size", f"{dp_size}")

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Input Length", f"{input_len} tok")
        with c2:
            st.metric("Output Length", f"{output_len} tok")

    with col2:
        st.subheader("Action")

        button_disabled = not is_valid

        run_button = st.button(
            "Run Benchmark", width="stretch", type="primary", disabled=button_disabled
        )

    st.markdown("---")

    # Run benchmark when button clicked
    if run_button and not st.session_state.benchmark_running:
        st.session_state.benchmark_running = True
        st.session_state.benchmark_requested = True
        st.rerun()

    if st.session_state.benchmark_running and st.session_state.benchmark_requested:
        st.session_state.benchmark_requested = False

        # Create tabs for results display
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["Results", "Configuration", "Stdout", "Debug", "Logs"]
        )

        try:
            # Create benchmark config
            cfg = BenchmarkConfig(
                username=str(st.session_state.username),
                model_name=model_name,
                dtype=DType(dtype),
                max_model_len=max_model_len,
                input_len=input_len,
                output_len=output_len,
                num_prompts=num_prompts,
                gpu_memory_util=gpu_memory_util,
                n_gpus_required=n_gpus_required,
                quantization=Quant(quantization),
                max_concurrency=num_concurrency,
                tp_size=tp_size,
                dp_size=dp_size,
                request_rate=request_rate,
                gpu_type=GPUType(gpu_type),
            )

            # Submit to backend
            resp = requests.post(f"{BACKEND_URL}/submit", json=cfg.model_dump())

            if resp.status_code != 200:
                raise Exception(f"Submit failed: {resp.text}")

            data = resp.json()

            data = BenchTaskResponse(**data)

            task_id = data.id
            status = data.status
            gpu_assigned = data.gpu_assigned

            status_holder = st.info(
                f"Your Benchmarking Task has been acknowledged. Task ID is {task_id}. "
                f"It has been scheduled to GPU {gpu_assigned}"
            )

            progress_bar = st.progress(0)
            progress_text = st.empty()

            progress_value = 0.0
            result = None

            start_time = time.time()
            TIMEOUT = 300  # 10 min

            while True:
                if time.time() - start_time > TIMEOUT:
                    progress_text.error(
                        "Timeout exceeded for polling. No response from backend"
                    )
                    break
                try:
                    res = requests.get(f"{BACKEND_URL}/status/{task_id}", timeout=2)

                    if res.status_code != 200:
                        progress_text.error("Failed to fetch status")
                        break

                    data = res.json()
                    data = BenchTaskResponse(**data)

                    status = data.status

                    # ---- STATUS HANDLING ----
                    if status == "queued":
                        if data.gpu_assigned:
                            progress_text.info(
                                f"Your task has been queued for GPU ID/s: {data.gpu_assigned}"
                            )
                        else:
                            progress_text.info(f"Your task has been queued")

                        progress_value = min(progress_value + 0.02, 0.2)

                    elif status == "assigned":
                        if gpu_assigned:
                            progress_text.info(
                                f"Your task has been assigned to GPU ID: {data.gpu_assigned}"
                            )
                        progress_value = max(progress_value, 0.3)

                    elif status == "running":
                        progress_text.info("Running benchmark...")
                        progress_value = min(progress_value + 0.05, 0.9)

                    elif status == "completed":
                        progress_bar.progress(1.0)
                        progress_text.success("Benchmark Completed!")

                        result = data.result
                        break

                    elif status == "failed":
                        progress_text.error("Failed")
                        result = data.result
                        break

                    progress_bar.progress(progress_value)

                except Exception as e:
                    progress_text.error(f"Polling error: {str(e)}")
                    break

                time.sleep(2)

            status_holder.empty()

            #append jsonl history here;

            metrics = result.metrics if result else {}

            with tab1:
                if not result or status == "failed":
                    st.error("Benchmark failed.")

                    if result and result.error_msg != "":
                        st.subheader("Error Details")
                        st.code(result.error_msg, language="text")

                    st.info("Check Logs / Stdout tabs for more context.")
                else:
                    st.subheader("Benchmark Metrics")
                    if metrics and len(metrics.keys()) > 0:
                        # Display metrics in a more organized way
                        col1, col2 = st.columns(2)

                        with col1:
                            st.write("**Request Metrics**")
                            if "successful_requests" in metrics:
                                st.metric(
                                    "Successful Requests",
                                    int(metrics["successful_requests"]),
                                )
                            if "benchmark_duration_sec" in metrics:
                                st.metric(
                                    "Duration (s)",
                                    f"{metrics['benchmark_duration_sec']:.2f}",
                                )
                            if "request_throughput" in metrics:
                                st.metric(
                                    "Request Throughput (req/s)",
                                    f"{metrics['request_throughput']:.2f}",
                                )

                        with col2:
                            st.write("**Token Metrics**")
                            if "total_input_tokens" in metrics:
                                st.metric(
                                    "Total Input Tokens",
                                    int(metrics["total_input_tokens"]),
                                )
                            if "total_generated_tokens" in metrics:
                                st.metric(
                                    "Total Generated Tokens",
                                    int(metrics["total_generated_tokens"]),
                                )
                            if "total_token_throughput" in metrics:
                                st.metric(
                                    "Total Token Throughput (tok/s)",
                                    f"{metrics['total_token_throughput']:.2f}",
                                )

                        st.divider()

                        # Latency metrics (TTFT, TPOT, ITL)
                        col1, col2 = st.columns(2)

                        with col1:
                            st.write("**Time to First Token (TTFT) - ms**")
                            if "median_ttft_ms" in metrics:
                                st.metric("Median", f"{metrics['median_ttft_ms']:.2f}")
                            # if "p99_ttft_ms" in metrics:
                            # st.metric("P99", f"{metrics['p99_ttft_ms']:.2f}")
                            # if "p95_ttft_ms" in metrics:
                            # st.metric("P95", f"{metrics['p95_ttft_ms']:.2f}")

                        with col2:
                            st.write("**Time Per Output Token (TPOT) - ms**")
                            if "median_tpot_ms" in metrics:
                                st.metric("Median", f"{metrics['median_tpot_ms']:.2f}")
                            # if "p99_tpot_ms" in metrics:
                            # st.metric("P99", f"{metrics['p99_tpot_ms']:.2f}")
                            # if "p95_tpot_ms" in metrics:
                            # st.metric("P95", f"{metrics['p95_tpot_ms']:.2f}")

                        col3, col4 = st.columns(2)

                        with col3:
                            st.write("**Inter-token Latency (ITL) - ms**")
                            if "median_itl_ms" in metrics:
                                st.metric("Median", f"{metrics['median_itl_ms']:.2f}")
                            # if "p99_itl_ms" in metrics:
                            # st.metric("P99", f"{metrics['p99_itl_ms']:.2f}")
                            # if "p95_itl_ms" in metrics:
                            # st.metric("P95", f"{metrics['p95_itl_ms']:.2f}")

                        with col4:
                            st.write("**End-to-End Latency (E2EL) - ms**")
                            if "median_e2el_ms" in metrics:
                                st.metric("Median", f"{metrics['median_e2el_ms']:.2f}")
                            # if "p99_e2el_ms" in metrics:
                            # st.metric("P99", f"{metrics['p99_e2el_ms']:.2f}")
                            # if "p95_e2el_ms" in metrics:
                            # st.metric("P95", f"{metrics['p95_e2el_ms']:.2f}")

                        # gpu_metrics deprecated in v6+
                        # st.divider()
                        # st.subheader("GPU Metrics")

                        # gpu_metrics = metrics.get("gpu_metrics", {})
                    else:
                        st.warning("No metrics were extracted. Check Logs on Server")

                    st.divider()
                    runtime = result.runtime_sec or 0
                    st.metric("Total Runtime", f"{runtime:.2f} seconds")

            with tab2:
                st.subheader("Benchmark Configuration")
                if result:
                    config = result.config
                    st.json(config)
                else:
                    st.warning(
                        "No configuration available because the benchmark did not complete."
                    )

            with tab3:
                st.subheader("Raw Stdout Output")

                bench_logs = ""

                if result:
                    bench_logs = result.bench_logs

                if bench_logs != "":
                    st.code(bench_logs, language="text")
                else:
                    st.write("No Logs to display for this task. Likely Failure")

            with tab4:
                if metrics:
                    st.subheader("Debug Information")
                    st.write(f"**Metrics Found**: {len(metrics)}")
                    st.write(f"**Metrics Keys**: {list(metrics.keys())}")
                    st.write("**Full Metrics Dictionary**:")
                    st.json(metrics if metrics else {"message": "No metrics extracted"})

            with tab5:
                st.subheader("Process Logs")
                process_logs = ""

                if result:
                    process_logs = result.process_logs

                if process_logs != "":
                    st.code(process_logs, language="text")
                else:
                    st.write("No Logs to display for this task. Likely Failure")
            # Export results button
            st.divider()
            st.subheader("Export Results")

            if result:
                col1, col2 = st.columns(2)
                with col1:
                    json_str = result.model_dump_json(indent=2)
                    st.download_button(
                        label="Download as JSON",
                        data=json_str,
                        file_name=f"benchmark_{model_name.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                    )

                with col2:
                    # Create CSV for metrics
                    if result.metrics:
                        csv_data = "Metric,Value\n"
                        for k, v in result.metrics.items():
                            if k == "gpu_metrics":
                                continue
                            csv_data += f"{k},{v}\n"

                        st.download_button(
                            label="Download Metrics as CSV",
                            data=csv_data,
                            file_name=f"metrics_{model_name.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                        )
            else:
                st.info("Export is available only for successful benchmark runs.")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error("Stack trace:")
            st.code(str(e), language="text")
            import traceback

            st.code(traceback.format_exc(), language="text")
        finally:
            # Reset benchmark running flag
            st.session_state.benchmark_running = False
            st.session_state.benchmark_requested = False

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray; font-size: 12px;'>
        vLLM Benchmark Tool | ©️2026
        </div>
        """,
        unsafe_allow_html=True,
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

        # Normalize column names
        rename_map = {
            "taskID": "task_id",
            "task_status": "status",
            "return_code": "returncode",
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

        # Username filter (include nulls if needed)
        if "username" in df.columns:
            df = df[
                (df["username"] == st.session_state.username) | (df["username"].isna())
            ]

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
        df = df[df["timestamp"].notna()]

    # Sort latest first
    df = df.sort_values("timestamp", ascending=False, na_position="last")

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

        selected_model = st.selectbox("Filter by Model", options=model_options, index=0)

    with col2:
        min_throughput = st.number_input(
            "Min Throughput (req/sec)", min_value=0.0, value=0.0, step=1.0
        )

    col3, col4 = st.columns(2)

    with col3:
        if "timestamp" in df.columns:
            date_range = st.date_input(
                "Filter by Date Range",
                value=(df["timestamp"].min().date(), df["timestamp"].max().date()),
            )
        else:
            date_range = None

    with col4:
        gpu_threshold = st.slider("Min Avg GPU Utilization (%)", 0, 100, 0)

    # ------------------------
    # Apply Filters
    # ------------------------
    filtered_df = df.copy()

    # Model filter
    if selected_model != "All Models":
        filtered_df = filtered_df[filtered_df["model"] == selected_model]

    # Throughput filter (FIXED)
    throughput_col = None
    if "request_throughput" in filtered_df.columns:
        throughput_col = "request_throughput"
    elif "total_token_throughput" in filtered_df.columns:
        throughput_col = "total_token_throughput"

    if throughput_col:
        filtered_df = filtered_df[
            filtered_df[throughput_col].fillna(0) >= min_throughput
        ]

    # GPU Util filter
    if "avg_gpu_util_percent" in filtered_df.columns:
        filtered_df = filtered_df[
            filtered_df["avg_gpu_util_percent"].fillna(0) >= gpu_threshold
        ]

    # Date filter
    if (
        isinstance(date_range, tuple)
        and len(date_range) == 2
        and "timestamp" in filtered_df.columns
    ):
        start_date, end_date = date_range

        filtered_df = filtered_df[
            (filtered_df["timestamp"].dt.date >= start_date)
            & (filtered_df["timestamp"].dt.date <= end_date)
        ]

    st.divider()

    # ------------------------
    # Display Results
    # ------------------------
    st.write(f"### Showing {len(filtered_df)} runs")

    st.dataframe(filtered_df, width="stretch", hide_index=True)

    st.divider()
