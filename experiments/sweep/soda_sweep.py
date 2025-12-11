#!/usr/bin/env python3
"""
SodaAnalyzer sweep helper

Runs a small grid over batch sizes, sequence lengths, and maximum new tokens
using the SodaAnalyzer + ModelTracer pipeline with an HF model. Swap the lists
below to try different shapes or precisions.
"""

import os
import sys
from itertools import product
from pathlib import Path
from typing import Dict

import torch
import json 
from datetime import datetime 

from soda import ModelTracer, SodaAnalyzer
from soda.common import utils
from experiments.sweep.summarize_soda_sweep import summarize as summarize_soda_sweep
from experiments.sweep.config import GEN_PARAMS, PRECISION_MAP, SWEEP_CONFIG_MAP

def ensure_env_loaded() -> None:
    """Exit early if env.sh was not sourced."""
    if not os.environ.get("SODA_ENV_LOADED"):
        print("Error: SODA environment not loaded.")
        print("Please run: source env.sh")
        sys.exit(1)

def filter_sweep_config(sweep_config: Dict, sweep_model_filter: str) -> Dict:
    # Parse "export SWEEP_MODEL_FILTER=a,b,c" into ["a", "b", "c"], strip whitespaces and drop empty strings
    requested_models = [m.strip() for m in sweep_model_filter.split(",") if m.strip()] 
    filtered_sweep_config = {} # Populate this with intersection of requested_models and sweep_config
    avail_model_str = ", ".join(sweep_config.keys())

    # Scan through requested_models and add to filtered_sweep_config if found in sweep_config
    for model in requested_models:
        if model in sweep_config:
            filtered_sweep_config[model] = sweep_config[model]
        else:
            log = f"Warning: Model '{model}' not found in sweep config for {sweep_mode}.\nAvailable models: {avail_model_str}"
            print(log)

    assert filtered_sweep_config, f"Error: No valid models found in sweep config for {sweep_mode}.\nAvailable models: {avail_model_str}"

def main() -> None:
    ensure_env_loaded()

    gpu_suffix = utils.get_gpu()

    compile_type = GEN_PARAMS["compile_type"]
    device = GEN_PARAMS["device"]
    warmup = GEN_PARAMS["inference_warmup"]

    assert "SWEEP_MODE" in os.environ, "Error: SWEEP_MODE environment variable is not set. Please set export SWEEP_MODE=<prefill|decode|fp8|debug|all> in your shell."
    sweep_mode = os.environ["SWEEP_MODE"]  # Use export SWEEP_MODE=<prefill|decode|fp8|debug|all> in your shell.
    sweep_model_filter = os.environ.get("SWEEP_MODEL_FILTER")  # Use export SWEEP_MODEL_FILTER=gpt2,llama3_1b,qwen_moe in your shell.

    # FP8 requires H100/H200
    if sweep_mode == "fp8":
        assert gpu_suffix in ("H100", "H200"), f"Error: FP8 requires H100/H200 GPU. Detected: {gpu_suffix}"

    # Get sweep config and precision
    sweep_config = SWEEP_CONFIG_MAP[sweep_mode]
    precision = PRECISION_MAP[sweep_mode]

    # Filter out models from sweep config based on SWEEP_MODEL_FILTER 
    if sweep_model_filter:
        filtered_sweep_config = filter_sweep_config(sweep_config, sweep_model_filter)
        sweep_config = filtered_sweep_config

    sweep_config_str = ", ".join(sweep_config.keys())
    print(f"Sweeping models in {sweep_mode} sweep config: {sweep_config_str}")

    for config_name, config in sweep_config.items():
        model = config["model_name"]
        batch_sizes = config["batch_sizes"]
        seq_lens = config["seq_lens"]
        max_new_toks = config["max_new_toks"]

        # Group sweep outputs under base output; ModelTracer will create group/name
        print(f"\n=== Running config: {config_name} ({model}) with precision={precision}")

        experiment_group_dir = None
        for bs, sl, mt in product(batch_sizes, seq_lens, max_new_toks):
            print(f"\n\n\n=== Running sweep point: batch_size={bs}, seq_len={sl}, max_new_tokens={mt}")
            cli_args = [
                "--model", model, "--batch-size", str(bs), "--seq-len", str(sl), "--max-new-tokens", str(mt),
                "--precision", precision, "--compile-type", compile_type, "--device", device,
                "--warmup", warmup,

                # Extra parser knobs (fusion + microbench) left at defaults:
                # "--fusion", "2",
                # "--prox-score", "1.0",
                # "--seed", "42",
                # "--version",

                # NOTE: for SodaAnalyzer
                # "--microbench", # DONOT use microbench 
                # "--warmup", "10", # 10 is ok 
                # "--runs", "5", # This doesn't matter 
            ]
            args = utils.parse_and_validate_args(cli_args)

            try:
                tracer = ModelTracer(args=args)
                tracer.run()

                analyzer = SodaAnalyzer(tracer=tracer, args=args)
                report_path = analyzer.run()
                print(f"Report saved to: {report_path}")

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"Skipping batch_size={bs}, seq_len={sl}, max_new_tokens={mt} due to OOM: {e}")
                    
                    # FIX: Generate a report.json that MATCHES soda.py structure
                    run_output_dir = tracer.output_dir
                    run_output_dir.mkdir(parents=True, exist_ok=True)
                    
                    oom_report = {
                        "metadata": {
                            "model_name": model,
                            "timestamp": datetime.now().isoformat(),
                            "config": {
                                "batch_size": bs,
                                "seq_len": sl,
                                "max_new_tokens": mt,
                                "precision": precision,
                                "compile_type": compile_type,
                                "device": device,
                                "gpu_name": gpu_suffix
                            }
                        },
                        "performance_metrics": {
                            "inference_time_ms": "OOM",
                            "error": str(e)
                        }
                    }
                    
                    with open(run_output_dir / "report.json", "w") as f:
                        json.dump(oom_report, f, indent=4)

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                raise

        # Summarize the experiment group 
        experiment_group_dir = tracer.experiment_group_dir
        summarize_soda_sweep(experiment_group_dir, gpu_name_override=gpu_suffix, max_tok_override=None)

if __name__ == "__main__":
    main()
