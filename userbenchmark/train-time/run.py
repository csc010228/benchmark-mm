import argparse
import itertools
import json
import time
from datetime import datetime
from typing import List
from pathlib import Path
import shlex
import textwrap

from ..utils import add_path, dump_output, get_output_dir, get_output_json, REPO_PATH

from torchbenchmark import ModelTask

with add_path(REPO_PATH):
    from torchbenchmark._components._impl.workers.subprocess_rpc import (
        ChildTraceException,
        UnserializableException,
    )
    from torchbenchmark.util.experiment.instantiator import (
        list_models,
        load_model,
        BenchmarkModel,
        TorchBenchModelConfig,
    )
    from torchbenchmark.util.experiment.metrics import (
        get_model_test_metrics,
        TorchBenchModelMetrics,
        NANOSECONDS_PER_MILLISECONDS
    )
    from torchbenchmark.util.extra_args import (
        TEST_STAGE
    )

WORKER_TIMEOUT = 3600  # seconds
BS_FIELD_NAME = "batch_size"

BM_NAME = "train-time"
DEFAULT_ROUNDS = 3
DEFAULT_BATCH_SIZE = 1
DEFAULT_ITERATIONS = 15

WARMUP_ROUNDS = 0


timestamp = datetime.now().strftime("%Y%m%d%H%M%S")


def generate_model_config(model_name: str, args: argparse.Namespace, output_dir: Path) -> List[TorchBenchModelConfig]:
    devices = ["cpu", "cuda"]
    tests = ["train", "eval"]
    cfgs = itertools.product(*[devices, tests])
    result = [
        TorchBenchModelConfig(
            name=model_name,
            device=device,
            test=test,
            batch_size=args.batch_size,
            extra_args=[],
            extra_env=parse_env(args.envs) if args.envs != "" else None,
            output_dir=output_dir.joinpath(model_name),
            skip=False,
        )
        for device, test in cfgs
    ]
    return result


def parse_args(args: List[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--rounds",
        default=DEFAULT_ROUNDS,
        type=int,
        help="Number of rounds to run.",
    )
    parser.add_argument(
        "-i",
        "--iterations",
        default=DEFAULT_ITERATIONS,
        type=int,
        help="Number of iterations every round.",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=DEFAULT_BATCH_SIZE,
        type=int,
        help="Batch size to use for the model.",
    )
    parser.add_argument(
        "-m",
        "--models",
        default="",
        help="Specify the models to run, default (empty) runs all models.",
    )
    parser.add_argument(
        "-e",
        "--envs",
        default="",
        help=(
            "Set the environment variables for the model execution. "
            "The format should be a space-separated list of key-value pairs, e.g., "
            "\"KEY1=value1 KEY2=\\\"value2 with spaces\\\" KEY3=value3,with,commas\". "
            "Supports quoted values and special characters. "
            "Example: --envs \"CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=backend:native,max_split_size_mb:128\"."
        ),
    )
    parser.add_argument("-d", "--device", default="cuda", help="Specify the device.")
    parser.add_argument(
        "-o", "--output", type=str, help="The default output json file."
    )
    args = parser.parse_args(args)
    return args


def parse_env(env_str: str) -> dict:
    """
    解析复杂的环境变量字符串，将其转换为字典。
    
    支持的格式：
    - key1=value1,key2="value2 with spaces",key3="value3,with,commas"
    - 支持引号和转义字符。

    参数:
        env_str (str): 环境变量字符串。

    返回:
        dict: 解析后的字典。
    """
    if not env_str:
        return {}

    env_dict = {}
    # 使用 shlex 分词器解析，支持引号和转义字符
    tokens = shlex.split(env_str)
    for token in tokens:
        if "=" in token:
            key, value = token.split("=", 1)  # 按等号分隔，最多分隔一次
            env_dict[key.strip()] = value.strip()
        else:
            raise ValueError(f"Invalid environment variable format: {token}")
    return env_dict


def reduce_results(full_results):
    ub_metrics = {}
    for round_data in full_results:
        for model_data in round_data:
            model_name = model_data["cfg"]["name"]
            raw_metrics = model_data["raw_metrics"]

            # 初始化模型的指标字典
            if model_name not in ub_metrics:
                ub_metrics[model_name] = {}

            # 遍历每个指标
            for metric, value in raw_metrics.items():
                if value is None:  # 跳过空值
                    continue
                if isinstance(value, list):  # 如果是列表，计算平均值
                    if len(value) > 2:
                        value.remove(max(value))
                        value.remove(min(value))
                    avg_value = sum(value) / len(value)
                else:  # 如果是单个值，直接使用
                    avg_value = value

                # 将平均值累加到结果中
                if metric not in ub_metrics[model_name]:
                    ub_metrics[model_name][metric] = []
                ub_metrics[model_name][metric].append(avg_value)

    # 计算最终的平均值
    for model_name, metrics in ub_metrics.items():
        for metric, values in metrics.items():
            ub_metrics[model_name][metric] = sum(values) / len(values)
    return ub_metrics


def generate_filter(args: argparse.Namespace):
    allowed_models = args.models
    if allowed_models:
        allowed_models = (
            allowed_models.split(",") if "," in allowed_models else [allowed_models]
        )
    allowed_devices = args.device
    allowed_devices = (
        allowed_devices.split(",") if "," in allowed_devices else [allowed_devices]
    )
    allowed_tests = "train"
    allowed_tests = (
        allowed_tests.split(",") if "," in allowed_tests else [allowed_tests]
    )

    def cfg_filter(cfg: TorchBenchModelConfig) -> bool:
        if cfg.device in allowed_devices and cfg.test in allowed_tests:
            if not allowed_models:
                return True
            else:
                return cfg.name in allowed_models
        return False

    return cfg_filter

run_times = []
forward_times = []
backward_times = []
optimizer_times = []

class RunTimer:
    def __enter__(self):
        self.t0 = time.time_ns()
        return self
        
    def __exit__(self, exc_type, exc_value, traceback):
        self.t1 = time.time_ns()
        run_times.append(self.t1 - self.t0)

class ForwardTimer:
    def __enter__(self):
        self.t0 = time.time_ns()
        return self
        
    def __exit__(self, exc_type, exc_value, traceback):
        self.t1 = time.time_ns()
        forward_times.append(self.t1 - self.t0)

class BackwardTimer:
    def __enter__(self):
        self.t0 = time.time_ns()
        return self
        
    def __exit__(self, exc_type, exc_value, traceback):
        self.t1 = time.time_ns()
        backward_times.append(self.t1 - self.t0)

class OptimizerTimer:
    def __enter__(self):
        self.t0 = time.time_ns()
        return self
        
    def __exit__(self, exc_type, exc_value, traceback):
        self.t1 = time.time_ns()
        optimizer_times.append(self.t1 - self.t0)

def load_model_with_timer(config: TorchBenchModelConfig) -> BenchmarkModel:
    model = load_model(config)
    model.add_context(RunTimer, TEST_STAGE.ALL)
    model.add_context(ForwardTimer, TEST_STAGE.FORWARD)
    model.add_context(BackwardTimer, TEST_STAGE.BACKWARD)
    model.add_context(OptimizerTimer, TEST_STAGE.OPTIMIZER)
    return model

def run(args: List[str]):
    args = parse_args(args)
    output_dir = get_output_dir(BM_NAME).joinpath(f"{timestamp}")
    output_dir.mkdir(exist_ok=True, parents=True)
    models = list_models()
    all_args = [args] * len(models)
    output_dirs = [output_dir] * len(models)
    cfgs = list(itertools.chain(*map(generate_model_config, models, all_args, output_dirs)))
    cfg_filter = generate_filter(args)
    # run a model cfg and get latencies
    full_results = []
    for _round in range(args.rounds):
        single_round_result = []
        for cfg in filter(cfg_filter, cfgs):
            print(f"[Round {_round}/{args.rounds}] Running {cfg}")
            try:
                task = load_model_with_timer(cfg)
                # get the model test metrics
                metrics: TorchBenchModelMetrics = get_model_test_metrics(
                    task, metrics=["latencies"], num_iter=args.iterations
                )
                raw_metrics = {
                    "run_times": [x / NANOSECONDS_PER_MILLISECONDS for x in run_times],
                    "forward_times": [x / NANOSECONDS_PER_MILLISECONDS for x in forward_times],
                    "backward_times": [x / NANOSECONDS_PER_MILLISECONDS for x in backward_times],
                    "optimizer_times": [x / NANOSECONDS_PER_MILLISECONDS for x in optimizer_times]
                }
                raw_metrics.update(metrics.__dict__)
                single_round_result.append(
                    {
                        "cfg": {key: str(value) for key, value in cfg.__dict__.items()},
                        "raw_metrics": raw_metrics
                    }
                )
            except NotImplementedError:
                # some models don't implement the test specified
                single_round_result.append(
                    {
                        "cfg": {key: str(value) for key, value in cfg.__dict__.items()},
                        "raw_metrics": "NotImplemented",
                    }
                )
            except ChildTraceException as exception:
                single_round_result.append(
                    {
                        "cfg": {key: str(value) for key, value in cfg.__dict__.items()},
                        "raw_metrics": str(exception),
                    }
                )
            except UnserializableException as exception:
                single_round_result.append(
                    {
                        "cfg": {key: str(value) for key, value in cfg.__dict__.items()},
                        "raw_metrics": exception.args_repr,
                    }
                )
            finally:
                # Remove task reference to trigger deletion in gc
                task = None
                run_times.clear()
                forward_times.clear()
                backward_times.clear()
                optimizer_times.clear()
        full_results.append(single_round_result)
    # reduce full results to metrics
    # log detailed results in the .userbenchmark/model-stableness/logs/ directory
    logs_fname = output_dir.joinpath(f"logs-{timestamp}.json")
    print(full_results)
    with open(logs_fname, "w") as f:
        json.dump(full_results, f, indent=4)
    # output userbenchmark metrics in the .userbenchmark/model-stableness directory
    ub_metrics = reduce_results(full_results)
    output_json = get_output_json(BM_NAME, ub_metrics)
    output_json["args"] = args.__dict__
    print(output_json)
    metrics_fname = output_dir.joinpath(f"metrics-{timestamp}.json")
    with open(metrics_fname, "w") as f:
        json.dump(output_json, f, indent=4)
    # dump_output(BM_NAME, output_json)
