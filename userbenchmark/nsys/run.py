import argparse
import itertools
import json
from datetime import datetime
from typing import List
from pathlib import Path
import shlex

from ..utils import add_path, dump_output, get_output_dir, get_output_json, REPO_PATH

from torchbenchmark import ModelTask

with add_path(REPO_PATH):
    from torchbenchmark._components._impl.workers.subprocess_rpc import (
        ChildTraceException,
        UnserializableException,
    )
    from torchbenchmark.util.experiment.instantiator import (
        list_models,
        TorchBenchModelConfig,
    )
    from torchbenchmark.util.experiment.metrics import (
        get_model_test_metrics,
        TorchBenchModelMetrics,
        NANOSECONDS_PER_MILLISECONDS
    )

WORKER_TIMEOUT = 3600  # seconds
BS_FIELD_NAME = "batch_size"

BM_NAME = "nsys"
DEFAULT_ROUNDS = 3
DEFAULT_BATCH_SIZE = 1
DEFAULT_ITERATIONS = 15
DEFAULT_ALLOCATOR = ""
DEFAULT_ALLOC_FUNC = "alloc"
DEFAULT_FREE_FUNC = "free"
DEFAULT_CACHE_INFO_FUNC = "cache_info"

WARMUP_ROUNDS = 0

ALL_NSYS_METRICS = ['cuda', 'nvtx', 'cublas', 'cublas-verbose', 'cusolver', 
           'cusolver-verbose', 'cusparse', 'cusparse-verbose', 'mpi', 'oshmem', 'ucx', 
           'osrt', 'cudnn', 'opengl', 'opengl-annotations', 'openacc', 'openmp', 
           'nvvideo', 'vulkan', 'vulkan-annotations', 'python-gil']


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
        '--metrics', 
        type=str, 
        default="all", 
        choices=ALL_NSYS_METRICS + ["all"], 
        help="Nsys metrics to be obtained, separated by commas.",
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
    parser.add_argument(
        "-a",
        "--allocator",
        default=DEFAULT_ALLOCATOR,
        help="The .so file of user defined allocator.",
    )
    parser.add_argument(
        "--alloc-func",
        default=DEFAULT_ALLOC_FUNC,
        help="Alloc function name of user defined allocator. Default is \"alloc\".",
    )
    parser.add_argument(
        "--free-func",
        default=DEFAULT_FREE_FUNC,
        help="Free function name of user defined allocator. Default is \"free\".",
    )
    parser.add_argument(
        "--cache_info-func",
        default=DEFAULT_CACHE_INFO_FUNC,
        help="Cache info function name of user defined allocator. Default is \"cache_info\".",
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

def init_model_task(config: TorchBenchModelConfig, metrics: str, output_dir, timeout: float = WORKER_TIMEOUT) -> ModelTask:
    if metrics == "all":
        metrics = ",".join(ALL_NSYS_METRICS)
    task = ModelTask(
        config.name,
        timeout=timeout,
        extra_env=config.extra_env,
        save_output_dir=config.output_dir,
        extra_args = ["nsys", "profile", "-t", metrics, "-o", output_dir.joinpath(f"{timestamp}")]
    )
    return task

def load_model_task_with_allocator_and_timer(
    task: ModelTask, config: TorchBenchModelConfig, allocator: str, alloc_func: str, free_func: str, cache_info_func: str
):
    if allocator != DEFAULT_ALLOCATOR:
        task.worker.run(
            f"""
            customized_alloc = torch.cuda.memory.CUDAPluggableAllocator('{allocator}', '{alloc_func}', '{free_func}')
            customized_alloc.set_cache_info_fn('{cache_info_func}')
            torch.cuda.memory.change_current_allocator(customized_alloc)
        """
        )
    if not task.model_details.exists:
        raise ValueError(
            f"Failed to import model task: {config.name}. Please run the model manually to make sure it succeeds, or report a bug."
        )
    task.make_model_instance(
        test=config.test,
        device=config.device,
        batch_size=config.batch_size,
        extra_args=config.extra_args,
    )
    task_batch_size = task.get_model_attribute(BS_FIELD_NAME)
    # check batch size if not measuring accuracy
    if (
        config.batch_size
        and (not config.batch_size == task_batch_size)
        and not task.get_model_attribute("accuracy")
    ):
        raise ValueError(
            f"User specify batch size {config.batch_size},"
            + f"but model {task.name} runs with batch size {task_batch_size}. Please report a bug."
        )

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
    outs_errs = []
    for _round in range(args.rounds):
        single_round_result = []
        single_round_out_err = []
        for cfg in filter(cfg_filter, cfgs):
            print(f"[Round {_round}/{args.rounds}] Running {cfg}")
            try:
                task = init_model_task(cfg, args.metrics, output_dir)
                # get the model test metrics
                with task.worker.watch_stdout_stderr() as get_output:
                    load_model_task_with_allocator_and_timer(task, cfg, args.allocator, args.alloc_func, args.free_func, args.cache_info_func)
                    metrics: TorchBenchModelMetrics = get_model_test_metrics(
                        task, metrics=["gpu_peak_mem"], nwarmup=0, num_iter=args.iterations
                    )
                    stdout, stderr = get_output()
                    single_round_out_err.append(
                        {
                            "cfg": {key: str(value) for key, value in cfg.__dict__.items()},
                            "stdout": stdout,
                            "stderr": stderr,
                        }
                    )
                raw_metrics = {}
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
        full_results.append(single_round_result)
        outs_errs.append(single_round_out_err)
    # reduce full results to metrics
    # log detailed results in the .userbenchmark/model-stableness/logs/ directory
    logs_fname = output_dir.joinpath(f"logs-{timestamp}.json")
    with open(logs_fname, "w") as f:
        json.dump(outs_errs, f, indent=4)
    raw_metrics_fname = output_dir.joinpath(f"raw_metrics-{timestamp}.json")
    with open(raw_metrics_fname, "w") as f:
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
