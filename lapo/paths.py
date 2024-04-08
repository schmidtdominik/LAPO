from pathlib import Path

MAX_DATA_CHUNKS = 80

storage_path = Path(".")
_expert_data_path = storage_path / "expert_data"
_experiment_results_path = storage_path / "exp_results"

assert (
    _expert_data_path.exists()
), f"Expert data dir: {_expert_data_path} does not exist"


def get_expert_data(env_name: str, test: bool) -> list[Path]:
    test_flag = "test" if test else "train"
    task_data_path = _expert_data_path / env_name / test_flag
    return sorted(task_data_path.iterdir(), key=lambda x: int(x.stem))[:MAX_DATA_CHUNKS]


def get_experiment_dir(exp_name):
    d = _experiment_results_path / exp_name
    d.mkdir(exist_ok=True, parents=True)
    return d


def get_models_path(exp_name: str):
    return get_experiment_dir(exp_name) / "idm_fdm.pt"


def get_latent_policy_path(exp_name):
    return get_experiment_dir(exp_name) / "latent_policy.pt"


def get_decoded_policy_path(exp_name):
    return get_experiment_dir(exp_name) / "decoded_policy.pt"