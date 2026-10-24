"""MLflow setup shared by the training scripts: local SQLite backend + artifacts under ./mlruns
(git-ignored). Browse runs with:  mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MLRUNS = ROOT / "mlruns"


def setup(experiment):
    import mlflow
    MLRUNS.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{(MLRUNS / 'mlflow.db').as_posix()}")
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(experiment, artifact_location=(MLRUNS / "artifacts" / experiment).as_uri())
    mlflow.set_experiment(experiment)
    return mlflow
