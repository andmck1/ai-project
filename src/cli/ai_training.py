from src.config.logging import create_logger
from src.ai.train_xgboost import run_experiment, ExperimentContext

import typer
from pathlib import Path
import yaml

logger = create_logger(__name__)

app = typer.Typer()


@app.command(name="run-experiment")
def run_mlflow_experiment(xgboost_config_path: Path) -> None:
    logger.info("Reading from file: " + str(xgboost_config_path.absolute()))
    with xgboost_config_path.open("r") as config_file:
        xgboost_config = yaml.safe_load(config_file)
        experiment_context = ExperimentContext(
            experiment_name=xgboost_config["experiment_name"],
            train_validation_split=xgboost_config["train_validation_split"],
            n_cross_validation_splits=xgboost_config[
                "n_cross_validation_splits"
            ],
            number_episodes_random_simulation=xgboost_config[
                "number_episodes_random_simulation"
            ],
            max_steps_random_simulation=xgboost_config[
                "max_steps_random_simulation"
            ],
            max_number_xgboost_runs=xgboost_config["max_number_xgboost_runs"],
            xgboost_parameter_search=xgboost_config["xgboost"][
                "parameter_search"
            ],
            max_test_simulation_episodes=xgboost_config[
                "max_test_simulation_episodes"
            ],
            max_n_test_simulation_steps=xgboost_config[
                "max_n_test_simulation_steps"
            ],
        )

    run_experiment(experiment_context)
