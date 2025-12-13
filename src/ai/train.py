from contextlib import redirect_stderr, redirect_stdout
import os
from src.config.logging import create_logger
from src.game.screen import Screen
from src.game.game import Game
from src.config.enums import Direction
from src.ai.data import create_input_row, init_data

# misc
from uuid import uuid4
from tqdm import tqdm
import random
from datetime import datetime as dt
from dataclasses import dataclass
from typing import Any
import logging
import multiprocessing as mp
from functools import partial

# data
import numpy as np
import pandas as pd

# plotting
import matplotlib
import matplotlib.axes
import matplotlib.pyplot as plt

# ml
import sklearn
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import (
    make_scorer,
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)
import xgboost
import mlflow
import mlflow.xgboost
import mlflow.data.pandas_dataset

N_CORES = 4

logger = create_logger(__name__)
logging.getLogger("mlflow").setLevel(logging.ERROR)


def simulate_game_run(
    _multiprocessing_dummy: int,
    max_steps: int,
    is_ai_random: bool,
    mdl: xgboost.sklearn.XGBModel | None = None,
    random_move_rate: float | None = None,
) -> tuple[pd.DataFrame, int, tuple[int, int], list[float | None]]:
    """
    Function designed to run as dispatch from multiprocessor.

    Simulates game run using given model settings at max frame speed.
    """
    # init
    data = init_data()
    inference_times: list[float | None] = []

    game_over = False
    frame_step = 0
    
    # create game with uncapped frame speed
    is_draw_enabled = False
    framerate = 0
    screen = Screen(is_draw_enabled=is_draw_enabled)
    game = Game(
        screen,
        "MAT501 - Coursework",
        is_draw_enabled=is_draw_enabled,
        framerate=framerate,
    )
    
    # run up to max number of steps
    for frame_step in range(max_steps):
        # for each snake
        for snake_idx in range(2):
            # get current state
            state = game.training_ai_get_current_snake_state()
            
            # get AI move
            direction = None
            result = None
            if is_ai_random:
                directions = list(Direction)
                random.shuffle(directions)
            else:
                if mdl is None:
                    continue
                if random_move_rate is None:
                    continue

                input_rows = []
                start_inference_time = dt.now()
                for direction in list(Direction):
                    input_row = create_input_row(
                        game.state.current_ai_snake, state, 0, direction
                    )
                    input_rows.append(input_row)
                input_rows = pd.DataFrame(input_rows).drop("reward", axis=1)

                if random.random() > random_move_rate:
                    directions = list(
                        np.array(list(Direction))[
                            mdl.predict(input_rows).argsort()[::-1]
                        ]
                    )
                    inference_time = (
                        dt.now() - start_inference_time
                    ).total_seconds()
                else:
                    inference_time = None
                    directions = list(Direction)
                    random.shuffle(directions)

                inference_times.append(inference_time)
            
            # step game forward, checking for end moves
            for direction in directions:
                result = game.training_ai_step(direction)
                if result != -1:
                    break
            
            # update data
            if result is None or isinstance(result, int):
                game_over = True
                break
            _, reward, game_over = result

            assert isinstance(direction, Direction)
            row = create_input_row(snake_idx, state, reward, direction)

            if len(data) > 0:
                data = pd.concat(
                    [data, pd.DataFrame([row])], ignore_index=True
                )
            else:
                data = pd.DataFrame([row])

            # not sure why I need this but I do
            game.state.increment_current_ai_index()

            if game_over:
                break

        if game_over:
            break

        game.training_ai_end_frame()

    snake_lengths = (
        len(game.state.snakes[0].body),
        len(game.state.snakes[1].body),
    )
    return data, frame_step, snake_lengths, inference_times


def simulate_game_with_random_inputs(
    num_episodes: int = 20, max_steps: int = 50
) -> tuple[
    pd.DataFrame, list[int], tuple[list[int], list[int]], list[float | None]
]:
    """
    Generate game data using random inputs to explore state space.

    Runs across multiple cores.
    """
    # init
    simulated_data_dfs = []
    all_n_frame_steps = []
    all_snake_lengths = []
    all_inference_times = []
    
    # run game
    simulate_game_run_wrapper_partial = partial(
        simulate_game_run, max_steps=max_steps, is_ai_random=True
    )
    with mp.Pool(N_CORES) as pool:
        results = list(
            pool.imap_unordered(
                simulate_game_run_wrapper_partial,
                tqdm(
                    range(num_episodes),
                    ncols=100,
                    leave=True,
                    desc="Simulating game with random inputs",
                ),
            )
        )

    # collect data
    for df, n_frame_steps, snake_lengths, inference_times in results:
        simulated_data_dfs.append(df)
        all_n_frame_steps.append(n_frame_steps)
        all_snake_lengths.append(snake_lengths)
        all_inference_times += inference_times

    simulated_data = pd.concat(simulated_data_dfs, ignore_index=True)
    max_snake_lengths = (
        [max([snake_length[0] for snake_length in all_snake_lengths])],
        [max([snake_length[1] for snake_length in all_snake_lengths])],
    )
    return (
        simulated_data,
        all_n_frame_steps,
        max_snake_lengths,
        all_inference_times,
    )


def simulate_game_with_model_inputs(
    mdl: xgboost.sklearn.XGBModel,
    random_move_rate: float = 0.05,
    num_episodes: int = 20,
    max_steps: int = 50,
    is_multiproc: bool = True,
) -> tuple[
    pd.DataFrame,
    list[int],
    tuple[list[int], list[int]],
    list[float | None],
]:
    """
    Collect simulation data using model to create inputs.

    Runs across multiple cores.
    """
    # init
    simulated_data_dfs = []
    all_n_frame_steps = []
    all_snake_lengths = []
    all_model_inference_times = []
    
    # run simulation
    simulate_game_run_wrapper_partial = partial(
        simulate_game_run,
        max_steps=max_steps,
        is_ai_random=False,
        mdl=mdl,
        random_move_rate=random_move_rate,
    )
    if is_multiproc:
        with mp.Pool(N_CORES) as pool:
            results = list(
                pool.imap_unordered(
                    simulate_game_run_wrapper_partial,
                    range(num_episodes),
                )
            )
    else:
        results = list(
            simulate_game_run_wrapper_partial(_) for _ in range(num_episodes)
        )
    
    # collect data
    for df, n_frame_steps, snake_lengths, inference_times in results:
        simulated_data_dfs.append(df)
        all_n_frame_steps.append(n_frame_steps)
        all_snake_lengths.append(snake_lengths)
        all_model_inference_times += inference_times

    simulated_data = pd.concat(simulated_data_dfs, ignore_index=True)
    max_snake_lengths = (
        [max([snake_length[0] for snake_length in all_snake_lengths])],
        [max([snake_length[1] for snake_length in all_snake_lengths])],
    )
    return (
        simulated_data,
        all_n_frame_steps,
        max_snake_lengths,
        all_model_inference_times,
    )


def train_model(
    model_name: str,
    X_data: pd.DataFrame,
    y_data: pd.DataFrame,
    params: dict[str, Any],
    n_cross_validations: int,
) -> tuple[dict, xgboost.sklearn.XGBModel]:
    """
    Run cross validation using YAML config.
    """
    match model_name:
        case "xgboost":
            mdl = xgboost.sklearn.XGBRegressor(**params)
        case "lasso":
            mdl = sklearn.linear_model.Lasso(**params)
        case "random_forest":
            mdl = sklearn.ensemble.RandomForestRegressor(**params, n_jobs=-1)
        case _:
            raise Exception(f"{model_name} is not supported")

    scores = cross_validate(
        mdl,
        X_data,
        y_data,
        cv=n_cross_validations,
        scoring={
            "r2_score": make_scorer(r2_score),
            "mean_squared_error": make_scorer(mean_squared_error),
            "mean_absolute_error": make_scorer(mean_absolute_error),
        },
    )
    return scores, mdl.fit(X_data, y_data)


def evaluate_model(
    scores: dict, y_true: pd.Series, predictions: pd.Series
) -> tuple[dict[str, float], matplotlib.axes.Axes]:
    """
    Create metrics and plots for model evaluation.
    """
    metrics = {
        "R2_avg": round(scores["test_r2_score"].mean(), 3),
        "R2_std": round(scores["test_r2_score"].std(), 3),
        "R2_valid": round(r2_score(y_true, predictions), 3),
        "RMSE_avg": round(
            np.sqrt(scores["test_mean_squared_error"]).mean(), 3
        ),
        "RMSE_std": round(np.sqrt(scores["test_mean_squared_error"]).std(), 3),
        "RMSE_valid": round(mean_squared_error(y_true, predictions) ** 0.5, 3),
        "MAE_avg": round(scores["test_mean_absolute_error"].mean(), 3),
        "MAE_std": round(scores["test_mean_absolute_error"].std(), 3),
        "MAE_valid": round(mean_absolute_error(y_true, predictions), 3),
    }
    plt = (y_true - predictions).plot(
        kind="hist",
        bins=100,
        title="Histogram of Residuals",
        xlabel="Residual",
        ylabel="Count",
    )
    return metrics, plt


def generate_run_name(now: dt) -> str:
    uuid_str = uuid4()
    now_str = now.strftime("%Y-%m-%d-%H-%M-%S")
    return f"{now_str}_{uuid_str}"


def sample_params(
    params: dict[str, dict[str, list[int | float] | bool | str]],
) -> dict[str, int | float]:
    """
    Sampler logic for random hyperparameter search.
    """
    sampled_params = dict()

    for param_key, param_info in params.items():
        assert isinstance(param_info["range"], list)
        assert isinstance(param_info["is_log"], bool)

        param_value_min = param_info["range"][0]
        param_value_max = param_info["range"][1]

        is_param_value_range_log_scaled = param_info["is_log"]
        param_value = None
        match param_info["type"]:
            case "int":
                assert isinstance(param_value_min, int)
                assert isinstance(param_value_max, int)
                if is_param_value_range_log_scaled:
                    param_value = np.exp(
                        np.random.uniform(
                            np.log(param_value_min), np.log(param_value_max)
                        )
                    )
                else:
                    param_value = random.randint(
                        param_value_min, param_value_max
                    )
            case "float":
                assert isinstance(param_value_min, float)
                assert isinstance(param_value_max, float)
                if is_param_value_range_log_scaled:
                    param_value = np.exp(
                        np.random.uniform(
                            np.log(param_value_min), np.log(param_value_max)
                        )
                    )
                else:
                    param_value = np.random.uniform(
                        param_value_min, param_value_max
                    )
            case _:
                param_value = None
        sampled_params[param_key] = param_value
    return sampled_params


@dataclass(kw_only=True)
class ExperimentContext:
    experiment_name: str
    train_validation_split: float
    n_cross_validation_splits: int
    number_episodes_random_simulation: int
    max_steps_random_simulation: int
    max_number_model_runs: int
    model_search: dict[
        str, dict[str, dict[str, list[int | float] | bool | str]]
    ]
    max_n_test_simulation_steps: int
    max_test_simulation_episodes: int


def mlflow_run(
    params: dict[str, int | float],
    model_name: str,
    experiment_id: str,
    start_time: dt,
    train_validation_split: float,
    n_cross_validation_splits: int,
    max_n_test_simulation_steps: int,
    max_test_simulation_episodes: int,
    data: pd.DataFrame,
    description: str,
    random_move_rate: float = 0.0,
) -> pd.DataFrame:
    """
    Single MLflow run.

    Split data for train/validation, train the model using cross validation,
    simulate game using model and finally collect any metrics and artifacts.
    """
    # Train/validation split
    x_cols = [col for col in data.columns if col != "reward"]
    y_col = ["reward"]
    X_train, X_test, y_train, y_test = train_test_split(
        data[x_cols],
        data[y_col],
        test_size=train_validation_split,
        shuffle=True,
    )
    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)
    y_train = pd.DataFrame(y_train)
    y_test = pd.DataFrame(y_test)

    run_name = generate_run_name(start_time)
    
    # MLflow warning verbosity workaround
    null = open(os.devnull, "w")
    with redirect_stdout(null), redirect_stderr(null):
        with mlflow.start_run(
            experiment_id=experiment_id,
            run_name=run_name,
            description=description,
            tags={"model_type": model_name},
        ):
            # train model
            start_time = dt.now()
            model_scores, trained_model = train_model(
                model_name=model_name,
                params=params,
                X_data=X_train,
                y_data=y_train,
                n_cross_validations=n_cross_validation_splits,
            )
            end_time = dt.now()
            model_from_random_simulation_train_time = end_time - start_time
            
            # log model
            match model_name:
                case "xgboost":
                    mlflow.xgboost.log_model(
                        trained_model, artifact_path="model"
                    )
                case _:
                    mlflow.sklearn.log_model(
                        trained_model, artifact_path="model"
                    )
            mlflow.log_params(params)
            
            # log datasets
            mlflow_train_dataset = mlflow.data.pandas_dataset.from_pandas(
                data.iloc[X_train.index, :]
            )
            mlflow_test_dataset = mlflow.data.pandas_dataset.from_pandas(
                data.iloc[X_test.index, :]
            )
            mlflow.log_inputs(
                [mlflow_train_dataset, mlflow_test_dataset],
                contexts=["train", "test"],
                tags_list=[None, None],
            )
            
            # create and log metrics and figures
            predictions = trained_model.predict(X_test)
            model_metrics, model_residuals_hist = evaluate_model(
                model_scores, y_test.iloc[:, 0], predictions
            )
            model_metrics["model_from_random_simulation_train_time_s"] = round(
                model_from_random_simulation_train_time.total_seconds() / 60, 3
            )
            model_residuals_hist_figure = model_residuals_hist.get_figure()
            mlflow.log_figure(
                model_residuals_hist_figure,
                "model_from_random_simulation_residuals_histogram.png",
            )
            plt.close()
            
            # simulate game with model
            (
                simulated_results,
                total_n_frames_per_run,
                max_snake_lengths_per_run,
                model_inference_times,
            ) = simulate_game_with_model_inputs(
                trained_model,
                random_move_rate,
                max_steps=max_n_test_simulation_steps,
                num_episodes=max_test_simulation_episodes,
                is_multiproc=False,
            )

            # collect metrics
            max_snake_lengths_arr = np.array(
                max_snake_lengths_per_run
            ).reshape(-1)
            model_inference_times_arr = np.array(
                [
                    inference_time
                    for inference_time in model_inference_times
                    if inference_time is not None
                ]
            )
            model_metrics = {
                **model_metrics,
                **{
                    "Median Run Length": round(
                        np.median(total_n_frames_per_run), 3
                    ),
                    "Average Run Length": round(
                        np.mean(total_n_frames_per_run), 3
                    ),
                    "Std Run Length": round(np.std(total_n_frames_per_run), 3),
                    "Max Run Length": round(np.max(total_n_frames_per_run), 3),
                    "Median End Snake Length": round(
                        np.median(max_snake_lengths_arr), 3
                    ),
                    "Average End Snake Length": round(
                        np.mean(max_snake_lengths_arr), 3
                    ),
                    "Std End Snake Length": round(
                        np.std(max_snake_lengths_arr), 3
                    ),
                    "Max End Snake Length": round(
                        np.max(max_snake_lengths_arr), 3
                    ),
                    "Average Model Inference Time": round(
                        np.mean(model_inference_times_arr), 3
                    ),
                    "Std Model Inference Time": round(
                        np.std(model_inference_times_arr), 3
                    ),
                    "Max Model Inference Time": round(
                        np.max(model_inference_times_arr), 3
                    ),
                },
            }
            mlflow.log_metrics(model_metrics)
            mlflow.log_table(simulated_results, "simulated_results.json")
            
            # model specific metrics and figures
            match model_name:
                case "xgboost":
                    feature_importances = pd.Series(
                        trained_model.feature_importances_,
                        index=X_train.columns,
                    ).sort_values(ascending=True)
                    feature_importances_plot = feature_importances.plot(
                        kind="barh",
                        title="Feature Importances",
                        xlabel="Importance",
                        ylabel="Feature",
                        figsize=(10, 20),
                    )
                    feature_importances_plot_figure = (
                        feature_importances_plot.get_figure()
                    )
                    mlflow.log_figure(
                        feature_importances_plot_figure,
                        "feature_importances.png",
                    )
                    plt.close()

        return simulated_results


def run_experiment(context: ExperimentContext) -> None:
    """
    Main function for running the whole experiment with a given experiment
    configuration.
    """
    logger.info("Running experiment")
    logger.info("Simulating game with random inputs")
    os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///mlflow.db"
    os.environ["MLFLOW_LOG_LEVEL"] = "ERROR"
    
    # autologging is too unpredictable
    mlflow.autolog(disable=True)
    
    # random simulation
    simulated_with_random_inputs_data, _, _, _ = (
        simulate_game_with_random_inputs(
            num_episodes=context.number_episodes_random_simulation,
            max_steps=context.max_steps_random_simulation,
        )
    )
    
    # mlflow verbosity workaround
    null = open(os.devnull, "w")
    with redirect_stdout(null), redirect_stderr(null):
        experiment = mlflow.get_experiment_by_name(context.experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(context.experiment_name)
        else:
            experiment_id = experiment.experiment_id
    
    # run experiment for each given model and parameters in hyperparameter
    # search
    all_simulated_results = pd.DataFrame()
    for model_key, params in context.model_search.items():
        runs_start_time = dt.now()
        sampled_params = [
            sample_params(params) for _ in range(context.max_number_model_runs)
        ]
        mlflow_run_wrapper = partial(
            mlflow_run,
            model_name=model_key,
            experiment_id=experiment_id,
            start_time=runs_start_time,
            train_validation_split=context.train_validation_split,
            n_cross_validation_splits=context.n_cross_validation_splits,
            data=simulated_with_random_inputs_data,
            max_test_simulation_episodes=context.max_test_simulation_episodes,
            max_n_test_simulation_steps=context.max_n_test_simulation_steps,
            description="Random Simulation Data",
        )

        with mp.Pool(N_CORES) as pool:
            simulated_results_dfs = list(
                tqdm(
                    pool.imap_unordered(
                        mlflow_run_wrapper,
                        sampled_params,
                    ),
                    desc="Hyperparameter Search with Random Model "
                    f"Data for {model_key} model",
                    leave=True,
                    total=len(sampled_params),
                ),
            )
        all_simulated_results = pd.concat(
            simulated_results_dfs, ignore_index=True
        )
    
    # In theory can run further training using trained models to generate
    # simulation data but disabled for now.
    # TODO - bug in simulation code for this, needs fixed
    # for model_key, params in context.model_search.items():
    #     runs_start_time = dt.now()
    #     sampled_params = [
    #         sample_params(params) for _ in range(context.max_number_model_runs)
    #     ]
    #     mlflow_run_wrapper = partial(
    #         mlflow_run,
    #         model_name=model_key,
    #         experiment_id=experiment_id,
    #         start_time=runs_start_time,
    #         train_validation_split=context.train_validation_split,
    #         n_cross_validation_splits=context.n_cross_validation_splits,
    #         data=all_simulated_results,
    #         max_test_simulation_episodes=context.max_test_simulation_episodes,
    #         max_n_test_simulation_steps=context.max_n_test_simulation_steps,
    #         description="Model Simulation Data",
    #     )
    #     with mp.Pool(N_CORES) as pool:
    #         simulated_results_dfs = list(
    #             tqdm(
    #                 pool.imap_unordered(
    #                     mlflow_run_wrapper,
    #                     sampled_params,
    #                 ),
    #                 desc="Hyperparameter Search with Model Simulation "
    #                 f"for {model_key} model",
    #                 leave=True,
    #                 total=len(sampled_params),
    #             ),
    #         )
