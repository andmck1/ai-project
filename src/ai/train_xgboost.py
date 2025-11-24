from src.config.logging import setup_logging
from src.game.screen import Screen
from src.game.game import Game
from src.config.enums import Direction
from src.ai.data import create_input_row

from uuid import uuid4
import numpy as np
import matplotlib
import matplotlib.axes
import matplotlib.pyplot as plt
import matplotlib.figure
from tqdm import tqdm
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import (
    make_scorer,
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)
import pandas as pd
import random
import xgboost
from xgboost import XGBRegressor
from datetime import datetime as dt
from dataclasses import dataclass
from typing import Any
import mlflow
import logging

setup_logging()
logging.getLogger("mlflow").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)


def init_data() -> pd.DataFrame:
    screen = Screen(is_draw_enabled=False)
    game = Game(
        screen,
        "MAT501 - Coursework",
        is_draw_enabled=False,
        framerate=0,
    )
    state = game.training_ai_get_current_snake_state()
    data_cols = (
        list(range(len(state) - 1))
        + [f"input_{direction.name}" for direction in list(Direction)]
        + [f"current_{direction.name}" for direction in list(Direction)]
        + ["reward"]
    )
    return pd.DataFrame(columns=pd.Index(data_cols))


def simulate_game_run(
    max_steps: int,
    is_ai_random: bool,
    mdl: xgboost.sklearn.XGBRegressor | None = None,
    random_move_rate: float | None = None,
) -> tuple[pd.DataFrame, int, tuple[int, int]]:
    data = init_data()
    is_draw_enabled = False
    framerate = 0
    screen = Screen(is_draw_enabled=is_draw_enabled)
    game = Game(
        screen,
        "MAT501 - Coursework",
        is_draw_enabled=is_draw_enabled,
        framerate=framerate,
    )
    game_over = False
    frame_step = 0
    for frame_step in range(max_steps):
        game.training_ai_frame()
        for snake_idx in range(2):
            state = game.training_ai_get_current_snake_state()

            result = None
            if is_ai_random:
                direction = list(Direction)[
                    random.randint(0, len(list(Direction)) - 1)
                ]
            else:
                if mdl is None:
                    continue
                if random_move_rate is None:
                    continue
                input_rows = []
                for direction in list(Direction):
                    input_row = create_input_row(
                        snake_idx, state, 0, direction, list(data.columns)
                    )
                    input_rows.append(input_row)
                input_rows = pd.DataFrame(input_rows).drop("reward", axis=1)
                if random.random() > random_move_rate:
                    direction = list(Direction)[
                        mdl.predict(input_rows).argmax()
                    ]
                else:
                    direction = list(Direction)[
                        random.randint(0, len(list(Direction)) - 1)
                    ]
            result = game.training_ai_step(direction)

            if result is None:
                game_over = True
                break
            _, reward, game_over = result

            row = create_input_row(
                snake_idx, state, reward, direction, data.columns
            )

            if len(data) > 0:
                data = pd.concat(
                    [data, pd.DataFrame([row])], ignore_index=True
                )
            else:
                data = pd.DataFrame([row])

            if game_over:
                break
        if game_over:
            break
        game.training_ai_end_frame()

    snake_lengths = (
        len(game.state.snakes[0].body),
        len(game.state.snakes[1].body),
    )
    return data, frame_step, snake_lengths


def simulate_game_with_random_inputs(
    num_episodes: int = 20, max_steps: int = 50
) -> tuple[pd.DataFrame, list[int], tuple[list[int], list[int]]]:
    simulated_data_dfs = []
    all_n_frame_steps = []
    all_snake_lengths = []
    for _ in tqdm(
        range(num_episodes),
        ncols=100,
        position=0,
        leave=True,
        desc="Simulating game",
    ):
        df, n_frame_steps, snake_lengths = simulate_game_run(
            max_steps, is_ai_random=True
        )
        simulated_data_dfs.append(df)
        all_n_frame_steps.append(n_frame_steps)
        all_snake_lengths.append(snake_lengths)

    simulated_data = pd.concat(simulated_data_dfs, ignore_index=True)
    max_snake_lengths = (
        [max([snake_length[0] for snake_length in all_snake_lengths])],
        [max([snake_length[1] for snake_length in all_snake_lengths])],
    )
    return simulated_data, all_n_frame_steps, max_snake_lengths


def simulate_game_with_model_inputs(
    mdl: xgboost.sklearn.XGBRegressor,
    random_move_rate: float = 0.05,
    num_episodes: int = 20,
    max_steps: int = 50,
) -> tuple[pd.DataFrame, int, tuple[int, int]]:
    simulated_data_dfs = []
    all_n_frame_steps = []
    all_snake_lengths = []
    for _ in tqdm(
        range(num_episodes),
        ncols=100,
        position=0,
        leave=False,
        desc="Simulating game",
    ):
        df, n_frame_steps, snake_lengths = simulate_game_run(
            max_steps,
            is_ai_random=False,
            mdl=mdl,
            random_move_rate=random_move_rate,
        )
        simulated_data_dfs.append(df)
        all_n_frame_steps.append(n_frame_steps)
        all_snake_lengths.append(snake_lengths)

    simulated_data = pd.concat(simulated_data_dfs, ignore_index=True)
    total_n_frames = len(all_n_frame_steps)
    max_snake_lengths = (
        max([snake_length[0] for snake_length in all_snake_lengths]),
        max([snake_length[1] for snake_length in all_snake_lengths]),
    )
    return simulated_data, total_n_frames, max_snake_lengths


def train_xgboost_model(
    X_data: pd.DataFrame,
    y_data: pd.DataFrame,
    xgb_kwargs: dict[str, Any],
    n_cross_validations: int,
) -> tuple[dict, xgboost.sklearn.XGBModel]:
    mdl = XGBRegressor(**xgb_kwargs)
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
    metrics = {
        "R2_avg": scores["test_r2_score"].mean(),
        "R2_std": scores["test_r2_score"].std(),
        "RMSE_avg": np.sqrt(scores["test_mean_squared_error"]).mean(),
        "RMSE_std": np.sqrt(scores["test_mean_squared_error"]).std(),
        "MAE_avg": scores["test_mean_absolute_error"].mean(),
        "MAE_std": scores["test_mean_absolute_error"].std(),
    }
    plt = (y_true - predictions).plot(kind="hist", bins=100)
    return metrics, plt


def generate_run_name(now: dt) -> str:
    uuid_str = uuid4()
    now_str = now.strftime("%Y-%m-%d-%H-%M-%S")
    return f"{now_str}_{uuid_str}"


def sample_xgboost_parameters(
    params: dict[str, dict[str, list[int | float] | bool | str]],
) -> dict[str, int | float]:
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
    max_number_xgboost_runs: int
    xgboost_parameter_search: dict[
        str, dict[str, list[int | float] | bool | str]
    ]


def run_experiment(context: ExperimentContext) -> None:
    logger.info("Running experiment")
    mlflow.autolog()
    # random_simulation_time = 0
    # model_from_random_simulation_train_time = 0

    # start_time = dt.now()
    logger.info("Simulating game with random inputs")
    simulated_with_random_inputs_data, _, _ = simulate_game_with_random_inputs(
        num_episodes=context.number_episodes_random_simulation,
        max_steps=context.max_steps_random_simulation,
    )
    # end_time = dt.now()
    # random_simulation_time = start_time - end_time

    # TODO - kfold

    experiment = mlflow.get_experiment_by_name(context.experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(context.experiment_name)
    else:
        experiment_id = experiment.experiment_id
    runs_start_time = dt.now()
    for _ in tqdm(
        range(context.max_number_xgboost_runs), desc="Hyperparameter Search"
    ):
        xgboost_params = sample_xgboost_parameters(
            context.xgboost_parameter_search
        )
        mlflow_run(
            experiment_id,
            runs_start_time,
            context.train_validation_split,
            context.n_cross_validation_splits,
            xgboost_params,
            simulated_with_random_inputs_data,
        )


def mlflow_run(
    experiment_id: str,
    start_time: dt,
    train_validation_split: float,
    n_cross_validation_splits: int,
    xgboost_parameters: dict[str, int | float],
    data: pd.DataFrame,
) -> None:
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
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name):
        start_time = dt.now()
        model_scores, model_from_random_simulation = train_xgboost_model(
            X_data=X_train,
            y_data=y_train,
            xgb_kwargs=xgboost_parameters,
            n_cross_validations=n_cross_validation_splits,
        )
        end_time = dt.now()
        model_from_random_simulation_train_time = end_time - start_time

        predictions = model_from_random_simulation.predict(X_test)
        model_metrics, model_residuals_hist = evaluate_model(
            model_scores, y_test.iloc[:, 0], predictions
        )
        model_metrics["model_from_random_simulation_train_time_s"] = round(
            model_from_random_simulation_train_time.total_seconds() / 60, 3
        )
        model_residuals_hist_fig = model_residuals_hist.get_figure()
        mlflow.log_figure(
            model_residuals_hist_fig,
            "model_from_random_simulation_residuals_histogram.png",
        )
        assert isinstance(model_residuals_hist_fig, matplotlib.figure.Figure)
        plt.close(model_residuals_hist_fig)

        mlflow.log_metrics(model_metrics)

    # max_snake_lengths_arr = np.array(max_snake_lengths_per_run).reshape(-1)
    # "Median Run Length": round(np.median(total_n_frames_per_run), 3),
    # "Average Run Length": round(np.mean(total_n_frames_per_run), 3),
    # "Std Run Length": round(np.std(total_n_frames_per_run), 3),
    # "Max Run Length": round(np.max(total_n_frames_per_run), 3),
    # "Median End Snake Length": round(np.median(max_snake_lengths_arr), 3),
    # "Average End Snake Length": round(np.mean(max_snake_lengths_arr), 3),
    # "Std End Snake Length": round(np.std(max_snake_lengths_arr), 3),
    # "Max End Snake Length": round(np.max(max_snake_lengths_arr), 3),
    # model_metrics["random_simulation_time (m)"] = round(
    #     random_simulation_time.total_seconds() / 60, 3
    # )
