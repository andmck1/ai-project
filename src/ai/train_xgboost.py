from dataclasses import dataclass
from typing import Any
from src.game.screen import Screen
from src.game.game import Game
from src.config.enums import Direction
from src.ai.data import create_input_row

import numpy as np
import matplotlib
import matplotlib.axes
import matplotlib.pyplot as plt
import matplotlib.figure
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pandas as pd
import random
import xgboost
from xgboost import XGBRegressor

import mlflow

import logging

logging.getLogger("mlflow").setLevel(logging.ERROR)


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
    total_n_frames = [
        len(n_frame_steps) for n_frame_steps in all_n_frame_steps
    ]
    max_snake_lengths = (
        [
            max([snake_length[0] for snake_length in _snake_lengths])
            for _snake_lengths in all_snake_lengths
        ],
        [
            max([snake_length[1] for snake_length in _snake_lengths])
            for _snake_lengths in all_snake_lengths
        ],
    )
    return simulated_data, total_n_frames, max_snake_lengths


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
    X_train: pd.DataFrame, y_train: pd.DataFrame, xgb_kwargs: dict[str, Any]
) -> xgboost.sklearn.XGBRegressor:
    # TODO add k-fold
    mdl = XGBRegressor(**xgb_kwargs)
    mdl.fit(X_train, y_train)
    return mdl


def evaluate_model(
    y_true: pd.DataFrame,
    predictions: pd.DataFrame,
    total_n_frames_per_run: list[int],
    max_snake_lengths_per_run: tuple[list[int], list[int]],
) -> tuple[dict[str, float], matplotlib.axes.Axes]:
    max_snake_lengths_arr = np.array(max_snake_lengths_per_run).reshape(-1)

    metrics = {
        "R2": r2_score(y_true, predictions),
        "RMSE": mean_squared_error(y_true, predictions),
        "MAE": mean_absolute_error(y_true, predictions),
        "Median Run Length": np.median(total_n_frames_per_run),
        "Average Run Length": np.mean(total_n_frames_per_run),
        "Std Run Length": np.std(total_n_frames_per_run),
        "Max Run Length": np.max(total_n_frames_per_run),
        "Median Snake Length": np.median(max_snake_lengths_arr),
        "Average Snake Length": np.mean(max_snake_lengths_arr),
        "Std Snake Length": np.std(max_snake_lengths_arr),
        "Max Snake Length": np.max(max_snake_lengths_arr),
    }

    plt = pd.Series(y_true.values.reshape(-1) - predictions).plot(
        kind="hist", bins=100
    )
    return metrics, plt


@dataclass(kw_only=True)
class ExperimentContext:
    number_episodes_random_simulation: int
    max_steps_random_simulation: int
    number_episodes_model_simulation: int
    max_steps_model_simulation: int


def run_experiment(context: ExperimentContext) -> None:
    mlflow.autolog()

    simulated_with_random_inputs_data, total_n_frames, max_snake_lengths = (
        simulate_game_with_random_inputs(
            num_episodes=context.number_episodes_random_simulation,
            max_steps=context.max_steps_random_simulation,
        )
    )

    # TODO - kfold
    x_cols = [
        col
        for col in simulated_with_random_inputs_data.columns
        if col != "reward"
    ]
    y_col = ["reward"]
    X_train, X_test, y_train, y_test = train_test_split(
        simulated_with_random_inputs_data[x_cols],
        simulated_with_random_inputs_data[y_col],
        test_size=0.33,
        shuffle=True,
    )

    # TODO - don't use asserts
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_train, pd.DataFrame)
    assert isinstance(y_test, pd.DataFrame)

    with mlflow.start_run(experiment_id="xgboost-snake"):
        model_from_random_simulation = train_xgboost_model(
            X_train=X_train, y_train=y_train, xgb_kwargs=dict(n_estimators=100)
        )
        predictions = model_from_random_simulation.predict(X_test)
        model_metrics, model_residuals_hist = evaluate_model(
            y_test, predictions, total_n_frames, max_snake_lengths
        )
        model_residuals_hist_fig = model_residuals_hist.get_figure()
        assert isinstance(model_residuals_hist_fig, matplotlib.figure.Figure)

        mlflow.log_metrics(model_metrics)
        mlflow.log_figure(
            model_residuals_hist_fig,
            "model_from_random_simulation_residuals_histogram.png",
        )

        plt.close(model_residuals_hist_fig)
