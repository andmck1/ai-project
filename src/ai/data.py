from src.config.enums import Direction, StateMember
from typing import Any
import pandas as pd

INPUT_ROW_COLUMNS = [
    "normalised_distance_to_wall_x",
    "normalised_distance_to_wall_y",
    "normalised_fruit_position_x_delta",
    "normalised_fruit_position_y_delta",
    "closest_segment_x_delta",
    "closest_segment_y_delta",
    "input_UP",
    "input_DOWN",
    "input_LEFT",
    "input_RIGHT",
    "current_UP",
    "current_DOWN",
    "current_LEFT",
    "current_RIGHT",
    "reward",
    "is_state_at_head_offset_(0, -1)_EDGE",
    "is_state_at_head_offset_(0, -1)_EMPTY",
    "is_state_at_head_offset_(0, -1)_MY_BODY",
    "is_state_at_head_offset_(0, -1)_OTHER_SNAKE_BODY",
    "is_state_at_head_offset_(0, -1)_OTHER_SNAKE_HEAD",
    "is_state_at_head_offset_(0, -1)_FOOD",
    "is_state_at_head_offset_(0, 1)_EDGE",
    "is_state_at_head_offset_(0, 1)_EMPTY",
    "is_state_at_head_offset_(0, 1)_MY_BODY",
    "is_state_at_head_offset_(0, 1)_OTHER_SNAKE_BODY",
    "is_state_at_head_offset_(0, 1)_OTHER_SNAKE_HEAD",
    "is_state_at_head_offset_(0, 1)_FOOD",
    "is_state_at_head_offset_(-1, 0)_EDGE",
    "is_state_at_head_offset_(-1, 0)_EMPTY",
    "is_state_at_head_offset_(-1, 0)_MY_BODY",
    "is_state_at_head_offset_(-1, 0)_OTHER_SNAKE_BODY",
    "is_state_at_head_offset_(-1, 0)_OTHER_SNAKE_HEAD",
    "is_state_at_head_offset_(-1, 0)_FOOD",
    "is_state_at_head_offset_(1, 0)_EDGE",
    "is_state_at_head_offset_(1, 0)_EMPTY",
    "is_state_at_head_offset_(1, 0)_MY_BODY",
    "is_state_at_head_offset_(1, 0)_OTHER_SNAKE_BODY",
    "is_state_at_head_offset_(1, 0)_OTHER_SNAKE_HEAD",
    "is_state_at_head_offset_(1, 0)_FOOD",
]


def init_data() -> pd.DataFrame:
    return pd.DataFrame(columns=pd.Index(INPUT_ROW_COLUMNS))


def create_state_member_text(
    current_snake_index: int, statemember: StateMember
) -> str | None:
    if current_snake_index == 0:
        if statemember == StateMember.SNAKE0:
            statemember_text = "MY_BODY"
        elif statemember == StateMember.SNAKE1:
            statemember_text = "OTHER_SNAKE_BODY"
        elif statemember == StateMember.SNAKE1_HEAD:
            statemember_text = "OTHER_SNAKE_HEAD"
        elif statemember == StateMember.SNAKE0_HEAD:
            return None
        else:
            statemember_text = str(statemember.name)
        assert statemember_text != StateMember.SNAKE0_HEAD.name
    else:
        if statemember == StateMember.SNAKE1:
            statemember_text = "MY_BODY"
        elif statemember == StateMember.SNAKE0:
            statemember_text = "OTHER_SNAKE_BODY"
        elif statemember == StateMember.SNAKE0_HEAD:
            statemember_text = "OTHER_SNAKE_HEAD"
        elif statemember == StateMember.SNAKE1_HEAD:
            return None
        else:
            statemember_text = str(statemember.name)
        assert statemember_text != StateMember.SNAKE1_HEAD.name

    return statemember_text


def create_input_row(
    current_snake_index: int,
    state: list[Any],
    reward: float,
    direction: Direction,
) -> dict[str, int | float]:
    row: dict[str, int | float] = {k: 0 for k in INPUT_ROW_COLUMNS}
    row["normalised_distance_to_wall_x"] = state[0]
    row["normalised_distance_to_wall_y"] = state[1]
    row["normalised_fruit_position_x_delta"] = state[2]
    row["normalised_fruit_position_y_delta"] = state[3]
    row["closest_segment_x_delta"] = state[4]
    row["closest_segment_y_delta"] = state[5]

    for direction_member in list(Direction):
        for statemember in list(StateMember):
            statemember_text = create_state_member_text(
                current_snake_index, statemember
            )
            if statemember_text is not None:
                row[
                    f"is_state_at_head_offset_{str(direction_member.value)}_{statemember_text}"
                ] = 0

    for i, _direction_member in zip(
        range(len(state) - 5, len(state) - 1), list(Direction)
    ):
        statemember_text = create_state_member_text(
            current_snake_index, state[i]
        )
        assert statemember_text is not None
        row[
            f"is_state_at_head_offset_{str(_direction_member.value)}_{statemember_text}"
        ] = 1

    row[f"current_{state[-1].name}"] = 1
    row[f"input_{direction.name}"] = 1
    row["reward"] = reward
    return row
