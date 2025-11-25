from src.cli.game import custom
from pathlib import Path

if __name__ == "__main__":
    mlflow_path = f"file://{str(Path('./_internal/xgboost-model').absolute())}"
    custom(
        ai_mlflow_model_path_or_uri=mlflow_path,
        is_draw_enabled=True,
        is_ai_random=False,
        is_ai_enabled=True,
        is_debug_enabled=True,
        framerate=10,
    )
