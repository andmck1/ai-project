# Versus Snake 
A versus Snake game as part of a research project into game AI where you can
compete against an AI to win at Snake.

![./assets/showcase.gif](Snake Game)

# Run EXE
The default exe provided in `dist` runs the game in normal mode (arrow key controls) infinitely against an xgboost
AI model. To exit press ESCAPE.

The game does take a minute or so load on first run.

# Prerequesites
- [uv](https://github.com/astral-sh/uv)

# Install
```psh
uv sync
```

# Run
To list available cli commands:
```psh
uv run python -m src.cli --help
```

To run the project with ai you first need to run a hyperparameter search
```psh
uv run python -m src.ai run-experiment "src/ai/experiment_configs/default.yaml"
```

And then select a model to use via an mlflow run:
```psh
uv run python -m src.cli game run normal "runs:/03325264f148448ab7fd1850fa5e33f2/model"
```

# Build
To build for release using pyinstaller you must first download the model of interest to eg `src/release/xgboost-model`:

```psh
uv run mlflow artifacts download --run-id 090472026d3d4ee189cbbd4cbde286cf --artifact-path model --dst-path ".\src\release\xgboost-model"
```

and then run:
```psh
uv run pyinstaller --add-data ".\src\release\xgboost-model\;xgboost-model" --add-data ".\.venv\Lib\site-packages\xgboost;xgboost" --add-binary ".\.venv\Lib\site-packages\xgboost\lib\xgboost.dll;bin" --name "snake" -y -p "." src/release/__main__.py
```

This will create an exe file in dist.
