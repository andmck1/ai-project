# Build
```python
uv run pyinstaller --add-data ".\src\release\xgboost-model\;xgboost-model" --add-data ".\.venv\Lib\site-packages\xgboost;xgboost" --add-binary ".\.venv\Lib\site-packages\xgboost\lib\xgboost.dll;bin" --name "snake" -y -p "." src/release/__main__.py
```
