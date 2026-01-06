@echo off
echo ========================================
echo Movie Box Office Pipeline - Clean & Rerun
echo ========================================
echo.

echo [1/3] Cleaning old outputs...
if exist "models\box_office_model.pkl" del /f "models\box_office_model.pkl" && echo   - Deleted models\box_office_model.pkl
if exist "dataset\data_cleaned.csv" del /f "dataset\data_cleaned.csv" && echo   - Deleted dataset\data_cleaned.csv
if exist "demo\model_comparison.csv" del /f "demo\model_comparison.csv" && echo   - Deleted demo\model_comparison.csv
echo.

echo [2/3] Activating conda environment...
call conda activate movie
echo.

echo [3/3] Running full pipeline...
echo This will take 20-35 minutes for 19k movies...
echo.
python main.py

echo.
echo ========================================
echo Pipeline Complete!
echo ========================================
echo.
echo To launch demo: python src\gradio_app.py
pause

