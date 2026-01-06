@echo off
echo ========================================
echo Cleaning Old Pipeline Outputs
echo ========================================
echo.

if exist "models\box_office_model.pkl" (
    del /f "models\box_office_model.pkl"
    echo ✓ Deleted models\box_office_model.pkl
) else (
    echo - models\box_office_model.pkl not found
)

if exist "dataset\data_cleaned.csv" (
    del /f "dataset\data_cleaned.csv"
    echo ✓ Deleted dataset\data_cleaned.csv
) else (
    echo - dataset\data_cleaned.csv not found
)

if exist "demo\model_comparison.csv" (
    del /f "demo\model_comparison.csv"
    echo ✓ Deleted demo\model_comparison.csv
) else (
    echo - demo\model_comparison.csv not found
)

echo.
echo ========================================
echo Cleanup Complete!
echo ========================================
echo.
echo To run pipeline: python main.py
echo Or use: clean_and_rerun.bat
pause

