@echo off


echo off

CALL C:\Users\YourUsername\anaconda3\Scripts\activate.bat
CALL conda activate hf_predictor

SET DB_HOST=localhost
SET DB_PORT=5432
SET DB_NAME=patients
SET DB_USER=postgres
