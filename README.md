# Smart Energy Data Viewer (Error handling + Unit testing)

## 1. Author
- Team: Python Group 6
- Name: Ecesu Yildiran [ESE 7226935], Stella (Sung Eun Kim) [ESE 7226046],Devashish Deshpande [ESE 7225975] 

## 2. Project Description
The aim of this work is to learn how to handle error in your application and testing princples

## 3. Technologies
- Python 3, unittest
- pandas, numpy, scikit-learn
- Git, GitHub (`feature/hometask4`)

## 4. How to Run
1. Install dependencies and testing requirements:
   pip install -r requirements.txt
2. Run the automated unit test suite across all subsystems:
   In python Terminal, execute: -m unittest discover tests
3. Verify custom exception behaviors and subsystem safety guardrails.

## 5. Data
- Subsystem 1: Custom exceptions (`DataIngestionError`, `DataProcessingError`) integrated into ingestion, database persistence, feature engineering, and models.
- Subsystem 2: Custom exceptions (`ModelTrainingError`, `ForecastError`, `ModelEvaluationError`) integrated into AI modeling, forecasting, data processing, and evaluation.
- Subsystem 3: Scenario generation, optimisation, and visualisation error handling.
- Test cases can be found in the individual files inside the tests folder. These are separated by subsystem


## 6. Video link
(https://www.loom.com/share/9c4df1187e6541a1b28f3fc676ecf162)
