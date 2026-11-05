"""Custom exception classes for Subsystem 2 (AI Modeling, Forecasting, and Evaluation)."""

class Subsystem2Error(Exception):
    """Base exception class for all Subsystem 2 errors."""
    pass

class ModelTrainingError(Subsystem2Error):
    """Raised when machine learning model training, fitting, or prediction fails."""
    pass

class ForecastError(Subsystem2Error):
    """Raised when demand forecasting calculations or time-series projections fail."""
    pass

class ModelEvaluationError(Subsystem2Error):
    """Raised when evaluating model performance metrics (e.g., RMSE, MAE) fails."""
    pass