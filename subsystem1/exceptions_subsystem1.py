class Subsystem1Error(Exception):
    """Base exception class for all Subsystem 1 errors."""
    pass

class DataIngestionError(Subsystem1Error):
    """Raised when loading or ingesting raw data files (e.g., CSV, JSON) fails."""
    pass

class DataProcessingError(Subsystem1Error):
    """Raised when data cleaning, filtering, or transformation fails."""
    pass