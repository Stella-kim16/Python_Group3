from typing import Dict                                
import pandas as pd                                    
from sklearn.linear_model import LinearRegression      
from sklearn.ensemble import RandomForestRegressor     


class ModelTrainer:
    def __init__(self):
        self.models = {                                
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        }
        self.is_trained = False

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, object]:
        for name, model in self.models.items():       
            model.fit(X_train, y_train)               
            print(f"  Trained: {name}")
        self.is_trained = True
        return self.models

    def get_models(self) -> Dict[str, object]:
        if not self.is_trained:
            raise RuntimeError("Call train() first.")
        return self.models


