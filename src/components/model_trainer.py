import os
import sys
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

# External models
from xgboost import XGBRegressor
from catboost import CatBoostRegressor


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def evaluate_models(self, X_train, y_train, X_test, y_test, models):
        report = {}

        for model_name, model in models.items():
            logging.info(f"Training {model_name}")

            model.fit(X_train, y_train)
            y_test_pred = model.predict(X_test)
            score = r2_score(y_test, y_test_pred)

            report[model_name] = score
            logging.info(f"{model_name} R2 score: {score}")

        return report

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting train and test arrays")

            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            models = {
                "LinearRegression": LinearRegression(),
                "RandomForest": RandomForestRegressor(
                    n_estimators=100, random_state=42
                ),
                "GradientBoosting": GradientBoostingRegressor(
                    random_state=42
                ),
                "XGBoost": XGBRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    verbosity=0
                ),
                "CatBoost": CatBoostRegressor(
                    iterations=500,
                    learning_rate=0.1,
                    depth=6,
                    verbose=False,
                    random_state=42
                )
            }

            model_report = self.evaluate_models(
                X_train, y_train, X_test, y_test, models
            )

            best_model_score = max(model_report.values())
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            logging.info(
                f"Best model: {best_model_name} with R2 score: {best_model_score}"
            )

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            logging.info("Best model saved successfully")

            return best_model_name, best_model_score

        except Exception as e:
            raise CustomException(e, sys)
