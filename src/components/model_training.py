import sys
import os
import dill
import mlflow
import mlflow.sklearn
from dataclasses import dataclass
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from src.exception import CustomException
from src.logger import logging


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, X_train, X_test, y_train, y_test):
        try:
            logging.info("Model Training started")

            # MLflow setup
            mlflow.set_tracking_uri('file:///D:/NE-EpiGuard/mlflow')
            mlflow.set_experiment('NE-EpiGuard-Disease-Prediction')

            with mlflow.start_run(run_name='Stacking_Ensemble_Production'):

                # Tuned base estimators
                estimators = [
                    ('xgb', XGBClassifier(
                        n_estimators=600,
                        max_depth=3,
                        learning_rate=0.1,
                        subsample=0.8,
                        colsample_bytree=1.0,
                        min_child_weight=5,
                        random_state=42,
                        eval_metric='mlogloss',
                        verbosity=0
                    )),
                    ('lgbm', LGBMClassifier(
                        n_estimators=400,
                        max_depth=3,
                        learning_rate=0.1,
                        num_leaves=31,
                        subsample=1.0,
                        colsample_bytree=0.8,
                        min_child_samples=30,
                        class_weight='balanced',
                        random_state=42,
                        verbosity=-1
                    )),
                    ('cat', CatBoostClassifier(
                        iterations=600,
                        depth=7,
                        learning_rate=0.1,
                        l2_leaf_reg=5,
                        bagging_temperature=1.0,
                        auto_class_weights='Balanced',
                        random_seed=42,
                        verbose=0
                    ))
                ]

                # Meta learner
                meta_learner = LogisticRegression(
                    class_weight='balanced',
                    max_iter=1000,
                    random_state=42
                )

                # Stacking ensemble
                stacking = StackingClassifier(
                    estimators=estimators,
                    final_estimator=meta_learner,
                    cv=3,
                    n_jobs=1
                )

                logging.info("Training stacking ensemble...")
                sample_weights = compute_sample_weight('balanced', y_train)
                stacking.fit(X_train, y_train)

                # Evaluate
                y_pred = stacking.predict(X_test)
                macro_f1  = f1_score(y_test, y_pred, average='macro')
                accuracy  = accuracy_score(y_test, y_pred)

                logging.info(f"Macro F1: {macro_f1}")
                logging.info(f"Accuracy: {accuracy}")

                # MLflow log
                mlflow.log_param('base_models', 'XGBoost+LightGBM+CatBoost')
                mlflow.log_param('meta_learner', 'LogisticRegression')
                mlflow.log_param('cv_folds', 3)
                mlflow.log_metric('macro_f1', round(macro_f1, 4))
                mlflow.log_metric('accuracy', round(accuracy, 4))

                # Save model
                os.makedirs(
                    os.path.dirname(self.model_trainer_config.trained_model_file_path),
                    exist_ok=True
                )
                with open(self.model_trainer_config.trained_model_file_path, 'wb') as f:
                    dill.dump(stacking, f)

                logging.info("Model saved to artifacts/model.pkl")
                logging.info("Model Training completed")

                return macro_f1, accuracy

        except Exception as e:
            raise CustomException(e, sys)