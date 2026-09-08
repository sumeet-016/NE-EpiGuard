import sys
import os
import pandas as pd
import numpy as np
import dill
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from src.exception import CustomException
from src.logger import logging
from src.components.feature_engineering import FeatureEngine
from src.utils import (
    save_object,
    apply_ordinal_encoding,
    apply_cyclic_encoding,
    apply_log_transformer
)


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')
    target_encoder_file_path:   str = os.path.join('artifacts', 'le_target.pkl')


class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numeric_cols = [
                'age', 'water_quality_index', 'ph', 'turbidity_ntu',
                'dissolved_oxygen_mg_l', 'bod_mg_l', 'fecal_coliform_per_100ml',
                'total_coliform_per_100ml', 'tds_mg_l', 'nitrate_mg_l',
                'fluoride_mg_l', 'arsenic_ug_l', 'open_defecation_rate',
                'sewage_treatment_pct', 'avg_temperature_c', 'avg_rainfall_mm',
                'avg_humidity_pct', 'month_sin', 'month_cos',
                'symptom_count', 'source_treatment_risk',
                'age_hygiene_risk', 'contamination_index'
            ]

            preprocessor = ColumnTransformer([
                ('scaler', StandardScaler(), numeric_cols)
            ], remainder='passthrough')

            logging.info("Preprocessor object created")
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df  = pd.read_csv(test_path)
            logging.info(f"Train: {train_df.shape}, Test: {test_df.shape}")

            # Step 1 — Cyclic encoding
            train_df = apply_cyclic_encoding(train_df)
            test_df  = apply_cyclic_encoding(test_df)

            # Step 2 — Ordinal encoding
            train_df = apply_ordinal_encoding(train_df)
            test_df  = apply_ordinal_encoding(test_df)

            # Step 3 — Log transform
            train_df = apply_log_transformer(train_df)
            test_df  = apply_log_transformer(test_df)

            # Step 4 — Feature engineering
            fe = FeatureEngine()
            train_df = fe.fit_transform(train_df)
            test_df  = fe.transform(test_df)

            # Step 5 — Target encoding
            target_col = 'disease'
            le = LabelEncoder()
            y_train = le.fit_transform(train_df[target_col])
            y_test  = le.transform(test_df[target_col])

            save_object(self.transformation_config.target_encoder_file_path, le)
            logging.info("Target encoder saved")

            # Step 6 — Drop target
            X_train = train_df.drop(columns=[target_col])
            X_test  = test_df.drop(columns=[target_col])

            # Step 7 — Scale
            preprocessor = self.get_data_transformer_object()
            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed  = preprocessor.transform(X_test)

            save_object(self.transformation_config.preprocessor_obj_file_path, preprocessor)
            logging.info("Preprocessor saved")

            return (
                X_train_transformed,
                X_test_transformed,
                y_train,
                y_test,
                self.transformation_config.preprocessor_obj_file_path,
                self.transformation_config.target_encoder_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)