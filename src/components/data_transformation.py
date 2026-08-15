import sys
import os
import pandas as pd
import numpy as np
import dill
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from src.exception import CustomException
from src.logger import logging
from src.components.feature_engineering import FeatureEngine


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')
    target_encoder_file_path:   str = os.path.join('artifacts', 'le_target.pkl')


class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            # Ordinal columns with risk order
            handwashing_cats   = ['Never', 'Sometimes', 'Always']
            water_treatment_cats = ['Untreated', 'Boiled', 'Filtered', 'Chlorinated']
            water_source_cats  = ['River', 'Pond', 'Open Well', 'Rainwater', 'Tanker', 'Borewell', 'Piped']
            season_cats        = ['Winter', 'Summer', 'Monsoon', 'Post-Monsoon']

            ordinal_cols = [
                'handwashing_practice',
                'water_treatment',
                'water_source',
                'season'
            ]

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

            ordinal_pipeline = Pipeline([
                ('ordinal', OrdinalEncoder(categories=[
                    handwashing_cats,
                    water_treatment_cats,
                    water_source_cats,
                    season_cats
                ]))
            ])

            numeric_pipeline = Pipeline([
                ('scaler', StandardScaler())
            ])

            preprocessor = ColumnTransformer([
                ('ordinal', ordinal_pipeline, ordinal_cols),
                ('numeric', numeric_pipeline, numeric_cols)
            ], remainder='passthrough')

            logging.info("Preprocessor object created")
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)



    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df  = pd.read_csv(test_path)
            logging.info("Train and test data loaded")


            # Cyclical encoding for month
            for df in [train_df, test_df]:
                df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
                df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
                df.drop(columns=['month'], inplace=True)


            # Log transform skewed columns
            log_cols = ['fecal_coliform_per_100ml', 'total_coliform_per_100ml']
            for df in [train_df, test_df]:
                for col in log_cols:
                    df[col] = np.log1p(df[col])


            # Feature engineering
            fe = FeatureEngine()
            train_df = fe.fit_transform(train_df)
            test_df  = fe.transform(test_df)


            # Target encoding
            target_col = 'disease'
            le = LabelEncoder()
            y_train = le.fit_transform(train_df[target_col])
            y_test  = le.transform(test_df[target_col])



            # Save target encoder
            os.makedirs(os.path.dirname(
                self.transformation_config.target_encoder_file_path),
                exist_ok=True
            )

            with open(self.transformation_config.target_encoder_file_path, 'wb') as f:
                dill.dump(le, f)

            logging.info("Target encoder saved")



            # Drop target from features
            X_train = train_df.drop(columns=[target_col])
            X_test  = test_df.drop(columns=[target_col])



            # Fit preprocessor
            preprocessor = self.get_data_transformer_object()
            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed  = preprocessor.transform(X_test)
            logging.info("Preprocessing done")



            # Save preprocessor
            with open(self.transformation_config.preprocessor_obj_file_path, 'wb') as f:
                dill.dump(preprocessor, f)


            logging.info("Preprocessor saved to artifacts/preprocessor.pkl")

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