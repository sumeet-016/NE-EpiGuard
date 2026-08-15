import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngine(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        try:
            logging.info("Feature Engineering started")

            data = X.copy()

            # Symptom Count
            symptom_columns = [
                'symptom_diarrhea', 'symptom_vomiting', 'symptom_fever',
                'symptom_abdominal_pain', 'symptom_dehydration', 'symptom_jaundice',
                'symptom_bloody_stool', 'symptom_skin_rash'
            ]
            data['symptom_count'] = data[symptom_columns].sum(axis=1)

            # Source Treatment Risk
            data['source_treatment_risk'] = (
                6 - data['water_source']
            ) + (3 - data['water_treatment'])

            # Age Hygiene Risk
            data['age_hygiene_risk'] = data['age'] * (2 - data['handwashing_practice'])

            # Contamination Index
            data['contamination_index'] = (
                data['fecal_coliform_per_100ml'] +
                data['total_coliform_per_100ml'] +
                data['turbidity_ntu'] / 10
            )

            logging.info(f"Feature Engineering completed — shape: {data.shape}")
            return data

        except Exception as e:
            raise CustomException(e, sys)