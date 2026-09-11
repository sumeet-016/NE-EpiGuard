import sys
import os
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import (
    load_object,
    apply_cyclic_encoding,
    apply_ordinal_encoding,
    apply_log_transformer,
    doctor_threshold,
    disease_doctor_map
)
from src.components.feature_engineering import FeatureEngine


class PredictPipeline:
    def __init__(self):
        model_path        = os.path.join("artifacts", "model.pkl")
        preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
        target_enc_path   = os.path.join("artifacts", "le_target.pkl")

        logging.info("Loading model, preprocessor and target encoder")
        self.model        = load_object(file_path=model_path)
        self.preprocessor = load_object(file_path=preprocessor_path)
        self.le_target    = load_object(file_path=target_enc_path)
        logging.info("All artifacts loaded successfully")

    def predict(self, features: pd.DataFrame):
        try:
            data = features.copy()

            # Step 1 — Cyclic encoding
            data = apply_cyclic_encoding(data)

            # Step 2 — Ordinal encoding
            data = apply_ordinal_encoding(data)

            # Step 3 — Log transform
            data = apply_log_transformer(data)

            # Step 4 — Feature engineering
            fe   = FeatureEngine()
            data = fe.transform(data)

            # Step 5 — Scale
            data_transformed = self.preprocessor.transform(data)

            # Step 6 — Predict
            prediction     = self.model.predict(data_transformed)
            probabilities  = self.model.predict_proba(data_transformed)
            max_prob       = float(np.max(probabilities))
            disease_name   = self.le_target.inverse_transform(prediction)[0]

            logging.info(f"Prediction: {disease_name}, Probability: {max_prob}")

            # Step 7 — Doctor recommendation
            doctor_info = None
            if max_prob >= doctor_threshold and disease_name != 'No_Disease':
                doctor_info = disease_doctor_map.get(disease_name)

            return {
                'disease':     disease_name,
                'probability': round(max_prob * 100, 2),
                'probabilities': {
                    self.le_target.inverse_transform([i])[0]: round(float(p) * 100, 2)
                    for i, p in enumerate(probabilities[0])
                },
                'doctor_info': doctor_info
            }

        except Exception as e:
            raise CustomException(e, sys)