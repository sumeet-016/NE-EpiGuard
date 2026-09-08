import os
import sys
import dill
from src.exception import CustomException
from src.logger import logging
import numpy as np


# ─── Ordinal Mappings ────────────────────────────────────────────
handwashing_map = {
    'Never': 0, 'Sometimes': 1,
    'Always': 2
}

water_treatment_map = {
    'Untreated': 0, 'Boiled': 1,
    'Filtered': 2, 'Chlorinated': 3
}

water_source_map= {
    'River': 0, 'Pond': 1, 'Open Well': 2,
    'Rainwater': 3, 'Tanker': 4, 'Borewell': 5,
    'Piped': 6
}

season_map = {
    'Winter': 0, 'Summer': 1,
    'Monsoon': 2, 'Post-Monsoon': 3
}

state_map = {
    'Arunachal Pradesh': 0, 'Assam': 1, 'Manipur': 2,
    'Meghalaya': 3, 'Mizoram': 4, 'Nagaland': 5,
    'Sikkim': 6, 'Tripura': 7
}


# ─── Disease Doctor Map ──────────────────────────────────────────
disease_doctor_map = {
    'Cholera':       {'specialist': 'Gastroenterologist / Infectious Disease', 'urgency': 'HIGH — seek care within 24 hours'},
    'Typhoid':       {'specialist': 'General Physician / Infectious Disease',  'urgency': 'MEDIUM — within 48 hours'},
    'Hepatitis_A':   {'specialist': 'Hepatologist / Gastroenterologist',       'urgency': 'MEDIUM'},
    'Hepatitis_E':   {'specialist': 'Hepatologist',                            'urgency': 'MEDIUM — especially if pregnant'},
    'Leptospirosis': {'specialist': 'Infectious Disease Specialist',           'urgency': 'HIGH — seek care within 24 hours'},
    'Dysentery':     {'specialist': 'Gastroenterologist',                      'urgency': 'MEDIUM'},
    'Giardiasis':    {'specialist': 'General Physician',                       'urgency': 'LOW-MEDIUM'},
    'No_Disease':    {'specialist': None,                                      'urgency': None}
}

doctor_threshold = 0.65


# ─── Save / Load ─────────────────────────────────────────────────
def save_object(file_path, obj):
    
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        logging.info(f"Saving object to: {file_path}")
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
   
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file path {file_path} does not exist.")
            
        logging.info(f"Loading object from: {file_path}")
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)


# ─── Encoding Functions ──────────────────────────────────────────
def apply_cyclic_encoding(data):
    try:
        data = data.copy()
        data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
        data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)

        data = data.drop(columns = ['month'])
        logging.info('Cyclic encoding completed')
        return data

    except Exception as e:
        raise CustomException(e, sys)


def apply_ordinal_encoding(data):
    try:

        data = data.copy()

        data['handwashing_practice']    = data['handwashing_practice'].map(handwashing_map)
        data['water_treatment']         = data['water_treatment'].map(water_treatment_map)
        data['water_source']            = data['water_source'].map(water_source_map)
        data['season']                  = data['season'].map(season_map)
        data['state']                   = data['state'].map(state_map)

        logging.info('Ordinal Encoding Completed')
        return data

    except Exception as e:
        raise CustomException(e, sys)


def apply_log_transformer(data):
    try:
        data = data.copy()

        log_cols = [
            'fecal_coliform_per_100ml',
            'total_coliform_100ml'
        ]

        for col in log_cols:
            data[col] = np.log1p(data)

        logging.info('Log Transformation Completed')
        return data

    except Exception as e:
        raise CustomException(e, sys)