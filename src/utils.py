import os
import sys
import dill
from src.exception import CustomException
from src.logger import logging

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