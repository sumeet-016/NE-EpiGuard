import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging


@dataclass
class DataIngestionConfig:
    train_data_path:    str = os.path.join('artifacts', 'train.csv')
    test_data_path:     str = os.path.join('artifacts', 'test.csv')
    raw_data_path:      str = os.path.join('artifacts', 'data.csv')
    dataset_path:       str = os.path.join('Notebooks', 'Data', 'northeast_waterborne.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion started")

        try:
            data = pd.read_csv(self.ingestion_config.dataset_path)
            logging.info(f"Raw dataset loaded - shape: {data.shape}" )


            # Column Drop
            columns_drop = [
                'district', 'latitude', 'longitude',
                'population_density', 'gender'  
            ]
            data = data.drop(columns=columns_drop)
            logging.info(f"Columns Dropped: {columns_drop}")
            logging.info(f"Shape after Drop: {data.shape}")


            # Save raw data
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            data.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info("Raw data saved to artifacts")


            # Train Test split
            train_set, test_set = train_test_split(
                data,
                test_size=0.2,
                random_state=42,
                stratify=data['disease']
            )
            logging.info(f"Train size: {train_set.shape}, Test size: {test_set.shape}")


            # Save train and test data
            train_set.to_csv(self.ingestion_config.train_data_path, index=False)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False)
            logging.info("Train and Test data are saved in artifacts")


            logging.info("Data Ingestion Completed sucessfully")


            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )


        except Exception as e:
            raise CustomException(e, sys)