import sys
from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_training import ModelTrainer


def main():
    try:
        # Step 1 — Data Ingestion
        logging.info("Training pipeline started")
        ingestion = DataIngestion()
        train_path, test_path = ingestion.initiate_data_ingestion()

        # Step 2 — Data Transformation
        transformation = DataTransformation()
        X_train, X_test, y_train, y_test, _, _ = transformation.initiate_data_transformation(
            train_path, test_path
        )

        # Step 3 — Model Training
        trainer = ModelTrainer()
        macro_f1, accuracy = trainer.initiate_model_trainer(
            X_train, X_test, y_train, y_test
        )

        print("Training complete")
        print("Macro F1 :", round(macro_f1, 4))
        print("Accuracy :", round(accuracy, 4))

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    main()