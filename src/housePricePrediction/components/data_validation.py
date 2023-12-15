import os
from housePricePrediction.logging import logger
from housePricePrediction.entity import DataValidationConfig
import pandas as pd

class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config
    
    def validate_all_files_exist(self)-> bool:
        try:
            validation_status = None

            all_files = os.listdir(os.path.join("artifacts","data_ingestion"))

            for file in all_files:
                if file not in self.config.ALL_REQUIRED_FILES:
                    validation_status = False
                    with open(self.config.STATUS_FILE, 'w') as f:
                        f.write(f"Validation status: {validation_status}")
                else:
                    validation_status = True
                    with open(self.config.STATUS_FILE, 'w') as f:
                        f.write(f"Validation status: {validation_status}")

            return validation_status
        
        except Exception as e:
            raise e

    def validate_features_exists(self)-> bool:
        try:
            validation_status = None

            xlsx_file = os.path.join("artifacts","data_ingestion","Pune_Real_Estate_Data.xlsx")

            raw_data = pd.read_excel(xlsx_file)

            for col in self.config.ALL_REQUIRED_FEAT:
                if col not in raw_data.columns:
                    with open(self.config.STATUS_FILE, 'w') as f:
                        f.write(f"Validation status: {False} for feature {col}.")
                    return False                    
                else:
                    validation_status = True
                    with open(self.config.STATUS_FILE, 'w') as f:
                        f.write(f"Validation status: {validation_status}")

            return validation_status

        except Exception as e:
            raise e