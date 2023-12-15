from pathlib import Path
from housePricePrediction.constants import *
from housePricePrediction.utils.common import read_yaml, create_directories
from housePricePrediction.entity import DataIngestionConfig, DataValidationConfig, DataTransformationConfig, ModelTrainerConfig

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config

    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,            
            STATUS_FILE=config.STATUS_FILE,
            ALL_REQUIRED_FILES=config.ALL_REQUIRED_FILES,
            ALL_REQUIRED_FEAT=config.ALL_REQUIRED_FEAT 
        )

        return data_validation_config

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            random_seed=config.random_seed,
            test_ratio=config.test_ratio,
            cat_feat=config.cat_feat,
            num_feat=config.num_feat
        )

        return data_transformation_config

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.params.TrainingArguments

        create_directories([config.root_dir])

        model_trainer_config = ModelTrainerConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            model_ckpt = config.model_ckpt,
            lasso_cv= params.lasso_cv,
            lasso_max_iter= params.lasso_max_iter,
            lasso_eps= params.lasso_eps,
            sgd_max_iter= params.sgd_max_iter,
            sgd_tol= params.sgd_tol     
        )

        return model_trainer_config