from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig():
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path

@dataclass(frozen=True)
class DataValidationConfig():
    root_dir: Path
    STATUS_FILE: Path
    ALL_REQUIRED_FILES: list
    ALL_REQUIRED_FEAT: list

@dataclass(frozen=True)
class DataTransformationConfig():
    root_dir: Path
    data_path: Path
    random_seed: int
    test_ratio: float
    cat_feat: list
    num_feat: list

@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    data_path: Path
    model_ckpt: Path
    lasso_cv: int
    lasso_max_iter: int 
    lasso_eps: float
    sgd_max_iter: int
    sgd_tol: float