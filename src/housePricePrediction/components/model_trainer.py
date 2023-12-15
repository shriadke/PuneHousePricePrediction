import os
import pandas as pd
import pickle
import numpy as np
from housePricePrediction.logging import logger
import tqdm

from sklearn.linear_model import LinearRegression, LassoLars, LarsCV, SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV

from housePricePrediction.entity import ModelTrainerConfig

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def print_results(self, y_pred, y_act):
        mse = np.mean((y_pred - y_act)**2)
        logger.info(f"[INFO] MSE: {mse}")

        rmse = np.sqrt(mse)
        logger.info(f"[INFO] RMSE: {rmse}")
    
    def train(self):
        train_path = os.path.join(self.config.data_path,"data/train-v1.csv")
        train_df = pd.read_csv(train_path)
        test_path = os.path.join(self.config.data_path,"data/test-v1.csv")
        test_df = pd.read_csv(test_path)

        features = list(train_df.columns)
        label = features.pop(-1)
        logger.info(features)
        logger.info(label)

        X_train =train_df[features]
        X_test =test_df[features]

        y_train = train_df[label]
        y_test = test_df[label]

        logger.info(f"Column orrder : \n {features}")
        logger.info(f"Label column is : \n {label}")
        logger.info(f"Data Shape: \n")
        logger.info(f"Train features:  {X_train.shape},  Label: {y_train.shape}")
        logger.info(f"Test features: {X_test.shape},  Label:  {y_test.shape}")

        lass_reg_model = LarsCV(cv=self.config.lasso_cv, max_iter=self.config.lasso_max_iter, eps =self.config.lasso_eps)
        lass_reg_model.fit(X_train, y_train)

        y_pred_train = lass_reg_model.predict(X_train)
        y_pred_test = lass_reg_model.predict(X_test)
        logger.info("Training Performance:")
        self.print_results(y_pred_train, y_train)
        logger.info(f"Training R2: {lass_reg_model.score(X_train, y_train)}")
        logger.info("*****************************************")
        logger.info("Testing Performance:")
        self.print_results(y_pred_test, y_test)
        logger.info(f"Testing R2: {lass_reg_model.score(X_test, y_test)}")  

        model_path = os.path.join(self.config.root_dir,self.config.model_ckpt)
        os.makedirs(os.path.dirname(model_path), exist_ok=True) 

        pickle.dump(lass_reg_model, open(model_path, "wb"))


