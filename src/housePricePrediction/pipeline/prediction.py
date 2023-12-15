import os
import pandas as pd
import pickle
import numpy as np
import re
from housePricePrediction.logging import logger
import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import TargetEncoder
from sklearn.linear_model import LinearRegression, LassoLars, LarsCV, SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV

from housePricePrediction.config.configuration import ConfigurationManager
from housePricePrediction.utils.data_utils import *

class PredictionPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_model_evaluation_config()

    def clean_raw_data(self, clean_df):
        
        clean_df["Area_sqft"] = clean_df["Area_sqft"].apply(lambda x:get_area(x))
        clean_df["Area_sqft"].fillna(value=clean_df["Area_sqft"].mean(), inplace=True)
        #print(clean_df.info())
        clean_df = clean_df.apply(lambda x:x.str.lower() if x.dtypes == "O" else x)
        clean_df = clean_df.apply(lambda x:x.str.strip() if x.dtypes == "O" else x)

        clean_df["No_of_Bedroom"] = clean_df["Type"].apply(lambda x:get_bedroom_size(x))
        clean_df["No_of_Bedroom"].fillna(value=np.round(clean_df["No_of_Bedroom"].mean()), inplace=True)

        clean_df["Township_Size_Ordinal"] = clean_df["Area_township"].apply(lambda x:get_township_size(x)).map(get_ts_size_map()).fillna(0)
        clean_df["Loc_trend"] = clean_df.Location.map(get_loc_trend_map()).fillna(0)

        clean_df["Loc_tag"] = clean_df.Location.map(get_loc_tag_map()).fillna("unknown")
        clean_df["Loc_tag_ordinal"] = clean_df.Location.map(get_area_idx_dict()).fillna(0)

        # dummy_cat = pd.get_dummies(clean_df[["hasClubHouse","hasEduFacility",	"hasHospital",	"hasMall",	"hasParkOrJogTrack",	"hasPool",	"hasGym"]], drop_first=True, dtype=int)
        # clean_df = pd.concat([clean_df, dummy_cat], axis=1)

        clean_df = clean_df.drop(["Area_township","Type", "Loc_tag"], axis=1)

        return clean_df

    def predict(self, test_df):
        if test_df is None:
            return None
        test_df = self.clean_raw_data(test_df)
        print("test data" , test_df)
        all_feat = ['Location', 'Area_sqft', 'Developer', 'Name', 'No_of_Bedroom', 'Township_Size_Ordinal', 'Loc_trend', 
                    'Loc_tag_ordinal', 'hasClubHouse_yes', 'hasEduFacility_yes', 'hasHospital_yes', 'hasMall_yes', 'hasParkOrJogTrack_yes', 
                    'hasPool_yes', 'hasGym_yes']

        cat_feat= ["Location", "Developer", "Name"]
        num_feat= ["Area_sqft","Loc_trend"]

        encoder_path = os.path.join(self.config.data_path,"feat","tgt_encoder.pkl")
        scalar_path = os.path.join(self.config.data_path,"feat","train_scaler.pkl")
        model_path = os.path.join(self.config.model_path)

        with open(encoder_path, "rb") as enc_file:
            tgt_encoder = pickle.load(enc_file)
        test_df[cat_feat] = tgt_encoder.transform(test_df[cat_feat])

        num_feat = cat_feat + num_feat

        with open(scalar_path, "rb") as enc_file:
            train_scalar = pickle.load(enc_file)

        test_df[num_feat] = train_scalar.transform(test_df[num_feat])

        with open(model_path, "rb") as enc_file:
            lass_reg_model = pickle.load(enc_file)

        logger.info(f"Column order : \n {test_df.columns}")
        y_pred_test = lass_reg_model.predict(test_df[all_feat])
        #print(y_pred_test.shape)
        price_mil = y_pred_test[0]
        print(f"The price for given property is: Rs. {price_mil} Million or Rs. {price_mil*10} Lakhs")

        return price_mil