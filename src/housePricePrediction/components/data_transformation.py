import os
import numpy as np
import pandas as pd
import re
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import TargetEncoder
from sklearn.linear_model import LinearRegression

from housePricePrediction.logging import logger
from housePricePrediction.entity import DataTransformationConfig

from housePricePrediction.utils.data_utils import *


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config


    def clean_raw_data(self, raw_df):
        clean_df = raw_df.copy().drop(["Sr. No.", "Location", "Description"], axis = 1)
        clean_df = clean_df.rename(columns={"Sub-Area": "Location", "Propert Type" : "Type", "Property Area in Sq. Ft." : "Area_sqft", "Price in lakhs" : "Price_Lakhs", "Price in Millions" : "Price_Mil",
                                            "Company Name" : "Developer", "TownShip Name/ Society Name" : "Name", "Total TownShip Area in Acres" : "Area_township", "ClubHouse" : "hasClubHouse",
                                            "School / University in Township ": "hasEduFacility", "Hospital in TownShip": "hasHospital", "Mall in TownShip" : "hasMall", "Park / Jogging track" : "hasParkOrJogTrack",
                                            "Swimming Pool" : "hasPool", "Gym" : "hasGym" })
        clean_df["Price_Mil"] = clean_df["Price_Mil"].apply(lambda x:round(float(x), 4) if str(x).replace(".", "", 1).isdigit() else np.NAN).astype("float64")
        
        # This can be changed and replaced with some real mean value through config.
        clean_df["Price_Mil"] = clean_df["Price_Mil"].replace([np.NAN], 9.5)
        clean_df["Price_Mil"] = clean_df["Price_Mil"].replace([92.300, 93.000], [9.230, 9.300])

        clean_df["Area_sqft"] = clean_df["Area_sqft"].apply(lambda x:get_area(x))
        clean_df["Area_sqft"].fillna(value=clean_df["Area_sqft"].mean(), inplace=True)

        clean_df = clean_df.apply(lambda x:x.str.lower() if x.dtypes == "O" else x)
        clean_df = clean_df.apply(lambda x:x.str.strip() if x.dtypes == "O" else x)

        clean_df["No_of_Bedroom"] = clean_df["Type"].apply(lambda x:get_bedroom_size(x))
        clean_df["No_of_Bedroom"].fillna(value=np.round(clean_df["No_of_Bedroom"].mean()), inplace=True)

        clean_df["Township_Size_Ordinal"] = clean_df["Area_township"].apply(lambda x:get_township_size(x)).map(get_ts_size_map()).fillna(0)
        clean_df["Loc_trend"] = clean_df.Location.map(get_loc_trend_map()).fillna(0)

        clean_df["Loc_tag"] = clean_df.Location.map(get_loc_tag_map()).fillna("unknown")
        clean_df["Loc_tag_ordinal"] = clean_df.Location.map(get_area_idx_dict()).fillna(0)

        dummy_cat = pd.get_dummies(clean_df[["hasClubHouse","hasEduFacility",	"hasHospital",	"hasMall",	"hasParkOrJogTrack",	"hasPool",	"hasGym"]], drop_first=True, dtype=int)
        clean_df = pd.concat([clean_df, dummy_cat], axis=1)

        clean_df = clean_df.drop(["Area_township","Type", "Price_Lakhs", "Loc_tag", "hasClubHouse","hasEduFacility",	"hasHospital",	"hasMall",	"hasParkOrJogTrack",	"hasPool",	"hasGym"], axis=1)

        return clean_df
    
    def create_train_test_data(self, clean_df):
        
        y = clean_df["Price_Mil"]
        X = clean_df.drop(["Price_Mil"], axis=1)

        test_ratio = self.config.test_ratio
        random_seed = self.config.random_seed

        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=test_ratio, random_state=random_seed)
        return X_train, X_test, y_train, y_test

    def get_target_encodings(self, df, feat, fit_enc=False, dataset="train", save_encoder=False, encoder_path=None):
        if (encoder_path is not None) and (os.path.exists(encoder_path)) and (not fit_enc):
            # Define encoder            
            with open(encoder_path, "rb") as enc_file:
                tgt_encoder = pickle.load(enc_file)
        else:
            tgt_encoder = TargetEncoder(target_type="continuous")
        
        if dataset == "test":
            df[feat] = tgt_encoder.transform(df[feat])
        else:            
            # Fit and transform the training data
            if fit_enc:
                df[feat] = tgt_encoder.fit_transform(df[feat], df["Price_Mil"])
            else:
                if encoder_path is not None:
                    df[feat] = tgt_encoder.transform(df[feat])
        
        if save_encoder and encoder_path is not None:
            logger.info("Saved target encoder\n")
            os.makedirs(os.path.dirname(encoder_path), exist_ok=True)
            pickle.dump(tgt_encoder, open(encoder_path, 'wb'))

        return df

    def get_scalar_data(self, df, feat, fit_scalar=False, dataset="train", save_scalar=False, scalar_path=None):
        if (scalar_path is not None) and (os.path.exists(scalar_path)) and (not fit_scalar):
            # Define encoder            
            with open(scalar_path, "rb") as enc_file:
                train_scalar = pickle.load(enc_file)
        else:
            train_scalar = StandardScaler()
        
        if dataset == "test":
            df[feat] = train_scalar.transform(df[feat])
        else:            
            # Fit and transform the training data
            if fit_scalar:
                df[feat] = train_scalar.fit_transform(df[feat])
            else:
                if scalar_path is not None:
                    df[feat] = train_scalar.transform(df[feat])
        
        if save_scalar and scalar_path is not None:
            logger.info("Saved Standard Scalar\n")
            os.makedirs(os.path.dirname(scalar_path), exist_ok=True)
            pickle.dump(train_scalar, open(scalar_path, 'wb'))

        return df



    def convert(self):

        """
        Convert raw data into train and test set
        """
        xlsx_file = os.path.join(self.config.data_path,"Pune_Real_Estate_Data.xlsx")
        raw_data = pd.read_excel(xlsx_file)

        clean_df = self.clean_raw_data(raw_data)      

        X_train, X_test, y_train, y_test = self.create_train_test_data(clean_df)

        full_train_df = pd.concat([X_train, y_train], axis=1)
        full_test_df = pd.concat([X_test, y_test], axis=1)

        cat_feat = self.config.cat_feat       
        encoder_path = os.path.join(self.config.root_dir,"feat","tgt_encoder.pkl")
        # Fit and transform the training data
        full_train_df = self.get_target_encodings(full_train_df, cat_feat, fit_enc=True, dataset="train", save_encoder=True, encoder_path=encoder_path)
        # Transform on test data, for unseen data a mean of group can be assigned
        full_test_df = self.get_target_encodings(full_test_df, cat_feat, fit_enc=False, dataset="test", save_encoder=False, encoder_path=encoder_path)

        num_feat = cat_feat + self.config.num_feat
        scalar_path = os.path.join(self.config.root_dir,"feat","train_scaler.pkl")
        full_train_df = self.get_scalar_data(full_train_df, num_feat, fit_scalar=True, dataset="train", save_scalar=True, scalar_path=scalar_path)
        full_test_df = self.get_scalar_data(full_test_df, num_feat, fit_scalar=False, dataset="test", save_scalar=False, scalar_path=scalar_path)

        # Save test and train files
        
        train_path = os.path.join(self.config.root_dir,"data","train-v1.csv")
        test_path = os.path.join(self.config.root_dir,"data","test-v1.csv")
        os.makedirs(os.path.dirname(train_path), exist_ok=True)
        os.makedirs(os.path.dirname(test_path), exist_ok=True)

        full_train_df.to_csv(train_path, index=False)
        full_test_df.to_csv(test_path, index=False)
