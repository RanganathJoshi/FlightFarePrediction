import os
import sys
import pandas as pd
import numpy as np

from dataclasses import dataclass
from src.FlightPricePrediction.exception import customexception
from src.FlightPricePrediction.logger import logging

from sklearn.compose import ColumnTransformer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder,StandardScaler,OneHotEncoder
from src.FlightPricePrediction.utils.utils import save_object
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_transformation(self):
        try:

            logging.info("Data Transformation started")
            cat_col=['Total_Stops']
            one_hot_cols=['Airline', 'Source', 'Destination']
            num_col=['Journey_day', 'Journey_month', 'Dep_hour', 'Dep_minute', 'Arrival_hour', 'Arrival_minute', 'dur_hours', 'dur_minutes']

            stops_cat=['non-stop', '1 stop', '2 stops', '3 stops', '4 stops']
            logging.info("Initiating Pipeline")

            num_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scalar',StandardScaler())
                ]
            )

            cat_pipeline=Pipeline(
                steps=[
                    ('ordinal_encoding',OrdinalEncoder(categories=[stops_cat])),
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('scalar',StandardScaler())
                ]
            )

            processor=ColumnTransformer([
                ('onehot',OneHotEncoder(sparse=False,drop='first'),one_hot_cols),
                ('num_pipeline',num_pipeline,num_col),
                ('cat_pipeline',cat_pipeline,cat_col)
            ],remainder='passthrough')



            return processor
        
        except Exception as e:
            logging.info("Exception occures")
            raise customexception(e,sys)
        

    
    def initialize_data_transformation(self,train_path,valid_path):
        try:
            train_df=pd.read_csv(train_path)
            valid_df=pd.read_csv(valid_path)

            logging.info("read train and test data complete")
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head : \n{valid_df.head().to_string()}')
            
            processing_obj=self.get_transformation()

            target_column='Price'
            train_df.drop(columns='Unnamed: 0',inplace=True)
            valid_df.drop(columns='Unnamed: 0',inplace=True)

            input_feature_train_df = train_df.drop(columns=target_column,axis=1)
            target_feature_train_df=train_df[target_column]

            input_feature_valid_df = valid_df.drop(columns=target_column,axis=1)
            target_feature_valid_df=valid_df[target_column]

            input_feature_train_arr=processing_obj.fit_transform(input_feature_train_df)

            input_feature_valid_arr=processing_obj.transform(input_feature_valid_df)

            logging.info("Applying preprocessing object on training and testing datasets.")

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            valid_arr = np.c_[input_feature_valid_arr, np.array(target_feature_valid_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=processing_obj
            )

            logging.info("Transformation completed")
            return (train_arr,valid_arr)
            
        except Exception as e:
            logging.info("erroe occured while applying data transformation")

            raise customexception(e,sys)