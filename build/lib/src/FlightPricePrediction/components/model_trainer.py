import pandas as pd
import numpy as np
import os
import sys
from src.FlightPricePrediction.logger import logging
from src.FlightPricePrediction.exception import customexception
from dataclasses import dataclass
from src.FlightPricePrediction.utils.utils import save_object
from src.FlightPricePrediction.utils.utils import evaluate_model

from sklearn.linear_model import LinearRegression, Ridge,Lasso,ElasticNet
from sklearn.ensemble import RandomForestRegressor

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("E:/iNeuron/End-end-project_2/artifacts/","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_training(self,train_array,test_array):
        try:
            logging.info('Splitting Dependent and Independent variables')
            x_train,y_train,x_test,y_test=(train_array[:,:-1],train_array[:,-1],test_array[:,:-1],test_array[:,-1])
            models={
                'LinearRegression':LinearRegression(),
                'Lasso':Lasso(),
                'Ridge':Ridge(),
                'ElasticNet':ElasticNet(),
                'RandomForest':RandomForestRegressor()
            }

            model_report:dict=evaluate_model(x_train,y_train,x_test,y_test,models)
            print(model_report)
            print("\n====================================================================")
            logging.info(f"Model Report : {model_report}")

            best_model_score=max(model_report.values())
            best_model_name=list(model_report.keys())[np.argmax(list(model_report.values()))]

            print(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            best_model=models[best_model_name]
            save_object(
                 file_path=self.model_trainer_config.trained_model_file_path,
                 obj=best_model
            )

        except Exception as e:
            logging.info("Exception occured at Model Training")
            raise customexception(e,sys)