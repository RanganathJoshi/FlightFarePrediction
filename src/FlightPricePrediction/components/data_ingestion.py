import pandas as pd
import numpy as np
from src.FlightPricePrediction.logger import logging
from src.FlightPricePrediction.exception import customexception

import os
import sys
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from pathlib import Path

class DataIngestionConfig:
    data_path:str=os.path.join("artifacts",'raw.csv')
    train_path:str=os.path.join("artifacts",'train_data.csv')
    validation_data:str=os.path.join("artifacts",'valid_data.csv')
    #test_data:str=os.path.join("E:/iNeuron/End-end-project_2/artifacts/",'test_data.csv')

class Ingestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion Started")
        try:
            data=pd.read_csv(Path(os.path.join("notebooks/data","train.csv")))
            #test=pd.read_csv('e:\\iNeuron\\End-end-project_2\\notebooks\\data\\test.csv')
            logging.info("Data has been fed for splitting")

            os.makedirs(os.path.dirname(os.path.join(self.ingestion_config.data_path)),exist_ok=True)
            data.to_csv(self.ingestion_config.data_path,index=False)
            logging.info("Uploaded the dataset in artifacts folder")
            #os.makedirs(os.path.dirname(os.path.join(self.ingestion_config.test_data)),exist_ok=True)
            #test.to_csv(self.ingestion_config.test_data,index=False)
            logging.info("Splitting the dataset")

            train_data,valid_data=train_test_split(data,test_size=0.25,random_state=145)
            logging.info("train valid split done")

            train_data.to_csv(self.ingestion_config.train_path,index=False)
            valid_data.to_csv(self.ingestion_config.validation_data,index=False)

            logging.info("Data Ingestion completed")

            return (self.ingestion_config.train_path,self.ingestion_config.validation_data)
            

        except Exception as e:
            logging.info("Exception during occured at data ingestion stage")
            raise customexception(e,sys)
            


