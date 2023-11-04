import os
import sys
import numpy as np
from src.FlightPricePrediction.logger import logging
from src.FlightPricePrediction.exception import customexception
from src.FlightPricePrediction.components.data_ingestion import Ingestion
from src.FlightPricePrediction.components.data_transformation import DataTransformation
from src.FlightPricePrediction.components.model_trainer import ModelTrainer
import pandas as pd

obj=Ingestion()

train_data_path,valid_data_path=obj.initiate_data_ingestion()
data_transformation=DataTransformation()
train_data,valid_data=data_transformation.initialize_data_transformation(train_data_path,valid_data_path)
np.save("E:\\iNeuron\\End-end-project_2\\artifacts\\prod_train.npy",train_data)
np.save("E:\\iNeuron\\End-end-project_2\\artifacts\\prod_valid.npy",valid_data)
model_trainer=ModelTrainer()
model_trainer.initiate_model_training(train_data,valid_data)

