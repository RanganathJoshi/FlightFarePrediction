import os
import sys
import pandas as pd
from src.FlightPricePrediction.exception import customexception
from src.FlightPricePrediction.logger import logging
from src.FlightPricePrediction.utils.utils import load_object

class predictPileline:
    def __init__(self):
        pass
    def predict(self):
        try:
            processor_path=os.path.join("artifacts",'preprocessor.pkl')
            model_path=os.path.join("artifacts",'model.pkl')
            processor=load_object(processor_path)
            model=load_object(model_path)

            processor.transform