import os
import sys
import pandas as pd
from src.FlightPricePrediction.exception import customexception
from src.FlightPricePrediction.logger import logging
from src.FlightPricePrediction.utils.utils import load_object

class predictPileline:
    def __init__(self):
        pass
    def predict(self,feature):
        try:
            processor_path="E:/iNeuron/End-end-project_2/artifacts/preprocessor.pkl"
            model_path="E:/iNeuron/End-end-project_2/artifacts/model.pkl"
            processor=load_object(processor_path)
            model=load_object(model_path)

            scaled_data=processor.transform(feature)
            pred=model.predict(scaled_data)

            return pred
        
        except Exception as e:
            logging.info("Error occured while predicting")
        
class customData:
    def __init__(self,Airline:str, Source:str, Destination:str, Total_Stops:str,Journey_day:float, Journey_month:float, Dep_hour:float, Dep_minute:float,Arrival_hour:float, Arrival_minute:float, dur_hours:float, dur_minutes:float):
        self.Airline=Airline
        self.Source=Source
        self.Destination=Destination
        self.Total_Stops=Total_Stops
        self.Journey_day=Journey_day
        self.Journey_month=Journey_month
        self.Dep_hour=Dep_hour
        self.Dep_minute=Dep_minute
        self.Arrival_hour=Arrival_hour
        self.Arrival_minute=Arrival_minute
        self.dur_hours=dur_hours
        self.dur_minutes=dur_minutes

    def get_data_as_df(self):
        try:
            custom_data={
                'Airline':[self.Airline],
                'Source':[self.Source],
                'Destination':[self.Destination],
                'Total_Stops':[self.Total_Stops],
                'Journey_day':[self.Journey_day],
                'Journey_month':[self.Journey_month],
                'Dep_hour':[self.Dep_hour],
                'Dep_minute':[self.Dep_minute],
                'Arrival_hour':[self.Arrival_hour],
                'Arrival_minute':[self.Arrival_minute],
                'dur_hours':[self.dur_hours],
                'dur_minutes':[self.dur_minutes]
            }

            df=pd.DataFrame(custom_data)
            logging.info('DataFrame Gathered')
            return df
        except Exception as e:
            logging.info("error occured while customizing data")
            raise customexception(e,sys)
