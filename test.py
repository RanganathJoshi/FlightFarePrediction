"""import os

path="notebooks/research.ipynb"

dir,file=os.path.split(path)

os.makedirs(dir,exist_ok=True)

with open(path,"w") as f:
    pass"""
    
    

import pickle
import numpy as np
from src.FlightPricePrediction.pipelines.prediction_pipeline import customData,predictPileline

custdataobj=customData('Jet Airways','Banglore','New Delhi','1 stop',18,3,11,40,5,5,17,25)

data=custdataobj.get_data_as_df()
pre_path="E:/iNeuron/End-end-project_2/artifacts/preprocessor.pkl"
model_path="E:/iNeuron/End-end-project_2/artifacts/model.pkl"

def loading(pre_path):
    with open(pre_path,'rb') as pre_obj:
        return pickle.load(pre_obj)
process=loading(pre_path)
model=loading(model_path)
scaled=process.transform(data)
#print(data)
#print(scaled)
pred=model.predict(scaled)
print(np.round(pred,decimals=2))


<div class="form-group">
  <label for="clarity">Clarity:</label>
  <select id="clarity" name="clarity">
    <option value="I1">I1</option>
    <option value="SI2">SI2</option>
    <option value="SI1">SI1</option>
    <option value="VS2">VS2</option>
    <option value="VS1">VS1</option>
    <option value="VVS2">VVS2</option>
    <option value="VVS1">VVS1</option>
    <option value="IF">IF</option>
  </select>
</div>