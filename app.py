from src.FlightPricePrediction.pipelines.prediction_pipeline import customData,predictPileline
from flask import Flask,render_template,jsonify,request
import numpy as np

app=Flask(__name__)

@app.route('/predict',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('form.html')
    
    else:
        data=customData(
            Airline=request.form.get('Airline'),
            Source=request.form.get('Source'),
            Destination=request.form.get('Destination'),
            Total_Stops=request.form.get('Total_Stops'),
            Journey_day=float(request.form.get('Journey_day')),
            Journey_month=float(request.form.get('Journey_month')),
            Dep_hour=float(request.form.get('Dep_hour')),
            Dep_minute=float(request.form.get('Dep_minute')),
            Arrival_hour=float(request.form.get('Arrival_hour')),
            Arrival_minute=float(request.form.get('Arrival_minute')),
            dur_hours=float(request.form.get('dur_hours')),
            dur_minutes=float(request.form.get('dur_minutes'))
            )
        
        final_data=data.get_data_as_df()
        predict_pipeline=predictPileline()
        pred=predict_pipeline.predict(final_data)
        result=np.round(pred[0],2)


        return render_template('result.html',final_result=result)

        #execution begin
if __name__ == '__main__':
    app.run()