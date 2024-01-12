import flask
from flask import Flask,request,render_template,jsonify
from src.pipeline.prediction_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application


@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/Time',methods=['GET','POST'])

def predict_datapoint():
    if request.method=='GET':
        return render_template('form.html')
    
    else:
        data=CustomData(
            Delivery_person_Age=request.form.get('Delivery_person_Age'),
            Delivery_person_Ratings=request.form.get('Delivery_person_Ratings'),
            Distance_in_km=request.form.get('Distance_in_km'),
            Vehicle_condition=request.form.get('Vehicle_condition'),
            Weather_conditions = request.form.get('Weather_conditions'),
            Road_traffic_density= request.form.get('Road_traffic_density'),
            City = request.form.get('City'),
            Type_of_vehicle=request.form.get('Type_of_vehicle'),
            Festival=request.form.get('Festival'),
            multiple_deliveries=request.form.get('multiple_deliveries')
       )
        final_new_data=data.get_data_as_dataframe()
        predict_pipeline=PredictPipeline()
        pred=predict_pipeline.predict(final_new_data)

        results=round(pred[0],2)

        return render_template('results.html',final_result=results)

if __name__=="__main__":
    app.run(host='0.0.0.0',debug=True)