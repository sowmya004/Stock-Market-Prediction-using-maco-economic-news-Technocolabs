import numpy as np
from flask import Flask, request,jsonify,render_template
import pickle
from _datetime import date
import pandas as pd

app = Flask(__name__,template_folder='templates')
model = pickle.load(open('new_model_pickle','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        int_features= [float(x) for x in request.form.values()]
        final_features= [np.array(int_features)]
        prediction = model.predict(final_features)
        output = round(prediction[0],3)
        Day = request.form['Day']
        Month = request.form['Month']
        Year = request.form['Year']
        StockName = request.form['StockName']
        Volume = request.form['Volume']
        Open = request.form['Open']
        Low = request.form['Low']
        High = request.form['High']
        Negative = request.form['Negative']
        Neutral = request.form['Neutral']
        Positive = request.form['Positive']
        Date = Year+"/"+Month+"/"+Day
       # return "Day: "+ Day + "\t"+ "Month: "+ Month + "\t" + "Year: "+ Year + "\t" + "Volume: "+ Volume + "\t" + "Open: "+ Open + "\t" + "Low" + Low + "\t" + "High" + High +"\t" + "Negative: "+ Negative + "\t" + "Neutral: "+ Neutral + "\t" + "Positive: "+ Positive + "\n" +'Next Day stock close price for is $ {}'.format(output)
        return render_template('predict.html', Year = Year, Month= Month, Day=Day, StockName= StockName, Volume=Volume, 
        Open= Open, Low=Low, High=High,Negative=Negative, Neutral=Neutral ,Positive=Positive,
        prediction_text='Predicted close price for '+ Date + ' is ${}'.format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

@app.route('/twitter-stock-data')
def vmd_timestamp():
    df1 = pd.read_csv('Twitter_stock_final_dataset (1).csv')
    df = df1[:100]
    return render_template('df.html',tables=[df.to_html(classes='data')], titles=df.columns.values)

@app.route('/actual-vs-predicted')
def act_vs_pred():
    df = pd.read_csv('act_vs_pred.csv')
    return render_template('df.html',tables=[df.to_html(classes='data')], titles=df.columns.values)

if __name__ == "__main__":
    app.run(debug=True)
