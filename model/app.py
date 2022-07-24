import csv
import pandas as pd
from flask import Flask, request, send_file, render_template
import pickle
import os

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''

    file=request.files['file']
    data = pd.read_csv(file)

    new_file=data.drop(['date', 'HumidityRatio'], axis=1)

    prediction=model.predict(new_file)
    data['Occupancy']=prediction

    # final = csv.writer(data)

    # final_csv = data + '.csv'

    final = data.to_csv('roomoccupancy.csv', index=False)


    output = round(prediction[0], 2)
    if output:
        return render_template('index.html', prediction_text='Your output file is downloaded in your device named as "roomoccupancy.csv"')
        # return render_template('index.html', tables=[data.to_html()], titles=[''])
        # return send_file(final, as_attachment=True)
    else:
        return render_template('index.html', prediction_text='somthing went wrong')
    # if prediction == 0:
    #     return render_template('index.html', tables=[data.to_html()], titles=[''])
    # else:
    #     return render_template('index.html', tables=[data.to_html()], titles=[''])

if __name__ == "__main__":
    app.run(debug=True)

