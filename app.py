import numpy as np
import pandas as pd
import pickle
from flask import Flask, request, jsonify, render_template
from waitress import serve

app = Flask(__name__)

# Load model and preprocessing components
with open('model_components.pkl', 'rb') as file:
    loaded_components = pickle.load(file)

scaler = loaded_components['simple scaler']
model = loaded_components['simple model']
features = loaded_components['simple features']

@app.route('/')
def home():
    return render_template('./index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input = dict(request.form)

    def to_numeric(key, value, numeric_features = features ):
        if key not in numeric_features:
            return value
        try:
            return float(value)
        except:
            return np.nan
        
    input_data = {key: to_numeric(key, value) for key, value in input.items()}

    df = pd.DataFrame([input_data])
    df = df[features]
    df = pd.DataFrame(scaler.transform(df), columns=df.columns)

    prediction = model.predict(df)
    prediction = int(np.round(prediction[0], 0)) 

    return render_template('./index.html', prediction_text=f'Predikert oppholdslengde er {prediction} dager')


if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=8080)
