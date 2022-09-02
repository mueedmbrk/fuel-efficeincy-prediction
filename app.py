import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from keras.models import load_model

# Creating Flask app
app = Flask(__name__)

# Loading the pickle model
model = load_model('model.h5')


# Home page landing
@app.route('/')
def home():
    return render_template('index.html')


# predicting the results from html form
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """
    For rendering results on HTML GUI
    """
    int_features = [int(x) for x in request.form.values()]
    final_features = np.array(int_features)
    values = final_features.reshape(1, -1)
    print(final_features.shape)
    prediction = model.predict(values)

    return render_template('index.html',
                           prediction_text='The fuel efficiency of vehicle in MPG is:  {}'.format(prediction))


# prediction using api call
@app.route('/calculate', methods=['POST'])
def calculate():
    json_ = request.json
    query_df = pd.DataFrame(json_)
    prediction = model.predict(query_df)
    return jsonify({'prediction': list(prediction)})


# main function
if __name__ == "__main__":
    app.run(debug=True)
