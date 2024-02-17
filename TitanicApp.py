from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__, static_url_path='/static',static_folder='static')
logmodel = pickle.load(open('classifier.pkl', 'rb'))
# scaler = pickle.load(open('scalared.pickle', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        passenger_id = float(request.form['Passengerid'])
        age = float(request.form['Age'])
        fare = float(request.form['Fare'])
        pclass = float(request.form['Pclass'])   

    except ValueError:
        return render_template('index.html', prediction_text="Please enter valid numerical values.")

    input_data = [passenger_id, age, fare, pclass]

    # Reshape features into a 2D array
    input_data_as_numpy_array = np.asarray(input_data)

    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = logmodel.predict(input_data_reshaped)

    # Display prediction result
    if prediction[0] == 0:
        output = "Dead"
    if prediction[0] == 1:
        output = "Alive"

    return render_template('index.html', prediction_text='This Passenger is : {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
