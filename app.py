from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('Drug.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Define categorical mappings
sex_mapping = {'male': 0, 'female': 1}
bp_mapping = {'HIGH': 0, 'NORMAL': 1, 'LOW': 2}
cholesterol_mapping = {'HIGH': 0, 'NORMAL': 1}

# Define route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Define route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input values from the form
    age = int(request.form['Age'])
    sex = sex_mapping[request.form['Sex']]
    bp = bp_mapping[request.form['BP']]
    cholesterol = cholesterol_mapping[request.form['Cholesterol']]
    na_to_k = float(request.form['Na_to_K'])
    
    # Make prediction
    input_features = np.array([[age, sex, bp, cholesterol, na_to_k]])
    prediction = model.predict(input_features)
    
    # Render the result template with prediction
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)



