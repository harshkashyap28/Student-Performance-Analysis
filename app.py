from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('model/student_performance_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    data = request.form
    gender = data['gender']
    race_ethnicity = data['race_ethnicity']
    parental_level_of_education = data['parental_level_of_education']
    lunch = data['lunch']
    test_preparation_course = data['test_preparation_course']
    math_score = float(data['math_score'])
    reading_score = float(data['reading_score'])
    writing_score = float(data['writing_score'])
    
    # Create a DataFrame from the form data
    input_data = pd.DataFrame({
        'gender': [gender],
        'race/ethnicity': [race_ethnicity],
        'parental level of education': [parental_level_of_education],
        'lunch': [lunch],
        'test preparation course': [test_preparation_course],
        'math score': [math_score],
        'reading score': [reading_score],
        'writing score': [writing_score]
    })
    
    # Perform necessary preprocessing (e.g., one-hot encoding)
    input_data = pd.get_dummies(input_data)
    
    # Align the input data with the model's feature names
    model_columns = model.feature_names_in_
    input_data = input_data.reindex(columns=model_columns, fill_value=0)
    
    # Make a prediction
    prediction = model.predict(input_data)[0]
    
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
