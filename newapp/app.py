from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np


app = Flask(__name__)

with open('random_forest_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    
    try:
        features = [
            int(request.form['Age']),                         
            int(request.form['Gender']),                      
            int(request.form['Ethnicity']),                   
            int(request.form['ParentalEducation']),           
            float(request.form['StudyTimeWeekly']),           
            int(request.form['Absences']),                    
            int(request.form['Tutoring']),                    
            int(request.form['ParentalSupport']),             
            int(request.form['Extracurricular']),             
            int(request.form['Sports']),                     
            int(request.form['Music']),                       
            int(request.form['Volunteering']),                
            float(request.form['GPA'])                        
        ]
    except ValueError:
        return "Please enter valid values for all fields."

    
    features_array = np.array([features])

    
    prediction = model.predict(features_array)

    
    return render_template('index.html', prediction_text=f'Predicted Grade: {prediction[0]}')


if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)
 
