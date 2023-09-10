from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

# Lists of selectable options for teams and cities
teams = [
    'Sunrisers Hyderabad',
    'Mumbai Indians',
    'Royal Challengers Bangalore',
    'Kolkata Knight Riders',
    'Kings XI Punjab',
    'Chennai Super Kings',
    'Rajasthan Royals',
    'Delhi Capitals'
]

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
       'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
       'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
       'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
       'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
       'Sharjah', 'Mohali', 'Bengaluru']

# Load the pre-trained machine learning model
with open('pipe.pkl', 'rb') as model_file:
    model = pickle.load(model_file)


@app.route('/')
def home():
    return render_template('index.html', teams=teams, cities=cities)


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get input values from the form
        batting_team = request.form['batting_team']
        bowling_team = request.form['bowling_team']
        city = request.form['city']
        target = int(request.form['target'])
        current_score = int(request.form['current_score'])
        overs_completed = int(request.form['overs_completed'])
        wickets_out = int(request.form['wickets_out'])

        # Create a dictionary with input values
        input_data = {
            'batting_team': batting_team,
            'bowling_team': bowling_team,
            'city': city,
            'runs_left': target - current_score,
            'balls_left': (120 - (overs_completed * 6)),
            'wickets': 10 - wickets_out,
            'total_runs_x': target,
            'crr': current_score/overs_completed,
            'rrr': (target-current_score)/(20-overs_completed)
        }

        # Convert input data to a DataFrame for prediction
        input_df = pd.DataFrame([input_data])

        # Use the pre-trained model to make predictions
        win_probabilities = model.predict_proba(input_df)

        # Prepare the response
        response = {
            'batting_team': batting_team,
            'bowling_team': bowling_team,
            'win_probability': {
                batting_team: round(win_probabilities[0][1] * 100, 2),
                bowling_team: round(win_probabilities[0][0] * 100, 2)
            }
        }

        return render_template('result.html', response=response)


if __name__ == '__main__':
    app.run(debug=True)