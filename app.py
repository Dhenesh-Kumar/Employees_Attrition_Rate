from flask import Flask, render_template, url_for, request
import pandas as pd
import pickle

app = Flask(__name__)

with open('finalModel_randForest.pkl', 'rb') as f:
    df4 = pickle.load(f)


@app.route('/')
def home():
    return render_template('EmployeeAttritionPredHome.html', emp_list=df4['EmployeeID'])


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        emp_id = request.form['emp_id']
        row = df4[df4['EmployeeID'] == int(emp_id)]

        prediction = ""
        emp_name = ""
        errorMessage = ""

        if row.empty:
            errorMessage = "No employee found for the entered ID"
        else:
            if row['pred'].values[0] == 0:
                prediction = "No"
            else:
                prediction = "Yes"
            emp_name = row['EmployeeName'].values[0]

        return render_template('EmployeeAttritionPredOutput.html', prediction=prediction, emp_name=emp_name, emp_id=emp_id, errorMessage=errorMessage,emp_list=df4['EmployeeID'])


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
