import numpy as np
from flask import Flask, request, jsonify, render_template
from joblib import load
app = Flask(__name__)
model = load('nb.save')

@app.route('/')
def home():
    return render_template('hm2.html')

@app.route('/predict',methods=['POST'])
def predict():
    org=request.form.to_dict()
    y_test=list(org.values())
    x_test=y_test[3:7]
    x_test=list(map(int, x_test)) 
    x_test= np.array(x_test)
    x_test=x_test.reshape(1,4)
    print(org)
    print(y_test)
    print(x_test)
    prediction = model.predict(x_test)
    print(prediction)
    output=prediction[0]
    if(output == 0):
        sug="level 0 - You are fine!"
    elif(output==1):
        sug="level 1 - Need some medication!"
    else:
        sug="level 2 - Consult doctor!"
    return render_template('hm2.html', name='Name : {}'.format(org['name']), age='Age : {}'.format(org['age']), gender='Gender : {}'.format(org['gender']), SBP='SBP : {}'.format(x_test[0][0]), DBP='DBP : {}'.format(x_test[0][1]), Pulse='Pulse : {}'.format(x_test[0][2]), Temperature='Temperature : {}'.format(x_test[0][3]), prediction='Status : {}'.format(sug), bloodgroup='Blood Group : {}'.format(org['blood_group']), phno='Phone No : {}'.format(org['phno']))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])
    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run()