from flask import Flask, render_template, url_for, request
from Predict_Function import clean_string, cosine_sim_vectors, predict_function

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    #message = request.form['message']

    #if request.method == 'POST':
    text = request.form['message']
    pred= predict_function(text)

    return render_template('result.html', prediction=pred)


if __name__ == '__main__':
    app.run(debug=True)