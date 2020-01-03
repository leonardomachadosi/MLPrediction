import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
upmodel = pickle.load(open('uppredictionmodel.pkl', 'rb'))

@app.route('/predict_api/up', methods=['POST'])
def predictUp():
    data = request.get_json(force=True)
    prediction = upmodel.predict([np.array(list(data.values()))])
    output = prediction[0]
    return jsonify(int(output))


if __name__ == "__main__":
    app.run(debug=True)
