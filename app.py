from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

# Load the model at startup
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        try:
            gender = int(request.form['gender'])
            parental_level = int(request.form['parental_level_of_education'])
            test_prep = int(request.form['test_preparation_course'])
            reading_score = float(request.form['reading_score'])
            writing_score = float(request.form['writing_score'])

            input_features = [gender, parental_level, test_prep, reading_score, writing_score]
            prediction = model.predict([input_features])[0]
        except Exception as e:
            prediction = f"Error: {str(e)}"
    
    return render_template('form.html', prediction=prediction)

@app.route('/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json(force=True)

        input_features = [
            data['gender'],
            data['parental_level_of_education'],
            data['test_preparation_course'],
            data['reading_score'],
            data['writing_score']
        ]

        prediction = model.predict([input_features])
        return jsonify({'Predicted Class': int(prediction[0])})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
