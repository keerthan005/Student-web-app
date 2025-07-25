from flask import render_template, request

@app.route('/form', methods=['GET', 'POST'])
def form():
    prediction = None
    if request.method == 'POST':
        gender = int(request.form['gender'])
        parental_level = int(request.form['parental_level_of_education'])
        test_prep = int(request.form['test_preparation_course'])
        reading_score = float(request.form['reading_score'])
        writing_score = float(request.form['writing_score'])

        input_features = [gender, parental_level, test_prep, reading_score, writing_score]
        prediction = model.predict([input_features])[0]

    return render_template('form.html', prediction=prediction)
