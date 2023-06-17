import os
import random

import pandas as pd
from flask import Flask, request, render_template, url_for


app = Flask(__name__)
audio_dir = 'static/Train/raw'
audio_files = os.listdir(os.path.join(audio_dir))
random.shuffle(audio_files)
df = pd.read_csv('static/Train/data_list.csv')


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        decision = request.form['decision']
        filename = request.form['filename']

        # Get the correct decision for the current audio file.
        is_correct = check_decision_correctness(decision, filename)

        if is_correct:
            flash_class = 'flash-green'
            message = 'Correct!'
        else:
            flash_class = 'flash-red'
            message = 'Incorrect!'

        return render_template('feedback.html', flash_class=flash_class, message=message, redirect_url=url_for('index'))

    if len(audio_files) > 0:
        audio_file = os.path.join(audio_dir, audio_files.pop())
        return render_template('index.html', audio_file=audio_file)

    # All audio files have been evaluated, display a thank you page or redirect as needed
    return render_template('index.html')


def check_decision_correctness(decision, filename):
    # Replace with your own logic to check if the decision is correct
    global df
    file_id = os.path.basename(filename).split('.')[0]
    correct_decision = df[df['ID'] == file_id]['Disease category'].values[0]

    print(correct_decision, decision, file_id)

    if correct_decision == 5 and decision == 'Normal':
        return True
    if decision == 'Abnormal':
        return True
    return False


if __name__ == '__main__':
    # app.debug = True
    app.run()
