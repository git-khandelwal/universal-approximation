from flask import Flask, render_template, request, redirect, url_for, send_file
import torch
from train import Pipeline
import numpy as np
import matplotlib.pyplot as plt
import io
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/train', methods=['GET', 'POST'])
def train():
    if request.method == 'POST':
        # Get the form data
        limits = int(request.form['limits'])
        input_size = float(request.form['input_size'])
        lr = float(request.form['lr'])
        epochs = int(request.form['epochs'])
        user_function = request.form['function']

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            funcApprox = eval(f"lambda x: {user_function}", {"np": np})
        except Exception as e:
            return f"Error in the function: {str(e)}"
        pipeline = Pipeline(funcApprox, limits, input_size, lr, device)

        # Training the model
        pipeline.train(epochs)


        
        global pipeline_instance
        pipeline_instance = pipeline

        # After training is complete, redirect to display the graph
        return redirect(url_for('plot'))
    
    return render_template('train.html')

@app.route('/plot')
def plot():
    arr, y_actual, y_pred = pipeline_instance.pred()  # Get predictions after training

    # Plot predicted vs actual
    plt.scatter(arr, y_actual, label='True Function', color='b')
    plt.scatter(arr, y_pred, label='Predicted Function', color='r')
    plt.legend()

    # Save the figure to a BytesIO object and send it as a response
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    return send_file(img, mimetype='image/png')


if __name__ == "__main__":
    app.run(debug=True)


