# Universal Approximation Theorem

This project demonstrates the **Universal Approximation Theorem** using neural networks to approximate various continuous functions. It is implemented using **PyTorch** and showcases how neural networks can approximate functions like `sin(x)`, `exp(x)`, and composite functions such as `exp(x) + x - x²`.

---

## 📚 Introduction

The **Universal Approximation Theorem** states that a feedforward neural network with a single hidden layer containing a finite number of neurons can approximate any continuous function on compact subsets of Real Numbers, under mild assumptions on the activation function.

This project illustrates the theorem by:

1. **Generating synthetic datasets** to simulate continuous functions.
2. **Training neural networks** to approximate these continuous functions.
3. **Visualizing the approximation performance** to demonstrate model effectiveness.
4. **Performing hyperparameter tuning** to optimize model performance.

---

## ⚙️ Project Structure

Here is the breakdown of the project files and their functionalities:

```plaintext
universal-approximation-theorem
├── app.py              # Flask backend script to serve the model
├── main.py             # Generates synthetic datasets and visualizes results
├── train.py            # Trains neural networks using a custom pipeline
├── dataset.py          # Custom dataset class implementation
├── models.py           # Defines the neural network architecture
├── requirements.txt    # List of required dependencies
└── README.md           # This project README
```

## 🚀 Try It Yourself

You can try out the project live by visiting the following link:

**[Try the project here](http://51.21.67.93/)**

---

## 📂 Installation

To run this project locally, follow the steps below:

1. **Clone the repository**:

    ```bash
    git clone https://github.com/your-username/universal-approximation.git
    cd universal-approximation
    ```

2. **Install dependencies**:

    Make sure to have Python 3.6+ installed, and then run:

    ```bash
    pip install -r requirements.txt
    ```

3. **Run the application**:

    - Start the Flask backend:

      ```bash
      python app.py
      ```

    - Go to localhost:5000 and train your model. Once the model is trained, it returns the graph of actual/predicted values.

---

## 📈 Watch the Demo

https://github.com/user-attachments/assets/34bbed49-d3d7-4fd3-8466-99bc2a6ac132

---

## 📈 Visualizations

The project includes visualizations of the model's performance in approximating various functions and is displayed once the model completes training.

---

## 🔧 Hyperparameter Tuning

The neural network models in this project include hyperparameter tuning to optimize the approximation of the functions. The tuning process focuses on parameters like:

- Number of neurons in the hidden layer(To be added in the future)
- Learning rate
- Batch size

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
