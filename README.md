# Netflix-returns-prediction
This project focuses on predicting Netflix stock returns using a Long Short-Term Memory (LSTM) neural network. The objective is to model temporal dependencies in financial time-series data and evaluate the model’s predictive performance using RMSE.
Project Overview

# Stock price movements are inherently sequential and non-linear. Traditional models struggle to capture long-term dependencies, making LSTM networks a strong choice for financial time-series forecasting.
In this project:
* Historical Netflix stock data is preprocessed
* Returns are calculated and scaled
* An LSTM model is trained to predict future returns
* Model performance is evaluated using RMSE


**Key Highlights:**

* Time-series forecasting using LSTM
* Feature scaling and sequence creation
* Model evaluation using RMSE
* Clean, reproducible Jupyter Notebook workflow

---

# Technologies Used

* **Python**
* **TensorFlow / Keras**
* **NumPy**
* **Pandas**
* **Scikit-learn**
* **Matplotlib**

---

# Workflow

1. **Data Loading & Preprocessing**

   * Historical Netflix stock data
   * Handling missing values
   * Feature scaling using `StandardScaler` / `MinMaxScaler`

2. **Sequence Creation**

   * Converting time-series data into supervised learning format
   * Sliding window approach

3. **Model Architecture**

   * LSTM layer(s)
   * Dense output layer
   * Optimized for small datasets

4. **Model Training**

   * Train-test split
   * Loss optimization using MSE

5. **Evaluation**

   * RMSE calculation (scaled & inverse-scaled values)
   * Prediction visualization

---

# LSTM Model Architecture

* **LSTM Layer** – captures long-term dependencies
* **Dense Layer** – final prediction output
* **Loss Function** – Mean Squared Error (MSE)
* **Optimizer** – Adam

---

# Evaluation Metric

Root Mean Squared Error (RMSE):

```python
lstm_rmse = np.sqrt(mean_squared_error(y_test_actual, lstm_predictions))
print("LSTM RMSE:", lstm_rmse)
```

RMSE is calculated both on:

* **Scaled values**
* **Inverse-transformed (actual stock values)**

---

# Results & Insights

* LSTM effectively captures Netflix stock price trends
* Performs well even with limited data (~1000 rows)
* Demonstrates the strength of deep learning in time-series forecasting

---

# Future Improvements

* Add technical indicators (RSI, MACD, Moving Averages)
* Hyperparameter tuning
* Compare with ARIMA / Prophet models
* Deploy using Streamlit

