# Facebook-Prophet-with-LLM-Embeddings
Description
The project integrates two core components:

Time Series Forecasting with Prophet: 

Facebook Prophet is used to handle the time series component of the flight delay data, capturing trends, seasonality, and making future delay predictions.

Categorical Feature Embedding with BERT (PyTorch):

Pre-trained BERT models from Hugging Face are used to generate embeddings for categorical features such as airport pairs (e.g., JFK-LAX) and airline names. These embeddings are then combined with the time series data to provide a more holistic prediction model.

Usage
Clone the repository and install the required dependencies:

git clone https://github.com/yourusername/prophet-pytorch-flight-delay.git

cd prophet-pytorch-flight-delay

pip install -r requirements.txt

Modify the prophet_pytorch_flight_delay.py file to load your custom flight delay dataset, replacing the example data.

Run the script:

python prophet_pytorch_flight_delay.py

This will train the hybrid Prophet-PyTorch model and use it to predict flight delays for future time periods.

Example Data Format

The input dataset should be a pandas DataFrame with at least the following columns:

ds: A column containing datetime values representing the time index for Prophet.
y: The target column, representing the flight delay value (or another time series value you are trying to predict).
categorical_data: Categorical values such as airport pairs, airline names, etc.

```python
# This is a Python code snippet
import numpy as np
import pandas as pd

data = pd.DataFrame({
    'ds': pd.date_range(start='2022-01-01', periods=100, freq='D'),
    'y': np.random.rand(100),  # Replace with actual time series flight delay data
    'categorical_data': ['JFK-LAX', 'LAX-SFO'] * 50  # Example categorical data
})

print(data.head())
```

Output

The model will output forecasted delays for future time periods based on the trained model, including uncertainty intervals (yhat_lower, yhat_upper).

Notes

The hybrid model allows leveraging both the temporal dynamics in the time series data and the complex relationships in categorical features like airport pairs.
The predictions are made using Prophet for time series forecasting and PyTorch for categorical feature embedding.
plotly is used for visualizing the forecast output interactively.

License

This project is licensed under the MIT License - see the LICENSE file for details.
