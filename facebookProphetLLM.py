'''
A comprehensive PyTorch implementation of a hybrid approach where a pre-trained Large Language Model (LLM) is used to 
generate embeddings, which are then fine-tuned alongside Facebook Prophet's time series components. This code will use the 
pre-trained LLM embeddings for categorical features (such as airports, airlines), and Facebook Prophet will handle the time 
series forecasting part.

The implementation assumes you have access to a pre-trained LLM model like BERT or GPT from Hugging Face, which will be used to 
embed the categorical features. After generating embeddings, these will be combined with Facebook Prophet's seasonal and trend 
components for time series forecasting.

Dataset Preparation:

TimeSeriesDataset: A custom dataset class that tokenizes categorical features (e.g., airport codes, airlines) using a pre-trained 
LLM's tokenizer (from Hugging Face) and prepares time series data.
LLM Embedding Model:

LLMEmbeddingModel: This model is based on a pre-trained language model (e.g., BERT) to create embeddings for categorical features. 
The output embeddings are reduced to a size of 128 and combined with time series data.

ProphetTimeSeriesModel: This class wraps Facebook Prophet for fitting and predicting time series flight delays.

CombinedModel: This model combines the LLM-generated embeddings with time series data (from Prophet) to produce a final delay 
prediction. The time series data and embeddings are concatenated and passed through a fully connected layer.

Training Loop:
The model is trained with a simple MSE loss function, iterating over batches of data to optimize both the LLM embeddings and the 
time series components.

Prophet Forecasting:

Finally, the ProphetTimeSeriesModel is used to fit the time series delay data and predict future delays using Facebook Prophet.

Note:
You will need to install transformers, torch, fbprophet, and other dependencies for this code to run.
The dataset here is mocked for demonstration. Replace time_series_data and categorical_data with actual flight delay data.
You can adjust the loss function or model architecture depending on whether you're predicting continuous delay times or binary 
outcomes (delay/no delay).
This code combines both the power of LLM-based embeddings and Prophet for time series forecasting, creating a hybrid model capable 
of handling categorical and temporal features effectively.

Andrew Kiruluta, 09/14/2024
'''

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
import pandas as pd
from prophet import Prophet
import numpy as np
import argparse
import datetime

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Custom dataset class to handle time series and categorical data
class TimeSeriesDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=16):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        ts_data = row['time_series_data'] # For Facebook Prophet, e.g. flight delay time series
        categorical_data = row['categorical_data'] # e.g., airport, airline, etc.
        
        # Tokenize categorical data (e.g., airport, airline) using LLM tokenizer
        tokens = self.tokenizer(categorical_data, max_length=self.max_len, padding='max_length', truncation=True, return_tensors="pt")

        return {
            'input_ids': tokens['input_ids'].squeeze(),
            'attention_mask': tokens['attention_mask'].squeeze(),
            'time_series_data': torch.tensor(ts_data, dtype=torch.float32),
            'delay': torch.tensor(row['delay'], dtype=torch.float32)
        }

# LLM-based embedding model
class LLMEmbeddingModel(nn.Module):
    def __init__(self, pretrained_model_name):
        super(LLMEmbeddingModel, self).__init__()
        self.llm = AutoModel.from_pretrained(pretrained_model_name)
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(self.llm.config.hidden_size, 128) # Output embedding size is 128

    def forward(self, input_ids, attention_mask):
        outputs = self.llm(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output # Take the pooled output from LLM (CLS token)
        pooled_output = self.dropout(pooled_output)
        embeddings = self.fc(pooled_output)
        return embeddings

# Prophet-based time series model
class ProphetTimeSeriesModel:
    def __init__(self):
        self.model = Prophet()

    def fit(self, df):
        self.model.fit(df)

    def predict(self, future):
        return self.model.predict(future)

# Combined Model for LLM Embeddings + Prophet Time Series Forecasting
class CombinedModel(nn.Module):
    def __init__(self, pretrained_model_name):
        super(CombinedModel, self).__init__()
        self.llm_model = LLMEmbeddingModel(pretrained_model_name)
        self.fc = nn.Linear(128 + 1, 1) # Combine LLM embeddings (128) with time series component (1)
    
    def forward(self, input_ids, attention_mask, time_series_data):
        # Get LLM embeddings
        llm_embeddings = self.llm_model(input_ids, attention_mask)
        
        # Combine LLM embeddings with time series data
        combined_input = torch.cat((llm_embeddings, time_series_data.unsqueeze(1)), dim=1)
        output = self.fc(combined_input)
        
        return output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Transformer model')
    parser.add_argument('--train', type=bool, default=False, help='self-supervised baseline training')
    args = parser.parse_args()
    
    # Data preparation (mock dataset structure, replace with actual data)
    data = pd.DataFrame({
    'time_series_data': np.random.rand(100),  # Replace with actual time series flight delay data
    'categorical_data': ['JFK-LAX', 'LAX-SFO'] * 50,  # Replace with actual categorical features
    'delay': np.random.randint(0, 2, 100)  # Binary flight delay labels
    })


    # Initialize tokenizer and dataset
    pretrained_model_name = 'bert-base-uncased' # Replace with any pre-trained LLM from Hugging Face
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
    dataset = TimeSeriesDataset(data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Initialize combined model
    model = CombinedModel(pretrained_model_name)
    criterion = nn.MSELoss() # For delay prediction (could be another loss function depending on task)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Training loop (simplified)
    model.train()
    for epoch in range(5):
        total_loss = 0
        for batch in dataloader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            time_series_data = batch['time_series_data']
            delays = batch['delay']

            optimizer.zero_grad()

            # Forward pass through the combined model
            outputs = model(input_ids, attention_mask, time_series_data)
            loss = criterion(outputs.squeeze(), delays)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {total_loss/len(dataloader)}')

    # Using Prophet for final prediction
    # Assume 'df' is a dataframe with time series data for flight delays (Prophet format: ds, y)
    df = pd.DataFrame({
        'ds': pd.date_range(start='2022-01-01', periods=365, freq='D'),
        'y': np.random.rand(365) # Replace with actual US airline delay time series
    })
    
    data = pd.DataFrame({
        'ds': pd.date_range(start='2022-01-01', periods=100, freq='D'),  # Replace with actual datetime values
        'y': np.random.rand(100),  # Replace with actual time series flight delay data
        'categorical_data': ['JFK-LAX', 'LAX-SFO'] * 50,  # Replace with actual categorical features
        'delay': np.random.randint(0, 2, 100)  # Binary flight delay labels
    })
   
    # Initialize and fit the Prophet model
    prophet_model = ProphetTimeSeriesModel()
    prophet_model.fit(data)

    # Predict future delays using Prophet
    future = prophet_model.model.make_future_dataframe(periods=30) # Predict next 30 days
    forecast = prophet_model.predict(future)

    # Display forecasted delays
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
