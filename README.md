# MovieSentimentAnalysis
An implementation of Bidirectional LSTM to analyze movie sentiments.
# Algorithm
In this project we train a Bidirectional Long Short-Term Memory (LSTM) network on movie sentiment reviews dataset. This is done because Bidirectional LSTMs run both forwards and backwards, unlike Unidirectional LSTM, which only move forward.
This helps the network to understand the "context" in the text better.

Also, Dropout is added to the Dense layer after BiLSTM. This is because by adding drop out for LSTM cells, there is a chance for forgetting something that should not be forgotten.
Apart from this, data cleaning is also done in order to convert both the datasets into a common structure and remove N/A values. 
# Datasets

# Environment
Language: Python 3 <p>
Libraries: Keras, NLTK, Pandas
