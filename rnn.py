
#########################################
#############  Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the data set
dataset_train = pd.read_csv("Google_Stock_Price_Train.csv")
training_set = dataset_train.iloc[:, 1:2].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
training_set_scaled = sc.fit_transform(training_set)


# Creating data structure
X_train = []
y_train = []
timesteps = 60 # How many days before to predict that day's stock price?

for i in range(timesteps, training_set.shape[0]):
    X_train.append(training_set_scaled[i-timesteps:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


#########################################
#############  Building RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# RNN initialization
regressor = Sequential()
units = 60

regressor.add(LSTM(units = units, return_sequences = True, input_shape=(X_train.shape[1], 1)))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = units, return_sequences = True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = units, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = units))
regressor.add(Dropout(0.2))

regressor.add(Dense(units = 1))

regressor.compile(optimizer='adam', loss='mse')

regressor.fit(X_train, y_train, epochs=100, batch_size=32)

#regressor.save('model.h5')
#regressor.summary() 


#########################################
#############  Predictions

# Importing the data set
dataset_test = pd.read_csv("Google_Stock_Price_Test.csv")
test_set = dataset_test.iloc[:, 1:2].values
   
dataset_all = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_all[len(dataset_all)-len(dataset_test)-timesteps:].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)

# Creating data structure
X_test = []

for i in range(timesteps, inputs.shape[0]):
    X_test.append(inputs[i-timesteps:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predicted = regressor.predict(X_test)
predicted = sc.inverse_transform(predicted)


#########################################
#############  Visuzalizations

plt.plot(test_set, color ='red', label= 'Google Stock Price')
plt.plot(predicted, color ='blue', label= 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
#plt.figure()
plt.show()