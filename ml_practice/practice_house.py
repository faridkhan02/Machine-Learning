# import libarries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import sys
np.set_printoptions(threshold=sys.maxsize)


#Importing DataSet 
df =pd.read_csv(r"D:\telegram\1st,2nd - SLR\1st,2nd - SLR\SLR - House price prediction\House_data.csv")
space=df['sqft_living']
price=df['price']

x = np.array(space).reshape(-1, 1)
y = np.array(price)

#Splitting the data into Train and Test
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=1/3, random_state=0)

#Fitting simple linear regression to the Training Set
from sklearn.linear_model import LinearRegression 
regressor = LinearRegression()
regressor.fit(xtrain, ytrain)

#Predicting the prices
pred = regressor.predict(xtest)

## First graph
print("Opening Graph 1...")
plt.scatter(xtrain, ytrain, color='red')
plt.plot(xtrain, regressor.predict(xtrain), color='blue')
plt.title("Visuals for Training Dataset")
plt.xlabel("Space")
plt.ylabel("Price")
plt.show()
print("Graph 1 closed.")

# Second graph
print("Opening Graph 2...")
plt.scatter(xtest, ytest, color='red')
plt.plot(xtrain, regressor.predict(xtrain), color='blue')
plt.title("Visuals for Test Dataset")
plt.xlabel("Space")
plt.ylabel("Price")
plt.show()
print("Graph 2 closed.")

# Save model
import pickle
import os

with open("house_price_model.pkl", "wb") as file:
    pickle.dump(regressor, file)

print("✅ Model saved successfully!")
print("Location:", os.path.abspath("house_price_model.pkl"))