import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor, HuberRegressor)
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
import lightgbm as lgb 
import xgboost as xgb
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

# laod dataset 

data = pd.read_csv(r"D:\telegram\USA_Housing.csv")

# preprocessing
x = data.drop(['Price','Address'],axis=1)
y = data['Price']

# split data 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
# define model 
models = {
    'LinearRegression':LinearRegression(),
    'RobustRegression':HuberRegressor(),
    'RidgeRegression':Ridge(),
    'LassoRegression':Lasso(),
    'PolynomialRegression':Pipeline([
       ('poly',PolynomialFeatures(degree=2)),
        ('Linear',LinearRegression())
    ]),
    'SGDRegressor':SGDRegressor(),
    'ANN':MLPRegressor(hidden_layer_sizes=(100,),max_iter=1000),
    'RandomForest':RandomForestRegressor(),
    "SVM":SVR(),
    "LGBM":lgb.LGBMRegressor(),
    "XGboost":xgb.XGBRegressor(),
    "KNN":KNeighborsRegressor()
}
# Train the model
results=[]
for name,model in models.items():
    model.fit(x_train,y_train)
    y_pred=model.predict(x_test)

    mae=mean_absolute_error(y_test,y_pred)
    mse=mean_squared_error(y_test,y_pred)
    r2=r2_score(y_test,y_pred)


    results.append({
        'Model':name,
        "MAE":mae,
        "MSE":mse,
        "R2":r2
    })
    with open(f'{name}.pkl', 'wb') as f:
        pickle.dump(model,f)

# convert result to the  data frame 
results_df=pd.DataFrame(results)
results_df.to_csv("model_evalution_results.csv",index=False)
print("Models have  been trained and saved as pickle files .Evalution results has been saved by the model_evalution_results.csv")
