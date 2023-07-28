from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
import pandas as pd
import numpy as np 
import pickle5 as pickle
import bz2file as bz2
import joblib

# Save
# encoder_filename = "models/encoder.save"
# joblib.dump(encoder, encoder_filename)
# # Load
# encoder = joblib.load("models/encoder.save")

#bz2
# data = bz2.BZ2File(file, ‘rb’)
# data = pickle.load(data)
# return data

# def decompress_pickle(file):
# data = bz2.BZ2File(file, ‘rb’)
# data = pickle.load(data)
# return data

# model = decompress_pickle(‘filename.pbz2’)


dataframe = pd.read_csv('./data/dataset_scrape.csv')

def cleanDataframe (dataframe):
    """
    Removes duplicates, removes rows with empty value, fills empty area_terrace and n_facades values with median. 
    """
    df=dataframe
    #Removing duplicates:
    df.drop_duplicates(inplace = True) #removing the duplicated rows in dataframe
    #Removing the rows with empty value
    df.dropna(subset=['region'], how='any',inplace=True)
    df.dropna(subset=['living_area'], how='any',inplace=True)
    df.dropna(subset=['state_building'], how='any',inplace=True)
    df.drop(df[(df.type_property == "APARTMENT") & (df.floor.isnull())].index, inplace=True)
    df.dropna(subset=['area_garden'], how='any',inplace=True)
    #Filling empty values
    x = (df[(df.terrace==1) & (df.area_terrace.notnull())].area_terrace.median())
    df["area_terrace"].fillna(x, inplace = True)
    x = (df[(df.n_facades.notnull())].n_facades.median())
    df["n_facades"].fillna(x, inplace = True)
    #Getting rif of columns with high percentage of null value
    df.drop("street", axis='columns', inplace=True)
    df.drop("floor", axis='columns', inplace=True)
    df.drop("n_floorCount", axis='columns', inplace=True)
    return(df)

def processDataframe (dataframe):
    """
    Deletes unwanted columns, deals with Postcode, scales numerical values and sets dummy values on string categorical values.
    """
    from sklearn.preprocessing import scale
    df=dataframe
    #Droping some unwanted columns
    # df.drop(['url','locality','type_transaction','state_building','subtype_property'], axis='columns', inplace=True)
    df.drop(['url','region','province','district','locality','type_transaction','state_building','subtype_property'], axis='columns', inplace=True)
    #Changing postalCode value to lower range
    df.postalCode=(df.postalCode/1000)
    #Scaling some values
    col = ['n_rooms','living_area', 'area_terrace','area_garden','land_surface','postalCode','n_facades']
    df[col] = scale(df[col])
    #Setting dummy values on string categorical values
    df = pd.get_dummies(df,columns = ['type_property'], drop_first=True)
    return(df)

def saveDataframe (dataframe):
    dataframe.to_csv("./data/dataset_ALL.csv", index=False, encoding="utf-8")
    # dataframe.to_csv("./data/dataset_APARTMENT.csv", index=False, encoding="utf-8")
    # dataframe.to_csv("./data/dataset_HOUSE.csv", index=False, encoding="utf-8")

def simpleLinearReg (dataframe):
    from sklearn.linear_model import LinearRegression
    df=dataframe
    X = df['living_area'].values
    y = df['price'].values
    X_train, X_test, y_train, y_test = train_test_split(X,y, random_state= 42)
    X_train = X_train.reshape(-1,1)
    y_train = y_train.reshape(-1,1)
    X_test = X_test.reshape(-1,1)
    y_test = y_test.reshape(-1,1)
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    pred = regressor.predict(X_test)
    rmse = np.sqrt(MSE(y_test, pred))
    scoreTrain=regressor.score(X_train, y_train)
    scoreTest=regressor.score(X_test, y_test)
    print("Simple linear regression train score is:",scoreTrain)
    print("Simple linear regression test test is:",scoreTest)
    print("RMSE: % f" %(rmse))

def multiLinearReg (dataframe):
    from sklearn.linear_model import LinearRegression
    df=dataframe
    X = df.drop(['price'],axis=1).to_numpy()
    y = df.price.to_numpy().reshape(-1 , 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)
    regressor = LinearRegression()
    regressor.fit(X_train,y_train)
    pred = regressor.predict(X_test)
    rmse = np.sqrt(MSE(y_test, pred))
    print("Multiple linear regression train score is:",regressor.score(X_train, y_train)) 
    print("Multiple linear regression train test is:",regressor.score(X_test, y_test))
    print("RMSE: % f" %(rmse))

def decisionTreeRegressor (dataframe):
    from sklearn.tree import DecisionTreeRegressor
    df=dataframe
    X = df.drop(['price'],axis=1).to_numpy()
    y = df.price.to_numpy().reshape(-1 , 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)
    regressor = DecisionTreeRegressor(criterion="squared_error",random_state=0)
    regressor.fit(X_train, y_train)
    pred = regressor.predict(X_test)
    rmse = np.sqrt(MSE(y_test, pred))
    print("DecisionTreeRegressor train score is:",regressor.score(X_train, y_train))
    print("DecisionTreeRegressor test score is:",regressor.score(X_test, y_test))
    print("RMSE: % f" %(rmse))

def xgbRegressor (dataframe):
    import xgboost as xg
    df=dataframe
    X = df.drop(['price'],axis=1).to_numpy()
    y = df.price.to_numpy().reshape(-1 , 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)
    regressor = xg.XGBRegressor(objective ='reg:linear',n_estimators = 10, seed = 123)       
    regressor.fit(X_train, y_train)
    #RMSE Computation
    pred = regressor.predict(X_test)
    rmse = np.sqrt(MSE(y_test, pred))
    print("XGBRegressor train score is:",regressor.score(X_train, y_train))
    print("XGBRegressor test score is:",regressor.score(X_test, y_test))
    print("RMSE: % f" %(rmse))

def randomForestRegressor (dataframe):
    from sklearn.ensemble import RandomForestRegressor
    df=dataframe
    X = df.drop(['price'],axis=1).to_numpy()
    y = df.price.to_numpy().reshape(-1 , 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)
    regressor = RandomForestRegressor(random_state=3)
    # regressor = RandomForestRegressor()
    regressor.fit(X_train, y_train)
    pred = regressor.predict(X_test)
    rmse = np.sqrt(MSE(y_test, pred))
    # saving the model
    # pickle_out = open("./models/randomForestRegressor.pickle","wb")
    # pickle.dump(regressor, pickle_out)
    # pickle_out.close()
    with bz2.BZ2File("./models/randomForestRegressor" + '.pbz2', 'w') as f:
        pickle.dump(regressor, f)
    print("Random Forest Regressor train score is:",regressor.score(X_train, y_train))
    print("Random Forest Regressor test score is:",regressor.score(X_test, y_test))
    print("RMSE: % f" %(rmse))    


print("Dataset size before cleaning",dataframe.shape)
df=cleanDataframe(dataframe)
print("Dataset size after cleaning",df.shape)
df=processDataframe(dataframe)
print("Dataset size after processing",df.shape)
print("columns",df.columns)

# saveDataframe(df)
# simpleLinearReg(df)
# multiLinearReg(df)
# decisionTreeRegressor(df)
# xgbRegressor(df)
randomForestRegressor(df)
