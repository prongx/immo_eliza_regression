""" contains all the code used to predict a new house's price."""
import pickle5 as pickle
import bz2file as bz2


def predict_new_data(df):
    pickle_in = bz2.BZ2File("./models/randomForestRegressor.pbz2",'rb')
    regressor = pickle.load(pickle_in)
    pred = regressor.predict(df)   
    return pred