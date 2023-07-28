"""preprocessing the user input data before using them in the trained model"""
import pandas as pd
from sklearn.preprocessing import scale

def preprocess_new_data(input):
    df = pd.DataFrame([input.model_dump()])
    #Changing postalCode value to lower range
    df.postalCode=(df.postalCode/1000)
    #Transforming type_property column into type_property_HOUSE dummy(0 1)
    if "HOUSE" in df["type_property"].values:
        type_property_HOUSE = [1]
    else:
        type_property_HOUSE = [0]
    df["type_property_HOUSE"] = type_property_HOUSE
    #Deleting type_property value
    del df["type_property"]
    #Scaling some values
    col = ['n_rooms','living_area', 'area_terrace','area_garden','land_surface','postalCode','n_facades']
    df[col] = scale(df[col])
    return df