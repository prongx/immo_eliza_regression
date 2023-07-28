"""In your app.py file, create an API that contains:
    A route at / that accept:
        GET request returning a string to explain what the POST expects (data and format).
    A route at /predict that accept:
        POST request that receives the data of a house in JSON format.
"""

from fastapi import FastAPI, HTTPException
# from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field
from typing import Optional, Union
# import pandas as pd
# import numpy as np 
# from sklearn.preprocessing import scale
# import pickle5 as pickle
# import bz2file as bz2
import src.prediction 
import src.preprocessing

app = FastAPI() 

# pickle_in = open("/models/model.pickle","rb")
# model = pickle.load(pickle_in)
# pickle_in.close()

class dataInput(BaseModel):
    postalCode: Union[float, str] = None
    type_property: str = None
    n_rooms: Union[float, str] = None
    living_area: Union[float, str] = None
    equipped_kitchen: Union[float, str] = None
    furnished: Union[float, str] = None
    fireplace: Union[float, str] = None
    terrace: Union[float, str] = None
    area_terrace: Union[float, str] = None
    garden: Union[float, str] = None
    area_garden: Union[float, str] = None
    land_surface: Union[float, str] = None
    n_facades: Union[float, str] = None
    swimming_pool: Union[float, str] = None

    # columns Index(['postalCode', 'price', 'n_rooms', 'living_area', 'equipped_kitchen',
    #    'furnished', 'fireplace', 'terrace', 'area_terrace', 'garden',
    #    'area_garden', 'land_surface', 'n_facades', 'swimming_pool',
    #    'type_property_HOUSE'],
    #   dtype='object')

@app.get('/')
def read_root():
    return """ Hello. Input the property values as stated below:
    {
    "area": int,
    "property-type": "APARTMENT" | "HOUSE",
    "rooms-number": int,
    "zip-code": int,
    "land-area": Optional[int],
    "garden": Optional[bool],
    "garden-area": Optional[int],
    "equipped-kitchen": Optional[bool],
    "full-address": Optional[str],
    "swimming-pool": Optional[bool],
    "furnished": Optional[bool],
    "open-fire": Optional[bool],
    "terrace": Optional[bool],
    "terrace-area": Optional[int],
    "facades-number": Optional[int],
    }
    """

@app.post('/predict')
async def inputdata(input: dataInput):
    requiredFields = {'postalCode', 'type_property', 'n_rooms', 'living_area', 'equipped_kitchen',
                       'furnished', 'fireplace', 'terrace', 'area_terrace', 'garden', 'area_garden','land_surface', 'n_facades','swimming_pool'}
    missingFields = requiredFields - input.model_dump(exclude_none=True).keys()
    if missingFields:
        raise HTTPException(status_code=400, detail= f"You forgot to input these fields: {', '.join(missingFields)}.") 
    else:
       df = src.preprocessing.preprocess_new_data(input)
       pred = src.prediction.predict_new_data(df)
    return {f"result: {pred[0]}"}
 