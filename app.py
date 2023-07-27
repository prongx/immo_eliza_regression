"""In your app.py file, create an API that contains:
    A route at / that accept:
        GET request returning a string to explain what the POST expects (data and format).
    A route at /predict that accept:
        POST request that receives the data of a house in JSON format.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Union

app = FastAPI() 

# pickle_in = open("/models/model.pickle","rb")
# model = pickle.load(pickle_in)
# pickle_in.close()

class dataInput(BaseModel):
    postalCode: Union[float, str]
    type_property: Union[float, str] = None
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

@app.get('/')
def read_root():
    return """ Hello. Input the property values as stated below:
    postalCode: number [integer]
    type_property: "HOUSE" or "APARTMENT"
    n_rooms: number [integer]
    living_area: number [integer]
    equipped_kitchen: boolean [0 or 1]
    furnished: boolean [0 or 1]
    fireplace: boolean [0 or 1]
    terrace: boolean [0 or 1]
    area_terrace: number [integer]
    garden: boolean [0 or 1]
    area_garden: number [integer]
    land_surface: number [integer]
    n_facades: number [integer]
    swimming_pool: boolean [0 or 1]"""

@app.post('/predict')
async def inputdataoutput(input: dataInput):
    requiredFields = {'postalCode', 'type_property', 'n_rooms', 'living_area', 'equipped_kitchen',
                       'furnished', 'fireplace', 'terrace', 'area_terrace', 'garden', 'area_garden','land_surface', 'n_facades','swimming_pool'}
    missingFields = requiredFields - input.model_dump(exclude_none=True).keys()
    if missingFields:
        raise HTTPException(status_code=400, detail= f"You forgot to input these fields: {', '.join(missingFields)}.") 

    return input

