from bs4 import BeautifulSoup
import requests
import json
import re
import pandas as pd
import concurrent
import time

def get_links(pagesToSearch=1):
    """
    Function that gets links of all properties set in propertiesToSearch list variable. 
    Parameter pagesToSearch: Set the number according to how many pages you want to search thru each of properties set in functions propertiesToSearch variable.
    Variable propertiesToSearch(list): Set according to what properties types you want to look for. 
    """

    propertiesToSearch = ["apartment","house"]
    s = requests.Session()
    listOfLinks = []
    listOfLinksFinal = []

    for prop in propertiesToSearch:
        for x in range(1, pagesToSearch+1):
            search_url = f"https://www.immoweb.be/en/search/{prop}/for-sale?countries=BE&isNewlyBuilt=false&isAPublicSale=false&page={x}&orderBy=relevance"
            r = s.get(search_url)
            soup = BeautifulSoup(r.text,"html.parser")
            for elem in soup.find_all("a", attrs = {"class": "card__title-link"}):
                listOfLinks.append(elem.get('href'))
            listOfLinks = listOfLinks[0:len(listOfLinks)-30] # 30 last links on each page are "recommended properties", throwing those out. 
            listOfLinksFinal = listOfLinksFinal + listOfLinks # Appending the first 30 properties to the output list.
            listOfLinks.clear()
    return(listOfLinksFinal)

def get_data(url):
    s = requests.Session()
    req= s.get(url)
    if req.status_code == 200: # Checking if the link is still valid

        soup= BeautifulSoup(req.text, "html.parser")
        data = soup.find('div',attrs={"class":"container-main-content"}).script.text
        Raw_data_InList= re.findall(r"window.classified = (\{.*\})", data)
        Raw_data_InDict = json.loads(Raw_data_InList[0]) # Data dictionary
        property_dict = {"url":url}


        try:
            property_dict["region"] = Raw_data_InDict["property"]['location']['region']
        except KeyError:
            property_dict["region"] = 0
        except TypeError:
            property_dict["region"] = 0

        try:
            property_dict["province"] = Raw_data_InDict["property"]['location']['province']
        except KeyError:
            property_dict["province"] = 0
        except TypeError:
            property_dict["province"] = 0    

        try:
            property_dict["district"] = Raw_data_InDict["property"]['location']['district']
        except KeyError:
            property_dict["district"] = 0
        except TypeError:
            property_dict["district"] = 0  

        try:
            property_dict["postalCode"] = Raw_data_InDict["property"]['location']['postalCode']
        except KeyError:
            property_dict["postalCode"] = 0
        except TypeError:
            property_dict["postalCode"] = 0 

        try:
            property_dict["locality"] = Raw_data_InDict["property"]['location']['locality']
        except KeyError:
            property_dict["locality"] = 0
        except TypeError:
            property_dict["locality"] = 0

        try:
            property_dict["street"] = Raw_data_InDict["property"]['location']['street']
        except KeyError:
            property_dict["street"] = 0
        except TypeError:
            property_dict["street"] = 0           
            
        try:
            property_dict["Type_property"] = Raw_data_InDict["property"]["type"]
        except KeyError:
            property_dict["Type_property"] = 0
        except TypeError:
            property_dict["Type_property"] = 0
        try:
            property_dict["subtype_property"] = Raw_data_InDict["property"]["subtype"]
        except KeyError:
            property_dict["subtype_property"] = 0
        except TypeError:
            property_dict["subtype_property"] = 0

        try:
            property_dict["price"] = Raw_data_InDict["price"]["mainValue"]
        except KeyError:
            property_dict["price"] = 0
        except TypeError:
            property_dict["price"] = 0
        
        try:
            property_dict["type_transaction"] = Raw_data_InDict["transaction"]["type"]
        except KeyError:
            property_dict["type_transaction"] = 0
        except TypeError:
            property_dict["type_transaction"] = 0
        
        try:
            property_dict["floor"] = Raw_data_InDict["property"]['location']["floor"]
        except KeyError:
            property_dict["floor"] = 0
        except TypeError:
            property_dict["floor"] = 0


        try:
            property_dict["n_rooms"] = Raw_data_InDict["property"]["bedroomCount"]
        except KeyError:
            property_dict["n_rooms"] = 0
        except TypeError:    
            property_dict["n_rooms"] = 0

        try:
            property_dict["living_area"] = Raw_data_InDict["property"]["netHabitableSurface"]
        except KeyError:
            property_dict["living_area"] = 0
        except TypeError:
            property_dict["living_area"] = 0

        try:
            if bool(Raw_data_InDict["property"]["kitchen"]['type']):
                property_dict["equipped_kitchen"] = 1
            else:
                property_dict["equipped_kitchen"] = 0
        except KeyError:
            property_dict["equipped_kitchen"] = 0
        except TypeError: 
            property_dict["equipped_kitchen"] = 0

        try:
            if bool(Raw_data_InDict["transaction"]["sale"]["isFurnished"]):
                property_dict["furnished"] = 1
            else:
                property_dict["furnished"] = 0
        except KeyError:
            property_dict["furnished"] = 0
        except TypeError:
            property_dict["furnished"] = 0

        try:
            if bool(Raw_data_InDict["property"]["fireplaceExists"]):
                property_dict["fireplace"] = 1
            else:
                property_dict["fireplace"] = 0
        except KeyError:
            property_dict["fireplace"] = 0
        except TypeError:
            property_dict["fireplace"] = 0
        
        try:
            if bool(Raw_data_InDict["property"]["hasTerrace"]):
                property_dict["terrace"] = 1
                property_dict["area_terrace"] = Raw_data_InDict["property"]["terraceSurface"]
            else:
                property_dict["terrace"] = 0
                property_dict["area_terrace"] = 0
        except KeyError:
            property_dict["terrace"] = 0
        except TypeError:
            property_dict["terrace"] = 0

        try:
            if bool(Raw_data_InDict["property"]["hasGarden"]):
                property_dict["garden"] = 1
                property_dict["area_garden"] = Raw_data_InDict["property"]["gardenSurface"]
            else:
                property_dict["garden"] = 0
                property_dict["area_garden"] = 0 
        except KeyError:
            property_dict["garden"] = 0
        except TypeError:
            property_dict["garden"] = 0

        try:
            if Raw_data_InDict["property"]["land"]["surface"] != None:
                property_dict["land_surface"] = Raw_data_InDict["property"]["land"]["surface"]
            else:
                property_dict["land_surface"] = 0
        except KeyError:
            property_dict["land_surface"] = 0
        except TypeError: 
            property_dict["land_surface"] = 0
        
        try:
            property_dict["n_facades"] = Raw_data_InDict["property"]["building"]["facadeCount"]
        except KeyError:
            property_dict["n_facades"] = 0
        except TypeError:
            property_dict["n_facades"] = 0

        try:
            property_dict["n_floorCount"] = Raw_data_InDict["property"]["building"]["floorCount"]
        except KeyError:
            property_dict["n_floorCount"] = 0
        except TypeError:
            property_dict["n_floorCount"] = 0    
        
        try:
            if bool(Raw_data_InDict["property"]["hasSwimmingPool"]):
                property_dict["swimming_pool"] = 1
            else:
                property_dict["swimming_pool"] = 0
        except KeyError:
            property_dict["swimming_pool"] = 0
        except TypeError:
            property_dict["swimming_pool"] = 0
            
        try:
            property_dict["state_building"] = Raw_data_InDict["property"]["building"]["condition"]
        except KeyError:
            property_dict["state_building"] = 0
        except TypeError:
            property_dict["state_building"] = 0

    else: # If the link is not valid, pass just the link url into the dictionary
        property_dict = {"url":url}

    return property_dict

data_immo = []
def save(new_data):
        '''This function saves the information acquired from the previous functions and store them in a csv file in the disk.'''
        data_immo.append(new_data)
        dataframe_immo = pd.DataFrame(data_immo)
        dataframe_immo.to_csv(".data/dataset-immo.csv", index=False, encoding="utf-8")
        print(dataframe_immo)
        return dataframe_immo

start = time.time() # Scraping links for later usage.
links = get_links(250) # Set the number according to how many pages you want to search thru each of properties set in get_links function's propertiesToSearch variable.
end = time.time()
print("Gathering links time: {:.6f}s".format(end-start))

# with open('links.txt', 'w') as f:
#     for item in links:
#         # write each item on a new line
#         f.write("%s\n" % item)

# with open('links.txt', 'r') as f:
#     links = [line.strip() for line in f]

start = time.time() # Using concurrency to spead up the scraping process. 
with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor: 
    futures = [executor.submit(get_data, link) for link in links]
    for future in concurrent.futures.as_completed(futures):
        save(future.result())        
end = time.time()
print("Time taken for gathering data from {} links: {:.6f}s".format(len(links),end-start))