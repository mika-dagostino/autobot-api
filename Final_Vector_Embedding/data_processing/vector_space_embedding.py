from datetime import datetime
import os


import pickle
import json
from model_loader import get_model  

from constants import *

"""
index = faiss.IndexFlatL2(dim)
index.add(embeddings)  # Add vectors to the Faiss index
faiss.write_index(index, "indexBody")
"""

def find_param(param_list_dir,param):

    for filename in os.listdir(param_list_dir):
        file_path = os.path.join(param_list_dir, filename)

        #filename = Body.txt
        if os.path.isfile(file_path):
            if filename[:-4] == param:
                with open(file_path, "r") as file:
                    loaded_list = file.read().split(",")
                return loaded_list


def embed_value_from_param(vector_database_dir,param_list_dir,params:list):

    model = get_model()
    current_time = datetime.now().strftime('_%H_%M')
    for param in params:
        
        embeddings = model.encode(find_param(param_list_dir,param))

        database_dir = r"vector_database/" + str(param) 
        with open(database_dir, "wb") as file:  # "wb" mode writes in binary
            pickle.dump(embeddings, file) 
    
    with open(vector_database_dir + "date", "w") as file:  # "wb" mode writes in binary
        file.write(current_time)


if __name__ == "__main__":
    embed_value_from_param(VECTOR_DATABASE_DIR,UNIQUE_VALUES_DIR,PARAMS)
