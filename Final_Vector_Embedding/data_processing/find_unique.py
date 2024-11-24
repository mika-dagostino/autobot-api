import pandas as pd
import pickle


from constants import *

# Load the CSV file
df = pd.read_csv(CSV_DIR)


"""
["Year",
"Make",
"Model",
"Body",
"Doors",
"ExteriorColor",
"InteriorColor",
"EnginerCylinders",
"Transmission",
"Miles",
"SellingPrice",
"MarketClass",
"PassengerCapacity",
"Drivetrain",
"Engine_Description",
"CityMPG",
"HighwayMPG"]
"""


from constants import *
print(PARAMS)


def determine_unique_values():
    for col in PARAMS:
        unique_values = df[col].unique().tolist()
        if not unique_values:
            ValueError()
        unique_values_str = [str(item) for item in unique_values]

        file_name = r"unique_values_param/" + col + ".txt"
        df[col] = df[col].fillna(0) 


        with open(file_name, "w") as file:

            for i in range(0,len(unique_values_str) - 1):  
                if unique_values[i]:
                    if unique_values[i] == "nan":
                        continue
                    file.write(str(unique_values[i]) + ",")

            if unique_values[len(unique_values)-1]:
                if unique_values[len(unique_values) - 1] == "nan":
                        continue
                file.write(str(unique_values[len(unique_values)-1]))

        # print("There are " + str(len(unique_values)) + "Unique values:" + str(unique_values))

if __name__ == "__main__":
    determine_unique_values()