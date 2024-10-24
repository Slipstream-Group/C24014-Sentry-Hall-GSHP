import pandas as pd
import os 
import numpy as np 

""" Take the output file from the TRNSYS model and turn it into a Pandas dataframe"""
#--------------------------------------------------------
#Import the CSV file that has been written from TRNSYS 
cwd=os.getcwd()
file=os.path.join(cwd,"results_hrly.out")
datadf=pd.read_csv(file, skiprows=[0], sep='\t', header=0, nrows=int(8760*25)) 
#datadf=pd.read_csv(file, skiprows=[0], sep='\t', header=0, skipfooter=23, engine='python') 
#strip spaces -----------------------
datadf.rename(columns=lambda x: x.strip(), inplace=True)
#drop the last column (empty) -------------
datadf.drop(datadf.columns[len(datadf.columns)-1], axis=1, inplace=True)

#nan_df = datadf.isna()
#print(nan_df)


#Unit Conversions
for column in datadf.columns:
    #convert Powers from kJ/hr to kW
    if column.startswith("p_"):
        datadf[column] = datadf[column] * 0.0002777778
    #convert heat transfer from kJ/hr to kBtu/hr
    elif column.startswith("q_"):
        datadf[column] = datadf[column] * 0.0009478171
    #convert temps from C to F
    elif column.startswith("t_"):
        datadf[column] = datadf[column].apply(lambda x: (x * 9/5) + 32)

#add a column for date - having it start at 1/1/2023 00:00, but can change it 
#datadf['date'] = pd.to_datetime(datadf['Period'],unit='h', origin='2022-12-31 23:00')



print(datadf.head())
hold=0