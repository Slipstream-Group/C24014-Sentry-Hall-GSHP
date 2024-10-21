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
#strip spaces
datadf.rename(columns=lambda x: x.strip(), inplace=True)
# drop the last column (empty) 
datadf.drop(datadf.columns[len(datadf.columns)-1], axis=1, inplace=True)

nan_df = datadf.isna()
print(nan_df)


hold=0