import pandas as pd
import os 
import numpy as np, scipy as sp
from matplotlib import pyplot as plt 
import seaborn as sns

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

# Import Cambium data and define combustion emissions factor
df_cambium = pd.read_csv(os.path.join(cwd,"emissionsfactors_longformat.csv"),
                         header = 0)
datadf['year'] = df_cambium['year']
datadf['MidCase_co2e'] = df_cambium['MidCase_co2e'] # kg per MWh
datadf['95by2050_co2e'] = df_cambium['95by2050_co2e'] # kg per MWh

natgas_co2e = 53.06 # kg per million Btu

# Define financial information
dollarPerkWh = 0.08
dollarPerMBtu = 9.78
GHX_DollarPerft = 33.96
HeatPmp_DollarPerTon = 1849
BldgSide_DollarPersf = 22
discount_rate = 0.05  # Discount rate for investment
escalation_rate = 0.03 # Escalation rate for utility costs


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

# Import power information for baseline case air cooled chiller (already in kW)
p_baseline_chlr = pd.read_csv(os.path.join(cwd,'baseline_chlr_power.csv'), header=0)
datadf['p_chlr_baseline'] = p_baseline_chlr['p_chlr_baseline']

# Make some calculations for QC checks
datadf['COP_clg'] = datadf['q_cool']/3.412/datadf['p_hp_clg']
datadf['COP_htg'] = datadf['q_heat']/3.412/datadf['p_hp_htg']
datadf['COP_hrc'] = ((datadf['q_cool_htrec'] + datadf['q_heat_htrec'])/3.412/
                     datadf['p_hp_dhr'])

#add a column for date - having it start at 1/1/2026 00:00
datadf['date'] = pd.to_datetime(datadf['Period'],unit='h', origin='2025-12-31 23:00')


print(datadf.head())
hold=0

# Calculate hourly co2 emissions

datadf['GSHP_co2e_MidCase'] = ((datadf['p_hp_hc'] + datadf['p_hp_dhr'])*
                               datadf['MidCase_co2e']*0.001 + datadf['q_boiler']
                               *1.25*1.25*natgas_co2e*0.001)
# Assuming 80% boiler efficiency and 20% distribution losses

datadf['GSHP_co2e_95by2050'] = ((datadf['p_hp_hc'] + datadf['p_hp_dhr'])*
                               datadf['95by2050_co2e']*0.001 + datadf['q_boiler']
                               *1.25*natgas_co2e*0.001)
datadf['baseline_co2e_MidCase'] = (0.001 * datadf['p_chlr_baseline'] * datadf['MidCase_co2e'] + 
                                   0.001 * datadf['q_heat'] * 1.25 * natgas_co2e)




# Calculate hourly operational costs

datadf['OPEX_baseline'] = (datadf['p_chlr_baseline']*dollarPerkWh + 
                           datadf['q_heat']*0.001*1.25*dollarPerMBtu)
# Only doing 20% distribution losses because the cost of steam reported
# already factors in the boilers themselves

datadf['OPEX_GSHP'] = ((datadf['p_hp_hc'] + datadf['p_hp_dhr'] + 
                        datadf['p_pump_BHE'])*dollarPerkWh)

annual = datadf.groupby('year')

'''
Some QC checks - plot COPs of heating and cooling heat pumps and heat recovery
chiller, some descriptive statistics
'''

clg_QC = datadf.loc[datadf['p_hp_clg']!=0,('year', 't_GHX_out', 't_chw_s', 
                                           'q_cool', 'p_hp_clg', 'COP_clg')]

htg_QC = datadf.loc[datadf['p_hp_htg']!=0,('year', 't_GHX_out', 't_hw_s',
                                           'q_heat', 'p_hp_htg', 'COP_htg')]

htrec_QC = datadf.loc[datadf['p_hp_dhr']!=0, ('year','p_hp_dhr','q_cool_htrec',
                                              'q_heat_htrec', 'COP_hrc')]

plt.scatter(clg_QC['t_GHX_out'],clg_QC['COP_clg'], marker = '+', s=1)
plt.scatter(clg_QC['q_cool'],clg_QC['COP_clg'], marker = '+', s=1)
sns.kdeplot(clg_QC['COP_clg'], cumulative=True)
clg_QC['COP_clg'].describe(percentiles=[.05,.1,.9,.95])

plt.scatter(htg_QC['t_GHX_out'],htg_QC['COP_htg'], color='r', marker = '+', s=1)
plt.scatter(htg_QC['q_heat'],htg_QC['COP_htg'], color='r', marker = '+', s=1)
sns.kdeplot(htg_QC['COP_htg'], cumulative=True)
htg_QC['COP_htg'].describe(percentiles=[.05,.1,.9,.95])


htrec_QC['cop_cool_hrc'] = htrec_QC['q_cool_htrec']/3.412/htrec_QC['p_hp_dhr']
htrec_QC['cop_heat_hrc'] = htrec_QC['q_heat_htrec']/3.412/htrec_QC['p_hp_dhr']
htrec_QC['total_clg_htg'] = htrec_QC['q_cool_htrec'] - htrec_QC['q_heat_htrec']
htrec_QC['COP_hrc'] = htrec_QC['total_clg_htg']/3.412/htrec_QC['p_hp_dhr']
plt.scatter(htrec_QC['q_cool_htrec'], htrec_QC['cop_cool_hrc'], marker='+')
plt.scatter(htrec_QC['q_heat_htrec'], htrec_QC['cop_cool_hrc'], marker='+')
plt.scatter(htrec_QC['total_clg_htg'], htrec_QC['COP_hrc'], marker='+')


'''
Collate and plot results: Greenhouse gas and financial
'''

# GHG results
plotresults_ghg = pd.DataFrame(index=annual.groups.keys(), columns = 
                              ['baseline_co2e_MidCase',
                               'GSHP_co2e_MidCase',
                               'GSHP_co2e_95by2050'])

emissions_lifetime = {'Baseline': np.sum(annual['baseline_co2e_MidCase'].sum())/1000,
                      'GSHP_MidCase': np.sum(annual['GSHP_co2e_MidCase'].sum())/1000,
                      'GSHP_95by2050': np.sum(annual['GSHP_co2e_95by2050'].sum())/1000}

for c in plotresults_ghg.columns:
        plotresults_ghg[c] = annual[c].sum()/1000 # Convert to metric tons
        
#Filter down to 2026, 2035, and 2050 for ghg plot
plotresults_ghg = plotresults_ghg.loc[[2026, 2035, 2050]] 
baseline = plotresults_ghg['baseline_co2e_MidCase']
gshp_midcase = plotresults_ghg['GSHP_co2e_MidCase']
gshp_95by2050 = plotresults_ghg['GSHP_co2e_95by2050']

x = np.arange(1,4)
w = 0.25

fig, ax = plt.subplots(layout='constrained')
ax.bar(x - w, baseline,
        width = w, label = 'Baseline', color='r')
ax.bar(x, gshp_midcase, width=w, label='Geothermal Mid Case')
ax.bar(x + w, gshp_95by2050, width=w, label='Geothermal 95% Decarb', color='g')
ax.set_ylabel('Annual emissions [metric tons CO2e]')
ax.set_xticks(x, ['2025', '2035', '2050'])
ax.legend(loc='best', ncols=1)


# Financial results
plotresults_financial = pd.DataFrame(index = annual.groups.keys(), columns = 
                                     ['OPEX_baseline', 'OPEX_GSHP'])
for c in plotresults_financial.columns:
    plotresults_financial[c] = annual[c].sum()
    
plotresults_financial.loc[:, 'net_savings'] = plotresults_financial.loc[:, 'OPEX_baseline']-plotresults_financial.loc[:, 'OPEX_GSHP']

# Add initial costs in 2025
plotresults_financial.loc[2025] = [749171, 853227, -104056]
plotresults_financial.sort_index(inplace=True)
plotresults_financial['cumulative'] = np.zeros(26)
plotresults_financial.loc[2025,'cumulative'] = plotresults_financial.loc[2025,'net_savings']

for i in list(range(2026,2051)):
    plotresults_financial.loc[i, 'cumulative'] = \
    plotresults_financial.loc[i-1, 'cumulative'] + \
    plotresults_financial.loc[i, 'net_savings']

    
    
# Apply discount and escalation factors

def present_value(money, i, j, e, n):
    return money*((1+e)**n)*((1+j)**n)*((1+i)**(-n))

'''
for n0, n1 in list(zip(list(range(2026,2051)), list(range(1,26)))):
    for c in plotresults_financial.columns:
        (plotresults_financial.loc[n0, c] = 
         plotresults_funancial.loc[n0, c]*(1+discount_rate)**n1*())
'''
    
fig1, ax1 = plt.subplots(layout='constrained') 

ax1.plot(plotresults_financial.index, 
         plotresults_financial['cumulative'])
ax1.yaxis.set_major_formatter('${x:,.0f}')
ax1.yaxis.set_tick_params(which='major')
ax1.grid(visible=True, which='major')


                  
         