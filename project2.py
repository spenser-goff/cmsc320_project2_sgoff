# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 15:34:29 2021

@author: spens
"""
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statistics 
import math
from statistics import median
import sqlite3, pandas as pd
#Create a database and connect to it
sqlite_file = 'lahman2014.sqlite'
conn = sqlite3.connect(sqlite_file)
cur = conn.cursor()


#Create a database and connect to it
sqlite_file = 'lahman2014.sqlite'
conn = sqlite3.connect(sqlite_file)


# Part 1
# Problem 1
#Use multiple left joins which means payroll data before 1985 will be missing
#Leave as NaN so it can be filled later if data is found 
query = """WITH tempTable
AS (WITH tempTable2 
    AS (SELECT Salaries.yearID, Salaries.teamID, SUM(Salaries.salary) as total_payroll
        FROM Salaries
        GROUP BY Salaries.yearID, Salaries.teamID)
    SELECT Teams.yearID, Teams.teamID, Teams.franchID, tempTable2.total_payroll, (CAST(Teams.W as REAL) / Teams.G * 100) as win_per, Teams.W, Teams.G
    FROM Teams
    LEFT JOIN tempTable2
    ON tempTable2.yearID = Teams.yearID AND tempTable2.teamID = Teams.teamID)
SElECT Teams.yearID, Teams.franchID, Teams.teamID, tempTable.total_payroll, tempTable.win_per
FROM Teams
LEFT JOIN tempTable
ON (tempTable.yearID = Teams.yearID AND tempTable.teamID = Teams.teamID)"""

df = pd.read_sql(query, conn)

#Part 2
#Problem 4
time_periods = pd.cut(np.array([i for i in range (1990,2015)]), 5)
time_periods = time_periods.unique()
temp = df.franchID.tolist()
franchID_periods = []
for per in time_periods:
    franch_lst2 = []
    lst = []
    for ind in df.index:
        if df['yearID'][ind] in per:
           lst.append(df['franchID'][ind])    
    [franch_lst2.append(x) for x in lst if x not in franch_lst2]
    franchID_periods.append(franch_lst2)
df_mast = []
years = []
df_90to14 = df.loc[df['yearID'] > 1989]
for i in range(5):
    df_temp = pd.DataFrame()
    franch_temp = []
    
    mean_pay_temp = []
    mean_win_temp = []
    per = time_periods[i]
    iv = str(math.ceil(per.left)) + " - " + str(math.floor(per.right))
    years.append(iv)
    
    for franch in franchID_periods[i]:
        franch_temp.append(franch)
        franch_per_pay_lst = []
        franch_per_win_lst = []
        for index, row in df_90to14.iterrows():
            if row['yearID'] in per and row['franchID'] == franch:
                franch_per_pay_lst.append(row['total_payroll'])                
                franch_per_win_lst.append(row['win_per'])
        mean_pay = sum(franch_per_pay_lst) / len(franch_per_pay_lst)
        mean_win = sum(franch_per_win_lst) / len(franch_per_win_lst)
        mean_pay_temp.append(mean_pay)
        mean_win_temp.append(mean_win)

    df_temp['franchID'] = franch_temp
    df_temp['mean_payroll'] = mean_pay_temp
    df_temp['mean_win_per'] = mean_win_temp
    df_mast.append(df_temp)

for i in range(5):
    df_temp = df_mast[i]
    sns.lmplot(x="mean_payroll", y="mean_win_per", data=df_temp, fit_reg=False, hue='franchID', legend=False).set(title=years[i])
    for i in range(df_temp.shape[0]):
        plt.text(x=df_temp.mean_payroll[i]+0.3,y=df_temp.mean_win_per[i]+0.3,s=df_temp.franchID[i], 
          fontdict=dict(color='red',size=8),
          bbox=dict(facecolor='yellow',alpha=0.5))
    sns.regplot(x="mean_payroll", y="mean_win_per", data=df_temp)    
    
    
#Problem 3
#Part 5
df_90to14 = df.loc[df['yearID'] > 1989]
df_90to14 = df_90to14.reset_index()
df_90to14 = df_90to14.drop(columns=['index'])
avg_payroll_by_year = []
std_dev_payroll_by_year = []
years = [i for i in range(1990,2015)]
for year in years:
    std_dev_list = []
    count = 0
    payroll_acc = 0
    for index, row in df_90to14.iterrows():
        if row['yearID'] == year:
            std_dev_list.append(row['total_payroll'])
            payroll_acc += row['total_payroll']
            count += 1
    avg_payroll_by_year.append(payroll_acc / count)
    std_dev_payroll_by_year.append(statistics.stdev(std_dev_list))
df_std_payroll = pd.DataFrame()
df_std_payroll['Year'] = years
df_std_payroll['Average Payroll'] = avg_payroll_by_year
df_std_payroll['Std Dev Payroll'] = std_dev_payroll_by_year

def standardized_payroll(tot_payroll, avg_payroll, stdev_payroll):
    return ((tot_payroll - avg_payroll) / stdev_payroll)
std_payroll_lst = []
for i,row1 in df_std_payroll.iterrows(): 
    for j, row2 in df_90to14.iterrows():
        if row1['Year'] == row2['yearID']:
            std_payroll_lst.append(standardized_payroll(row2['total_payroll'], row1['Average Payroll'], row1['Std Dev Payroll']))
            
df_90to14['standardized_payroll'] = std_payroll_lst
df_90to14

#Part 3
#Problem 7
array = df_90to14.to_numpy()
y = array[:,4]
y = y.astype(str).astype(float)
x = array[:,5]
x = x.astype(str).astype(float)
p1 = np.polyfit(x, y, 1)
p1

exp_wins_lst = []
for ind, row in df_90to14.iterrows():
    exp_wins_lst.append(50 + 2.7 * row['standardized_payroll'])
df_90to14['exp_wins'] = exp_wins_lst
    

#Part 3
#Problem 8
eff_lst = []
for ind, row in df_90to14.iterrows():
    eff_lst.append(row['win_per'] - row['exp_wins'])
df_90to14['eff'] = eff_lst

df_90to14_oak =  df_90to14.loc[(df_90to14['franchID'] == "OAK")]
df_90to14_bos =  df_90to14.loc[(df_90to14['franchID'] == "BOS")]
df_90to14_atl =  df_90to14.loc[(df_90to14['franchID'] == "ATL")]
df_90to14_tbd =  df_90to14.loc[(df_90to14['franchID'] == "TBD")]
df_90to14_nyy =  df_90to14.loc[(df_90to14['franchID'] == "NYY")]
frames = [df_90to14_oak, df_90to14_bos, df_90to14_atl, df_90to14_tbd, df_90to14_nyy]
result = pd.concat(frames)
result = result.reset_index()
result = result.drop(columns=['index'])
df_90to14 = df.loc[df['yearID'] > 1989]
result_temp1 = result.loc[result['yearID'] >= 2000]
result_temp2 = result.loc[result['yearID'] <= 2005]
#merged_inner = pd.merge(left=survey_sub, right=species_sub, left_on='species_id', right_on='species_id')
result_inner = pd.merge(left=result_temp1, right=result_temp2, left_on='yearID', right_on='yearID')
result_inner = result_inner.drop(columns=['exp_wins_x','teamID_x', 'franchID_x', 'total_payroll_x', 'win_per_x', 'standardized_payroll_x', 'eff_x'] )
result_inner = result_inner.drop_duplicates()

sns.lmplot(x="yearID", y="eff_y", data=result_inner, hue="franchID_y", fit_reg=False)
sns.lmplot(x="yearID", y="eff", data=result, hue="franchID", fit_reg=False)
#plt.show()/
result_most_eff = result.loc[result['eff'] > 11]
result_most_eff = result_most_eff.reset_index()
result_most_eff = result_most_eff.drop(columns=['index'])