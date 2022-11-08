# Code source: Jaques Grobler
# License: BSD 3 clause

import math
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

'''
2022-08-30 using repeats for 4 times, so take mean 4 wells for 1 test
2022-10-06 one 1 well for 1 test
'''
def dataMean(data, nRow):
    data = data.reshape(nRow, -1)
    data = np.mean(data, axis=1)
    data = data.reshape(-1, 1)
    return data

conc = np.loadtxt('2022-08-30_POCT_Conc.txt')
conc2 = np.loadtxt('2022-11-01_POCT_Conc.txt')
conc2 = np.flipud(conc2)
#  conc = conc.reshape(-1, 1)
nRow = len(conc)
nRow2 = len(conc2)

data_1st = np.loadtxt('2022-08-30_POCT_data.txt')
data_2nd = np.loadtxt('2022-10-06_CR3022 Spike-in in RIPA_2nd.txt')
data_3rd = np.loadtxt('2022-10-06_CR3022 Spike-in in RIPA_3rd.txt')
data_4th = np.loadtxt('2022-11-01_Antibody 20X CR3022 spike-in_1st.txt')
data_5th = np.loadtxt('2022-11-01_Antibody 20X CR3022 spike-in_2nd.txt')
data_6th = np.loadtxt('2022-11-01_Antibody 20X CR3022 spike-in_3rd.txt')


data_1st = dataMean(data_1st, nRow)
data_2nd = dataMean(data_2nd, nRow)
data_3rd = dataMean(data_3rd, nRow)
data_4th = dataMean(data_4th, nRow2)
data_5th = dataMean(data_5th, nRow2)
data_6th = dataMean(data_6th, nRow2)

dic1 = { 'Concentration': conc, 'Signal': data_1st.flatten(), 'File':1 }
dic2 = { 'Concentration': conc, 'Signal': data_2nd.flatten(), 'File':2}
dic3 = { 'Concentration': conc, 'Signal': data_3rd.flatten(), 'File':3}
dic4 = { 'Concentration': conc2, 'Signal': data_4th.flatten(), 'File':4}
dic5 = { 'Concentration': conc2, 'Signal': data_5th.flatten(), 'File':5}
dic6 = { 'Concentration': conc2, 'Signal': data_6th.flatten(), 'File':6}

df1 = pd.DataFrame(dic1)
df2 = pd.DataFrame(dic2)
df3 = pd.DataFrame(dic3)
df4 = pd.DataFrame(dic4)
df5 = pd.DataFrame(dic5)
df6 = pd.DataFrame(dic6)

frames = [df4, df5, df6]
#  frames = [df1, df2, df3, df4, df5, df6]
df = pd.concat(frames)
df = df.loc[~((df['Concentration'] == 1e-2) |
              (df['Concentration'] == 100) |
              (df['Concentration'] == 500))]
df['conc_log'] = np.log(df['Concentration'])
df = df.sort_values(by=['File'])

#  breakpoint()
# ------ linear regression -------
regr = linear_model.LinearRegression()
# Train the model using the training sets
#  regr.fit(diabetes_X_train, diabetes_y_train)
X_train = df['conc_log'].values.reshape(-1, 1)
Y_train = df['Signal'].values.reshape(-1, 1)
regr.fit(X_train, Y_train)
# Make predictions using the testing set
data_y_pred = regr.predict(X_train)

dev = Y_train - data_y_pred
SD = math.sqrt(np.mean(dev ** 2))
LOD = 3.3 * SD / regr.coef_
LOD = LOD[0][0]

#  fmri = sns.load_dataset("fmri")
#  fmri.head()
#  sns.lineplot(data=fmri, x="timepoint", y="signal", hue="event")
#  breakpoint()
#  sns.lineplot(data=df, x='Concentration', y='Signal', hue='File')
print(df)
df.reset_index(inplace=True)
print(df)
sns.lineplot(data=df, x='Concentration', y='Signal')
#  sns.lineplot(data=df, x="Concentration", y="Signal",
             #  units='File', markers=True, lw=0)
            #  x=byColumn, y="Titer", hue="vaccineCurrent", marker='o',
            #  markers=True, dashes=False,units="Participant", estimator=None, lw=0,
            #  markersize=8


#  pv = df.pivot(index='Concentration', columns='File', values='Signal')
#  flights_wide = flights.pivot("year", "month", "passengers")
pv = df.pivot('Concentration', 'File', 'Signal')
#  pv = df.stack()
print(pv)
#  sns.lineplot(data=df, x='Concentration', y='Signal')
#  sns.lineplot(data=pv, x='Concentration', y='Signal')
#  sns.lineplot(data=pv, x='Concentration', y=4)
plt.show()
breakpoint()
sns.set_theme(style='white')
fig, ax = plt.subplots()
#  ax.set(xscale="log", yscale='log')
#  sns.lineplot(data=df, x='Concentration', y='Signal',
             #  marker='o', ci='sd', ax=ax)
plt.scatter(df['Concentration'], df['Signal'], color='C0')
plt.plot(df['Concentration'], data_y_pred, color="C1", linewidth=3)
plt.title('LOD: 3.3x$\u03C3$/slop = '+ ' '+ str(round(LOD, 2)) + ' pM')
plt.xlabel('Spike Concentration (pM)')
plt.ylabel('Bead number per 100 X 100 $\mu$$m^2$')
#  plt.xscale('log')
#  ax.set_xlim(0.01, 25)
ax.set_xscale('log')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
#  ax.spines['bottom'].set_visible(False)
#  ax.spines['left'].set_visible(False)
plt.show()

breakpoint()
data = np.hstack((data_1st, data_2nd, data_3rd))

data2 = np.array(data).copy()
conc2 = np.array(conc).copy()

selectNum = 4
data = data[0:selectNum]
conc = conc[0:selectNum]
conc2 = conc2[0:selectNum]
data2 = data2[0:selectNum]

#  breakpoint()
#  print(conc)
conc = [math.log(x) for x in conc]
#  print(conc)

data = data.transpose()
data = data.reshape(-1, 1)
#  data = data.reshape(-1, 1)
conc = np.tile(np.array([conc]), (1, 3))
conc = conc.reshape(-1, 1)

breakpoint()
# ------ linear regression -------
regr = linear_model.LinearRegression()
# Train the model using the training sets
#  regr.fit(diabetes_X_train, diabetes_y_train)
regr.fit(conc, data)
# Make predictions using the testing set
data_y_pred = regr.predict(conc)

dev = data - data_y_pred
SD = math.sqrt(np.mean(dev ** 2))
LOD = 3.3 * SD / regr.coef_
LOD = LOD[0][0]

print(len(data))
print(SD)
#  breakpoint()
conc = [math.exp(x) for x in conc]
print(conc)
# the LOD 
print('LOD: \n', LOD)
# The coefficients
print("Coefficients: \n", regr.coef_)
# Plot outputs

sns.set_theme(style='darkgrid')
print(conc)
#  conc = conc.flatten()
print(conc)
print(data)
#  plt.scatter(conc, data, color="C0")
#  breakpoint()

#  plt.boxplot(data2.transpose(), labels=conc2)
plt.errorbar(conc2, np.mean(data2, axis=1), yerr=np.std(data2,axis=1))
#  breakpoint()
#  plt.plot(conc, data_y_pred, color="C1", linewidth=3)
plt.title('LOD: 3.3x$\u03C3$/slop = '+ ' '+ str(round(LOD, 2)) + ' pM')
plt.xlabel('Spike Concentration (pM)')
#  plt.text(np.mean(conc), max(data), 'LOD: 3.3x$\u03C3$/slop = '+ ' '+ str(round(LOD, 2)),\
          #  horizontalalignment='center', size='x-large', color='k', weight='semibold')

plt.ylabel('Bead number per 100 X 100 $\mu$$m^2$')
#  plt.xticks(())
#  plt.yticks(())
#  import pdb; pdb.set_trace()
plt.xscale('log')
plt.show()

#  breakpoint()
