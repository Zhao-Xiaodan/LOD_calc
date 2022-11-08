# Code source: Jaques Grobler
# License: BSD 3 clause
from os import listdir, path
from pathlib import Path

import math
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.utils import resample

'''
read data from expData folder
'''
def calc_LOD(df):
    """
    using linear regression
    """
    df['conc_log'] = np.log(df['Concentration'])
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
    return LOD, data_y_pred

def resample_pdFormat(df):
    """
    using sklearn library
    """
    df_wide = df.pivot(columns='Concentration', values='Signal')
    df_wide = df_wide.apply(lambda x: pd.Series(x.dropna().values))

    #  boot = resample(data, replace=True, n_samples=4, random_state=1)

    col = df_wide.columns

    boot = [np.nan] * len(col)
    for i in range(len(col)):
        data = df_wide[col[i]].values
        boot[i] = resample(data, replace=True,
                           n_samples=len(data),
                           random_state=None)

    df_boot = pd.DataFrame(np.array(boot).transpose())
    df_boot.columns = col
    df_melt = pd.melt(df_boot, value_name='Signal')
    return df_melt

def dataMean(data, nRow):
    data = data.reshape(nRow, -1)
    data = np.mean(data, axis=1)
    data = data.reshape(-1, 1)
    return data

# load txt data and its conc condition
rawData_ls = []
infoData_ls = []
basepath = Path(__file__).parent
#  basepath = basepath / "expData_all"
basepath = basepath / "expData_part"

for file in listdir(basepath):
    if file.endswith(".txt"):
        rawData_ls.append(file)
    elif file.endswith(".xlsx"):
        infoData_ls.append(file)

rawData_ls.sort()
infoData_ls.sort()

data = []
conc = []
fileNum = []
for i, file in enumerate(rawData_ls):
    fileName = rawData_ls[i].split(".")[0]
    #  fileName = rawData_ls[i].strip(".txt")
    if fileName != infoData_ls[i].strip(".xlsx"):
        print(f"File name is wrong with {file} or {infoData_ls[i]}")
        break

    rawData = np.loadtxt(path.join(basepath, rawData_ls[i]))
    testInfo = pd.read_excel(path.join(basepath, infoData_ls[i]),
                         header=None, engine="openpyxl").values
    # nRow for 08-30 4 well 1 POCT for one condition
    nRow = len(testInfo)
    rawData = rawData.reshape(nRow, -1)
    meanData = np.mean(rawData, axis=1)
    for info in testInfo:
        c = info[0].split("pM")[0]
        c = float(c)
        conc.append(c)
    temp = [f"File_{i}"] * nRow
    fileNum.append(temp)

    data.append(meanData.tolist())
#  # something not correct with the reason beyond my understand
#  data = np.array(data).flatten().tolist()
#  fileNum = np.array(fileNum).flatten().tolist()
data = sum(data, [])
fileNum = sum(fileNum, [])

dic = {'Concentration': conc, 'Signal': data, 'FileNum': fileNum}
df = pd.DataFrame(dic)


df = df.loc[~((df['Concentration'] == 1e-2) |
              (df['Concentration'] == 100) |
              (df['Concentration'] == 500))]
df = df.sort_values(by=['FileNum'])

#  N = 10000
N = 100
LOD_ls = []
for i in range(N):
    df_melt = resample_pdFormat(df)
    LOD, data_y_pred = calc_LOD(df_melt)
    LOD_ls.append(LOD)

LOD_mean = np.mean(LOD_ls)
LOD_std = np.std(LOD_ls)
print(f"LOD mean: {LOD_mean}, LOD std: {LOD_std}")
breakpoint()

# ------ plot ---------
LOD, data_y_pred = calc_LOD(df)
sns.set_theme(style='white')
fig, ax = plt.subplots(figsize=(3,2))
#  ax.set(xscale="log", yscale='log')
sns.lineplot(data=df, x="Concentration", y="Signal",
             err_style="bars", marker='o', markersize=10, lw=0)
sns.scatterplot(data=df, x='Concentration', y='Signal')
#  sns.lineplot(data=df, x='Concentration', y='Signal', marker='o', lw=0)
             #  marker='o', markersize=10, ci='sd', lw=0, ax=ax)
plt.plot(df['Concentration'], data_y_pred, color="C1", linewidth=2)
#  plt.title('LOD: 3.3x$\u03C3$/slop = '+ ' '+ str(round(LOD, 2)) + ' pM')
plt.xlabel('CR3022 Concentration (pM)')
#  plt.ylabel('Bead number per 100 X 100 $\mu$$m^2$')
#  plt.xscale('log')
#  ax.set_xlim(0.01, 25)
ax.set_xscale('log')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
#  ax.spines['bottom'].set_visible(False)
#  ax.spines['left'].set_visible(False)

#  plt.savefig("LOD_inset.svg")
plt.show()
# --- Bootstraping -------





breakpoint()
