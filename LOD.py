# Code source: Jaques Grobler
# License: BSD 3 clause

import math
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
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
#  conc = conc.reshape(-1, 1)
nRow = len(conc)

data_1st = np.loadtxt('2022-08-30_POCT_data.txt')
data_2nd = np.loadtxt('2022-10-06_CR3022 Spike-in in RIPA_2nd.txt')
data_3rd = np.loadtxt('2022-10-06_CR3022 Spike-in in RIPA_3rd.txt')

data_1st = dataMean(data_1st, nRow)
data_2nd = dataMean(data_2nd, nRow)
data_3rd = dataMean(data_3rd, nRow)

data = np.hstack((data_1st, data_2nd, data_3rd))

selectNum = 4
data = data[0:selectNum]
conc = conc[0:selectNum]

#  print(conc)
conc = [math.log(x) for x in conc]
#  print(conc)

#  breakpoint()
data = data.transpose()
data = data.reshape(-1, 1)
#  data = data.reshape(-1, 1)
conc = np.tile(np.array([conc]), (1, 3))
conc = conc.reshape(-1, 1)

# ------ linear regression -------
regr = linear_model.LinearRegression()
# Train the model using the training sets
#  regr.fit(diabetes_X_train, diabetes_y_train)
regr.fit(conc, data)
# Make predictions using the testing set
data_y_pred = regr.predict(conc)

dev = data - data_y_pred
SD = math.sqrt(np.mean(dev ** 2) / len(data))
LOD = 3.3 * SD / regr.coef_
LOD = LOD[0][0]

print(len(data))
print(SD)
breakpoint()
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
plt.scatter(conc, data, color="C0")
#  breakpoint()
plt.plot(conc, data_y_pred, color="C1", linewidth=3)
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
