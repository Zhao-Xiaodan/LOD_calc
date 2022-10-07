# Code source: Jaques Grobler
# License: BSD 3 clause

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

'''
2022-08-30 using repeats for 4 times, so take mean 4 wells for 1 test
2022-10-06 one 1 well for 1 test
'''
conc = np.loadtxt('2022-08-30_POCT_Conc.txt')
#  conc = np.loadtxt('concentration.txt')
SMS = np.loadtxt('2022-08-30_POCT_data.txt')
selectNum = -1

#  import pdb; pdb.set_trace()
conc = conc[0:selectNum]
SMS = SMS[0:selectNum*4, :]
#  conc = conc[selectNum:]
#  SMS = SMS[selectNum:, :]

print(conc)
#  print(SMS)
#  import pdb; pdb.set_trace()

concData = np.tile(np.array([conc]), (1, 16))
#  concData = np.tile(np.array([conc]).transpose(), (1, 16))
print(concData)

breakpoint()
#  print(f"transpose(1,4){ concData }")
SMS = SMS.flatten().reshape(-1, 1)
#  SMS = SMS.flatten()
#  print(f"flatten and reshape{SMS}")
#  breakpoint()
concData = concData.flatten().reshape(-1, 1)
#  import pdb; pdb.set_trace()

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
#  regr.fit(diabetes_X_train, diabetes_y_train)
regr.fit(concData, SMS)
#  import pdb; pdb.set_trace()

# Make predictions using the testing set
SMS_y_pred = regr.predict(concData)

dev = SMS - SMS_y_pred
SD = np.mean(dev ** 2)
LOD = 3.3 * SD / regr.coef_
LOD = LOD[0][0]

# the LOD 
print('LOD: \n', LOD)
# The coefficients
print("Coefficients: \n", regr.coef_)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(concData, SMS))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(SMS, SMS_y_pred))

# Plot outputs
plt.style.use('seaborn')

plt.scatter(concData, SMS, color="black")
plt.plot(concData, SMS_y_pred, color="blue", linewidth=3)
plt.xlabel('Spike Concentration (pM)')
plt.text(np.mean(conc), max(SMS), 'LOD: 3.3x$\u03C3$/slop = '+ ' '+ str(round(LOD, 2)),\
          horizontalalignment='center', size='x-large', color='k', weight='semibold')

plt.ylabel('Bead number per 100 X 100 $\mu$$m^2$')
#  plt.xticks(())
#  plt.yticks(())
#  import pdb; pdb.set_trace()

plt.show()

