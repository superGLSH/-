import pandas as pd
import numpy as np


data = pd.read_csv('task-1-stocks.csv')
sts = np.array(data.columns.values)
data = np.array(data.iloc[:])
n = data.shape[0]
tdata = data.transpose()
rs = ((np.roll(tdata, -1) - tdata) / tdata)[:, :-1]
rsM = rs.sum(1) / (n - 1)
rs = rs[rsM > 0]
sts = sts[rsM > 0]
rsM = rsM[rsM > 0]
rsD = np.array([np.sqrt((n - 1) / (n - 2) * np.sum(np.square(rs[i] - rsM[i]))) for i in range(rsM.shape[0])])
df = pd.DataFrame(np.array((rsM, rsD)), columns=sts)
df.to_csv('preprocessed.csv', index=None)
