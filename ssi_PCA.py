import h5py
import numpy as np
import pandas as pd
import sys
from sklearn.decomposition import PCA

import matplotlib as mpl
mpl.use('TkAgg')

import matplotlib.pyplot as plt

f = h5py.File('../../CS229/ssi71.h5','r')

#Use sys parameters if you want to import a different time range! If not, defaults are given.
if len(sys.argv) < 4:
    print("Parameters: <output_file.csv> <start_index> <end_index> <start_time>")
    print(len(sys.argv))
    output_file = "ssi_pressure_labels.csv"
    start_index = 3368179 #1529169147562
    end_index = 3460119 #1529173744553
    start_time = 1529169147562
    #start_time = 1529175944249
else:
    print("sys:", sys.argv)
    output_file = sys.argv[1]
    start_index = int(sys.argv[2])
    end_index = int(sys.argv[3])
    start_time = int(sys.argv[4])

n_sensors = 4
sampling_rate = 20 #picks one in every 20 ratings

altitude_barometer = np.array(f['df'].get('block1_values')[2])[::sampling_rate]
raw_pressure = np.array(f['df'].get('block1_values')[start_index:end_index,45:49]).T[::,::sampling_rate]
time = np.array(f['df'].get('axis1')[start_index:end_index]).T[::sampling_rate]
temp = np.array(f['df'].get('block1_values')[start_index:end_index,7]).T[::sampling_rate]
press = np.array(f['df'].get('block1_values')[start_index:end_index,49]).T[::sampling_rate]
other_features = raw_pressure = np.array(f['df'].get('block1_values')[start_index:end_index,2:22]).T[::,::sampling_rate]


def is_bad_time(timestamp, time_ranges, shouldp):
    if shouldp:
        print(timestamp)
        print(time_ranges[0][0])
    for time_range in time_ranges:
        if timestamp > time_range[0] and timestamp < time_range[1]:
            return True
    return False

def set_labels(arr,bad_times, time_arr):
    for i in range(len(time_arr)):
        if is_bad_time(time_arr[i],bad_times, i == 892):
            arr[i]=0
    print(arr[892])
    return arr

# """ ssi_pressure_labels.csv"""
bad_time_ranges = [
(1529170069945000000, 1529170089030000000),
(1529170260176000000, 1529170319689000000),
(1529170941689000000, 1529171168560000000),
(1529171797418000000, 1529171852313000000),
(1529172292538000000, 1529172336343000000),
(1529172859979000000, 1529172991451000000),
(1529173465386000000, 1529173516745000000)
]
accepted = np.ones(len(time))
accepted = set_labels(accepted,bad_time_ranges,time)

training_set = pd.DataFrame(columns=["time", "temperature", "pressure"])
training_set["time"] = time
training_set["temperature"] = temp
training_set["pressure"] = press

k = 5

for j in range(1,k):
    name = "press_change" + str(j)
    training_set[name] = [abs(press[i] - press[i-j]) if i > k - 1 else 0 for i in range(0,len(press))]

for i in range(0, len(other_features)):
    training_set["feature" + str(i + 2)] = other_features[i]



# training_set["press_change1"] = [abs(press[i] - press[i-1]) if i > 0 else 0 for i in range(0,len(press))]
# training_set["press_change2"] = [abs(press[i] - press[i-2]) if i > 1 else 0 for i in range(0,len(press))]
# training_set["press_change3"] = [abs(press[i] - press[i-3]) if i > 2 else 0 for i in range(0,len(press))]
training_set["var"] = [np.var(press[j-5:j]) if j >= 5 else 0 for j in range(len(press))]

from sklearn import preprocessing
data_scaled = pd.DataFrame(preprocessing.scale(training_set),columns = training_set.columns)

# PCA
pca = PCA(n_components=.80)
pca.fit_transform(data_scaled)

components = pd.DataFrame(pca.components_, columns=data_scaled.columns)
explained_var = pca.explained_variance_/sum(pca.explained_variance_)

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print (components)
    print(explained_var)

for row in range(0, len(components)):
    max_comp = components.idxmax(1).ix[row]
    print("Component " + str(row) + " with % explained variance " +  str(explained_var[row]) + ":")
    print(max_comp, components[max_comp][row])
    print(components.columns[components.ix[row].argsort()][::-1])

plt.plot(np.cumsum(pca.explained_variance_/sum(pca.explained_variance_)), '--o')
plt.savefig("pca_plot1.png")

print(pca.explained_variance_ratio_)
training_set["label"] = accepted



#training_set.to_csv(output_file)
