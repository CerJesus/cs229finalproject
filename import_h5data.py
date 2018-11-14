import h5py
import numpy as np
import pandas as pd

ERROR_REJECTION_DT = 0.5
ERROR_REJECTION_VEL = 10
ERROR_REJECTION_SD = 3.4
slope_coef = 940.9411
slope_pow = -.8097

f = h5py.File('../../CS229/ssi71.h5','r')
start_index = 3368179 #1529169147562
end_index = 3460119 #1529173744553
n_sensors = 4
sampling_rate = 20 #picks one in every 20 ratings

altitude_barometer = np.array(f['df'].get('block1_values')[2])[::sampling_rate]
raw_pressure = np.array(f['df'].get('block1_values')[start_index:end_index,45:49]).T[::,::sampling_rate]
time = np.array(f['df'].get('axis1')[start_index:end_index]).T[::sampling_rate]
temp = np.array(f['df'].get('block1_values')[start_index:end_index,7]).T[::sampling_rate]
press = np.array(f['df'].get('block1_values')[start_index:end_index,49]).T[::sampling_rate]

print(np.size(temp))

last_accepted_pressure = np.zeros((4))
last_accepted_time = np.zeros((4))
#bmp_enabled = np.zeros((4))


def velocity_check(df_index, raw_pressure, time):
    if df_index == 0:
        for j in range(0, n_sensors):
            last_accepted_pressure[j] = raw_pressure[j][df_index]
            last_accepted_time[j] = time[df_index]
        return np.ones((4))

        #set defaults
    accepts = np.zeros((4))
    for i in range(0,n_sensors):
        slope = slope_coef*np.power(last_accepted_pressure[i],slope_pow)
        ts = ERROR_REJECTION_DT + (time[df_index] - last_accepted_time[i])
        #print(raw_pressure[i])
        vel = slope*(abs(raw_pressure[i][df_index]- last_accepted_pressure[i]) - ERROR_REJECTION_SD*5)/ts
        if (vel <= ERROR_REJECTION_VEL): #and bmp_enabled[i]):
            #if velocity is within threshold and sensor enabled, accept
            accepts[i] = 1
            last_accepted_time[i] = time[df_index]
            last_accepted_pressure[i] = raw_pressure[i][df_index]
            #smooth the data that is not rejected

    return accepts

# accepted = np.zeros((n_sensors, end_index - start_index))
# print(accepted)
# for k in range(0, end_index - start_index):
#     new_accepts = velocity_check(k, raw_pressure, time)
#     for j in range(0, n_sensors):
#         accepted[j][k] = new_accepts[j]
#         if new_accepts[j] == 1:
#             print(time[k])

accepted = np.ones(len(time))

def set_labels(arr, start_time, a, b):
    a_index = int((a - start_time)/(50*20))
    b_index = int((b - start_time)/(50*20))
    print(a_index, b_index)
    arr[a_index:b_index] = np.zeros((b_index - a_index))
    return(arr)

start_time = 1529169147562

bad_time_ranges = [
(1529170069945, 1529170089030),
(1529170260176, 1529170319689),
(1529170941689, 1529171168560),
(1529171797418, 1529171852313),
(1529172292538, 1529172336343),
(1529172859979, 1529172991451),
(1529173465386, 1529173516745)
]

for i in range(0, len(bad_time_ranges)):
    accepted = set_labels(accepted, start_time, bad_time_ranges[i][0], bad_time_ranges[i][1])

training_set = pd.DataFrame(columns=["time", "temperature", "pressure", "press_change", "label"])
training_set["time"] = time
training_set["temperature"] = temp
training_set["pressure"] = press
training_set["press_change"] = [press[i] - press[i-1] if i > 0 else 0 for i in range(0,len(press))]
training_set["label"] = accepted

training_set.to_csv("ssi_pressure_labels.csv")

"""

        time            index
start   1529169147562   3368179
end     1529173744553   3460119

labelling: these are *time* ranges where sensor readings are bad:
1529170069945 - 1529170089030
1529170260176 - 1529170319689
1529170941689 - 1529171168560
1529171797418 - 1529171852313
1529172292538 - 1529172336343
1529172859979 - 1529172991451
1529173465386 - 1529173516745

features:
- previous X readings
- previous altitudes
- raw_temp
- press ()
"""


"""
FEATURES:

time
altitude_gps
altitude_barometer
last_accepted
last_rejected


"""
