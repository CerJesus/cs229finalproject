import h5py
import numpy as np
import pandas as pd

ERROR_REJECTION_DT = 0.5
ERROR_REJECTION_VEL = 10
ERROR_REJECTION_SD = 3.4
slope_coef = 940.9411
slope_pow = -.8097

f = h5py.File('../../CS229/ssi71.h5','r')
print(f['df']['block1_values'][3])

altitude_barometer = np.array(f['df'].get('block1_values')[2])
raw_pressure = np.array(f['df'].get('block1_values')[45:48])
time = np.array(f['df'].get('axis1'))
n_sensors = 4

last_accepted_pressure = np.zeros((4))
last_accepted_time = np.zeros((4))
#bmp_enabled = np.zeros((4))


def velocity_check(df_index, raw_pressure, time):
    if df_index == 0:
        for j in range(0, n_sensors):

            last_accepted_pressure[j] = raw_pressure[j]
            last_accepted_time[j] = time[df_index]
        return np.ones((4))

        #set defaults
    accepts = np.zeros((4))
    for i in range(0,n_sensors):
        slope = slope_coef*last_accepted_pressure[i]^slope_pow
        ts = ERROR_REJECTION_DT + (time[df_index] - last_accepted_time[i])
        vel = slope*(abs(raw_pressure[df_index][i] - last_accepted_pressure[i]) - ERROR_REJECTION_SD*5)/ts
        if (vel <= ERROR_REJECTION_VEL and bmp_enabled[i]):
            #if velocity is within threshold and sensor enabled, accept
            accepts[i] = 1
            last_accepted_time[i] = time[df_index]
            last_accepted_pressure[i] = raw_pressure[df_index][i]
            #smooth the data that is not rejected

    return accepts

accepted = np.zeros((len(raw_pressure)))
for k in range(0, len(time)):
    new_accepts = velocity_check(k, raw_pressure, time)
    print(new_accepts)
    for j in range(0, n_sensors):
        accepted[k][j] = new_accepts[j]

print(accepted)

"""
FEATURES:

time
altitude_gps
altitude_barometer
last_accepted
last_rejected


"""
