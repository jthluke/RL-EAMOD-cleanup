import json
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import pickle
import math

od_data = np.load('od_matrix.npy')
duration_data = np.load('duration_matrix.npy')
distance_data = np.load('distance_matrix.npy')
# remove transit zones
od_data = od_data[:-3, :-3, :]
duration_data = duration_data[:-3, :-3, :]
distance_data = distance_data[:-3, :-3, :]
# set remaing nans to avg
avg_duration = int(np.nanmean(duration_data))
duration_data[np.isnan(duration_data)] = avg_duration
avg_distance = int(np.nanmean(distance_data))
distance_data[np.isnan(distance_data)] = avg_distance

delta_c = 1.25 # energy step [kWh] 0.75
time_granularity = 0.25 # in h
cost_per_mile = 2.75 # in $
cost_per_hour = 33 # in $ as per https://www.taxi-calculator.com/taxi-rate-san-francisco/271
cost_per_timestep = cost_per_hour * time_granularity
beta = 0.6 # in $ according to https://www.bls.gov/regions/west/news-release/averageenergyprices_sanfrancisco.htm#:~:text=San%20Francisco%20area%20households%20paid,per%20therm%20spent%20last%20year -> gives cost for kwh (0.30$/kwh)) + 0.30$ for maintenance 
chevy_bolt_capacity = 60 # in kWh
chevy_bolt_usable_capacity = chevy_bolt_capacity * 0.6 # never go below 20% or above 80% of charge
charger_capacity = 22.5 # assuming 22.5 kW Chargers
charge_time_per_delta_c = math.ceil((delta_c/charger_capacity)/time_granularity)
chevy_bolt_range = 230 # range in mi for mild city trips according to https://media.chevrolet.com/media/us/en/chevrolet/2022-bolt-euv-bolt-ev.detail.html/content/Pages/news/us/en/2021/feb/0214-boltev-bolteuv-specifications.html
chevy_bolt_usable_range = chevy_bolt_range*0.6*0.9 # never go below 20% or above 80% of charge and assume 10% less efficient because of range https://cleantechnica.com/2017/10/13/autonomous-cars-shorter-range-due-high-power-consumption-computers/
chevy_bolt_kwh_per_mi = chevy_bolt_usable_capacity/chevy_bolt_usable_range
# print(chevy_bolt_kwh_per_mi)
energy_distance = np.ceil(((distance_data * chevy_bolt_kwh_per_mi)/delta_c).max(axis=2))
energy_distance[energy_distance==0] = 1 # we should always use energy to satisfy a trip
np.save('energy_distance.npy', energy_distance)
# print(np.sum(energy_distance==1.))
# print(energy_distance.max())
duration_data = np.round(duration_data/(3600*time_granularity)) # convert travel time from sec to h
duration_data[duration_data==0] = 1. # it should always take time to satisfy a trip
data_timespan = od_data.shape[2]
episode_length = int(data_timespan/time_granularity)
fleet_size = 116616 # got number from Justin Lukes optimization with boundary:283905, without boundary:116616
number_chargelevels = int(chevy_bolt_usable_capacity/delta_c)
number_spatial_nodes = 190
print(number_chargelevels)

new_tripAttr = []
new_reb_time = []
new_total_acc = []
for origin in range(duration_data.shape[0]):
    for destination in range(duration_data.shape[1]):
        for ts in range(episode_length):
            attr = defaultdict()
            attr['time_stamp'] = ts
            attr['origin'] = origin
            attr['destination'] = destination
            attr['demand'] = round(od_data[origin,destination,int(ts*time_granularity)]*time_granularity) # create equal distributed demand over granular time
            attr['price'] = cost_per_mile * distance_data[origin,destination,int(ts*time_granularity)] + cost_per_timestep * duration_data[origin,destination,int(ts*time_granularity)]
            new_tripAttr.append(attr)

            reb = defaultdict()
            reb['time_stamp'] = ts
            reb['origin'] = origin
            reb['destination'] = destination
            reb['reb_time'] = int(duration_data[origin,destination,int(ts*time_granularity)])
            new_reb_time.append(reb)

for hour in range(24):
    acc = defaultdict()
    acc['hour'] = hour
    acc['acc'] =  math.ceil(fleet_size)
    new_total_acc.append(acc)
new_data = defaultdict()
new_data['demand'] = new_tripAttr
new_data['rebTime'] = new_reb_time
new_data['totalAcc'] = new_total_acc
new_data['chargelevels'] = number_chargelevels
new_data['spacialNodes'] = number_spatial_nodes
new_data['chargeTime'] = charge_time_per_delta_c
new_data['episodeLength'] = episode_length
print(episode_length)
print(number_chargelevels)

with open('SF_190.json', 'w') as f:
    json.dump(new_data, f)