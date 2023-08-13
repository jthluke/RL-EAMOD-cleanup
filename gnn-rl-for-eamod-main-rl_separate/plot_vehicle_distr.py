import pickle
import numpy as np
import matplotlib.pyplot as plt
import math
import wandb

problem_folder = 'SF_5_clustered'
clusters = 5
episode_length = 48
time_granularity = 0.25
charge_levels = 26

# problem_folder = 'Toy'
# clusters = 2
# episode_length = 20
# time_granularity = 1
# charge_levels = 5

api = wandb.Api()
run = api.run("e-amod/e-amod/2q19uiuy")
run.file(f"./saved_files/ckpt/{problem_folder}/acc.p").download(replace=True)
run.file(f"./saved_files/ckpt/{problem_folder}/acc_spatial.p").download(replace=True)
run.file(f"./saved_files/ckpt/{problem_folder}/new_charging_vehicles.p").download(replace=True)
run.file(f"./saved_files/ckpt/{problem_folder}/n_customer_vehicles_spatial.p").download(replace=True)
run.file(f"./saved_files/ckpt/{problem_folder}/new_rebalancing_vehicles.p").download(replace=True)
run.file(f"./saved_files/ckpt/{problem_folder}/satisfied_demand.p").download(replace=True)



charge_algo = "off_peak_one_step_charging_"
# charge_algo = "empty_to_full_"
spatial_algo = "uniform_distr"
# spatial_algo = "rl_heur"
# charge_algo = "mpc_48"
# spatial_algo = ""
solution_approach = charge_algo  + spatial_algo
with open(f"./saved_files/ckpt/{problem_folder}/acc.p", "rb") as file:
        acc = pickle.load(file)
with open(f"./saved_files/ckpt/{problem_folder}/acc_spatial.p", "rb") as file:
        acc_spatial = pickle.load(file)
with open(f"./saved_files/ckpt/{problem_folder}/new_charging_vehicles.p", "rb") as file:
    new_charging_vehicles = pickle.load(file)
with open(f"./saved_files/ckpt/{problem_folder}/new_rebalancing_vehicles.p", "rb") as file:
    new_rebalancing_vehicles = pickle.load(file)
with open(f"./saved_files/ckpt/{problem_folder}/satisfied_demand.p", "rb") as file:
    satisfied_demand = pickle.load(file)

od_data = np.load('data/'+problem_folder+'/od_matrix.npy')
od_data = od_data[:,:,8:-4]
try:
    charging_stations = np.load('data/'+problem_folder+'/charging_stations.npy')
except:
    charging_stations = np.ones(clusters)*6000

# problem_folder = 'SF_5_clustered_sparsified'
# charging_stations *= 1.2
# od_data[:3,:,:] = (0.95*od_data[:3,:,:]).astype(np.int32)
# od_data[3:,:,:] *= 0

# problem_folder += "/flip"
# od_data = np.zeros((2,2,20))
# od_data[1,0,:10] = 90
# od_data[0,1,10:] = 90
# try:
#     charging_stations = np.load('data/'+problem_folder+'/charging_stations.npy')
# except:
#     charging_stations = np.ones(clusters)*6000

# for t in range(episode_length+1):
#     baseline_value = 93240
#     for n in range(clusters):
#         acc_check = 0
#         for c in range(charge_levels):
#             print(n,c,t)
#             acc_check += acc[n,c][t]
#         assert acc_check - acc_spatial[n][t] < 1
# assert False
sum_rebal = 0
customer_temp = 0
for n in range(clusters):
    y_values_idle = [0]
    y_values_charging = []
    y_values_charging_stations = []
    y_values_rebal = []
    y_values_customer = []
    y_values_customer_cars = []
    y_values_sum = [0]
    y_values_demand = [0]
    x_values = []
    partial_sum = 0
    sum_demand = 0
    for t in range(episode_length+1):
        y_values_charging.append(new_charging_vehicles[n][t])
        y_values_charging_stations.append(charging_stations[n])
        y_values_rebal.append(new_rebalancing_vehicles[n][t])
        sum_rebal += new_rebalancing_vehicles[n][t]
        partial_sum += new_rebalancing_vehicles[n][t]
        y_values_customer.append(satisfied_demand[n][t])
        if t<episode_length:
            y_values_sum.append(acc_spatial[n][t])
            y_values_demand.append(round(0.95*od_data[n,:,int((t)*time_granularity)].sum()*time_granularity))
            y_values_idle.append(acc_spatial[n][t]-new_charging_vehicles[n][t+1]-new_rebalancing_vehicles[n][t+1]-satisfied_demand[n][t+1])
        x_values.append(t)
    plt.plot(x_values, y_values_idle, label = "idling")
    plt.plot(x_values, y_values_charging, label = "charging")
    plt.plot(x_values, y_values_charging_stations, label = "charging stations")
    plt.plot(x_values, y_values_rebal, label = "rebal")
    plt.plot(x_values, y_values_customer, label = "customer")
    # plt.plot(x_values, y_values_sum, label = "sum")
    plt.plot(x_values, y_values_demand, label = "demand")
    plt.xlabel("Time")
    plt.ylabel("Number Vehicles")
    plt.title("Region " + str(n) + " " + solution_approach)
    plt.legend()
    # plt.show()
    plt.ylim([0, 25000])
    plt.savefig(f"./plots/{problem_folder}/{episode_length}/{solution_approach}/node_{n}.png")
    # print(f"./plots/{problem_folder}/{episode_length}/{solution_approach}/node_{n}.png")
    plt.close()

for n in range(clusters):
    # for c in range(charge_levels):
    for c in range(3):
        y_values_charge_level = [0]
        x_values = []
        for t in range(episode_length+1):
            if t<episode_length:
                y_values_charge_level.append(acc[(n,c)][t])
            x_values.append(t)
        plt.plot(x_values, y_values_charge_level, label = f"charge: {c}")
    plt.xlabel("Time")
    plt.ylabel("Number Vehicles")
    plt.title("Region " + str(n) + " " + solution_approach)
    plt.legend()
    # plt.show()
    plt.savefig(f"./plots/{problem_folder}/{episode_length}/{solution_approach}/charge_distr_node_{n}.png")
    # print(f"./plots/{problem_folder}/{episode_length}/{solution_approach}/node_{n}.png")
    plt.close()