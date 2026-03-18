import random
import numpy as np
from matplotlib.pyplot import cm
import csv
from decimal import Decimal
from Configuration import Configuration
from Environment import Environment
from HttpClient import HttpClient
from CmdManager import CmdManager
from DdqnAgent import DoubleDeepQNetwork
import matplotlib.pyplot as plt
import shutil
import argparse
import tensorflow as tf
import os
import hydra
from omegaconf import DictConfig, OmegaConf
import mlflow
import optuna
import time
from datetime import datetime
import plotly.express as px

# Copies a CICFlowMeter output file to a new location.
# This is used to store the network flow metrics collected during an experiment step
def copy_cic_step_file(config, path_to_save, episode, step):
    try:
        cic_file_name = f"Episdode {episode} - Step {step} - CIC results.csv"
        destination = os.path.join(path_to_save, "cic", cic_file_name)
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        shutil.copyfile(config.cic_output_file_path, destination)
    except Exception as e:
        print(f"(Warning) Could not copy CIC file: {e}")




# Randomly selects an attack type from a predefined list of attacks.
def get_attack_type():
    # available_attacks = ["ICMP", "TCP", "UDP", "SYN"] # TDOO: Uncomment if random attack type
    available_attacks = ["ICMP"] # TODO: for testing purposes, currently just using ICMP attacks
    attack_type_index = random.randint(0, len(available_attacks) - 1)
    return available_attacks[attack_type_index]

def get_basic_metrics_headers():
    headers = ["tx_bytes",
               "rx_bytes",
               "bandwidth",
               "tx_packets",
               "rx_packets",
               "tx_packets_len",
               "rx_packets_len",
               "delivered_pkts",
               "loss_pct",
               "is_connected",
               "pkts_s",
               "bytes_s"]
    return headers

def get_network_metrics_headers():
    headers = ["avg_latency_s",
               "avg_packet_transmission_time_s",
               "throughput_bps",
               "avg_jitter_s"]
    return headers

SWITCHES_BW_HEADERS = None

def save_file_with_headers(filepath, data, headers, fmt='%.18e'):
    with open(filepath, 'w') as result_file:
        wr = csv.writer(result_file)
        wr.writerow(headers)
        np.savetxt(result_file, data, delimiter=',', fmt=fmt)


# Checks the network state data for anomalies such as NaN, infinite values, or negative metrics.
# If any issues are found, a warning file is generated to log the detected problems for further inspection.
def generate_warning_file_if_necessary(config, file_name, new_state):
    headers = get_basic_metrics_headers()
    headers.remove("bandwidth")
    warnings = ""
    for host in new_state ['host'].keys():
        for header in headers:
            val = new_state['host'][host][header]
            if np.insan(val):
                warnings += f"\nINSAN: new_state['host'][{host}][{header}]={val}"
            elif np.isinf(val):
                warnings += f"\nISINF: new_state['host'][{host}][{header}]={val}"
            elif val < 0:
                 warnings += f"\nNEGATIVE: new_state['host'][{host}][{header}]={val}"
    if warnings:
        warning_file = f"{config.current_train_folder}/{file_name}"
        with open(warning_file, 'w') as f:
            f.write(warnings)

def setup_directories(base_dir):
    ##Creates the folder structure for a training run
    for folder in ["data", "figs", "cic", "configs"]:
        os.makedirs(os.path.join(base_dir, folder), exist_ok=True)


def save_episode_plots(current_run_dir, episode, ep_rews, ep_lats, ep_loss, ep_jits, episode_hosts_bw):
    ##Generates and saves all PNG and HTML plots for a given episode
    def save_plot_png(data, title, color, filename):
        plt.figure(figsize=(10,6))
        plt.plot(data, color=color, label=title)
        plt.title(f"Episode {episode} - {title}")
        plt.xlabel("Steps")
        plt.ylabel("Value")
        plt.legend()
        plt.savefig(os.path.join(current_run_dir, "figs", f"Episode_{episode}_{filename}.png"))

        plt.close()


    save_plot_png(ep_rews,  "Reward",      "blue",   "Reward")
    save_plot_png(ep_lats,  "Latency",     "green",  "Latency")
    save_plot_png(ep_loss,  "Packet Loss", "orange", "Packet_Loss")
    save_plot_png(ep_jits,  "Jitter",      "purple", "Jitter")


    # Multi-line bandwidth plot
    plt.figure(figsize=(10, 6))
    for host, values in episode_hosts_bw.items():
        plt.plot(values, label=f"Host {host}")
    plt.title(f"Episode {episode} - Hosts Bandwidth")
    plt.xlabel("Steps")
    plt.ylabel("bps")
    plt.legend(loc='upper right', fontsize='small')
    plt.savefig(os.path.join(current_run_dir, "figs",
                             f"Episode_{episode}_Hosts_BW.png"))
    plt.close() 


     # Interactive Plotly HTML
    fig_lat = px.line(y=ep_lats,
                      title=f"Interactive Latency Ep {episode}",
                      labels={'y': 'Latency (s)', 'x': 'Steps'})
    fig_lat.write_html(os.path.join(current_run_dir, "figs",
                                    f"Episode_{episode}_latency_interactive.html"))



## CORE EXPERIMENT FUNCTION


def run_experiment(cfg:DictConfig, trial=None):

    with mlflow.start_run(nested=True):
        ## Parameter initilization
        if trial:
            w_lat = trial.suggest_float("w_latency", 0.1, 5.0)
            w_jit = trial.suggest_float("w_jitter", 0.1, 5.0)
            lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

            gamma = trial.suggest_float("gamma", 0.7, 0.99)
            epsilon_decay = trial.suggest_float("epsilon_decay", 0.990, 0.9999)
        else:
            w_lat = cfg.reward_weights.w_latency
            w_jit = cfg.reward_weights.w_jitter
            lr = cfg.hyperparameters.learning_rate
            gamma = cfg.hyperparameters.get("gamma", 0.5)
            epsilon_decay = cfg.epsilon_decay
        
        mlflow.log_params({
            "w_lat": w_lat, "w_jit": w_jit, "lr":lr, "gamma": gamma, "epsilon_decay": epsilon_decay,
        })
        mlflow.set_tag("topology", cfg.hosts_topo)
        mlflow.set_tag("mlflow.note.content", f"QoS run on {cfg.hosts.topo} | lr={lr:.5f}")

        ## Folder setup

        timestamp       = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        current_run_dir = f"results/train_{timestamp}"
        setup_directories(current_run_dir)

        config_path = os.path.join(current_run_dir, "configs", "config_used.yaml")
        with open(config_path, "w") as f:
            f.write(OmegaConf.to_yaml(cfg))
        mlflow.log_artifact(config_path)



        ### ENVIRONMENT & AGENT INIT

        config = Configuration(
            cfg.hosts_topo, cfg.episodes, cfg.steps,
            epsilon_decay,               # uses tuned value when in tune mode
            cfg.nbr_controlled_switches
        )
        env         = Environment(config)
        cmd         = CmdManager(config)
        http_client = HttpClient(config)

        # FIX 1 — learning_rate and gamma are passed to the DDQN constructor
        ddqn_agent = DoubleDeepQNetwork(
            config, env, http_client,
            is_controlled=False,
            is_prefilled_actions=False,
            gamma=gamma,
            learning_rate=lr          # ← was missing in original code
        )

        total_rewards_per_episode = []

        try:
            for episode in range(1, config.episodes + 1):

                tot_episode_reward = 0
                current_state      = env.reset()
                env.update_hosts()

                episode_hosts_bw = {h: [] for h in env.hosts}
                ep_rews, ep_lats, ep_jits, ep_loss = [], [], [], []
                ep_sw_bw_matrix = []
                sw_headers      = None

                env.perform_setup(http_client, cfg.get('attackers', []))
                ddqn_agent.set_actions(env.ACTIONS)
                cmd.start_network_in_background(
                    env.servers, env.attacker_hosts,
                    config.hosts_topo_file_name,
                    config.nbr_controlled_switches
                )

                print(f"(RL) Episode {episode} - Starting simulation...")
                time.sleep(20)

                env.update_hosts_ips(http_client)
                env.update_interfaces(http_client.get_switches_interfaces())
                tshark_ids = env.get_tshark_interfaces_ids(cmd)

                sender_receiver_relation  = {h: random.choice(env.servers)        for h in env.normal_hosts}
                attacker_victim_relation  = {a: random.choice(env.victim_servers)  for a in env.attacker_hosts}
                attack_types              = {a: get_attack_type()                  for a in env.attacker_hosts}

                current_state = env.get_state(
                    config, cmd, http_client, tshark_ids,
                    sender_receiver_relation, attacker_victim_relation, attack_types
                )

                # --------------------------------------------------------------
                # STEP LOOP
                # --------------------------------------------------------------
                for step in range(1, config.steps + 1):

                    state_vec          = env.transform_state_dict_to_normalized_vector(current_state)
                    action, is_predicted = ddqn_agent.action(step, state_vec)

                    new_state, base_reward, done, loss_val, delay, latency, jitter = \
                        env.apply_action_controlled_switches(
                            config, cmd, http_client, tshark_ids,
                            sender_receiver_relation, attacker_victim_relation,
                            attack_types, action, is_predicted
                        )

                    # FIX 4: log raw components so reward scale issues are visible
                    reward_val        = base_reward - (w_lat * latency) - (w_jit * jitter)
                    tot_episode_reward += reward_val

                    global_step = (episode - 1) * config.steps + step

                    mlflow.log_metric("latency_step",     latency,          step=global_step)
                    mlflow.log_metric("jitter_step",      jitter,           step=global_step)
                    mlflow.log_metric("packet_loss_pct",  loss_val * 100,   step=global_step)
                    mlflow.log_metric("base_reward_step", base_reward,      step=global_step)
                    mlflow.log_metric("reward_step",      reward_val,       step=global_step)
                    # FIX 4: log penalty terms individually so scaling problems
                    # are immediately visible in the MLflow chart
                    mlflow.log_metric("penalty_latency",  w_lat * latency,  step=global_step)
                    mlflow.log_metric("penalty_jitter",   w_jit * jitter,   step=global_step)

                    ep_rews.append(reward_val)
                    ep_lats.append(latency)
                    ep_jits.append(jitter)
                    ep_loss.append(loss_val)

                    for h in env.hosts:
                        val_bw = new_state['host'].get(h, {}).get('bandwidth', 0)
                        episode_hosts_bw[h].append(val_bw)

                    # Switch bandwidth matrix
                    sw_row, current_sw_headers = [], []
                    for category in ['routing', 'controlled']:
                        for src in new_state[category]:
                            for dst in new_state[category][src]:
                                sw_row.append(new_state[category][src][dst]['bw'])
                                if sw_headers is None:
                                    current_sw_headers.append(f"{src}->{dst}")
                    ep_sw_bw_matrix.append(sw_row)
                    if sw_headers is None:
                        sw_headers = current_sw_headers

                    copy_cic_step_file(config, current_run_dir, episode, step)

                    ddqn_agent.store(
                        state_vec, action, reward_val,
                        env.transform_state_dict_to_normalized_vector(new_state),
                        done
                    )

                    if len(ddqn_agent.memory) > ddqn_agent.batch_size:
                        ddqn_agent.experience_replay(ddqn_agent.batch_size)

                    current_state = new_state
                    if done:
                        break

                # --------------------------------------------------------------
                # END OF EPISODE
                # --------------------------------------------------------------
                print(f"(RL) Episode {episode} finished. Saving results...")

                sw_bw_path = os.path.join(current_run_dir, "data",
                                          f"Episode_{episode}_switches_bw.csv")
                if ep_sw_bw_matrix and sw_headers:
                    np.savetxt(sw_bw_path, np.array(ep_sw_bw_matrix),
                               delimiter=",",
                               header=",".join(sw_headers),
                               comments='')

                save_episode_plots(current_run_dir, episode,
                                   ep_rews, ep_lats, ep_loss,
                                   ep_jits, episode_hosts_bw)

                total_rewards_per_episode.append(tot_episode_reward)
                mlflow.log_metric("episode_total_reward",
                                  tot_episode_reward, step=episode)

                cmd.stop_network()

            # FIX 5 — log_artifacts called ONCE after all episodes, not inside the loop
            mlflow.log_artifacts(current_run_dir)

        except Exception as e:
            mlflow.set_tag("status", "FAILED")
            mlflow.log_param("error_message", str(e))
            print(f"CRITICAL ERROR: {e}")
            try:
                cmd.stop_network()
            except Exception:
                pass
            raise   # FIX 6 — bare raise preserves the original traceback

        avg_reward = float(np.mean(total_rewards_per_episode))
        mlflow.log_metric("final_average_reward", avg_reward)
        mlflow.set_tag("status", "OK")
        return avg_reward


# ---------------------------------------------------------------------------
# MAIN ENTRY POINT
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):

    # FIX 2 — called exactly once, here at startup
    mlflow.enable_system_metrics_logging()
    mlflow.set_experiment("QoSentry_Optimization")

    if cfg.mode == "tune":
        print("\n[INFO] Starting Optuna autotuning...")

        # FIX 3 — parent run opened HERE so that nested=True inside
        #          run_experiment actually creates a proper parent-child hierarchy
        with mlflow.start_run(run_name="optuna_study"):
            mlflow.set_tag("mode", "tune")
            mlflow.log_param("n_trials", cfg.tune_trials)

            study = optuna.create_study(direction="maximize")
            study.optimize(
                lambda trial: run_experiment(cfg, trial),
                n_trials=cfg.tune_trials
            )

            best = study.best_trial
            print(f"\n[RESULT] Best trial #{best.number}  avg_reward={best.value:.4f}")
            for k, v in best.params.items():
                print(f"  {k}: {v}")
                mlflow.log_param(f"best_{k}", v)
            mlflow.log_metric("best_avg_reward", best.value)

        # Save best params as YAML for subsequent training runs
        with open("best_params.yaml", "w") as f:
            OmegaConf.save(config=OmegaConf.create(study.best_params), f=f)
        print("[INFO] Best params saved to best_params.yaml")

    else:
        print("\n[INFO] Starting standard training...")
        # Single run — no nesting needed
        with mlflow.start_run(run_name="standard_train"):
            mlflow.set_tag("mode", "train")
            run_experiment(cfg)

    print("\n[FINISH] Run completed.")


if __name__ == '__main__':
    main()

# # The main block initializes the reinforcement learning environment and manages the experiment workflow.
# # Key steps include:
# # - Parsing command-line arguments for experiment configurations.
# # - Initializing the network environment, RL agent, and other components.
# # - Running multiple training episodes where the RL agent interacts with the environment.
# # - Logging results and visualizing metrics such as packet loss, delay, and bandwidth usage
# if __name__ == '__main__':

#     # Parses command-line arguments to allow the user to customize the experiment setup.
#     # Configurable parameters include the number of episodes, steps, and the epsilon decay rate.
#     # Validation checks ensure the parameters are within acceptable ranges.
#     parser = argparse.ArgumentParser(description="Main",
#                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#     parser.add_argument("-a", "--attackers", help="Attacker hosts names. E.g: [h1]", required=False)
#     parser.add_argument("-e", "--episodes", help="Number of episodes. E.g: 50", required=False)
#     parser.add_argument("-s", "--steps", help="Number of steps. E.g: 100", required=False)
#     parser.add_argument("-ed", "--epsilon-decay", help="Epsilon decay. E.g: 0.999", required=False)
#     parser.add_argument("-ncs", "--nbr-controlled-switches", help="The number of controlled switches in the network", required=False)
#     parser.add_argument("-c", "--controlled", action="store_true",
#                         help="Whether to control action taking")
#     parser.add_argument("-pfa", "--prefilled-actions", action="store_true",
#                         help="Whether to use prefilled actions, from file 'prefilled-actions.txt'")
#     parser.add_argument("-htf", "--hosts-topo-file",
#                         help="When given, the provided JSON file in the 'input-data' folder will be used. E.g: hosts-topology-6hosts",
#                         required=False, default="hosts-toplogy-6hosts")
#     # Initializes the simulation environment and its components.
#     # This involves setting up the network topology, controlled switches, and hosts.
#     config = vars(parser.parse_args())
#     is_controlled = config['controlled']
#     is_prefilled_actions = config['prefilled_actions']
#     if is_controlled and is_prefilled_actions:
#         raise Exception("Please use either '--controlled' flag or '--prefilled-actions' flag, but not both!")
#     pre_set_attackers = []
#     if not (config['attackers'] is None or config['attackers'] == '' or config['attackers'] == '[]'):
#         pre_set_attackers = config['attackers'].lstrip("[").rstrip("]").split(',')
#     if is_controlled:
#         print('(Reinforcement) ================> Main Started with "controlled actions"')
#     elif is_prefilled_actions:
#         print('(Reinforcement) ================> Main Started with "prefilled actions"')
#     else:
#         print('(Reinforcement) ================> Main Started')
#     hosts_topo_file_name = 'hosts-toplogy-6hosts.json'
#     if not ('hosts_topo_file' not in config or config['hosts_topo_file'] is None or config[
#         'hosts_topo_file'] == ''):
#         hosts_topo_file_name = config['hosts_topo_file']
#         if not hosts_topo_file_name.lower().endswith(".json"):
#             hosts_topo_file_name += ".json"
#     episodes = 50
#     if not ('episodes' not in config or config['episodes'] is None or config['episodes'] == ''):
#         episodes = int(config['episodes'])
#         print(f'(Reinforcement) ==================> Episodes: {episodes}')
#     steps = 100
#     if not ('steps' not in config or config['steps'] is None or config['steps'] == ''):
#         steps = int(config['steps'])
#         print(f'(Reinforcement) ==================> Steps: {steps}')
#     epsilon_decay = 0.999
#     if not ('epsilon_decay' not in config or config['epsilon_decay'] is None or config[
#         'epsilon_decay'] == ''):
#         epsilon_decay = float(config['epsilon_decay'])
#         if epsilon_decay >= 1 or epsilon_decay <= 0.1:
#             raise Exception("Epsilon decay must be in the range ]0.1, 1[!")
#         print(f'(Reinforcement) ==================> Epsilon decay: {epsilon_decay}')
#     nbr_controlled_switches = 4
#     if not ('nbr_controlled_switches' not in config or config['nbr_controlled_switches'] is None or config['nbr_controlled_switches'] == ''):
#         nbr_controlled_switches = int(config['nbr_controlled_switches'])
#         if nbr_controlled_switches < 4:
#             raise Exception(f"Number of controlled switches set to a ({nbr_controlled_switches}) which is lower than 4. Min value is 4!")
#         if nbr_controlled_switches > 99:
#             raise Exception(f"Number of controlled switches set to a ({nbr_controlled_switches}) which is more than 99. max value is 99!")
#         print(f'(Reinforcement) ==================> Number of controlled switches: {nbr_controlled_switches}')

#     config = Configuration(hosts_topo_file_name, episodes, steps, epsilon_decay, nbr_controlled_switches)
#     env = Environment(config)
#     cmd = CmdManager(config)
#     http_client = HttpClient(config)
#     tot_rewards = 0
#     total_rewards_per_episode = []
#     epsilons = []
#     ddqn_agent = DoubleDeepQNetwork(config, env, http_client, is_controlled, is_prefilled_actions)

#     global_vars_to_print = {
#         "max_attacker": {},
#         "max_host": {},
#         "max_server": {},
#     }
#     for header in get_basic_metrics_headers():
#         global_vars_to_print["max_attacker"][header] = 0
#         global_vars_to_print["max_host"][header] = 0
#         global_vars_to_print["max_server"][header] = 0

#     for header in get_network_metrics_headers():
#         global_vars_to_print["max_host"][header] = 0

#     # Iterates through the specified number of episodes for training the reinforcement learning agent.
#     # Each episode involves multiple interaction steps where the agent takes actions and receives feedback.
#     # Traffic and performance metrics are collected and saved at the end of each episode.
#     for episode in range(1, env.episodes + 1):
#         tot_rewards = 0
#         episode_index = episode - 1
#         current_state = env.reset()

#         episode_rewards = []
#         ddqn_agent.episode_loss = []
#         ddqn_agent.episode_loss = []
#         episode_avg_packet_loss = []
#         episode_avg_real_delays = []
#         episode_avg_latencys = []
#         episode_avg_jitters = []

#         print(f'(Reinforcement) ==================> Episode {episode} Started')

#         env.update_hosts()

#         env.perform_setup(http_client, pre_set_attackers)

#         ddqn_agent.set_actions(env.ACTIONS)

#         cmd.start_network_in_background(env.servers, env.attacker_hosts, config.hosts_topo_file_name, nbr_controlled_switches)

#         env.update_hosts_ips(http_client)

#         env.update_interfaces(http_client.get_switches_interfaces())

#         tshark_interfaces_ids = env.get_tshark_interfaces_ids(cmd)

#         sender_receiver_relation = {}
#         for host in env.normal_hosts:
#             server_index = random.randint(0, len(env.servers) - 1)
#             server = env.servers[server_index]
#             sender_receiver_relation[host] = server

#         attacker_victim_relation = {}
#         attack_types = {}
#         for attacker in env.attacker_hosts:
#             victim_server_index = random.randint(0, len(env.victim_servers) - 1)
#             victim_server = env.victim_servers[victim_server_index]
#             attacker_victim_relation[attacker] = victim_server
#             attack_types[attacker] = get_attack_type()

#         # variables for each host

#         attacker_state_variables = {}
#         for attacker in env.attacker_hosts:
#             cols = env.NBR_HOST_STATE_METRICS + 1
#             attacker_state_variables[attacker] = {
#                 'filename': f'attacker_{attacker}_attackType_{attack_types[attacker]}.csv',
#                 'data': np.empty((env.steps, cols), dtype=object)
#             }
#             attacker_state_variables[attacker]['data'][:, 0:(cols - 1)] = 0.0
#             attacker_state_variables[attacker]['data'][:, (cols - 1)] = ""
#         server_state_variables = {}
#         for server in env.servers:
#             attacker_suffix = ""
#             for attacker in env.attacker_hosts:
#                 if attacker_victim_relation[attacker] == server:
#                     attacker_suffix = f"{attacker_suffix}_attacker_{attacker}_type_{attack_types[attacker]}"
#             server_state_variables[server] = {
#                 'filename': f'server_{server}{attacker_suffix}.csv',
#                 'data': np.zeros((env.steps, env.NBR_HOST_STATE_METRICS))
#             }
#         normal_host_state_variables = {}
#         for host in env.normal_hosts:
#             cols = env.NBR_HOST_STATE_METRICS + env.nbr_of_network_metrics + 1
#             normal_host_state_variables[host] = {
#                 'filename': f'host_{host}.csv',
#                 'data': np.empty((env.steps, cols), dtype=object)
#             }
#             normal_host_state_variables[host]['data'][:, 0:(cols - 1)] = 0.0
#             normal_host_state_variables[host]['data'][:, (cols - 1)] = ""

#         switches_bw_variables = {
#             'filename': f'switches_bw.csv',
#             'data': np.zeros((env.steps, env.nbr_routing_switches + (env.nbr_controlled_switches * env.nbr_controlled_switches)))
#         }

#         episode_hosts_bw = {}
#         for host in env.hosts:
#             episode_hosts_bw[host] = {'data': []}

#         print(f'(Reinforcement) ====================> Init Step Started')

#         new_state = env.get_state(config, cmd, http_client, tshark_interfaces_ids, sender_receiver_relation,
#                                   attacker_victim_relation, attack_types)
#         current_state = new_state
#         env.last_recorded_delay = env.calculate_delay(current_state)
#         env.last_recorded_latency = env.calculate_latency(current_state)
#         env.last_recorded_jitter = env.calculate_jitter(current_state)
#         env.before_last_recorded_delay = env.last_recorded_delay

#         for i in range(1, 1):
#             print(f'(Reinforcement) ====================> Init Step Started - Additional {i}')
#             new_state = env.get_state(config, cmd, http_client, tshark_interfaces_ids, sender_receiver_relation,
#                                       attacker_victim_relation, attack_types)
#             current_state = new_state
#             print(f'(Reinforcement) <==================== Init Step Ended - Additional {i}')
#         print(current_state)

#         print(f'(Reinforcement) <==================== Init Step Ended')

#         # Executes a series of steps within each episode.
#         # During each step, the RL agent selects an action, and the environment updates its state accordingly.
#         for step in range(1, env.steps + 1):

#             # The RL agent selects an action either based on its policy or by exploration (random actions).
#             # The selected action is applied to the environment, which responds with a new state and reward.
#             # The action's effectiveness is evaluated based on the resulting network performance metrics.
#             print(f'(Reinforcement) ====================> Step {step} (of episode {episode}) Started')

#             action, is_predicted = ddqn_agent.action(step, env.transform_state_dict_to_normalized_vector(current_state))

#             new_state, reward, done, avg_packet_loss, avg_real_delays, avg_latency, avg_jitter = env.apply_action_controlled_switches(
#                 config, cmd, http_client, tshark_interfaces_ids, sender_receiver_relation, attacker_victim_relation,
#                 attack_types, action, is_predicted)

#             episode_avg_packet_loss.append(avg_packet_loss)
#             episode_avg_real_delays.append(avg_real_delays)
#             episode_avg_latencys.append(avg_latency)
#             episode_avg_jitters.append(avg_jitter)
#             print(new_state)

#             generate_warning_file_if_necessary(config, f"Episode {episode} - Step {step} - Warning.txt", new_state)

#             tot_rewards += reward
#             episode_rewards.append(reward)

#             ddqn_agent.store(env.transform_state_dict_to_normalized_vector(current_state), action,
#                              reward, env.transform_state_dict_to_normalized_vector(new_state), done)

#             current_state = new_state

#             # Experience Replay
#             if len(ddqn_agent.memory) > ddqn_agent.batch_size:
#                 ddqn_agent.experience_replay(ddqn_agent.batch_size)
#             else:
#                 ddqn_agent.episode_loss.append(1)

#             if done or (step % ddqn_agent.update_target_each == 0):
#                 ddqn_agent.update_target_from_model()

#             do_break = False
#             if done or step == env.steps:
#                 total_rewards_per_episode.append(tot_rewards)
#                 epsilons.append(ddqn_agent.epsilon)
#                 do_break = True

#             step_index = step - 1

#             #############################################################################################################
#             # filling state information of each host in each step in order to be saved in a csv file after each episode #
#             #############################################################################################################
#             for attacker in env.attacker_hosts:
#                 arr = np.zeros(env.NBR_HOST_STATE_METRICS)
#                 i = 0
#                 for header in get_basic_metrics_headers():
#                     arr[i] = new_state['host'][attacker][header]
#                     i = i + 1
#                 attacker_state_variables[attacker]['data'][step_index, 0:env.NBR_HOST_STATE_METRICS] = arr
#                 ####################new_state['host']#####################
#                 for header in get_basic_metrics_headers():
#                     global_vars_to_print['max_attacker'][header] = max(
#                         global_vars_to_print['max_attacker'][header], new_state['host'][attacker][header])
#                 attacker_state_variables[attacker]['data'][step_index, env.NBR_HOST_STATE_METRICS] = str(http_client.get_host_path(attacker).json()['current'])
#             for server in env.servers:
#                 arr = np.zeros(env.NBR_HOST_STATE_METRICS)
#                 i = 0
#                 for header in get_basic_metrics_headers():
#                     arr[i] = new_state['host'][server][header]
#                     i = i + 1
#                 server_state_variables[server]['data'][step_index, 0:env.NBR_HOST_STATE_METRICS] = arr
#                 ####################new_state['host']#####################
#                 for header in get_basic_metrics_headers():
#                     global_vars_to_print['max_server'][header] = max(
#                         global_vars_to_print['max_server'][header], new_state['host'][server][header])

#             ######################################## normal host state variable##############################
#             for normal_host in env.normal_hosts:
#                 arr = np.zeros(env.NBR_HOST_STATE_METRICS)
#                 i = 0
#                 for header in get_basic_metrics_headers():
#                     arr[i] = new_state['host'][normal_host][header]
#                     i = i + 1
#                 normal_host_state_variables[normal_host]['data'][step_index, 0:env.NBR_HOST_STATE_METRICS] = arr
#                 arr = np.zeros(env.nbr_of_network_metrics)
#                 i = 0
#                 for header in get_network_metrics_headers():
#                     arr[i] = new_state['host'][normal_host]['non_server_data']['network_metrics'][header]
#                     i = i + 1
#                 normal_host_state_variables[normal_host]['data'][step_index, env.NBR_HOST_STATE_METRICS:(env.NBR_HOST_STATE_METRICS + env.nbr_of_network_metrics)] = arr
#                 normal_host_state_variables[normal_host]['data'][step_index, env.NBR_HOST_STATE_METRICS + env.nbr_of_network_metrics] = str(http_client.get_host_path(normal_host).json()['current'])
#                 ####################new_state['host']#####################
#                 for header in get_basic_metrics_headers():
#                     global_vars_to_print['max_host'][header] = max(
#                         global_vars_to_print['max_host'][header], new_state['host'][normal_host][header])
#                 for header in get_network_metrics_headers():
#                     global_vars_to_print['max_host'][header] = max(
#                         global_vars_to_print['max_host'][header], new_state['host'][normal_host]['non_server_data']['network_metrics'][header])
#             ######################################## Switches BW variables ##############################
#             if SWITCHES_BW_HEADERS is None:
#                 SWITCHES_BW_HEADERS = []
#                 for src_switch in new_state['routing'].keys():
#                     for dst_switch in new_state['routing'][src_switch].keys():
#                         SWITCHES_BW_HEADERS.append(f"{src_switch} -> {dst_switch}")
#                 for src_switch in new_state['controlled'].keys():
#                     for dst_switch in new_state['controlled'][src_switch].keys():
#                         SWITCHES_BW_HEADERS.append(f"{src_switch} -> {dst_switch}")
#             arr = np.zeros(env.nbr_routing_switches + (env.nbr_controlled_switches * env.nbr_controlled_switches))
#             i = 0
#             for src_switch in new_state['routing'].keys():
#                 for dst_switch in new_state['routing'][src_switch].keys():
#                     arr[i] = new_state['routing'][src_switch][dst_switch]['bw']
#                     i = i + 1
#             for src_switch in new_state['controlled'].keys():
#                 for dst_switch in new_state['controlled'][src_switch].keys():
#                     arr[i] = new_state['controlled'][src_switch][dst_switch]['bw']
#                     i = i + 1
#             switches_bw_variables['data'][step_index, :] = arr

#             for host in env.hosts:
#                 episode_hosts_bw[host]['data'].append(Decimal(http_client.get_host_bw(host).json()['bw']))

#             copy_cic_step_file(config, f"Episode {episode} - Step {step} - CIC results.csv")

#             print(f'(Reinforcement) <==================== Step {step} (of episode {episode}) Ended')

#             if do_break:
#                 break

#         for normal_host in env.normal_hosts:
#             headers = get_basic_metrics_headers() + get_network_metrics_headers() + ["current_path"]
#             save_file_with_headers(f"{config.data_folder}/Episode {episode} - {normal_host_state_variables[normal_host]['filename']}", normal_host_state_variables[normal_host]['data'], headers, fmt='%.18e,%.18e,%.18e,%.18e,%.18e,%.18e,%.18e,%.18e,%.18e,%.18e,%.18e,%.18e,%.18e,%.18e,%.18e,%.18e,%s')
#         for server in env.servers:
#             save_file_with_headers(f"{config.data_folder}/Episode {episode} - {server_state_variables[server]['filename']}", server_state_variables[server]['data'], get_basic_metrics_headers())
#         for attacker in env.attacker_hosts:
#             headers = get_basic_metrics_headers() + ["current_path"]
#             save_file_with_headers(f"{config.data_folder}/Episode {episode} - {attacker_state_variables[attacker]['filename']}", attacker_state_variables[attacker]['data'], headers, fmt='%.18e,%.18e,%.18e,%.18e,%.18e,%.18e,%.18e,%.18e,%.18e,%.18e,%.18e,%.18e,%s')
#         save_file_with_headers(f"{config.data_folder}/Episode {episode} - {switches_bw_variables['filename']}", switches_bw_variables['data'], SWITCHES_BW_HEADERS)
#         save_file_with_headers(f"{config.data_folder}/Episode {episode} - Actions", env.episode_actions_text_list, ["Action", "Message"], fmt='%s')
#         # Generates plots to visualize the performance of the RL agent across multiple episodes.
#         # Graphs include total reward per episode, packet loss, delay, and bandwidth usage.
#         fig1 = plt.figure(f"Episode {episode} Reward")
#         plt.plot(range(1, len(episode_rewards) + 1), episode_rewards, color='b', label='rewards')
#         plt.legend()
#         plt.xlim((1, env.steps))
#         plt.xlabel("Steps")
#         plt.ylabel("Reward")
#         plt.title(f"Episode {episode} Reward")
#         fig1.savefig(f"{config.figures_folder}/Episode {episode} - Reward.png")
#         plt.close(fig1)

#         fig1 = plt.figure(f"Episode {episode} Loss Function")
#         plt.plot(range(1, len(ddqn_agent.episode_loss) + 1), ddqn_agent.episode_loss, color='r', label='loss function')
#         plt.legend()
#         plt.xlim((1, env.steps))
#         plt.xlabel("Steps")
#         plt.ylabel("Loss function")
#         plt.title(f"Episode {episode} Loss Function")
#         fig1.savefig(f"{config.figures_folder}/Episode {episode} - Loss Function.png")
#         plt.close(fig1)

#         fig3_1 = plt.figure(f"Episode {episode} PKT loss")
#         plt.plot(range(1, len(episode_avg_packet_loss) + 1), [100 * x for x in episode_avg_packet_loss], color='b', label='pkt loss')
#         plt.legend()
#         plt.xlim((1, env.steps))
#         plt.xlabel("Steps")
#         plt.ylabel("PKT loss")
#         plt.title(f"Episode {episode} PKT loss")
#         fig3_1.savefig(f"{config.figures_folder}/Episode {episode} - PKT loss.png")
#         plt.close(fig3_1)

#         fig3_2 = plt.figure(f"Episode {episode} AVG delay")
#         plt.plot(range(1, len(episode_avg_real_delays) + 1), episode_avg_real_delays, color='r', label='avg delay')
#         plt.legend()
#         plt.xlim((1, env.steps))
#         plt.xlabel("Steps")
#         plt.ylabel("AVG delay")
#         plt.title(f"Episode {episode} AVG delay")
#         fig3_2.savefig(f"{config.figures_folder}/Episode {episode} - AVG delay.png")
#         plt.close(fig3_2)

#         fig3_3 = plt.figure(f"Episode {episode} AVG latency")
#         plt.plot(range(1, len(episode_avg_latencys) + 1), episode_avg_latencys, color='g', label='avg latency')
#         plt.legend()
#         plt.xlim((1, env.steps))
#         plt.xlabel("Steps")
#         plt.ylabel("AVG latency")
#         plt.title(f"Episode {episode} AVG latency")
#         fig3_3.savefig(f"{config.figures_folder}/Episode {episode} - AVG latency.png")
#         plt.close(fig3_3)

#         fig3_4 = plt.figure(f"Episode {episode} AVG jitter")
#         plt.plot(range(1, len(episode_avg_jitters) + 1), episode_avg_jitters, color='m', label='avg jitter')
#         plt.legend()
#         plt.xlim((1, env.steps))
#         plt.xlabel("Steps")
#         plt.ylabel("AVG jitter")
#         plt.title(f"Episode {episode} AVG jitter")
#         fig3_4.savefig(f"{config.figures_folder}/Episode {episode} - AVG jitter.png")
#         plt.close(fig3_4)

#         fig5 = plt.figure(f"Episode {episode} Hosts BW")
#         for host in env.hosts:
#             host_label = f'{host}'
#             if host in env.servers:
#                 host_label = f'{host_label} (server)'
#             elif host in env.attacker_hosts:
#                 host_label = f'{host_label} (attacker {attack_types[host]})'
#             plt.plot(range(1, len(episode_hosts_bw[host]['data']) + 1), episode_hosts_bw[host]['data'], label=host_label)
#         plt.legend()
#         plt.xlim((1, env.steps))
#         plt.xlabel("Steps")
#         plt.ylabel("BW")
#         plt.title(f"Episode {episode} Hosts BW")
#         fig5.savefig(f"{config.figures_folder}/Episode {episode} - Hosts BW")
#         plt.close(fig5)

#         fig6 = plt.figure(f"Episode {episode} Switches BW")
#         color = iter(cm.rainbow(np.linspace(0, 1, len(SWITCHES_BW_HEADERS))))
#         for i in range(len(SWITCHES_BW_HEADERS)):
#             switch_label = SWITCHES_BW_HEADERS[i]
#             c = next(color)
#             plt.plot(range(1, len(switches_bw_variables['data'][:,i]) + 1), switches_bw_variables['data'][:,i],
#                      label=switch_label, c=c)
#         plt.legend(loc='center left', bbox_to_anchor=(1, 0))
#         plt.xlim((1, env.steps))
#         plt.xlabel("Steps")
#         plt.ylabel("BW")
#         plt.title(f"Episode {episode} Switches BW")
#         fig6.savefig(f"{config.figures_folder}/Episode {episode} - Switches BW", bbox_inches='tight')
#         plt.close(fig6)

#         print(f'(Reinforcement) <================== Episode {episode} Ended')

#         plt.close('all')
#         cmd.stop_network()


#     ddqn_agent.save_model(f"{config.rl_models_folder}/rl_model")

#     fig = plt.figure(f"Results per Episode")
#     plt.plot(range(1, env.episodes + 1), total_rewards_per_episode, color='blue', label='Total rewards per episode')
#     plt.axhline(y=max(total_rewards_per_episode), color='r', linestyle='-', label='Max total reward')
#     eps_graph = [max(total_rewards_per_episode) * x for x in epsilons]
#     plt.plot(range(1, env.episodes + 1), eps_graph, color='g', linestyle='-', label='Epsilon')
#     plt.legend()
#     plt.xlabel("Episode")
#     plt.xlim((1, env.episodes))
#     plt.ylim((min(total_rewards_per_episode), 1.1 * max(total_rewards_per_episode)))
#     plt.title(f"Results per Episode")
#     fig.savefig(f"{config.figures_folder}/Last - total rewards and epsilon.png")
#     plt.close('all')

#     print(global_vars_to_print)

#     print('(Reinforcement) ================> Main Ended')