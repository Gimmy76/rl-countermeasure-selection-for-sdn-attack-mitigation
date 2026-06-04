"""
QoSentry Training Loop — Reinforcement Learning for DDoS Mitigation in SDN

CRITICAL TIMING CONSTRAINTS (Fixed by Paper — DO NOT MODIFY):
- TCP flow duration per step: 40 seconds (non-negotiable)
- DDoS attack duration: 40 seconds (non-negotiable)
- tshark capture post-flow: 15 seconds (non-negotiable)
- Network reset between steps: 2-3 seconds
→ MINIMUM per step: ~55 seconds (wall-clock time)
→ 50 episodes × 100 steps = 5,000 steps → ~76+ hours per trial

Optimization focus: Scapy processing, DDQN replay, logging (remaining 20-25% of time).
See TIMING_CONSTRAINTS.md for detailed timing analysis.
"""

import random
import numpy as np
import tensorflow as tf
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(2)
import csv
from decimal import Decimal
from Configuration import Configuration
from Environment import Environment
from HttpClient import HttpClient
from CmdManager import CmdManager
from DdqnAgent import DoubleDeepQNetwork
import matplotlib.pyplot as plt
import shutil
import os
import hydra
from omegaconf import DictConfig, OmegaConf
import mlflow
import optuna
import time
from datetime import datetime
import plotly.express as px
from collections import defaultdict
import json


# ---------------------------------------------------------------------------
# TIMING INSTRUMENTATION
# ---------------------------------------------------------------------------

class TimingTracker:
    """Tracks execution time of key components to identify bottlenecks."""
    def __init__(self):
        self.timings = defaultdict(list)
        self.trial_start = None
    
    def start_trial(self):
        self.trial_start = time.time()
        self.timings.clear()
    
    def log(self, component_name, duration_seconds):
        """Log timing for a component."""
        self.timings[component_name].append(duration_seconds)
        if duration_seconds > 5:  # Only print long operations
            print(f"[TIMING] {component_name}: {duration_seconds:.2f}s")
    
    def get_summary(self):
        """Generate timing summary for trial."""
        summary = {}
        total_time = 0
        for component in sorted(self.timings.keys()):
            times = self.timings[component]
            avg = sum(times) / len(times)
            total = sum(times)
            total_time += total
            summary[component] = {
                "count": len(times),
                "avg_s": round(avg, 2),
                "total_s": round(total, 1),
                "pct": round(100 * total / (total_time or 1), 1),
            }
        summary["TOTAL_TIME_S"] = round(total_time, 1)
        summary["TOTAL_TIME_H"] = round(total_time / 3600, 2)
        return summary
    
    def save_report(self, output_path):
        """Save timing report to JSON."""
        summary = self.get_summary()
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\n[TIMING] Report saved: {output_path}")
        print(json.dumps(summary, indent=2))

timing_tracker = TimingTracker()


# ---------------------------------------------------------------------------
# UTILITY FUNCTIONS
# ---------------------------------------------------------------------------

def copy_cic_step_file(config, path_to_save, episode, step):
    try:
        cic_file_name = f"Episode {episode} - Step {step} - CIC results.csv"
        destination = os.path.join(path_to_save, "cic", cic_file_name)
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        shutil.copyfile(config.cic_output_file_path, destination)
    except Exception as e:
        print(f"(Warning) Could not copy CIC file: {e}")


def get_attack_type():
    return "ICMP"


def get_basic_metrics_headers():
    return ["tx_bytes", "rx_bytes", "bandwidth", "tx_packets", "rx_packets",
            "tx_packets_len", "rx_packets_len", "delivered_pkts", "loss_pct",
            "is_connected", "pkts_s", "bytes_s"]


def get_network_metrics_headers():
    return ["avg_latency_s", "avg_packet_transmission_time_s",
            "throughput_bps", "avg_jitter_s"]


def setup_directories(base_dir):
    for folder in ["data", "figs", "cic", "configs", "models", "timing"]:
        os.makedirs(os.path.join(base_dir, folder), exist_ok=True)


def save_step_timing_row(timing_dir, row):
    file_path = os.path.join(timing_dir, "step_timing.csv")
    fieldnames = [
        "episode", "step", "step_total_s", "tcp_flow_s", "tshark_s",
        "netmetrics_s", "experience_replay_s", "logging_mlflow_s",
        "reset_mininet_s", "episode_elapsed_s"
    ]
    write_header = not os.path.exists(file_path)
    with open(file_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def save_episode_plots(current_run_dir, episode, ep_rews, ep_lats, ep_loss,
                       ep_jits, episode_hosts_bw):
    def save_plot_png(data, title, color, filename):
        plt.figure(figsize=(10, 6))
        plt.plot(data, color=color, label=title)
        plt.title(f"Episode {episode} - {title}")
        plt.xlabel("Steps")
        plt.ylabel("Value")
        plt.legend()
        plt.savefig(os.path.join(current_run_dir, "figs",
                                 f"Episode_{episode}_{filename}.png"))
        plt.close()

    save_plot_png(ep_rews, "Reward",      "blue",   "Reward")
    save_plot_png(ep_lats, "Latency",     "green",  "Latency")
    save_plot_png(ep_loss, "Packet Loss", "orange", "Packet_Loss")
    save_plot_png(ep_jits, "Jitter",      "purple", "Jitter")

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

    try:
        fig_lat = px.line(y=ep_lats,
                          title=f"Interactive Latency Ep {episode}",
                          labels={'y': 'Latency (s)', 'x': 'Steps'})
        fig_lat.write_html(os.path.join(current_run_dir, "figs",
                                        f"Episode_{episode}_latency_interactive.html"))
    except Exception as e:
        print(f"(Warning) Could not save interactive plot: {e}")


# ---------------------------------------------------------------------------
# CORE EXPERIMENT FUNCTION
# ---------------------------------------------------------------------------

def save_extra_plots(current_run_dir, episode, ep_throughput, ep_ddqn_loss, ep_sw_bw_matrix, sw_headers):
    """Plot aggiuntivi: throughput, DDQN loss, switch BW."""
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    # Throughput
    if ep_throughput:
        plt.figure(figsize=(10, 6))
        plt.plot(ep_throughput, color='teal', label='Avg Throughput (bps)')
        plt.title(f"Episode {episode} - Average Throughput")
        plt.xlabel("Steps")
        plt.ylabel("bps")
        plt.legend()
        plt.savefig(os.path.join(current_run_dir, "figs",
                                 f"Episode_{episode}_Throughput.png"))
        plt.close()

    # DDQN loss function
    if ep_ddqn_loss:
        plt.figure(figsize=(10, 6))
        plt.plot(ep_ddqn_loss, color='red', label='DDQN Loss')
        plt.title(f"Episode {episode} - DDQN Loss Function")
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(current_run_dir, "figs",
                                 f"Episode_{episode}_DDQN_Loss.png"))
        plt.close()

    # Switch BW heatmap
    if ep_sw_bw_matrix and sw_headers:
        try:
            matrix = np.array(ep_sw_bw_matrix, dtype=float)
        except (ValueError, TypeError):
            matrix = np.array([[float(v) if v != '' else 0.0 for v in row] for row in ep_sw_bw_matrix], dtype=float)
        plt.figure(figsize=(max(10, len(sw_headers)), 6))
        plt.imshow(matrix.T, aspect='auto', cmap='YlOrRd', interpolation='nearest')
        plt.colorbar(label='Bandwidth')
        plt.yticks(range(len(sw_headers)), sw_headers, fontsize=7)
        plt.xlabel("Steps")
        plt.title(f"Episode {episode} - Switch Bandwidth Heatmap")
        plt.tight_layout()
        plt.savefig(os.path.join(current_run_dir, "figs",
                                 f"Episode_{episode}_SwitchBW_Heatmap.png"))
        plt.close()

    # Attacker vs Benign loss (se loggato)
    print(f"[PLOTS] Extra plots saved for episode {episode}")

def run_experiment(cfg: DictConfig, trial=None):
    run_name = f"trial_{trial.number}" if trial is not None else "standard_train"
    nested_run = trial is not None

    with mlflow.start_run(run_name=run_name, nested=nested_run):
        timing_tracker.start_trial()
        if trial is not None:
            mlflow.set_tag("trial_number", trial.number)

        # ------------------------------------------------------------------
        # 1. PARAMETER INITIALIZATION
        # ------------------------------------------------------------------
        if trial:
            w_lat           = trial.suggest_float("w_latency",       0.5,   2.0)
            w_jit           = trial.suggest_float("w_jitter",        0.5,   2.0)
            w_loss          = trial.suggest_float("w_loss",           0.5,   2.0)
            tolerable_loss  = trial.suggest_float("tolerable_loss",   0.10, 0.45)
            threshold_loss  = trial.suggest_float("threshold_loss",   0.01,  0.10)
            lr              = trial.suggest_float("lr",               1e-4,  5e-2, log=True)
            gamma         = trial.suggest_float("gamma", 0.70, 0.99)
            epsilon_decay = trial.suggest_float("epsilon_decay", 0.9950, 0.9999)
            batch_size    = trial.suggest_categorical("batch_size", [128, 256, 512])
        else:
            w_lat          = cfg.reward.w_latency
            w_jit          = cfg.reward.w_jitter
            w_loss         = cfg.reward.w_loss
            tolerable_loss = cfg.reward.tolerable_loss
            threshold_loss = cfg.reward.threshold_loss
            lr            = cfg.training.learning_rate
            gamma         = cfg.training.discount_factor
            epsilon_decay = cfg.epsilon_decay
            batch_size    = 128

        mlflow.log_params({
            "w_lat": w_lat, "w_jit": w_jit, "w_loss": w_loss,
            "tolerable_loss": tolerable_loss, "threshold_loss": threshold_loss,
            "lr": lr, "gamma": gamma, "epsilon_decay": epsilon_decay, "batch_size": batch_size,
        })
        mlflow.set_tag("topology", cfg.hosts_topo)
        mlflow.set_tag("mlflow.note.content",
                       f"QoS run on {cfg.hosts_topo} | lr={lr:.5f}")

        # ------------------------------------------------------------------
        # 2. FOLDER SETUP
        # ------------------------------------------------------------------
        timestamp       = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        current_run_dir = f"results/train_{timestamp}"
        setup_directories(current_run_dir)
        timing_dir = os.path.join(current_run_dir, "timing")
        os.makedirs(timing_dir, exist_ok=True)

        config_path = os.path.join(current_run_dir, "configs", "config_used.yaml")
        with open(config_path, "w") as f:
            f.write(OmegaConf.to_yaml(cfg))
        mlflow.log_artifact(config_path)

        # ------------------------------------------------------------------
        # 3. ENVIRONMENT & AGENT INIT
        # ------------------------------------------------------------------
        config = Configuration(
            cfg.hosts_topo, cfg.episodes, cfg.steps,
            epsilon_decay,
            cfg.nbr_controlled_switches
        )
        env         = Environment(config)
        # Normalize weights so w_lat + w_jit = 2 (come nel paper: 1.0 + 1.0)
        # pesi diretti come paper
        env.w_lat          = w_lat
        env.w_jit          = w_jit
        env.w_loss         = w_loss
        env.tolerable_loss = tolerable_loss
        env.threshold_loss = threshold_loss
        cmd         = CmdManager(config)
        http_client = HttpClient(config)

        ddqn_agent = DoubleDeepQNetwork(
            config, env, http_client,
            is_controlled=False,
            is_prefilled_actions=False,
            gamma=gamma,
            learning_rate=lr,
            batch_size=batch_size
        )



        total_rewards_per_episode = []

        try:
            for episode in range(1, config.episodes + 1):

                tot_episode_reward = 0
                env.reset()
                env.update_hosts()

                episode_hosts_bw = {h: [] for h in env.hosts}
                ep_rews, ep_lats, ep_jits, ep_loss = [], [], [], []
                ep_throughput, ep_ddqn_loss = [], []
                ep_sw_bw_matrix = []
                sw_headers = None

                env.perform_setup(http_client, cfg.get('attackers', []))
                ddqn_agent.set_actions(env.ACTIONS)
                cmd.start_network_in_background(
                    env.servers, env.attacker_hosts,
                    config.hosts_topo_file_name,
                    config.nbr_controlled_switches
                )

                print(f"(RL) Episode {episode}/{config.episodes} - Network started, waiting stabilization...")

                reset_start = time.time()
                env.update_hosts_ips(http_client)
                env.update_interfaces(http_client.get_switches_interfaces())
                env.last_reset_mininet_duration = time.time() - reset_start
                tshark_ids = env.get_tshark_interfaces_ids(cmd)

                sender_receiver_relation = {h: random.choice(env.servers)       for h in env.normal_hosts}
                attacker_victim_relation = {a: random.choice(env.victim_servers) for a in env.attacker_hosts}
                attack_types             = {a: get_attack_type()                 for a in env.attacker_hosts}

                # Stato iniziale (usato come baseline per il calcolo dei delta di reward)
                current_state = env.get_state(
                    config, cmd, http_client, tshark_ids,
                    sender_receiver_relation, attacker_victim_relation, attack_types
                )

                # FIX CRITICO: inizializza i baseline dal primo stato reale.
                # Senza questo, al primo step di ogni episodio il delta è sempre
                # negativo (latency reale > 0 vs baseline 0.0) → reward sempre negativa
                # → episode_total_reward discende ad ogni episodio.
                env.last_recorded_latency = env.calculate_latency(current_state)
                env.last_recorded_jitter  = env.calculate_jitter(current_state)
                env.last_recorded_delay   = env.calculate_delay(current_state)

                print(f"(RL) Episode {episode} - Baseline: "
                      f"lat={env.last_recorded_latency:.4f}s "
                      f"jit={env.last_recorded_jitter:.4f}s "
                      f"delay={env.last_recorded_delay:.1f}ms")

                episode_start = time.time()
                # --------------------------------------------------------------
                # STEP LOOP — TIMING BREAKDOWN (per paper constraints)
                # ~40s: TCP flow execution (40s flow + 15s tshark capture = 55s min)
                # ~2-5s: Scapy pcap processing (NetMetricsCalculator)
                # ~2-5s: DDQN replay + prediction
                # ~1-2s: Logging + HTTP calls
                # Total: ~55-70s per step (non-reducible first 55s from paper)
                # See TIMING_CONSTRAINTS.md for details
                # --------------------------------------------------------------
                for step in range(1, config.steps + 1):
                    step_start = time.time()

                    print(f"(RL) Ep{episode} Step{step}/{config.steps}")

                    state_vec            = env.transform_state_dict_to_normalized_vector(current_state)
                    action, is_predicted = ddqn_agent.action(step, state_vec)

                    new_state, base_reward, done, loss_val, delay, latency, jitter = \
                        env.apply_action_controlled_switches(
                            config, cmd, http_client, tshark_ids,
                            sender_receiver_relation, attacker_victim_relation,
                            attack_types, action, is_predicted
                        )
                    network_simulation_duration = time.time() - step_start
                    timing_tracker.log("Network_Simulation_TCPFlowTshark", network_simulation_duration)

                    if done:
                        print(f"[DONE] Ep{episode} Step{step} "
                              f"lat={latency:.4f} jit={jitter:.4f} "
                              f"delay={delay:.1f}ms reward={base_reward:.4f}")

                    reward_val = float(np.clip(base_reward, -12.0, 18.0))  # range: w*[-2,-2,-2] a w*[+3,+3,+3] con w_max=2
                    tot_episode_reward += reward_val

                    global_step = (episode - 1) * config.steps + step

                    # Logging to MLflow
                    logging_start = time.time()
                    mlflow.log_metric("latency_step",     latency,        step=global_step)
                    mlflow.log_metric("jitter_step",      jitter,         step=global_step)
                    mlflow.log_metric("delay_step",       delay,          step=global_step)
                    mlflow.log_metric("packet_loss_pct",  loss_val * 100, step=global_step)
                    mlflow.log_metric("attacker_loss_pct", new_state["host"].get(env.attacker_hosts[0], {}).get("loss_pct", 0) * 100, step=global_step)
                    mlflow.log_metric("benign_loss_pct", float(np.mean([new_state["host"].get(h, {}).get("loss_pct", 0) for h in env.normal_hosts])) * 100, step=global_step)
                    mlflow.log_metric("base_reward_step", base_reward,    step=global_step)
                    mlflow.log_metric("reward_step",      reward_val,     step=global_step)
                    mlflow.log_metric("epsilon",          ddqn_agent.epsilon, step=global_step)
                    mlflow.log_metric("throughput_bps",    env.calculate_throughput(new_state), step=global_step)
                    # Attacker metrics (latency, jitter, throughput)
                    _att = env.attacker_hosts[0]
                    _att_metrics = new_state["host"].get(_att, {}).get("non_server_data", {}).get("network_metrics") or {}
                    mlflow.log_metric("attacker_latency_step",   float(_att_metrics.get("avg_latency_s", 0)),   step=global_step)
                    mlflow.log_metric("attacker_jitter_step",    float(_att_metrics.get("avg_jitter_s", 0)),    step=global_step)
                    mlflow.log_metric("attacker_throughput_bps", float(new_state["host"].get(_att, {}).get("non_server_data", {}).get("network_metrics", {}).get("throughput_bps", 0)), step=global_step)
                    logging_duration = time.time() - logging_start
                    timing_tracker.log("Logging_MLflow", logging_duration)

                    replay_duration = 0.0
                    if len(ddqn_agent.memory) > ddqn_agent.batch_size:
                        replay_start = time.time()
                        ddqn_agent.experience_replay(ddqn_agent.batch_size)
                        replay_duration = time.time() - replay_start
                        timing_tracker.log("DDQN_ExperienceReplay", replay_duration)

                    ep_rews.append(reward_val)
                    ep_lats.append(latency)
                    ep_jits.append(jitter)
                    ep_loss.append(loss_val)
                    ep_throughput.append(env.calculate_throughput(new_state))
                    ep_ddqn_loss.append(ddqn_agent.episode_loss[-1] if ddqn_agent.episode_loss else 0.0)

                    for h in env.hosts:
                        val_bw = new_state['host'].get(h, {}).get('bandwidth', 0)
                        episode_hosts_bw[h].append(float(val_bw))

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

                    next_state_vec = env.transform_state_dict_to_normalized_vector(new_state)
                    ddqn_agent.store(state_vec, action, reward_val, next_state_vec, done)

                    # FIX: experience_replay chiamato UNA SOLA VOLTA per step
                    # FIX: epsilon decay indipendente dalla replay
                    if ddqn_agent.decay_epsilon() < 1.0:
                        print(f"<------>  Epsilon: {ddqn_agent.epsilon:.5f}")

                    step_total = time.time() - step_start
                    step_record = {
                        "episode": episode,
                        "step": step,
                        "step_total_s": round(step_total, 3),
                        "tcp_flow_s": round(env.last_tcp_flow_duration, 3),
                        "tshark_s": round(env.last_tshark_duration, 3),
                        "netmetrics_s": round(getattr(cmd, 'last_netmetrics_duration', 0.0), 3),
                        "experience_replay_s": round(replay_duration, 3),
                        "logging_mlflow_s": round(logging_duration, 3),
                        "reset_mininet_s": round(getattr(env, 'last_reset_mininet_duration', 0.0), 3),
                        "episode_elapsed_s": round(time.time() - episode_start, 3),
                    }
                    save_step_timing_row(timing_dir, step_record)

                    # Aggiornamento periodico target network o su done
                    if step % ddqn_agent.update_target_each == 0 or done:
                        ddqn_agent.update_target_from_model()

                    current_state = new_state

                    if done:
                        print(f"(RL) Episode {episode} ended early at step {step}")
                        break

                # --------------------------------------------------------------
                # END OF EPISODE
                # --------------------------------------------------------------
                steps_run = len(ep_rews)
                print(f"(RL) Episode {episode} finished ({steps_run} steps). "
                      f"Total reward: {tot_episode_reward:.4f}")

                sw_bw_path = os.path.join(current_run_dir, "data",
                                          f"Episode_{episode}_switches_bw.csv")
                if ep_sw_bw_matrix and sw_headers:
                    np.savetxt(sw_bw_path, np.array(ep_sw_bw_matrix),
                               delimiter=",",
                               header=",".join(sw_headers),
                               comments='')

                save_extra_plots(current_run_dir, episode, ep_throughput, ep_ddqn_loss, ep_sw_bw_matrix, sw_headers)
                save_episode_plots(current_run_dir, episode,
                                   ep_rews, ep_lats, ep_loss,
                                   ep_jits, episode_hosts_bw)

                total_rewards_per_episode.append(tot_episode_reward)
                mlflow.log_metric("episode_total_reward", tot_episode_reward, step=episode)

                # Early stopping: ferma se reward media cala per 5 episodi consecutivi
                EARLY_STOP_PATIENCE = 999  # early stopping disabilitato
                EARLY_STOP_MIN_DELTA = 0.99  # soglia impossibile da raggiungere
                if len(total_rewards_per_episode) >= EARLY_STOP_PATIENCE * 2:
                    avg_recent   = float(np.mean(total_rewards_per_episode[-EARLY_STOP_PATIENCE:]))
                    avg_previous = float(np.mean(total_rewards_per_episode[-EARLY_STOP_PATIENCE*2:-EARLY_STOP_PATIENCE]))
                    degradation = (avg_previous - avg_recent) / (abs(avg_previous) + 1e-8)
                    if avg_recent < avg_previous and degradation > EARLY_STOP_MIN_DELTA:
                        print(f"[EARLY STOP] avg_reward ultimi {EARLY_STOP_PATIENCE} ep ({avg_recent:.4f}) "
                              f"< precedenti {EARLY_STOP_PATIENCE} ep ({avg_previous:.4f}) "
                              f"degradazione={degradation:.2%}. Stop.")
                        mlflow.set_tag("status", "EARLY_STOPPED")
                        break
                mlflow.log_metric("episode_steps_run",    steps_run,          step=episode)

                # Checkpoint modello ogni 5 episodi
                if episode % 5 == 0:
                    model_path = os.path.join(current_run_dir, "models",
                                              f"model_ep{episode}.keras")
                    ddqn_agent.save_model(model_path)
                    print(f"[CHECKPOINT] Model saved at episode {episode}: {model_path}")

                try:
                    cmd.stop_network()
                except Exception as _stop_err:
                    print(f"(RL) WARNING: stop_network episode {episode} failed: {_stop_err}")

            # Save timing report
            timing_report_path = os.path.join(current_run_dir, "timing_report.json")
            timing_tracker.save_report(timing_report_path)
            # Ensure the step-level timing CSV is saved in the timing folder
            print(f"[TIMING] Step-level timing file available in: {os.path.join(timing_dir, 'step_timing.csv')}")
            
            # FIX: log_artifacts UNA SOLA VOLTA dopo tutti gli episodi
            mlflow.log_artifacts(current_run_dir)

        except Exception as e:
            mlflow.set_tag("status", "FAILED")
            mlflow.log_param("error_message", str(e))
            print(f"CRITICAL ERROR in episode loop: {e}")
            try:
                cmd.stop_network()
            except Exception:
                pass
            raise

        avg_reward = float(np.mean(total_rewards_per_episode))
        mlflow.log_metric("final_average_reward", avg_reward)
        mlflow.set_tag("status", "OK")
        print(f"(RL) Experiment finished. avg_reward={avg_reward:.4f}")
        return avg_reward


# ---------------------------------------------------------------------------
# MAIN ENTRY POINT
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    import os as _os
    
    # Ogni worker ha il suo esperimento MLflow separato
    trial_id = _os.environ.get("TRIAL_ID", "_default").replace("_", "")
    experiment_name = "QoSentry_Optimization"
    
    # MLflow server centralizzato: tutti i trial scrivono allo stesso server
    mlflow.set_tracking_uri("http://127.0.0.1:5050")
    mlflow.set_experiment(experiment_name)
    
    # Cleanup MLflow rimosso: ucciderebbe run di altri trial paralleli

    if cfg.mode == "tune":
        print("\n[INFO] Starting Optuna autotuning...")
        _storage_path = _os.environ.get(
            'OPTUNA_STORAGE',
            f"sqlite:///{_os.getcwd()}/optuna_parallel.db"
        )
        _study_name = _os.environ.get('OPTUNA_STUDY_NAME', 'qosentry_parallel')
        print(f"[OPTUNA] storage={_storage_path} study={_study_name}")
        
        # Storage SQLite con timeout esteso per gestire scritture concorrenti
        _storage = optuna.storages.RDBStorage(
            url=_storage_path,
            engine_kwargs={
                "connect_args": {"timeout": 60},
                "pool_pre_ping": True,
                "pool_size": 1,
            },
            heartbeat_interval=60,
            failed_trial_callback=optuna.storages.RetryFailedTrialCallback(max_retry=3),
        )
        study = optuna.create_study(
            study_name=_study_name,
            storage=_storage,
            direction="maximize",
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        study.optimize(
            lambda trial: run_experiment(cfg, trial),
            n_trials=int(cfg.tune_trials),
        )

        try:
            best = study.best_trial
        except ValueError as exc:
            print(f"[WARNING] No completed trials: {exc}")
            return

        print(f"\n[RESULT] Best trial #{best.number} avg_reward={best.value:.4f}")
        for k, v in best.params.items():
            print(f"  {k}: {v}")

        with open("best_params.yaml", "w") as f:
            OmegaConf.save(config=OmegaConf.create(study.best_params), f=f)
        print("[INFO] Best params saved to best_params.yaml")
        # --- Ranking finale per metriche di rete ---
        completed = [t for t in study.trials if t.state.name == "COMPLETE"]
        if completed:
            print(f"\n[RANKING] Trials ordinati per metriche di rete finali:")
            client = mlflow.tracking.MlflowClient()
            trial_metrics = []
            for t in completed:
                run_name = f"trial_{t.number}"
                runs = client.search_runs(
                    experiment_ids=["1"],
                    filter_string=f"tags.mlflow.runName = \'{run_name}\'",
                    order_by=["attributes.start_time DESC"],
                    max_results=1
                )
                if runs:
                    r = runs[0]
                    trial_metrics.append({
                        "trial":       t.number,
                        "avg_reward":  t.value,
                        "latency":     r.data.metrics.get("latency_step",    float("inf")),
                        "jitter":      r.data.metrics.get("jitter_step",     float("inf")),
                        "packet_loss": r.data.metrics.get("packet_loss_pct", float("inf")),
                        "throughput":  r.data.metrics.get("throughput_bps",  0.0),
                    })
            if trial_metrics:
                hdr = f"  {'Trial':<8} {'AvgReward':<12} {'Latency(s)':<14} {'Jitter(s)':<12} {'PktLoss(%)':<12} {'Throughput(bps)'}"
                sep = f"  {'-'*75}"
                def row(m):
                    return (f"  #{m['trial']:<7} {m['avg_reward']:<12.4f} "
                            f"{m['latency']:<14.6f} {m['jitter']:<12.6f} "
                            f"{m['packet_loss']:<12.2f} {m['throughput']:.1f}")
                for label, key, rev in [
                    ("avg_reward DESC", "avg_reward", True),
                    ("latency ASC",     "latency",    False),
                    ("jitter ASC",      "jitter",     False),
                    ("packet_loss ASC", "packet_loss",False),
                    ("throughput DESC", "throughput", True),
                ]:
                    print(f"\n  [By {label}]")
                    print(hdr)
                    print(sep)
                    for m in sorted(trial_metrics, key=lambda x: x[key], reverse=rev):
                        print(row(m))

    else:
        print("\n[INFO] Starting standard training...")
        run_experiment(cfg)

    print("\n[FINISH] Run completed.")


if __name__ == '__main__':
    main()

