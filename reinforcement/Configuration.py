from datetime import datetime
import json
import os

class Configuration():

    def __init__(self, hosts_topo_file_name, episodes, steps, epsilon_decay,
                 nbr_controlled_switches, run_id=None):
        print("(Reinforcement) Configuration.__init__()")

        # API
        # Porta Flask dinamica per parallelizzazione
        _tid = os.environ.get("TRIAL_ID", "")
        _port = 5001  # default
        if _tid.startswith("_t"):
            try:
                _trial_num = int(_tid.replace("_t", ""))
                if 1 <= _trial_num <= 99:
                    _port = 5000 + _trial_num
                else:
                    raise ValueError(f"TRIAL_ID fuori range: {_trial_num}")
            except ValueError as e:
                import warnings
                warnings.warn(f"TRIAL_ID malformato ('{_tid}'): {e}. Uso porta 5001.")
                _port = 5001
        elif _tid != "":
            import warnings
            warnings.warn(f"TRIAL_ID formato inatteso ('{_tid}'): non inizia con '_t'. Uso porta 5001.")
        self.api_link = f"http://127.0.0.1:{_port}"

        # Network
        self.network_dir        = os.getcwd() + "/../network"
        self.network_entrypoint = f'{self.network_dir}/EntryPoint.py'
        self.network_command    = (
            f'/usr/bin/python3 {self.network_entrypoint} '
            f'--servers [SERVERS] --attackers [ATTACKERS] '
            f'--hosts-topo-file [HOSTS_FILE] '
            f'--nbr-controlled-switches [NBR_CONTROLLED_SWITCHES] '
            f'--manuel-receivers'
        )

        self.tmp_dir = f"/tmp/qosentry{os.environ.get('TRIAL_ID', '')}"

        os.makedirs(self.tmp_dir, exist_ok=True)
        self.network_log_dir = f'{self.tmp_dir}/network_logs'
        os.makedirs(self.network_log_dir, exist_ok=True)

        # TShark
        self.tshark_interfaces_command = 'tshark -D'
        self.tshark_pcap_file_name     = 'tshark_out.pcap'
        self.tshark_pcap_file_path     = f'{self.tmp_dir}/{self.tshark_pcap_file_name}'
        self.tshark_sniffing_command   = f'tshark -i any -f "tcp port 80" -w {self.tshark_pcap_file_path} -a filesize:524288'

        # CIC output — ora prodotto da NetMetricsCalculator invece che da CICFlowMeter Java.
        # Il percorso rimane lo stesso per compatibilità con Environment.read_cic_flow_file().
        self.cic_output_dir       = f'{self.tmp_dir}/cic_out'
        self.cic_output_file_path = f'{self.cic_output_dir}/{self.tshark_pcap_file_name}_Flow.csv'
        os.makedirs(self.cic_output_dir, exist_ok=True)

        # DITG logs (legacy, non usato attivamente)
        self.ditg_directory    = '/home/giannidon/D-ITG/bin'
        self.ditg_logs_file_path = f'{self.tmp_dir}/ITGRecv.log'
        self.ditg_logs_command = f'{self.ditg_directory}/ITGDec {self.ditg_logs_file_path}'

        # NetMetricsCalculator — ora fa anche il lavoro di CICFlowMeter.
        # Il flag -cic dice a NetMetricsCalculator dove scrivere il CSV equivalente a CIC.
        self.net_metrics_calculator_path = f'{os.getcwd()}/NetMetricsCalculator.py'
        self.net_metrics_result_file_path = f'{self.tmp_dir}/metrics.json'
        self.net_metrics_command = (
            f'python3 {self.net_metrics_calculator_path} '
            f'-s [SERVER_IP] -p [SERVER_PORT] -hip [HOSTS_IPS] '
            f'-t [DURATION] -b [BYTES] '
            f'-pcap {self.tshark_pcap_file_path} '
            f'-cic [CIC_OUTPUT]'   
        )

        # Results folder
        self.results_folder = os.getcwd() + "/results"
        os.makedirs(self.results_folder, exist_ok=True)

        if run_id is not None:
            self.running_time = run_id
        else:
            self.running_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

        self.current_train_folder = f"{self.results_folder}/train_{self.running_time}"
        os.makedirs(self.current_train_folder, exist_ok=True)

        self.figures_folder = self.current_train_folder + "/figs"
        os.makedirs(self.figures_folder, exist_ok=True)

        self.data_folder = self.current_train_folder + "/data"
        os.makedirs(self.data_folder, exist_ok=True)

        self.cic_folder = self.current_train_folder + "/cic"
        os.makedirs(self.cic_folder, exist_ok=True)

        self.rl_models_folder = self.current_train_folder + "/models"
        os.makedirs(self.rl_models_folder, exist_ok=True)

        self.rl_stats_folder = self.current_train_folder + "/rl_stats"
        os.makedirs(self.rl_stats_folder, exist_ok=True)

        self.configs_folder = self.current_train_folder + "/configs"
        os.makedirs(self.configs_folder, exist_ok=True)

        self.prefilled_actions_file = os.getcwd() + "/prefilled-actions.txt"

        # Network Hosts
        self.hosts_topo_file_name      = hosts_topo_file_name
        self.hosts_topo_file_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../input-data')
        self.hosts_topo_file_path      = f'{self.hosts_topo_file_directory}/{self.hosts_topo_file_name}'
        self.client_hosts_list         = []
        self.host_default_switch_relation   = {}
        self.router_to_host_relation        = {}
        self.host_to_router_relation        = {}
        self.router_switches_list           = []
        self.router_to_controlled_switch_relation   = {}
        self.controlled_switch_to_router_relation   = {}
        self.read_hosts_topology_file()

        # RL inputs
        self.episodes               = episodes
        self.steps                  = steps
        self.epsilon_decay          = epsilon_decay
        self.nbr_controlled_switches = nbr_controlled_switches

    def read_hosts_topology_file(self):
        import os as _cfg_os
        _sfx = _cfg_os.environ.get("TRIAL_ID", "").replace("_t", "t")
        print(f"(Reinforcement) ==> Reading hosts from {self.hosts_topo_file_path}")
        with open(self.hosts_topo_file_path) as f:
            data = json.load(f)
            self.hosts_raw_topo = data

        for host in self.hosts_raw_topo:
            if not host.startswith("h"):
                raise Exception(f"Invalid host name: {host}")
            self.client_hosts_list.append(host)
            _dp = self.hosts_raw_topo[host]['default_path_switch']
            _dp_sfx = f"{_dp}{_sfx}" if _sfx else _dp
            _router = self.hosts_raw_topo[host]['router_switch']
            _router_sfx = f"{_router}{_sfx}" if _sfx else _router
            self.host_default_switch_relation[host] = {
                'default_path_switch': _dp_sfx}
            self.router_to_host_relation[_router_sfx] = {'host': host}
            self.host_to_router_relation[host] = {'router': _router_sfx}
            self.router_switches_list.append(_router_sfx)
            self.router_to_controlled_switch_relation[_router_sfx] = {
                'controlled_switch': _dp_sfx}
            if _dp_sfx in self.controlled_switch_to_router_relation:
                self.controlled_switch_to_router_relation[_dp_sfx]['routers'].append(_router_sfx)
            else:
                self.controlled_switch_to_router_relation[_dp_sfx] = {'routers': [_router_sfx]}
