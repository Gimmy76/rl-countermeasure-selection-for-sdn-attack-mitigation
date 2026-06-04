import time
import os
import signal
import subprocess
import requests
import json

class CmdManager:

    def __init__(self, config):
        print("(Reinforcement) CmdManager.__init__()")
        self.config = config
        self.network_subprocess = None
        self.tshark_sniffing_subprocess = None

    def start_network_in_background(self, servers, attackers, hosts_topo_file_name, nbr_controlled_switches):
        print("(Reinforcement) ----> Starting network in background")
        
        # Pre-cleanup: kill any existing network processes and free the port
        self._pre_cleanup()
        
        cmd = self.config.network_command
        
        # Trasformiamo le liste ['hs'] in stringhe pulite hs
        servers_str = " ".join(servers) if isinstance(servers, list) else servers
        attackers_str = " ".join(attackers) if isinstance(attackers, list) else attackers
        
        cmd = cmd.replace('[SERVERS]', servers_str) \
            .replace('[ATTACKERS]', attackers_str) \
            .replace('[HOST_BW]', '3.1') \
            .replace('[NBR_CONTROLLED_SWITCHES]', str(nbr_controlled_switches)) \
            .replace('[HOSTS_FILE]', hosts_topo_file_name)
        
        # Usiamo il PYTHONPATH locale corretto
        path_prefix = f"sudo -E TRIAL_ID={os.environ.get('TRIAL_ID', '')} QOSENTRY_TMP={os.environ.get('QOSENTRY_TMP', '/tmp/qosentry')} PYTHONPATH=$PYTHONPATH:. "
        cmd = path_prefix + cmd
        
        print(f"(Reinforcement) ----> Executing {cmd}")
        os.makedirs(self.config.network_log_dir, exist_ok=True)
        stdout_path = os.path.join(self.config.network_log_dir, 'network_stdout.log')
        stderr_path = os.path.join(self.config.network_log_dir, 'network_stderr.log')

        stdout_file = open(stdout_path, 'a', encoding='utf-8')
        stderr_file = open(stderr_path, 'a', encoding='utf-8')
        self.network_subprocess = subprocess.Popen(
            cmd,
            shell=True,
            stdin=subprocess.PIPE,
            stdout=stdout_file,
            stderr=stderr_file,
            preexec_fn=os.setsid
        )

        print(f"(Reinforcement) Waiting up to 20 seconds for network process to start...")
        for retry in range(20):
            if self.network_subprocess.poll() is not None:
                break
            time.sleep(1)
        if self.network_subprocess.poll() is not None:
            print(f"(Reinforcement) ERROR: network process exited immediately with code {self.network_subprocess.poll()}")
            try:
                with open(stderr_path, 'r') as _f:
                    _tail = _f.readlines()[-30:]
                print(f"(Reinforcement) --- Last lines of {stderr_path} ---")
                print("".join(_tail))
            except Exception:
                pass
            raise RuntimeError("Network subprocess exited before startup")

        self._wait_for_flask_ready()
        self._wait_for_hosts_ready()
        print("(Reinforcement) --> Network started")

    def _pre_cleanup(self):
        """Pre-cleanup ISOLATO al singolo trial — non tocca altri trial paralleli."""
        import fcntl
        trial_id = os.environ.get("TRIAL_ID", "")
        suffix = trial_id.replace("_t", "t")
        # Lock per serializzare ovs-vsctl tra trial concorrenti ed evitare
        # interferenze sul demone OVS condiviso.
        _lock_path = f"/tmp/qosentry_ovs_cleanup_{suffix or 'default'}.lock"
        _lock_file = open(_lock_path, "w")
        try:
            fcntl.flock(_lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"(Reinforcement) WARNING: cleanup già in corso per trial '{suffix}', skip")
            _lock_file.close()
            return
        
        if trial_id.startswith("_t"):
            try:
                flask_port = 5000 + int(trial_id.replace("_t", ""))
            except ValueError:
                flask_port = 5001
        else:
            flask_port = 5001
        
        print(f"(Reinforcement) ----> Pre-cleanup ISOLATO trial='{trial_id}' porta={flask_port}")
        
        # Kill processi precedenti del proprio trial (solo EntryPoint e ApiManager)
        # NON usare pkill su TRIAL_ID perché ucciderebbe il processo corrente!
        if trial_id:
            try:
                flask_port_to_kill = flask_port
                subprocess.run(
                    f"sudo fuser -k {flask_port_to_kill}/tcp || true",
                    shell=True, timeout=10
                )
            except subprocess.TimeoutExpired:
                pass
        
        # Aspetta che la porta Flask sia libera prima di procedere
        import time as _time
        for _attempt in range(20):
            result = subprocess.run(
                f"sudo fuser {flask_port}/tcp 2>/dev/null",
                shell=True, capture_output=True, text=True
            )
            if not result.stdout.strip():
                print(f"(Reinforcement) ----> Porta {flask_port} libera")
                break
            occupying_pid = result.stdout.strip()
            print(f"(Reinforcement) ----> Porta {flask_port} occupata da PID {occupying_pid}, aspetto...")
            # Killa solo se è un processo vecchio (non il corrente)
            current_pid = str(os.getpid())
            if occupying_pid != current_pid:
                subprocess.run(f"sudo kill -9 {occupying_pid} 2>/dev/null", shell=True)
            _time.sleep(2)
        
        # Rimuovi SOLO i bridge OVS del proprio trial
        if suffix:
            try:
                cmd = (
                    f"for br in $(sudo ovs-vsctl list-br 2>/dev/null | grep '{suffix}$'); do "
                    f"  sudo ovs-vsctl --if-exists del-br $br 2>/dev/null; "
                    f"done"
                )
                subprocess.run(cmd, shell=True, timeout=15)
            except subprocess.TimeoutExpired:
                pass
        else:
            try:
                subprocess.run(
                    "sudo ovs-vsctl --if-exists del-br $(sudo ovs-vsctl list-br 2>/dev/null) || true",
                    shell=True, timeout=10
                )
            except subprocess.TimeoutExpired:
                pass
        
        # Rimuovi le interfacce veth orfane del proprio trial.
        # OVS le lascia nel kernel se EntryPoint e' stato killato brutalmente,
        # e EntryPoint del prossimo episodio crasha con "RTNETLINK File exists".
        # Pattern: <nome>t{N}-eth<X>  oppure  <nome>t{N}  (fine stringa)
        # suffix='t1' matcha s1t1-eth101, s101t1-eth1, nat0t1 ma NON t2/t3...
        if suffix:
            try:
                cmd_veth = (
                    "for iface in $(ip -o link show 2>/dev/null "
                    "| awk -F': ' '{print $2}' | cut -d'@' -f1 "
                    f"| grep -E '{suffix}(-eth|$)'); do "
                    "  sudo ip link delete \"$iface\" 2>/dev/null || true; "
                    "done"
                )
                subprocess.run(cmd_veth, shell=True, timeout=20)
                print(f"(Reinforcement) ----> Interfacce veth trial '{suffix}' rimosse")
            except subprocess.TimeoutExpired:
                print(f"(Reinforcement) WARNING: timeout rimozione veth '{suffix}'")
        
        time.sleep(2)
        print(f"(Reinforcement) <--- Pre-cleanup trial='{trial_id}' completato")
        try:
            fcntl.flock(_lock_file, fcntl.LOCK_UN)
        except Exception:
            pass
        _lock_file.close()

    def _wait_for_flask_ready(self, max_retries=60, retry_interval=0.5):
        """Aspetta che Flask sia pronto con retry veloci."""
        url = f"{self.config.api_link}/"
        
        for attempt in range(1, max_retries + 1):
            try:
                response = requests.get(url, timeout=2)
                if response.status_code == 200:
                    elapsed = (attempt * retry_interval)
                    print(f"(Reinforcement) --> Flask ready after {elapsed:.1f}s (attempt {attempt}/{max_retries})")
                    return
            except requests.exceptions.ConnectionError as e:
                if attempt % 10 == 0:  # Stampa ogni 10 tentativi per ridurre verbosità
                    elapsed = (attempt * retry_interval)
                    print(f"(Reinforcement) --> Waiting for Flask... {elapsed:.1f}s ({attempt}/{max_retries})")
            except Exception as e:
                if attempt % 10 == 0:
                    print(f"(Reinforcement) --> Flask connection error: {type(e).__name__}: {str(e)[:50]} ({attempt}/{max_retries})")
            
            time.sleep(retry_interval)
        

        print(f"(Reinforcement) WARNING: Flask not ready after {max_retries * retry_interval:.0f}s. Continuing anyway...")

    def _wait_for_hosts_ready(self, max_retries=90, retry_interval=1):
        """Aspetta che il primo host abbia IP valido e che il suo switch di default sia pronto."""
        first_host = self.config.client_hosts_list[0] if getattr(self.config, 'client_hosts_list', None) else 'h1'

        first_switch = None
        if getattr(self.config, 'host_default_switch_relation', None):
            first_switch = self.config.host_default_switch_relation.get(first_host, {}).get('default_path_switch')
        if not first_switch and getattr(self.config, 'router_switches_list', None):
            first_switch = self.config.router_switches_list[0]
        if not first_switch:
            first_switch = 's101'

        url_ip = f"{self.config.api_link}/host-ip/{first_host}"
        url_sw = f"{self.config.api_link}/get_dst_switches/{first_switch}"

        print(f"(RL) --> Readiness check: host={first_host}, switch={first_switch}, api={self.config.api_link}")
        for attempt in range(1, max_retries + 1):
            ip_ok = False
            sw_ok = False
            host_ip_text = "N/A"
            switch_text = "N/A"
            try:
                # Verifica IP host
                resp_ip = requests.get(url_ip, timeout=5)
                host_ip_text = resp_ip.text.strip()
                ip_ok = (resp_ip.status_code == 200
                         and host_ip_text
                         and host_ip_text != 'UNKNOWN'
                         and len(host_ip_text) <= 15
                         and all(c.isdigit() or c == '.' for c in host_ip_text))

                # Verifica switch pronti
                sw_resp = requests.get(url_sw, timeout=5)
                switch_text = sw_resp.text.strip()
                sw_ok = (sw_resp.status_code == 200 and switch_text != '')
                if sw_ok:
                    try:
                        sw_ok = 'dst_switches' in sw_resp.json()
                    except Exception:
                        sw_ok = False

                if ip_ok and sw_ok:
                    print(f"(RL) --> Network ready: {first_host}={host_ip_text}, switch {first_switch} OK (attempt {attempt})")
                    time.sleep(5)  # extra buffer
                    return

                print(f"(RL) --> Not ready yet: attempt {attempt}/{max_retries} - ip_ok={ip_ok}, sw_ok={sw_ok}")
                print(f"(RL) --> host-ip response: status={resp_ip.status_code}, body='{host_ip_text}'")
                print(f"(RL) --> get_dst_switches response: status={sw_resp.status_code}, body='{switch_text}'")
            except Exception as e:
                print(f"(RL) --> Wait error: {type(e).__name__}: {e} ({attempt}/{max_retries})")
            time.sleep(retry_interval)

        print(f"(RL) WARNING: Network not fully ready after {max_retries * retry_interval:.0f}s. Continuing anyway...")

    def stop_network(self):
        print(f"(Reinforcement) ----> Stopping network (killing process group)")
        try:
            pgid = os.getpgid(self.network_subprocess.pid)
            os.killpg(pgid, signal.SIGTERM)
            try:
                self.network_subprocess.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(pgid, signal.SIGKILL)
                self.network_subprocess.wait(timeout=5)
        except ProcessLookupError:
            pass
        except Exception as e:
            print(f"(Reinforcement) WARNING: stop_network error: {e}")
        time.sleep(2)
        # Cleanup isolato delle interfacce/bridge del proprio trial subito
        # dopo aver ucciso EntryPoint, cosi' il prossimo episodio parte pulito.
        try:
            self._pre_cleanup()
        except Exception as e:
            print(f"(Reinforcement) WARNING: post-stop cleanup error: {e}")
        print("(Reinforcement) <-- Network stopped")

    def get_tshark_interfaces(self):
        print(f"(Reinforcement) ----> Executing {self.config.tshark_interfaces_command}")
        return subprocess.Popen(self.config.tshark_interfaces_command, shell=True,
                                stdout=subprocess.PIPE).stdout.read().decode("ascii").split('\n')

    def start_tshark_sniffing(self, interfaces_ids):
        try:
            os.remove(self.config.tshark_pcap_file_path)
        except FileNotFoundError:
            pass
        cmd = self.config.tshark_sniffing_command.replace('[INTERFACES]', interfaces_ids)
        print(f"(Reinforcement) ----> Executing {cmd}")
        print(f"(Reinforcement) ----> PCAP will be saved to: {self.config.tshark_pcap_file_path}")
        self.tshark_sniffing_subprocess = subprocess.Popen(
            cmd, shell=True, stdin=subprocess.PIPE, preexec_fn=os.setsid)
        time.sleep(2)
        os.system(f'chmod 777 {self.config.tshark_pcap_file_path} 2>/dev/null || true')
        print("(Reinforcement) --> TShark sniffing started")

    def stop_tshark_sniffing(self):
        os.killpg(os.getpgid(self.tshark_sniffing_subprocess.pid), signal.SIGTERM)
        time.sleep(2)
        os.system(f'chmod 644 {self.config.tshark_pcap_file_path} 2>/dev/null || true')
        print("(Reinforcement) <-- TShark sniffing stopped")

    # run_cic() ELIMINATO — sostituito da NetMetricsCalculator con flag -cic

    def run_network_metrics_calculator(self, server_ip, server_port, hosts_ips, duration_s, packet_bytes):
        """
        Chiama NetMetricsCalculator.py che ora produce in un unico passaggio:
          - metrics.json  (latency, jitter, throughput, APTT per host)
          - cic_flow.csv  (equivalente alle colonne CIC usate da Environment.py)
        Nessuna JVM Java, nessun CICFlowMeter.
        """
        print('(Reinforcement) --> Running NetMetricsCalculator started')

        try:
            os.remove(self.config.net_metrics_result_file_path)
        except (FileNotFoundError, PermissionError):
            os.system(f'sudo rm -f {self.config.net_metrics_result_file_path}')
        os.system(f'sudo chmod -R 777 {self.config.tmp_dir} 2>/dev/null')

        if not self._is_valid_ip(server_ip):
            print(f"(Reinforcement) ERROR: Invalid server IP. Writing defaults.")
            self._write_default_metrics(hosts_ips)
            self._write_default_cic(hosts_ips, server_ip)
            return

        valid_ips = [ip.strip() for ip in hosts_ips if self._is_valid_ip(ip)]
        if not valid_ips:
            print("(Reinforcement) ERROR: No valid IPs. Writing defaults.")
            self._write_default_metrics(hosts_ips)
            self._write_default_cic(hosts_ips, server_ip)
            return

        cmd = self.config.net_metrics_command \
            .replace('[SERVER_IP]',   server_ip) \
            .replace('[SERVER_PORT]', str(server_port)) \
            .replace('[HOSTS_IPS]',   str(valid_ips).replace('\'', '').replace(' ', '')) \
            .replace('[DURATION]',    str(duration_s)) \
            .replace('[BYTES]',       str(packet_bytes)) \
            .replace('[CIC_OUTPUT]',  self.config.cic_output_file_path)

        import os as _os
        _qos_tmp = _os.environ.get('QOSENTRY_TMP', '/tmp/qosentry')
        cmd = f"sudo QOSENTRY_TMP={_qos_tmp} PYTHONPATH=$PYTHONPATH:. " + cmd
        print(f"(Reinforcement) ----> Executing {cmd}")
        start_time = time.time()
        subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE).communicate()
        self.last_netmetrics_duration = time.time() - start_time

        if not os.path.exists(self.config.net_metrics_result_file_path):
            print("(Reinforcement) WARNING: metrics.json missing. Writing defaults.")
            self._write_default_metrics(valid_ips)

        if not os.path.exists(self.config.cic_output_file_path):
            print("(Reinforcement) WARNING: cic_flow.csv missing. Writing defaults.")
            self._write_default_cic(valid_ips, server_ip)

        print('(Reinforcement) <-- NetMetricsCalculator finished')

    def _is_valid_ip(self, ip):
        ip = str(ip).strip()
        return bool(ip) and len(ip) <= 15 and all(c.isdigit() or c == '.' for c in ip)

    def _write_default_metrics(self, hosts_ips):
        default = {"avg_latency_s": 1.0, "avg_packet_transmission_time_s": 1.0,
                   "throughput_bps": 0.01, "avg_jitter_s": 1.0}
        data = {str(ip).strip(): default for ip in hosts_ips}
        path = self.config.net_metrics_result_file_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def _write_default_cic(self, hosts_ips, server_ip):
        import csv
        path = self.config.cic_output_file_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fields = ['Src IP','Dst IP','Dst Port','Flow Pkts/s','Flow Byts/s',
                  'Tot Fwd Pkts','Tot Bwd Pkts','TotLen Fwd Pkts','TotLen Bwd Pkts','ACK Flag Cnt']
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for ip in hosts_ips:
                w.writerow({'Src IP': str(ip).strip(), 'Dst IP': server_ip,
                            'Dst Port': '80', 'Flow Pkts/s': '0.0', 'Flow Byts/s': '0.0',
                            'Tot Fwd Pkts': '0', 'Tot Bwd Pkts': '0',
                            'TotLen Fwd Pkts': '0', 'TotLen Bwd Pkts': '0', 'ACK Flag Cnt': '0'})

    def read_ditg_logs(self):
        subprocess.Popen(self.config.ditg_logs_command, shell=True, stdin=subprocess.PIPE).communicate()