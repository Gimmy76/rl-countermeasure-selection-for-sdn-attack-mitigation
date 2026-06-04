# Flask
import os
import subprocess
import time

from flask import Flask, jsonify
import threading
import Shared as shared
from decimal import Decimal
import re

# Intermed Imports
from intermed.OvsIntermediateMininet import *
from intermed.OvsIntermediate import *


def get_trial_port():
    """Porta Flask univoca per trial: 5001 + indice."""
    trial_id = os.environ.get("TRIAL_ID", "")
    if trial_id.startswith("_t"):
        try:
            return 5000 + int(trial_id.replace("_t", ""))
        except ValueError:
            return 5001
    return 5001

# Trial isolation per parallelizzazione
_TRIAL_SFX = os.environ.get("TRIAL_ID", "").replace("_t", "t")

def _pid_path(prefix, host):
    """Pid file isolato per trial: evita conflitti tra trial paralleli."""
    sfx = f"_{_TRIAL_SFX}" if _TRIAL_SFX else ""
    return f"/tmp/{prefix}_{host}{sfx}.pid"

def _log_path(prefix, host):
    """Log file isolato per trial."""
    sfx = f"_{_TRIAL_SFX}" if _TRIAL_SFX else ""
    return f"/tmp/{prefix}_{host}{sfx}.log"


app = Flask(__name__)






@app.route("/")
def mininet_network_up_page():
    return "<p>Network is up!</p>"

@app.route("/get-host-names")
def get_host_names():
    global GLOBALS
    try:
        return jsonify([key for key in GLOBALS.network_spec.get('hosts', {})])
    except Exception:
        return jsonify([])

@app.route("/get-switches-interfaces")
def get_switches_interfaces():
    global GLOBALS
    try:
        if 'hosts' not in GLOBALS.network_spec:
            return jsonify([])
        return jsonify([shared.get_host_status(key)["dst_int"] for key in GLOBALS.network_spec['hosts']])
    except Exception:
        return jsonify([])

@app.route("/host-ip/<host_name>")
def get_ip_by_host_name(host_name):
    global GLOBALS
    try:
        ip = GLOBALS.net[host_name].IP()
        if ip is None or ip == '':
            return 'UNKNOWN'
        return ip
    except Exception as e:
        print(f'[Flask ERROR] get_ip_by_host_name({host_name}): {repr(e)}')
        return 'UNKNOWN'

@app.route("/host-status/<host_name>")
def get_host_status(host_name):
    global GLOBALS
    return shared.get_host_status(host_name)

@app.route("/host-status-connected/<host_name>")
def get_host_status_connected(host_name):
    global GLOBALS
    return str(shared.get_host_status(host_name)['connected'])

@app.route("/get_switch-status-connected/<src_switch>")
def get_switch_status_connected(src_switch):
    global GLOBALS
    return {
        dst: GLOBALS.network_spec['switches'][src_switch]['connections'][dst]['connected']
        for dst in GLOBALS.network_spec['switches'][src_switch]['connections']
    }

@app.route("/get_dst_switches/<src_switch>")
def get_dst_switches(src_switch):
    global GLOBALS
    return {'dst_switches': list(GLOBALS.network_spec['switches'][src_switch]['connections'].keys())}

@app.route("/get_switch_bw/<src_switch>/<dst_switch>")
def get_switch_bw(src_switch, dst_switch):
    global GLOBALS
    return {'bw': GLOBALS.network_spec['switches'][src_switch]['connections'][dst_switch]['bw']}

@app.route("/get_link_information/<src_switch>/<dst_switch>")
def get_link_information(src_switch, dst_switch):
    global GLOBALS
    try:
        link_info = {'tx_bytes': 0, 'rx_bytes': 0, 'bw': '0'}

        # Verifica che la connessione esista
        if src_switch not in GLOBALS.network_spec['switches']:
            return link_info
        if dst_switch not in GLOBALS.network_spec['switches'][src_switch]['connections']:
            return link_info

        conn = GLOBALS.network_spec['switches'][src_switch]['connections'][dst_switch]
        src_int = conn['src_int']
        dst_int = conn['dst_int']
        link_info['bw'] = conn['bw']

        # Leggi statistiche src
        raw = GLOBALS.net[src_switch].cmd(
            f'ovs-vsctl get interface {src_int} statistics'
        )
        if raw and '{' in raw:
            for stat in raw.replace("{","").replace("}","").split(","):
                item = stat.strip().split('=')
                if len(item) == 2 and item[0].strip() == 'tx_bytes':
                    link_info['tx_bytes'] = int(item[1].strip())
                    break

        # Leggi statistiche dst
        raw2 = GLOBALS.net[dst_switch].cmd(
            f'ovs-vsctl get interface {dst_int} statistics'
        )
        if raw2 and '{' in raw2:
            for stat in raw2.replace("{","").replace("}","").split(","):
                item = stat.strip().split('=')
                if len(item) == 2 and item[0].strip() == 'rx_bytes':
                    link_info['rx_bytes'] = int(item[1].strip())
                    break

        return link_info

    except Exception as e:
        print(f'[Flask ERROR] get_link_information({src_switch},{dst_switch}): {repr(e)}')
        return {'tx_bytes': 0, 'rx_bytes': 0, 'bw': '0'}

@app.route("/change-host-status/<host_name>")
def change_host_status(host_name):
    global GLOBALS
    host_status = shared.get_host_status(host_name)
    host_ip = host_status['ip']
    connected_switch = host_status['connected-switch']
    switch_port = host_status['switch-port']
    if host_status['connected']:
        GLOBALS.net[connected_switch].cmd(shared.get_host_switch_turn_off_link_command(host_ip, connected_switch))
        GLOBALS.network_spec['hosts'][host_name]['connected'] = False
        return f'the link of {host_name} is turned off successfully'
    else:
        GLOBALS.net[connected_switch].cmd(shared.get_host_switch_turn_on_link_command(host_ip, connected_switch, switch_port))
        GLOBALS.network_spec['hosts'][host_name]['connected'] = True
        return f'the link of {host_name} is turned on successfully'

@app.route("/get_host_path/<host_name>")
def get_host_path(host_name):
    global GLOBALS
    result = {
        'current': [],
        'default': GLOBALS.network_spec['hosts'][host_name]['default_path_switch'],
        'options': [],
        'router': GLOBALS.network_spec['hosts'][host_name]['router_switch']
    }
    for switch, active in GLOBALS.network_spec['hosts'][host_name]['current_path'].items():
        result['current'].append(switch) if active else result['options'].append(switch)
    return jsonify(result)

# ---------------------------------------------------------------------------
# MHDDoS — VM VERSION
# usa cmd() in background invece di makeTerm (no display su VM headless).
# Il PID viene salvato per un kill selettivo che NON uccide altri python3
# (es. TcpClient, TcpServer) sullo stesso host.
# ---------------------------------------------------------------------------

@app.route("/start-mhddos/<attacker_host>/<victim_host>/<attack_type>")
def start_mhddos_attack(attacker_host, victim_host, attack_type):
    global GLOBALS
    victim_ip = shared.get_host_status(victim_host)['ip']
    terminal_name = f'mhddos-{attacker_host}-{victim_host}'

    # Avvia MHDDoS in background e salva il PID nel file /tmp/mhddos_<host>.pid
    # dentro il namespace di rete dell'host Mininet
    pid_file = _pid_path('mhddos', attacker_host)
    GLOBALS.net[attacker_host].cmd(
        f"python3 {GLOBALS.mhddos_start_path} {attack_type} {victim_ip}:80 20 30000 & "
        f"echo $! > {pid_file}"
    )
    # Salviamo nome host + pid_file (non un oggetto xterm)
    GLOBALS.ddos_flooding_attacks[terminal_name] = {'host': attacker_host, 'pid_file': pid_file}
    log = f"Starting attack --> Attacker: {attacker_host} --> Victim: {victim_host}"
    print(f'(Network) ==> {log}')
    return log

@app.route("/stop-mhddos/<attacker_host>/<victim_host>")
def stop_mhddos_attack(attacker_host, victim_host):
    global GLOBALS
    terminal_name = f'mhddos-{attacker_host}-{victim_host}'

    if terminal_name in GLOBALS.ddos_flooding_attacks:
        entry = GLOBALS.ddos_flooding_attacks[terminal_name]
        host = entry['host']
        pid_file = entry['pid_file']
        # Kill selettivo: legge il PID salvato e uccide solo quel processo
        # NON usa killall python3 che ucciderebbe anche TcpClient/TcpServer
        GLOBALS.net[host].cmd(
            f"if [ -f {pid_file} ]; then "
            f"  pid=$(cat {pid_file}); "
            f"  kill $pid 2>/dev/null; "
            f"  pkill -P $pid 2>/dev/null; "
            f"  rm -f {pid_file}; "
            f"fi"
        )
        del GLOBALS.ddos_flooding_attacks[terminal_name]

    log = f"Stopping attack --> Attacker: {attacker_host} --> Victim: {victim_host}"
    print(f'(Network) ==> {log}')
    return log

# ---------------------------------------------------------------------------
# TCP flows — cmd() in background, kill selettivo tramite PID file
# ---------------------------------------------------------------------------

@app.route("/start-tcp-flow/<source_host>/<destination_host>/<duration_ms>")
def start_tcp_flow(source_host, destination_host, duration_ms):
    global GLOBALS
    thread = threading.Thread(
        target=start_tcp_flow_thread,
        args=(source_host, destination_host, duration_ms,),
        daemon=True
    )
    thread.start()
    log = f"Starting flow --> Sender: {source_host} --> Receiver: {destination_host}"
    print(f'(Network) ==> {log}')
    return log

def start_tcp_flow_thread(source_host, destination_host, duration_ms):
    """
    Avvia il client TCP usando popen() per maggior affidabilità.
    popen() assicura il processo sia nel namespace Mininet e ritorna il PID diretto.
    """
    global GLOBALS
    destination_host_status = shared.get_host_status(destination_host)
    source_terminal_name = f'tcp-flow-{source_host}-{destination_host}-src'
    duration_s = int(int(duration_ms) / 1000)
    pid_file = _pid_path('tcpflow', source_host)
    log_file = _log_path('tcpflow', source_host)
    
    print(f'(Network) ==> TCP flow {source_host}->{destination_host} for {duration_s}s')
    
    try:
        # USO popen() per lanciare il client TCP nel namespace Mininet
        with open(log_file, 'w') as logf:
            proc = GLOBALS.net[source_host].popen(
                f"python3 {GLOBALS.tcp_flow_client_file} -n {source_host}"
                f" -ip {destination_host_status['ip']} -t {duration_s} -np 1000",
                stdout=logf,
                stderr=subprocess.STDOUT,
                shell=False
            )
        
        # Salva il PID per later cleanup
        with open(pid_file, 'w') as pf:
            pf.write(str(proc.pid))
        
        GLOBALS.tcp_flows[source_terminal_name] = {
            'host': source_host,
            'pid': proc.pid,
            'pid_file': pid_file,
            'process': proc,
            'log_file': log_file
        }
        
        print(f'(Network) ==> TcpClient started {source_host}->{destination_host} (PID {proc.pid})')
        
    except Exception as e:
        print(f'(Network) ERROR: Failed to start TcpClient on {source_host}: {e}')

@app.route("/stop-all-tcp-flows")
def stop_all_tcp_flows():
    global GLOBALS
    for name, entry in list(GLOBALS.tcp_flows.items()):
        try:
            # Termina tramite il riferimento diretto del processo se disponibile
            if 'process' in entry:
                proc = entry['process']
                if proc.poll() is None:  # Se è ancora in esecuzione
                    proc.terminate()
                    proc.wait(timeout=2)
        except Exception as e:
            print(f'(Network) ==> Warning stopping tcp flow {name}: {e}')
        
        # Fallback: kill via command sul host Mininet tramite PID file
        try:
            host = entry['host']
            pid_file = entry['pid_file']
            GLOBALS.net[host].cmd(
                f"if [ -f {pid_file} ]; then "
                f"  pid=$(cat {pid_file}); "
                f"  kill $pid 2>/dev/null; "
                f"  rm -f {pid_file}; "
                f"fi"
            )
        except Exception as e:
            print(f'(Network) ==> Warning stopping tcp flow {name}: {e}')
    
    GLOBALS.tcp_flows.clear()
    print(f'(Network) ==> Stopped all TCP flows')
    return "Stopped all TCP flows"

# ---------------------------------------------------------------------------
# TCP receivers — cmd() in background, PID file per kill selettivo
# ---------------------------------------------------------------------------

@app.route("/reset-tcp-receivers")
def reset_tcp_receivers():
    """
    Avvia i server TCP sui receiver host usando popen() invece di cmd().
    popen() è preferibile perché:
    - Ritorna un oggetto Popen con controllo diretto
    - Assicura che il processo sia nel namespace Mininet
    - Permette stdout/stderr redirection affidabile
    """
    global GLOBALS
    server_port = 80
    info("(Network) ==> stopping server...\n")
    
    for host_name in GLOBALS.servers:
        check_port_used_and_kill_process(host_name, server_port)
    
    GLOBALS.tcp_receivers = []

    for host_name in GLOBALS.servers:
        ip = shared.get_host_status(host_name)['ip']
        pid_file = _pid_path('tcpserver', host_name)
        log_file = _log_path('tcpserver', host_name)
        
        try:
            # USO popen() per avere controllo diretto del processo nel namespace
            with open(log_file, 'w') as logf:
                proc = GLOBALS.net[host_name].popen(
                    f"python3 {GLOBALS.tcp_flow_server_file} -n {host_name} -ip {ip}",
                    stdout=logf,
                    stderr=subprocess.STDOUT,
                    shell=False
                )
            
            # Salva il PID nel file per later cleanup
            with open(pid_file, 'w') as pf:
                pf.write(str(proc.pid))
            
            GLOBALS.tcp_receivers.append({
                'host': host_name, 
                'pid': proc.pid,
                'pid_file': pid_file,
                'process': proc,
                'log_file': log_file
            })
            
            info(f"(Network) ==> TcpServer started on {host_name} (PID {proc.pid})\n")
            
        except Exception as e:
            info(f"(Network) ERROR: Failed to start TcpServer on {host_name}: {e}\n")
            print(f"[ERROR] TcpServer launch failed: {e}")
    
    time.sleep(2)  # Wait for servers to stabilize
    
    # DEBUGGING: Verifica che i server siano effettivamente in ascolto
    info("(Network) ==> DEBUGGING: Verifying TcpServer deployment\n")
    for receiver_entry in GLOBALS.tcp_receivers:
        host_name = receiver_entry['host']
        log_file = receiver_entry['log_file']
        
        # Leggi il log del server per diagnostica
        try:
            with open(log_file, 'r') as f:
                log_content = f.read()
                if log_content.strip():
                    info(f"(Network) ==> Log for {host_name}:\n{log_content}\n")
        except:
            pass
        
        # Verifica se il server è in ascolto sulla porta 80
        try:
            listening = GLOBALS.net[host_name].cmd(f"ss -lptn | grep ':80'")
            if listening:
                info(f"(Network) ==> {host_name} is listening on port 80:\n{listening}\n")
            else:
                info(f"(Network) WARNING: {host_name} is NOT listening on port 80\n")
        except Exception as e:
            info(f"(Network) ERROR checking listening ports on {host_name}: {e}\n")
        
        # Verifica il namespace Mininet
        try:
            ns_result = GLOBALS.net[host_name].cmd("ip netns identify || echo 'root'")
            info(f"(Network) ==> Namespace of {host_name}: {ns_result.strip()}\n")
        except Exception as e:
            info(f"(Network) ERROR checking namespace on {host_name}: {e}\n")
        
        # Verifica che l'IP sia configurato
        try:
            ip_result = GLOBALS.net[host_name].cmd("ip addr show | grep -E 'inet.*10.0'")
            info(f"(Network) ==> IPs on {host_name}: {ip_result.strip()}\n")
        except Exception as e:
            info(f"(Network) ERROR checking IPs on {host_name}: {e}\n")
    
    log = f"Resetting TCP for hosts: {GLOBALS.servers}"
    info(f'(Network) ==> {log}\n')
    return log

@app.route("/stop-tcp-receivers")
def stop_tcp_receivers():
    global GLOBALS
    server_port = 80
    info("(Network) ==> stopping server...\n")
    
    for receiver_entry in GLOBALS.tcp_receivers:
        host_name = receiver_entry['host']
        try:
            # Termina il processo tramite il riferimento diretto
            if 'process' in receiver_entry:
                proc = receiver_entry['process']
                if proc.poll() is None:  # Se è ancora in esecuzione
                    proc.terminate()
                    proc.wait(timeout=2)
                    info(f"(Network) ==> Terminated process for {host_name}\n")
        except Exception as e:
            info(f"(Network) WARNING: Error terminating process for {host_name}: {e}\n")
        
        # Fallback: kill via command sul host Mininet
        try:
            check_port_used_and_kill_process(host_name, server_port)
        except:
            pass
    
    GLOBALS.tcp_receivers = []
    log = f"Stopped TCP for hosts: {GLOBALS.servers}"
    info(f'(Network) ==> {log}\n')
    return log

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_pid_using_port(host_name, port):
    used_port_result = GLOBALS.net[host_name].cmd(f"ss -lptn 'sport = :{port}'")
    info(used_port_result + "\n")
    match = re.compile(r'pid=(\d+)').search(used_port_result)
    if match:
        pid_value = match.group(1)
        info(f"(Network) ==> port <{port}> in host <{host_name}> used by pid <{pid_value}>\n")
        return pid_value
    return 'None'

def check_port_used_and_kill_process(host_name, port):
    pid_value = get_pid_using_port(host_name, port)
    if pid_value != 'None':
        info(f"(Network) ==> killing pid <{pid_value}> on port <{port}> in <{host_name}>\n")
        info(GLOBALS.net[host_name].cmd(f"kill {pid_value}") + "\n")

@app.route("/get-host-interface-statistics/<host_name>")
def get_host_interface_statistics(host_name):
    global GLOBALS
    host_status = shared.get_host_status(host_name)
    return GLOBALS.net[host_status['router_switch']].cmd(
        f'ovs-vsctl get interface {host_status["dst_int"]} statistics'
    )

@app.route("/get-host-ifconfig/<host>")
def get_host_ifconfig(host):
    global GLOBALS
    return GLOBALS.net[host].cmd('ifconfig')

@app.route("/get-switch-statistics/<switch>/<interface_name>")
def get_switch_interface_statistics(switch, interface_name):
    global GLOBALS
    return GLOBALS.net[switch].cmd(f'ovs-vsctl get interface {interface_name} statistics')

@app.route("/get-host-bw/<host>")
def get_host_bw(host):
    global GLOBALS
    try:
        return jsonify({'bw': str(shared.get_host_status(host)['bw'])})
    except Exception as e:
        print(f'[Flask ERROR] get_host_bw({host}): {repr(e)}')
        return jsonify({'bw': '0'})

@app.route("/increase-host-bw/<host>/<change>")
def increase_host_bw(host, change):
    global GLOBALS
    host_spec = shared.get_host_status(host)
    new_bw = Decimal(host_spec['bw']) + Decimal(change)
    GLOBALS.network_spec['hosts'][host]['bw'] = new_bw
    GLOBALS.net[host].intf(f'{host}-eth0').config(bw=new_bw, smooth_change=False)
    sn = host_spec['connected-switch']
    GLOBALS.net[sn].intf(f'{sn}-eth{host_spec["switch-port"]}').config(bw=new_bw, smooth_change=False)
    print(f'(Network) ==> Increased bandwidth of {host} to {new_bw}')
    return 'Increased'

@app.route("/decrease-host-bw/<host>/<change>")
def decrease_host_bw(host, change):
    global GLOBALS
    host_spec = shared.get_host_status(host)
    new_bw = Decimal(host_spec['bw']) - Decimal(change)
    GLOBALS.network_spec['hosts'][host]['bw'] = new_bw
    GLOBALS.net[host].intf(f'{host}-eth0').config(bw=new_bw, smooth_change=False)
    sn = host_spec['connected-switch']
    GLOBALS.net[sn].intf(f'{sn}-eth{host_spec["switch-port"]}').config(bw=new_bw, smooth_change=False)
    print(f'(Network) ==> Decreased bandwidth of {host} to {new_bw}')
    return 'Decreased'

@app.route("/increase-switch-bw/<src_switch>/<dst_switch>/<change>")
def increase_switch_bw(src_switch, dst_switch, change):
    global GLOBALS
    current_bw = Decimal(GLOBALS.network_spec['switches'][src_switch]['connections'][dst_switch]['bw'])
    new_bw = current_bw + Decimal(change)
    GLOBALS.network_spec['switches'][src_switch]['connections'][dst_switch]['bw'] = new_bw
    GLOBALS.network_spec['switches'][dst_switch]['connections'][src_switch]['bw'] = new_bw
    src_int = GLOBALS.network_spec['switches'][src_switch]['connections'][dst_switch]['src_int']
    dst_int = GLOBALS.network_spec['switches'][src_switch]['connections'][dst_switch]['dst_int']
    GLOBALS.net[src_switch].intf(src_int).config(bw=new_bw, smooth_change=False)
    GLOBALS.net[dst_switch].intf(dst_int).config(bw=new_bw, smooth_change=False)
    print(f'(Network) ==> increase BW {src_switch}<->{dst_switch} to {new_bw}')
    return 'Switch bandwidth increased'

@app.route("/decrease-switch-bw/<src_switch>/<dst_switch>/<change>")
def decrease_switch_bw(src_switch, dst_switch, change):
    global GLOBALS
    current_bw = Decimal(GLOBALS.network_spec['switches'][src_switch]['connections'][dst_switch]['bw'])
    new_bw = current_bw - Decimal(change)
    GLOBALS.network_spec['switches'][src_switch]['connections'][dst_switch]['bw'] = new_bw
    GLOBALS.network_spec['switches'][dst_switch]['connections'][src_switch]['bw'] = new_bw
    src_int = GLOBALS.network_spec['switches'][src_switch]['connections'][dst_switch]['src_int']
    dst_int = GLOBALS.network_spec['switches'][src_switch]['connections'][dst_switch]['dst_int']
    GLOBALS.net[src_switch].intf(src_int).config(bw=new_bw, smooth_change=False)
    GLOBALS.net[dst_switch].intf(dst_int).config(bw=new_bw, smooth_change=False)
    print(f'(Network) ==> decrease BW {src_switch}<->{dst_switch} to {new_bw}')
    return 'Switch bandwidth decreased'

@app.route("/redirect_switch_flow/<host_name>/<dst_switch>")
def redirect_switch_flow(host_name, dst_switch):
    global GLOBALS
    host_mac = GLOBALS.network_spec['hosts'][host_name]['mac']
    server_mac = GLOBALS.network_spec['hosts'][GLOBALS.servers[0]]['mac']
    current_path = get_host_path(host_name)['current']
    default_switch = GLOBALS.network_spec['hosts'][host_name]['default_path_switch']
    commands = []
    for controlled_switch in current_path:
        GLOBALS.network_spec['hosts'][host_name]['current_path'][controlled_switch] = False
        commands.append(OvsOfctlDelFlowsCommand(GLOBALS.s0_switch,
                        OvsOfctlCommandArguments(mac_destination=host_mac)))
        if controlled_switch == default_switch:
            commands.append(OvsOfctlDelFlowsCommand(controlled_switch,
                            OvsOfctlCommandArguments(mac_source=host_mac, mac_destination=server_mac)))
        else:
            commands.append(OvsOfctlDelFlowsCommand(default_switch,
                            OvsOfctlCommandArguments(mac_source=host_mac, mac_destination=server_mac)))
            commands.append(OvsOfctlDelFlowsCommand(controlled_switch,
                            OvsOfctlCommandArguments(mac_source=host_mac, mac_destination=server_mac)))
            commands.append(OvsOfctlDelFlowsCommand(controlled_switch,
                            OvsOfctlCommandArguments(mac_source=server_mac, mac_destination=host_mac)))
    GLOBALS.network_spec['hosts'][host_name]['current_path'][dst_switch] = True
    s0_int = shared.get_interface_name(GLOBALS.s0_switch, dst_switch)
    commands.append(OvsOfctlAddFlowCommand(GLOBALS.s0_switch, OvsOfctlCommandArguments(
        priority=GLOBALS.highest_priority, mac_destination=host_mac,
        actions=[OvsCommandArgumentActionOutput(f"{GLOBALS.switch_interface_port_mapping[GLOBALS.s0_switch][s0_int]}")])))
    if dst_switch == default_switch:
        d_int = shared.get_interface_name(dst_switch, GLOBALS.s0_switch)
        commands.append(OvsOfctlAddFlowCommand(dst_switch, OvsOfctlCommandArguments(
            priority=GLOBALS.highest_priority, mac_source=host_mac, mac_destination=server_mac,
            actions=[OvsCommandArgumentActionOutput(f"{GLOBALS.switch_interface_port_mapping[dst_switch][d_int]}")])))
    else:
        dp_int = shared.get_interface_name(default_switch, dst_switch)
        commands.append(OvsOfctlAddFlowCommand(default_switch, OvsOfctlCommandArguments(
            priority=GLOBALS.highest_priority, mac_source=host_mac, mac_destination=server_mac,
            actions=[OvsCommandArgumentActionOutput(f"{GLOBALS.switch_interface_port_mapping[default_switch][dp_int]}")])))
        ds0 = shared.get_interface_name(dst_switch, GLOBALS.s0_switch)
        commands.append(OvsOfctlAddFlowCommand(dst_switch, OvsOfctlCommandArguments(
            priority=GLOBALS.highest_priority, mac_source=host_mac, mac_destination=server_mac,
            actions=[OvsCommandArgumentActionOutput(f"{GLOBALS.switch_interface_port_mapping[dst_switch][ds0]}")])))
        ddp = shared.get_interface_name(dst_switch, default_switch)
        commands.append(OvsOfctlAddFlowCommand(dst_switch, OvsOfctlCommandArguments(
            priority=GLOBALS.highest_priority, mac_source=server_mac, mac_destination=host_mac,
            actions=[OvsCommandArgumentActionOutput(f"{GLOBALS.switch_interface_port_mapping[dst_switch][ddp]}")])))
    for command in commands:
        GLOBALS.ovs.apply_command(command)
    return 'flow redirected'

def run_flask_thread():
    """Avvia Flask in production mode usando waitress (WSGI server thread-safe)."""
    try:
        import logging
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)
        try:
            from waitress import serve
            print(f"[INFO] Starting Flask with waitress on port {get_trial_port()}")
            serve(app, host='127.0.0.1', port=get_trial_port(), threads=4)
        except ImportError:
            print("[WARNING] waitress not installed, falling back to Flask dev server")
            app.run(host='127.0.0.1', debug=False, port=get_trial_port(),
                    use_reloader=False, threaded=True)
    except Exception as e:
        print(f"[FATAL] Flask startup error: {e}")
        import traceback
        traceback.print_exc()
        raise

def run_flask(_GLOBALS):
    """Avvia Flask in un thread daemon e aspetta che sia pronto."""
    global GLOBALS
    GLOBALS = _GLOBALS

    t = threading.Thread(target=run_flask_thread, daemon=True)
    t.start()

    import time
    import requests

    max_retries = 40
    retry_delay = 0.5

    for attempt in range(max_retries):
        try:
            response = requests.get(f'http://127.0.0.1:{get_trial_port()}/', timeout=1)
            if response.status_code == 200:
                print(f"[INFO] Flask is ready after {(attempt+1) * retry_delay:.1f}s")
                time.sleep(1)  # Extra buffer per stabilità
                return
        except Exception:
            pass
        
        time.sleep(retry_delay)
        if attempt % 5 == 4:  # Stampa ogni 5 tentativi
            print(f"[INFO] Waiting for Flask... ({attempt+1}/{max_retries} attempts)")
    
    # Se non è pronto dopo 20 secondi, stampa avviso ma continua
    print(f"[WARNING] Flask not ready after {max_retries * retry_delay:.0f}s, but continuing...")
    time.sleep(2)