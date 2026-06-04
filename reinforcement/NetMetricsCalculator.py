"""
NetMetricsCalculator.py — sostituisce CICFlowMeter + il vecchio NetMetricsCalculator.

Produce due output dallo stesso pcap in un unico passaggio:
  1. metrics.json  — latency, jitter, throughput, APTT per host
  2. cic_flow.csv  — colonne equivalenti a CICFlowMeter usate da Environment.py:
       Src IP, Dst IP, Dst Port, Flow Pkts/s, Flow Byts/s,
       Tot Fwd Pkts, Tot Bwd Pkts, TotLen Fwd Pkts, TotLen Bwd Pkts, ACK Flag Cnt

Velocizzazione: usa tshark -T fields invece di Scapy rdpcap().
tshark (C) è 10-100x più veloce di Scapy per pcap grandi.
tcp.analysis.ack_rtt fornisce la latency senza calcolo manuale seq/ack.
"""

import subprocess
import argparse
import os
import json
import csv
import threading
from collections import defaultdict
import sys
import time

VERBOSE = False
MAX_PERIOD = 1.0
MIN_BPS    = 0.01

def debug(msg):
    if VERBOSE:
        print(f'(NetMetrics) --> {msg}')

# ===========================================================================
# SEZIONE 0: estrazione campi dal pcap tramite tshark (sostituisce rdpcap)
# ===========================================================================

_TSHARK_FIELDS = [
    'ip.src', 'ip.dst',
    'tcp.srcport', 'tcp.dstport',
    'tcp.seq', 'tcp.ack',
    'tcp.flags.ack',         # 0 o 1
    'tcp.len',               # payload TCP (no header)
    'frame.len',             # frame completo
    'frame.time_epoch',      # timestamp epoch
    'tcp.analysis.ack_rtt',  # RTT data→ACK calcolato da tshark (su pacchetti ACK)
]

def read_pcap_tshark(pcap_file):
    """
    Estrae i campi TCP dal pcap tramite tshark.
    Restituisce list[dict]. Molto più veloce di Scapy rdpcap().
    Usa subprocess list (no shell=True) per evitare problemi con il separatore \t.
    """
    cmd = (
        ['tshark', '-r', pcap_file, '-Y', 'tcp', '-T', 'fields']
        + [arg for f in _TSHARK_FIELDS for arg in ('-e', f)]
        + ['-E', 'separator=\t', '-E', 'header=y', '-E', 'occurrence=f']
    )
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    rows = []
    lines = result.stdout.splitlines()
    if len(lines) < 2:
        return rows
    header = lines[0].split('\t')
    for line in lines[1:]:
        parts = line.split('\t')
        if len(parts) < len(header):
            continue
        rows.append(dict(zip(header, parts)))
    return rows


# ===========================================================================
# SEZIONE 1: calcolo metriche di rete (latency, jitter, throughput, APTT)
# ===========================================================================

def calculate_latency(rows, host, server, server_port):
    """
    Usa tcp.analysis.ack_rtt (campo nativo tshark) per la latency.
    Il campo è presente sugli ACK server→client che riconoscono i dati del client.
    Equivalente al calcolo manuale seq/ack del codice precedente, ma senza iterare
    due volte sul pcap.
    """
    rtts = []
    sp = str(server_port)
    for row in rows:
        if (row.get('ip.src') != server or
                row.get('ip.dst') != host or
                row.get('tcp.srcport') != sp):
            continue
        rtt_str = row.get('tcp.analysis.ack_rtt', '')
        if not rtt_str:
            continue
        try:
            rtt = float(rtt_str)
            if 0 < rtt < MAX_PERIOD:
                rtts.append(rtt)
        except ValueError:
            pass
    if rtts:
        avg = sum(rtts) / len(rtts)
        print(f"(NetMetrics) --> Latency: {avg:.6f}s over {len(rtts)} RTT samples for {host}")
        return min(avg, MAX_PERIOD)
    print(f"(NetMetrics) --> Latency: no tcp.analysis.ack_rtt for {host}, using MAX_PERIOD")
    return MAX_PERIOD


def calculate_aptt(rows, src, dst, dport, avg_pkt_size, period_s):
    total_bytes = 0
    seen_seq = set()  # (sport, seq) per distinguere connessioni TCP diverse
    dp = str(dport)
    for row in rows:
        if (row.get('ip.src') != src or
                row.get('ip.dst') != dst or
                row.get('tcp.dstport') != dp):
            continue
        seq = row.get('tcp.seq', '')
        sport = row.get('tcp.srcport', '')
        if not seq:
            continue
        key = (sport, seq)
        if key in seen_seq:
            continue
        seen_seq.add(key)
        try:
            total_bytes += int(row.get('frame.len', 0) or 0)
        except ValueError:
            pass
    if total_bytes > 0:
        aptt = avg_pkt_size / (total_bytes / period_s)
        print(f"(NetMetrics) --> APTT: {aptt:.6f}s")
        return min(aptt, MAX_PERIOD)
    return MAX_PERIOD


def calculate_throughput(rows, src, dst, dport, period_s):
    total_bytes = 0
    seen_seq = set()  # (sport, seq) per distinguere connessioni TCP diverse
    dp = str(dport)
    for row in rows:
        if (row.get('ip.src') != src or
                row.get('ip.dst') != dst or
                row.get('tcp.dstport') != dp):
            continue
        seq = row.get('tcp.seq', '')
        sport = row.get('tcp.srcport', '')
        if not seq:
            continue
        key = (sport, seq)
        if key in seen_seq:
            continue
        seen_seq.add(key)
        try:
            total_bytes += int(row.get('frame.len', 0) or 0)
        except ValueError:
            pass
    if total_bytes > 0:
        bps = (total_bytes * 8) / period_s
        print(f"(NetMetrics) --> Throughput: {bps:.1f} bps")
        return max(bps, MIN_BPS)
    return MIN_BPS


def calculate_jitter(rows, src, dst, dport):
    timestamps = []
    seen_seq = set()  # (sport, seq) per distinguere connessioni TCP diverse
    dp = str(dport)
    for row in rows:
        if (row.get('ip.src') != src or
                row.get('ip.dst') != dst or
                row.get('tcp.dstport') != dp):
            continue
        seq = row.get('tcp.seq', '')
        sport = row.get('tcp.srcport', '')
        if not seq:
            continue
        key = (sport, seq)
        if key in seen_seq:
            continue
        seen_seq.add(key)
        try:
            timestamps.append(float(row['frame.time_epoch']))
        except (ValueError, KeyError):
            pass
    if len(timestamps) < 2:
        print(f"(NetMetrics) --> Jitter: not enough packets for {src}, using MAX_PERIOD")
        return MAX_PERIOD
    timestamps.sort()
    deltas = [timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps))]
    avg_jitter = sum(deltas) / len(deltas)
    print(f"(NetMetrics) --> Jitter: {avg_jitter:.6f}s")
    return min(avg_jitter, MAX_PERIOD)


def calculate_metrics_for_host(data, rows, ip, server_ip, server_port,
                                pkt_size, duration_s):
    print(f"(NetMetrics) --> Processing host {ip}")
    data[ip] = {
        "avg_latency_s":                  calculate_latency(rows, ip, server_ip, server_port),
        "avg_packet_transmission_time_s": calculate_aptt(rows, ip, server_ip, server_port, pkt_size, duration_s),
        "throughput_bps":                 calculate_throughput(rows, ip, server_ip, server_port, duration_s),
        "avg_jitter_s":                   calculate_jitter(rows, ip, server_ip, server_port),
    }
    print(f"(NetMetrics) <-- Host {ip} done: {data[ip]}")


# ===========================================================================
# SEZIONE 2: calcolo colonne equivalenti a CICFlowMeter
# ===========================================================================

def build_cic_equivalent(rows, duration_s):
    """
    Costruisce una lista di dizionari con le stesse colonne usate da
    Environment.py quando legge il CSV di CICFlowMeter.
    Due passate sui rows (list di dict) invece di tre sullo stesso pcap Scapy.
    """
    flows = defaultdict(lambda: {
        'fwd_pkts': 0, 'bwd_pkts': 0,
        'fwd_bytes': 0, 'bwd_bytes': 0,
        'ack_flags': 0,
    })

    # Pass 1: pacchetti forward
    for row in rows:
        src   = row.get('ip.src', '')
        dst   = row.get('ip.dst', '')
        dport = row.get('tcp.dstport', '')
        if not src or not dst or not dport:
            continue
        pkt_len = int(row.get('frame.len', 0) or 0)
        flows[(src, dst, dport)]['fwd_pkts']  += 1
        flows[(src, dst, dport)]['fwd_bytes'] += pkt_len

    # Pass 2: pacchetti backward (ACK) rispetto ai flussi forward già noti
    for row in rows:
        src   = row.get('ip.src', '')
        dst   = row.get('ip.dst', '')
        sport = row.get('tcp.srcport', '')
        dport = row.get('tcp.dstport', '')
        if not src or not dst or not sport:
            continue
        pkt_len = int(row.get('frame.len', 0) or 0)
        ack_val = row.get('tcp.flags.ack', '0')
        is_ack  = ack_val in ('1', 'True', 'true')
        # Il backward di (dst→src:sport) corrisponde al flusso forward (dst, src, sport)
        fwd_key = (dst, src, sport)
        if fwd_key in flows:
            flows[fwd_key]['bwd_pkts']  += 1
            flows[fwd_key]['bwd_bytes'] += pkt_len
            if is_ack:
                flows[fwd_key]['ack_flags'] += 1

        # Controlla anche flusso con dport invertito per maggiore copertura
        fwd_key2 = (dst, src, dport)
        if fwd_key2 in flows and fwd_key2 != fwd_key:
            flows[fwd_key2]['bwd_pkts']  += 1
            flows[fwd_key2]['bwd_bytes'] += pkt_len
            if is_ack:
                flows[fwd_key2]['ack_flags'] += 1
    result = []
    for (src, dst, dport), f in flows.items():
        if f['fwd_pkts'] == 0 or duration_s == 0:
            continue
        result.append({
            'Src IP':          src,
            'Dst IP':          dst,
            'Dst Port':        str(dport),
            'Flow Pkts/s':     str(f['fwd_pkts']  / duration_s),
            'Flow Byts/s':     str(f['fwd_bytes']  / duration_s),
            'Tot Fwd Pkts':    str(f['fwd_pkts']),
            'Tot Bwd Pkts':    str(f['bwd_pkts']),
            'TotLen Fwd Pkts': str(f['fwd_bytes']),
            'TotLen Bwd Pkts': str(f['bwd_bytes']),
            'ACK Flag Cnt':    str(f['ack_flags']),
        })
    return result


def write_cic_csv(rows, output_path):
    """Scrive il CSV nello stesso formato letto da Environment.read_cic_flow_file()."""
    fieldnames = [
        'Src IP', 'Dst IP', 'Dst Port',
        'Flow Pkts/s', 'Flow Byts/s',
        'Tot Fwd Pkts', 'Tot Bwd Pkts',
        'TotLen Fwd Pkts', 'TotLen Bwd Pkts',
        'ACK Flag Cnt',
    ]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"(NetMetrics) --> CIC-equivalent CSV written: {output_path} ({len(rows)} flows)")


# ===========================================================================
# ENTRY POINT
# ===========================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="NetMetricsCalculator — replaces CICFlowMeter",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-s",    "--server-ip",                required=True)
    parser.add_argument("-p",    "--server-port",              required=True)
    parser.add_argument("-hip",  "--hosts-ips",                required=True,
                        help="E.g: [10.0.1.1,10.0.1.2]")
    parser.add_argument("-t",    "--sending-duration-seconds", required=True)
    parser.add_argument("-b",    "--packet-size-bytes",        required=True)
    parser.add_argument("-cic",  "--cic-output-path",          required=False,
                        default="",
                        help="Percorso dove scrivere il CSV equivalente a CICFlowMeter")
    parser.add_argument("-pcap", "--pcap-path",                required=False,
                        default="/tmp/qosentry/tshark_out.pcap",
                        help="Percorso del file pcap da analizzare")
    parser.add_argument("-nr",   "--no-results", action="store_true")

    cfg = vars(parser.parse_args())

    server_ip    = cfg['server_ip']
    server_port  = int(cfg['server_port'])
    hosts_ips    = cfg['hosts_ips'].lstrip("[").rstrip("]").split(',')
    duration_s   = int(cfg['sending_duration_seconds'])
    pkt_size     = int(cfg['packet_size_bytes'])
    cic_out_path = cfg['cic_output_path']
    pcap_file    = cfg['pcap_path']

    tmp_dir      = os.environ.get('QOSENTRY_TMP', '/tmp/qosentry')
    metrics_data = {}

    def _write_defaults():
        for ip in hosts_ips:
            metrics_data[ip.strip()] = {
                "avg_latency_s": 1.0,
                "avg_packet_transmission_time_s": 1.0,
                "throughput_bps": 0.01,
                "avg_jitter_s": 1.0,
            }

    print(f"(NetMetrics) --> Reading pcap: {pcap_file}")
    if not os.path.exists(pcap_file):
        print(f"(NetMetrics) ERROR: pcap not found: {pcap_file}")
        _write_defaults()
        if cic_out_path:
            write_cic_csv([], cic_out_path)
        if not cfg['no_results']:
            os.makedirs(tmp_dir, exist_ok=True)
            with open(os.path.join(tmp_dir, 'metrics.json'), 'w') as f:
                json.dump(metrics_data, f, indent=2)
        sys.exit(0)

    tshark_start = time.time()
    rows = read_pcap_tshark(pcap_file)
    print(f"(NetMetrics) --> {len(rows)} TCP rows extracted in {time.time() - tshark_start:.2f}s")

    if not rows:
        print("(NetMetrics) WARNING: no TCP packets found, using defaults")
        _write_defaults()
        if cic_out_path:
            write_cic_csv([], cic_out_path)
        if not cfg['no_results']:
            os.makedirs(tmp_dir, exist_ok=True)
            with open(os.path.join(tmp_dir, 'metrics.json'), 'w') as f:
                json.dump(metrics_data, f, indent=2)
        sys.exit(0)

    # --- Sezione 1: metriche per host (parallele) ---
    metrics_start = time.time()
    threads = []
    for ip in hosts_ips:
        ip = ip.strip()
        t = threading.Thread(
            name=f'thr-{ip}',
            target=calculate_metrics_for_host,
            args=(metrics_data, rows, ip, server_ip, server_port, pkt_size, duration_s)
        )
        threads.append(t)
        t.start()
    for t in threads:
        t.join(timeout=60)
        if t.is_alive():
            print(f"(NetMetrics) WARNING: thread {t.name} timed out, using defaults")
    print(f"(NetMetrics) --> Metrics calculation: {time.time() - metrics_start:.2f}s")

    # --- Sezione 2: CIC-equivalent CSV ---
    if cic_out_path:
        cic_start = time.time()
        cic_rows = build_cic_equivalent(rows, duration_s)
        write_cic_csv(cic_rows, cic_out_path)
        print(f"(NetMetrics) --> CIC CSV generation: {time.time() - cic_start:.2f}s")

    print(metrics_data)

    if not cfg['no_results']:
        metrics_file = os.path.join(tmp_dir, 'metrics.json')
        os.makedirs(tmp_dir, exist_ok=True)
        for _ip in hosts_ips:
            _ip = _ip.strip()
            if _ip not in metrics_data:
                print(f"(NetMetrics) WARNING: no data for {_ip}, using defaults")
                metrics_data[_ip] = {
                    "avg_latency_s": 1.0,
                    "avg_packet_transmission_time_s": 1.0,
                    "throughput_bps": 0.01,
                    "avg_jitter_s": 1.0,
                }
        with open(metrics_file, 'w') as f:
            json.dump(metrics_data, f, indent=2, default=str)
        print(f"(NetMetrics) --> metrics.json written: {metrics_file}")
