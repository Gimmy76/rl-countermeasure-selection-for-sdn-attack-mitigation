# Mininet
import numpy as np
from math import floor  # FIX: era "from tensorflow.python.ops.numpy_ops import floor"
                        # import interno di TF non necessario, causa warning e
                        # dipendenza inutile da TF nel processo Mininet
from mininet.link import TCLink
from mininet.net import Mininet
from mininet.node import OVSSwitch, Host
from mininet.link import TCLink
from mininet.term import makeTerm
from mininet.topo import Topo
from mininet.cli import CLI
from mininet.log import setLogLevel, info
import random
from decimal import Decimal
import subprocess

import Shared as shared

# Intermed Imports
from intermed.OvsIntermediateMininet import *
from intermed.OvsIntermediate import *
from intermed import OvsIntermediateConstants as consts

class NetworkTopo(Topo):

    def get_server_host_max_bw(self):
        global GLOBALS
        return (GLOBALS.max_host_bw - 0.1) * floor(len(GLOBALS.client_hosts_list) / 2) + 0.1

    def generate_host_cpu(self, host):
        global GLOBALS
        if host in GLOBALS.servers:
            return 3
        elif host in GLOBALS.attackers:
            return 3
        else:
            return 1

    def generate_host_bw(self, host):
        global GLOBALS
        if host in GLOBALS.servers:
            return self.get_server_host_max_bw()
        else:
            if host in GLOBALS.attackers:
                r = float((Decimal(random.randint(15, 20)) * Decimal('0.3')) + Decimal('0.1'))
            elif GLOBALS.unified_host_bandwidth is not None:
                r = GLOBALS.unified_host_bandwidth
            else:
                r = float((Decimal(random.randint(2, 10)) * Decimal('0.3')) + Decimal('0.1'))
            info(f"*** Init host {host} with bw = {r}  ***\n")
            return r

    def generate_switch_bw(self, switch, dst, attacker_default_switch=''):
        global GLOBALS
        if GLOBALS.unified_switch_bandwidth is not None:
            r = GLOBALS.unified_switch_bandwidth
        else:
            if switch == GLOBALS.s0_switch:
                r = float((Decimal(random.randint(3, 10)) * Decimal('0.3')) + Decimal('0.1'))
            else:
                r = float((Decimal(random.randint(16, 30)) * Decimal('0.3')) + Decimal('0.1'))
        info(f"*** Init switch {switch} {dst} with bw = {r}  ***\n")
        return r

    def get_controlled_switch_connections(self, switch, s0_bandwidthes, switches_bandwidthes):
        global GLOBALS
        connections = {}
        if switch != GLOBALS.s0_switch:
            connections[GLOBALS.s0_switch] = {
                'src_int': shared.get_interface_name(switch, GLOBALS.s0_switch),
                'dst_int': shared.get_interface_name(GLOBALS.s0_switch, switch),
                'bw': f'{s0_bandwidthes[f"{GLOBALS.s0_switch}-{switch}"]}'
            }
            for other_switch in GLOBALS.controlled_switches_list:
                if switch != other_switch:
                    connections[other_switch] = {
                        'src_int': shared.get_interface_name(switch, other_switch),
                        'dst_int': shared.get_interface_name(other_switch, switch),
                        'bw': f'{switches_bandwidthes[f"{switch}-{other_switch}"]}',
                    }
        else:
            for other_switch in GLOBALS.controlled_switches_list:
                connections[other_switch] = {
                    'src_int': shared.get_interface_name(GLOBALS.s0_switch, other_switch),
                    'dst_int': shared.get_interface_name(other_switch, GLOBALS.s0_switch),
                    'bw': f'{s0_bandwidthes[f"{GLOBALS.s0_switch}-{other_switch}"]}'
                }
        return connections

    def get_controlled_switch_interfaces(self, switch, special_interfaces=None):
        if special_interfaces is None:
            special_interfaces = []
        global GLOBALS
        interfaces = []
        if switch != GLOBALS.s0_switch:
            interfaces.append(shared.get_interface_name(switch, GLOBALS.s0_switch))
        for other_switch in GLOBALS.controlled_switches_list:
            if switch != other_switch:
                interfaces.append(shared.get_interface_name(switch, other_switch))
        if switch != GLOBALS.s0_switch:
            # Cerca con e senza suffisso trial
            _sfx = getattr(GLOBALS, '_trial_suffix', '')
            _switch_key = switch[:-len(_sfx)] if _sfx and switch.endswith(_sfx) else switch
            for router in GLOBALS.controlled_switch_to_router_relation.get(switch, GLOBALS.controlled_switch_to_router_relation.get(_switch_key, {'routers': []}))['routers']:
                interfaces.append(shared.get_interface_name(switch, router))
        if len(special_interfaces) > 0:
            interfaces.extend(special_interfaces)
        return interfaces

    def build(self, **_opts):
        global GLOBALS
        info("*** Creating switches\n")

        s0 = self.addSwitch(GLOBALS.s0_switch)

        # Legge i controlled switches dalla topologia
        # Se controlled_switches_list è già stato impostato da run_mininet (con suffisso),
        # non sovrascrivere — altrimenti leggi dalla topologia
        _sfx = getattr(GLOBALS, '_trial_suffix', '')
        if not _sfx:
            _topo_cs = list(dict.fromkeys(
                GLOBALS.host_default_switch_relation[h]['default_path_switch']
                for h in GLOBALS.host_default_switch_relation
            ))
            if _topo_cs:
                GLOBALS.controlled_switches_list = _topo_cs
            else:
                GLOBALS.controlled_switches_list = [f's1{i:02d}' for i in range(1, GLOBALS.nbr_controlled_switches + 1)]
        for switch in GLOBALS.controlled_switches_list:
            self.addSwitch(switch)

        for switch in GLOBALS.router_switches_list:
            self.addSwitch(switch)

        attacker_default_switch = GLOBALS.host_default_switch_relation[GLOBALS.attackers[0]]['default_path_switch']

        s0_bandwidthes = {}
        for switch in GLOBALS.controlled_switches_list:
            s0_bandwidthes[f"{GLOBALS.s0_switch}-{switch}"] = self.generate_switch_bw(GLOBALS.s0_switch, switch, attacker_default_switch)

        switches_bandwidthes = {}
        for switch in GLOBALS.controlled_switches_list:
            for other_switch in GLOBALS.controlled_switches_list:
                if switch != other_switch:
                    key = f"{switch}-{other_switch}"
                    if key not in switches_bandwidthes:
                        switches_bandwidthes[key] = self.generate_switch_bw(switch, other_switch)
                        switches_bandwidthes[f"{other_switch}-{switch}"] = switches_bandwidthes[key]

        hosts_bandwidthes = {}
        hosts_bandwidthes[GLOBALS.default_server] = self.generate_host_bw(GLOBALS.default_server)
        for host in GLOBALS.client_hosts_list:
            hosts_bandwidthes[host] = self.generate_host_bw(host)

        GLOBALS.network_spec['switches'] = {
            GLOBALS.s0_switch: {
                'ports': self.get_controlled_switch_interfaces(GLOBALS.s0_switch, [f"{GLOBALS.s0_switch}-eth0", f"{GLOBALS.s0_switch}-eth{GLOBALS.nbr_controlled_switches + 2}"]),
                'connections': self.get_controlled_switch_connections(GLOBALS.s0_switch, s0_bandwidthes, switches_bandwidthes)
            }
        }

        for switch in GLOBALS.controlled_switches_list:
            GLOBALS.network_spec['switches'][switch] = {
                'ports': self.get_controlled_switch_interfaces(switch),
                'connections': self.get_controlled_switch_connections(switch, s0_bandwidthes, switches_bandwidthes)
            }

        for router in GLOBALS.router_switches_list:
            controlled_switch = GLOBALS.router_to_controlled_switch_relation[router]['controlled_switch']
            GLOBALS.network_spec['switches'][router] = {
                'connections': {
                    controlled_switch: {
                        'src_int': shared.get_interface_name(router, controlled_switch),
                        'dst_int': shared.get_interface_name(controlled_switch, router),
                        'bw': f'{hosts_bandwidthes[GLOBALS.router_to_host_relation[router]["host"]]}',
                        'id': 1,
                        'connected': True,
                    }
                },
                'ports': [
                    shared.get_interface_name(router, "0"),
                    shared.get_interface_name(router, controlled_switch)
                ]
            }

        info("*** Creating hosts\n")

        GLOBALS.network_spec['hosts'] = {
            GLOBALS.default_server: {
                'ip': '10.0.1.101',
                'router_switch': GLOBALS.s0_switch,
                'src_int': f'{GLOBALS.default_server}-eth0',
                'dst_int': f'{GLOBALS.s0_switch}-eth0',
                'connected': True,
                'bw': f'{hosts_bandwidthes[GLOBALS.default_server]}',
                'mac': '00:00:00:00:01:00'
            }
        }

        for host in GLOBALS.client_hosts_list:
            data = {
                'ip': GLOBALS.hosts_raw_topo[host]['ip'],
                'router_switch': GLOBALS.hosts_raw_topo[host]['router_switch'],
                'src_int': host + '-eth0',
                'dst_int': GLOBALS.hosts_raw_topo[host]['router_switch'] + '-eth0',
                'connected': True,
                'bw': f'{hosts_bandwidthes[host]}',
                'mac': GLOBALS.hosts_raw_topo[host]['mac'],
                'current_path': {},
                'default_path_switch': GLOBALS.host_default_switch_relation[host]['default_path_switch']
            }
            for switch in GLOBALS.controlled_switches_list:
                data['current_path'][switch] = (switch == data['default_path_switch'])
            GLOBALS.network_spec['hosts'][host] = data

        for host in GLOBALS.network_spec['hosts'].keys():
            ip  = GLOBALS.network_spec['hosts'][host]['ip']
            cpu = self.generate_host_cpu(host)
            mac = GLOBALS.network_spec['hosts'][host]['mac']
            info(f"*** Init host {host}({cpu}, {ip}, {mac}) ***\n")
            self.addHost(host, ip=ip, mac=mac)

        info("*** Creating links\n")

        max_switch_queue_size = 10000000
        max_host_queue_size   = 10000000

        for src_switch in GLOBALS.controlled_switches_list:
            self.addLink(src_switch, s0,
                         intfName1=GLOBALS.network_spec['switches'][src_switch]['connections'][GLOBALS.s0_switch]['src_int'],
                         intfName2=GLOBALS.network_spec['switches'][src_switch]['connections'][GLOBALS.s0_switch]['dst_int'],
                         bw=float(GLOBALS.network_spec['switches'][src_switch]['connections'][GLOBALS.s0_switch]['bw']),
                         max_queue_size=max_switch_queue_size)

        for i in range(len(GLOBALS.controlled_switches_list) - 1):
            for j in range(i + 1, len(GLOBALS.controlled_switches_list)):
                src = GLOBALS.controlled_switches_list[i]
                dst = GLOBALS.controlled_switches_list[j]
                self.addLink(src, dst,
                             intfName1=shared.get_interface_name(src, dst),
                             intfName2=shared.get_interface_name(dst, src),
                             bw=float(GLOBALS.network_spec['switches'][src]['connections'][dst]['bw']),
                             max_queue_size=max_switch_queue_size)

        for src_switch in GLOBALS.router_switches_list:
            for dst_switch in GLOBALS.network_spec['switches'][src_switch]['connections'].keys():
                conn = GLOBALS.network_spec['switches'][src_switch]['connections'][dst_switch]
                self.addLink(src_switch, dst_switch,
                             intfName1=conn['src_int'], intfName2=conn['dst_int'],
                             bw=float(conn['bw']),
                             max_queue_size=max_switch_queue_size)

        hs = GLOBALS.default_server
        self.addLink(s0, hs,
                     intfName1=GLOBALS.network_spec['hosts'][hs]['dst_int'],
                     intfName2=GLOBALS.network_spec['hosts'][hs]['src_int'],
                     params2={'ip': "10.0.1.101/16"},
                     bw=float(GLOBALS.network_spec['hosts'][hs]['bw']),
                     max_queue_size=max_host_queue_size)

        for host in GLOBALS.client_hosts_list:
            router_switch = GLOBALS.network_spec['hosts'][host]['router_switch']
            self.addLink(router_switch, host,
                         intfName1=GLOBALS.network_spec['hosts'][host]['dst_int'],
                         intfName2=GLOBALS.network_spec['hosts'][host]['src_int'],
                         params2={'ip': f"{GLOBALS.network_spec['hosts'][host]['ip']}/16"},
                         bw=float(GLOBALS.network_spec['hosts'][host]['bw']),
                         max_queue_size=max_host_queue_size)


def run_mininet(_GLOBALS):
    global GLOBALS
    GLOBALS = _GLOBALS

    # Suffisso gestito in Shared.py — qui aggiorniamo solo s0_switch e controlled_switches_list
    import os
    _suffix = os.environ.get("TRIAL_ID", "").replace("_t", "t")
    if _suffix:
        GLOBALS.s0_switch = f"s0{_suffix}"
        GLOBALS.controlled_switches_list = [
            f"s1{i:02d}{_suffix}"
            for i in range(1, GLOBALS.nbr_controlled_switches + 1)
        ]
        GLOBALS._trial_suffix = _suffix
    # mn -c rimosso: distruggerebbe altri trial paralleli.
    # Il pre_cleanup mirato in CmdManager si occupa solo del proprio trial.
    pass

    topo = NetworkTopo()
    GLOBALS.net = Mininet(topo=topo, controller=None, switch=OVSSwitch,
                          waitConnected=False, link=TCLink, host=Host)

    # NAT con nome univoco per trial
    _nat_name = f'nat0{_suffix}' if _suffix else 'nat0'
    GLOBALS.net.addNAT(name=_nat_name).configDefault()
    GLOBALS.net.start()

    info("*** Testing network\n")

    for src_switch in GLOBALS.router_switches_list:
        for dst_switch in GLOBALS.network_spec['switches'][src_switch]['connections'].keys():
            if not GLOBALS.network_spec["switches"][src_switch]['connections'][dst_switch]["connected"]:
                shared.turn_down_link(
                    src_switch,
                    GLOBALS.network_spec['switches'][src_switch]['connections'][dst_switch]['src_int'],
                    dst_switch,
                    GLOBALS.network_spec["switches"][src_switch]['connections'][dst_switch]["dst_int"]
                )

    # VM: manual_receivers è sempre True → non si apre nessun xterm per ITGRecv
    if not GLOBALS.manual_receivers:
        host_name = GLOBALS.default_server
        terminal = makeTerm(GLOBALS.net[host_name],
                            title=f"Host {host_name} DITG-Receiver",
                            cmd=f"nice -n -20 {GLOBALS.ditg_directory}/ITGRecv")
        GLOBALS.net.terms += terminal
        GLOBALS.ditg_receivers.append(terminal)

    for switch in GLOBALS.net.switches:
        GLOBALS.switch_interface_port_mapping[switch.name] = {}
        for interface in switch.ports.keys():
            port = switch.ports[interface]
            GLOBALS.switch_interface_port_mapping[switch.name][f"{interface}"] = port
    print(GLOBALS.switch_interface_port_mapping)

    commands = []

    commands.append(shared.flood_arp_for_icmp_command(
        target=GLOBALS.s0_switch, priority=GLOBALS.server_switch_flood_priority))

    commands.extend(shared.init_arp_for_cotnrolled_switches(
        GLOBALS.controlled_switch_arp_priority,
        GLOBALS.controlled_switch_flood_priority,
        shared.build_switch_info_for_arp()))

    commands.extend(shared.init_arp_for_non_controlled_switches(
        GLOBALS.non_controlled_switch_arp_priority,
        GLOBALS.router_switches_list))

    commands.append(shared.init_flow_for_global_dns_from_server_switch(
        GLOBALS.s0_switch, GLOBALS.highest_priority,
        GLOBALS.global_dns, f"{GLOBALS.s0_switch}-eth{GLOBALS.nbr_controlled_switches + 2}"))

    commands.append(shared.init_flow_from_switch_to_direct_host_via_mac(
        GLOBALS.s0_switch, GLOBALS.highest_priority, GLOBALS.server_host))

    commands.extend(shared.init_flow_from_server_switch_to_controlled_switch_for_hosts(
        GLOBALS.s0_switch, GLOBALS.highest_priority))

    for host in GLOBALS.client_hosts_list:
        router_switch     = GLOBALS.network_spec['hosts'][host]['router_switch']
        controlled_switch = GLOBALS.network_spec['hosts'][host]['default_path_switch']

        rs_to_host_int = GLOBALS.network_spec['hosts'][host]['dst_int']
        rs_to_cs_int   = shared.get_interface_name(router_switch, controlled_switch)

        commands.append(OvsOfctlAddFlowCommand(router_switch, OvsOfctlCommandArguments(
            protocol=consts.OVS_PROTOCOL_IP,
            in_port=f"{GLOBALS.switch_interface_port_mapping[router_switch][rs_to_host_int]}",
            priority=GLOBALS.highest_priority,
            ip_destination=GLOBALS.global_dns,
            actions=[OvsCommandArgumentActionOutput(
                f"{GLOBALS.switch_interface_port_mapping[router_switch][rs_to_cs_int]}")])))

        commands.append(OvsOfctlAddFlowCommand(router_switch, OvsOfctlCommandArguments(
            in_port=f"{GLOBALS.switch_interface_port_mapping[router_switch][rs_to_host_int]}",
            priority=GLOBALS.highest_priority,
            mac_destination=GLOBALS.network_spec['hosts'][GLOBALS.server_host]['mac'],
            actions=[OvsCommandArgumentActionOutput(
                f"{GLOBALS.switch_interface_port_mapping[router_switch][rs_to_cs_int]}")])))

        commands.append(OvsOfctlAddFlowCommand(router_switch, OvsOfctlCommandArguments(
            priority=GLOBALS.highest_priority,
            mac_destination=GLOBALS.network_spec['hosts'][host]['mac'],
            actions=[OvsCommandArgumentActionOutput(
                f"{GLOBALS.switch_interface_port_mapping[router_switch][rs_to_host_int]}")])))

        cs_to_rs_int = shared.get_interface_name(controlled_switch, router_switch)
        cs_to_s0_int = shared.get_interface_name(controlled_switch, GLOBALS.s0_switch)

        commands.append(OvsOfctlAddFlowCommand(controlled_switch, OvsOfctlCommandArguments(
            protocol=consts.OVS_PROTOCOL_IP,
            priority=GLOBALS.highest_priority,
            ip_destination=GLOBALS.global_dns,
            actions=[OvsCommandArgumentActionOutput(
                f"{GLOBALS.switch_interface_port_mapping[controlled_switch][cs_to_s0_int]}")])))

        commands.append(OvsOfctlAddFlowCommand(controlled_switch, OvsOfctlCommandArguments(
            priority=GLOBALS.highest_priority,
            mac_source=GLOBALS.network_spec['hosts'][host]['mac'],
            mac_destination=GLOBALS.network_spec['hosts'][GLOBALS.server_host]['mac'],
            actions=[OvsCommandArgumentActionOutput(
                f"{GLOBALS.switch_interface_port_mapping[controlled_switch][cs_to_s0_int]}")])))

        commands.append(OvsOfctlAddFlowCommand(controlled_switch, OvsOfctlCommandArguments(
            priority=GLOBALS.highest_priority,
            mac_destination=GLOBALS.network_spec['hosts'][host]['mac'],
            actions=[OvsCommandArgumentActionOutput(
                f"{GLOBALS.switch_interface_port_mapping[controlled_switch][cs_to_rs_int]}")])))

    GLOBALS.ovs = OvsIntermediateMininet(GLOBALS.net, True, True)
    for command in commands:
        GLOBALS.ovs.apply_command(command)

    info("*** Running CLI\n")
    GLOBALS.cli = CLI(GLOBALS.net)

    info("*** Stopping network\n")
    GLOBALS.net.stop()