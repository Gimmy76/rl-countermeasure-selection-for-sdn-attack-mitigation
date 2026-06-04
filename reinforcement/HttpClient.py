import requests

class HttpClient():

    # Initializes the HttpClient class responsible for making HTTP requests to a network simulation API.
    # Stores the API base URL from the provided configuration object.
    def __init__(self, configuration):
        print("(Reinforcement) HttpClient.__init__()")
        self.api_link = configuration.api_link

    # Retrieves the list of interfaces available on all switches in the network.
    # The response is returned in JSON format.
    def get_switches_interfaces(self):
        try:
            resp = requests.get(f'{self.api_link}/get-switches-interfaces', timeout=10)
            return resp.json()
        except Exception as e:
            print(f'(HTTP) WARNING: get_switches_interfaces failed: {e}')
            return []

    def start_ditg_flow(self, source_host, destination_host, duration_ms):
        return requests.get(f'{self.api_link}/start-ditg-flow/{source_host}/{destination_host}/{duration_ms}')

    # Starts a TCP flow between two hosts for a specified duration.
    # Useful for simulating regular traffic in the network.
    def start_tcp_flow(self, source_host, destination_host, duration_ms):
        return requests.get(f'{self.api_link}/start-tcp-flow/{source_host}/{destination_host}/{duration_ms}', timeout=10)

    def stop_all_ditg_flows(self):
        return requests.get(f'{self.api_link}/stop-all-ditg-flows')

    def stop_all_tcp_flows(self):
        return requests.get(f'{self.api_link}/stop-all-tcp-flows', timeout=10)

    def start_ddos_flooding_attack(self, attacker_host, victim_host, attack_type):
        return requests.get(f'{self.api_link}/start-ddos-flooding/{attacker_host}/{victim_host}/{attack_type}')

    def stop_ddos_flooding_attack(self, attacker_host, victim_host):
        return requests.get(f'{self.api_link}/stop-ddos-flooding/{attacker_host}/{victim_host}')

    ## Starts an MH-DDOS attack from an attacker host to a victim host using a specified attack type.
    # This can be used to simulate various network attack scenarios for testing security measures.
    def start_mhddos_attack(self, attacker_host, victim_host, attack_type):
        return requests.get(f'{self.api_link}/start-mhddos/{attacker_host}/{victim_host}/{attack_type}')

    def stop_mhddos_attack(self, attacker_host, victim_host):
        return requests.get(f'{self.api_link}/stop-mhddos/{attacker_host}/{victim_host}')

    def reset_ditg_receivers(self):
        return requests.get(f'{self.api_link}/reset-ditg-receivers')

    def reset_tcp_receivers(self):
        return requests.get(f'{self.api_link}/reset-tcp-receivers', timeout=10)

    def stop_tcp_receivers(self):
        return requests.get(f'{self.api_link}/stop-tcp-receivers', timeout=10)

    def get_host_interface_statistics(self, host):
        return requests.get(f'{self.api_link}/get-host-interface-statistics/{host}', timeout=10)

    def get_ip_by_host_name(self, host):
        return requests.get(f'{self.api_link}/host-ip/{host}', timeout=10)

    def get_host_status_connected(self, host):
        return requests.get(f'{self.api_link}/host-status-connected/{host}', timeout=10)

    def get_host_bw(self, host):
        try:
            resp = requests.get(f'{self.api_link}/get-host-bw/{host}', timeout=10)
            if resp.status_code != 200 or not resp.content:
                raise ValueError(f'Invalid host_bw response: status={resp.status_code}, content={resp.text!r}')
            try:
                resp.json()
            except ValueError as e:
                raise ValueError(f'Invalid JSON in host_bw response: {e}') from e
            return resp
        except Exception as e:
            print(f'(HTTP) WARNING: get_host_bw failed: {e}')
            r = requests.models.Response()
            r._content = b'{"bw": "0"}'
            r.status_code = 200
            return r

    def increase_host_bw(self, host, change):
        return requests.get(f'{self.api_link}/increase-host-bw/{host}/{change}', timeout=10)

    def decrease_host_bw(self, host, change):
        return requests.get(f'{self.api_link}/decrease-host-bw/{host}/{change}', timeout=10)

    def get_switch_status_connected(self, src_switch):
        return requests.get(f'{self.api_link}/get_switch-status-connected/{src_switch}', timeout=10)

    def get_switch_bw(self, src_switch, dst_switch):
        try:
            return requests.get(
            f'{self.api_link}/get_switch_bw/{src_switch}/{dst_switch}',
            timeout=10
        )
        except requests.exceptions.RequestException as e:
            print(f'(HTTP) WARNING: get_switch_bw failed: {e}')
            r = requests.models.Response()
            r._content = b'{"bw": "0.1"}'
            r.status_code = 200
            return r
    def decrease_switch_bw(self, src_switch, dst_switch, change):
        return requests.get(f'{self.api_link}/decrease-switch-bw/{src_switch}/{dst_switch}/{change}', timeout=10)

    def increase_switch_bw(self, src_switch, dst_switch, change):
        return requests.get(f'{self.api_link}/increase-switch-bw/{src_switch}/{dst_switch}/{change}', timeout=10)

    def get_dst_switches(self, src_switch):
        try:
            return requests.get(
            f'{self.api_link}/get_dst_switches/{src_switch}',
            timeout=10
        )
        except requests.exceptions.RequestException as e:
            print(f'(HTTP) WARNING: get_dst_switches failed: {e}')
            r = requests.models.Response()
            r._content = b'{"dst_switches": []}'
            r.status_code = 200
            return r
    
    def get_link_information(self, src_switch, dst_switch):
        try:
            return requests.get(
            f'{self.api_link}/get_link_information/{src_switch}/{dst_switch}',
            timeout=10
        )
        except Exception as e:
            print(f'(HTTP) WARNING: get_link_information failed: {e}')
        # Ritorna un oggetto mock con json() funzionante
        import types
        r = requests.models.Response()
        r._content = b'{"tx_bytes": 0, "rx_bytes": 0, "bw": "0.1"}'
        r.status_code = 200
        return r
    
    def get_host_path(self, host_name):
        return requests.get(f'{self.api_link}/get_host_path/{host_name}', timeout=10)

    def redirect_switch_flow(self, host_name, dst_switch):
        return requests.get(f'{self.api_link}/redirect_switch_flow/{host_name}/{dst_switch}', timeout=10)