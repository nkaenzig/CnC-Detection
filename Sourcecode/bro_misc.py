"""
Master Thesis
Network Monitoring and Attack Detection

bro_misc.py
Helper functios used for analysing the Bro .log-files.

@author: Nicolas Kaenzig, D-ITET, ETH Zurich
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os
import tqdm
import ipaddress
import sys
import pandas as pd
from collections import Counter


def check_if_valid_IP(ip_addr):
    """
    Checks if given IP address is valid (for IPv4 & 6)

    :param ip_addr: String containing an IP to be checked
    :return: True if valid, False if invalid
    """
    try:
        ip = ipaddress.ip_address(ip_addr)
        # print('{} is a correct IP{} address.'.format(ip, ip.version))
    except ValueError:
        #print('address/netmask is invalid: {}'.format(ip_addr))
        return False
    except:
        # print('Usage : ip')
        return False
    return True


def get_conn_id(src_nr, dst_nr, nr_destinations):
    """
    bro_parsers.conn_log_extract() first enumerates all observed src and dst IP adresses to integers.
    This funcion calculates a uniue integer ID for each src-dst pair. ('Connection ID')

    :param src_nr: Number of the src IP
    :param dst_nr: Number of the dst IP
    :param nr_destinations: # of all observed dst IPs
    :return: connection ID
    """
    return src_nr * nr_destinations + dst_nr


def get_conn_id_from_ip(src_ip, dst_ip, nr_destinations, mapping_dicts):
    """
    This function calculates the connection ID that belongs to the specified src & dst IPs

    :param src_ip: String of the src IP
    :param dst_ip: String of the dst IP
    :param nr_destinations: # of all observed dst IPs
    :param mapping_dicts: mapping dictionary as returned by bro_parsers.conn_log_extract()
    :return: connection ID
    """
    src_nr = mapping_dicts['src_ip_to_nr_dict'][src_ip]
    dst_nr = mapping_dicts['dst_ip_to_nr_dict'][dst_ip]

    return get_conn_id(src_nr, dst_nr, nr_destinations)


def conn_id_to_src_and_dst_nr(conn_id, nr_destinations):
    """
    This function resolves a connection ID to the corresponding src & dst numbers

    :param conn_id: The connection ID to be resolved
    :param nr_destinations: # of all observed dst IPs
    :return: src IP-nr, dst IP-nr
    """
    src_nr = int(conn_id / nr_destinations)
    dst_nr = conn_id % nr_destinations

    return src_nr, dst_nr


def conn_id_to_src_and_dst_ip(conn_id, nr_destinations, mapping_dicts):
    """
    This function resolves a connection ID to the corresponding src & dst strings

    :param conn_id: The connection ID to be resolved
    :param nr_destinations: # of all observed dst IPs
    :param mapping_dicts: mapping dictionary as returned by bro_parsers.conn_log_extract()
    :return: src IP, dst IP
    """
    src_nr, dst_nr = conn_id_to_src_and_dst_nr(conn_id, nr_destinations)

    return mapping_dicts['src_nr_to_ip_dict'][src_nr], mapping_dicts['dst_nr_to_ip_dict'][dst_nr]


def get_conn_properties(conn_id, conn_properties_dict):
    """
    Get properties of a particular connection

    :param conn_id: ID of the connection
    :param conn_properties_dict: as returned by bro_parsers.conn_log_extract()
    :return: Properties of connection conn_id
    """
    return conn_properties_dict[conn_id]


def get_property_summary(conn_id, conn_properties_dict, key):
    """
    Get summary of all properties observed in connection conn_id

    :param conn_id: ID of the connection
    :param conn_properties_dict: as returned by bro_parsers.conn_log_extract()
    :param key: Name of the property to be summarized
    :return:
    """
    conn_properties = get_conn_properties(conn_id, conn_properties_dict)
    summary = []

    for prop in conn_properties:
        attribute = prop[key]
        if attribute in summary or attribute=='-':
            continue
        else:
            summary.append(attribute)

    return summary


def get_summary_of_all_properties(conn_id, conn_properties_dict):
    """
    Same as get_property_summary(), but gets summary of all properties

    """
    summary = {}
    for key in conn_properties_dict[conn_id]:
        summary[key] = get_property_summary(conn_id, conn_properties_dict, key)
    return summary


def write_properties_to_csv(conn_properties_dict, mapping_dicts, nr_destinations, csv_path):
    """
    Write the extracted connection properties (bro_parsers.conn_log_extract()) to a .csv file

    :param conn_properties_dict: see bro_parsers.conn_log_extract()
    :param mapping_dicts: see bro_parsers.conn_log_extract()
    :param nr_destinations: # of all observed dst IPs
    :param csv_path: Path of the new .csv file
    """
    print('Write connection properties to CSV ...')
    with open(os.path.join(csv_path, 'connections.csv'), 'w') as fp:
        fp.write('src_ip;dst_ip;src_dom;dst_dom;src_class;dst_class;nr_connections;conn_states;protocol;min_duration;max_duration;services;dst_ports\n')
        for conn_id, conn_properties in conn_properties_dict.items():
            src_ip, dst_ip = conn_id_to_src_and_dst_ip(conn_id, nr_destinations, mapping_dicts)
            if src_ip in mapping_dicts['ip_to_domain_dict']:
                src_dom = mapping_dicts['ip_to_domain_dict'][src_ip]
            else:
                src_dom = ''
            if dst_ip in mapping_dicts['ip_to_domain_dict']:
                dst_dom = mapping_dicts['ip_to_domain_dict'][dst_ip]
            else:
                dst_dom = ''
            conn_state_string = ", ".join('{} : {}'.format(k, v) for k, v in conn_properties['conn_state_counts'].items())
            # min_max_duration = '{}{}'.format(conn_properties['duration']['min_duration'], conn_properties['duration']['max_duration'])
            write_string = ';'.join([src_ip, dst_ip, src_dom, dst_dom, conn_properties['src_ip_class'],
                                                                    conn_properties['dst_ip_class'], conn_properties['nr_connections'],
                                                                    conn_state_string, conn_properties['proto'], conn_properties['duration']['min_duration'],
                                                                    conn_properties['duration']['max_duration'], conn_properties['service'],
                                                                    conn_properties['id.resp_p']])
            fp.write(write_string)


def merge_domain_dicts(domain_to_ip_dict_1, domain_to_ip_dict_2):
    """
    bro_parsers.extract_domains() can be run with different Bro .log files. This function can then be used to merge
    the different domain to IP dictionaries to one single dictionary.

    :param domain_to_ip_dict_1: First domain to IP dictionary
    :param domain_to_ip_dict_2: Second domain to IP dictionary
    :return: merged dictionary
    """
    keys_1 = domain_to_ip_dict_1.keys()
    keys_2 = domain_to_ip_dict_2.keys()

    for key_1 in keys_1:
        if key_1 in keys_2:
            # print('key_1 {} added to dict_2'.format(key_1))
            domain_to_ip_dict_2[key_1].union(domain_to_ip_dict_1[key_1])
        else:
            domain_to_ip_dict_2[key_1] = domain_to_ip_dict_1[key_1]
    return domain_to_ip_dict_2


def create_ip_to_domain_dict(domain_to_ip_dict):
    """
    This function converts a domain->IP mapping into a IP->domain mapping

    :param domain_to_ip_dict: Dictionary containg a domain->IP mapping
    :return: Dictionary with IP->domain mapping
    """
    ip_to_domain_dict = {}

    for domain, ips in domain_to_ip_dict.items():
        for ip in ips:
            if ip in ip_to_domain_dict:
                ip_to_domain_dict[ip].add(domain)
            else:
                ip_to_domain_dict[ip] = {domain}

    return ip_to_domain_dict


def check_multiple_ips_per_mac(mapping_dicts, split):
    """
    This function checks if multiple IPs are assigned to the same MAC address.
    MAC adresses with a unique IP will be written to unique_mac_ip_mappings.txt, and MACs with multiples IPs
    are written to nonunique_mac_ip_mappings.txt if split option is set. Otherwise all mappings will be written to
    a file called mac_ip_mappings.txt.

    :param mapping_dicts: Mapping dictionaries file as generated by conn_log_extract()
    :param split: See above description
    """
    if split:
        with open("unique_mac_ip_mappings.txt", "w") as fp1, open("nonunique_mac_ip_mappings.txt", "w") as fp2:
            for key, value in mapping_dicts['mac_to_ip_dict'].items():
                if len(value) == 1 or len(value) == 2:
                    fp1.write('{} - {}\n'.format(key, value))
                else:
                    fp2.write('{} - {}\n'.format(key, value))
    else:
        with open("mac_ip_mappings.txt", "w") as fp:
            for key, value in mapping_dicts['mac_to_ip_dict'].items():
                fp.write('{} - {}\n'.format(key, value))


def check_connections_within_subnet(conn_bool_matrix, mapping_dicts, conn_properties_dict):
    """
    Filter connections that go from one host to another host in the same subnet

    :param conn_bool_matrix: Generated by conn_log_extract()
    :param mapping_dicts: Generated by conn_log_extract()
    :param conn_properties_dict: Generated by conn_log_extract()
    """
    nr_destinations = conn_bool_matrix.shape[1]
    conn_bool_vector = conn_bool_matrix.flatten()

    with open("ipv4_interconnections.txt", "w") as fp1, open("ipv6_interconnections.txt", "w") as fp2:
        for conn_id, connected in enumerate(conn_bool_vector):
            if connected==False:
                continue
            else:
                src_ip, dst_ip = conn_id_to_src_and_dst_ip(conn_id, nr_destinations, mapping_dicts)
                src_mac = conn_properties_dict[conn_id]['orig_l2_addr']
                dst_mac = conn_properties_dict[conn_id]['resp_l2_addr']
                # connections between two ipv6 machines inside the same subnet - assume /64
                if ':' in src_ip or ':' in dst_ip:
                    if src_ip.split(':')[:4] == dst_ip.split(':')[:4]:
                        fp2.write('{} - {}\t{} - {}\n'.format(src_ip, dst_ip, src_mac, dst_mac))
                        #print('{} - {}'.format(src_ip, dst_ip))
                # connections between two ipv4 machines inside the same subnet - assume /24
                elif src_ip.split('.')[:3] == dst_ip.split('.')[:3]:
                    fp1.write('{} - {}\t{} - {}\n'.format(src_ip, dst_ip, src_mac, dst_mac))
                    #print('{} - {}'.format(src_ip, dst_ip))


def extract_ip_prefixes(ip_prefixes):
    """
    Splits the ip prefixes: e.g. ['10.7.0.0/16', '2a07:1182/32'] --> [['10','7'], ['2a07','1182']]
    prefix_lengths contains the len of the lists --> e.g. [['10','7'], ['2a07','1182']] --> [2,2]

    :param ip_prefixes:
    :return: Splitted prefixes and lengths of the prefix-lists --> e.g. [['10','7'], ['2a07','1182']] --> [2,2]
    """

    prefixes = []
    prefix_lengths = []
    for prefix in ip_prefixes:
        if '.' in prefix:
            prefix, len = prefix.split('/')
            index = int(int(len)/8)
            prefixes.append(prefix.split('.')[:index])
            prefix_lengths.append(index)
        elif ':' in prefix:
            prefix, len = prefix.split('/')
            index = int(int(len)/16)
            prefixes.append(prefix.split(':')[:index])
            prefix_lengths.append(index)
    return prefixes, prefix_lengths


def match_prefix(ip, prefix_list, prefix_lengths):
    """
    Checks if the specified IP matches one of the prefixes in the prefix_list

    :return: True, if a match is found
    """
    prefix_match = False
    for prefix, len in zip(prefix_list, prefix_lengths):
        if len <= 4:
            if ip.split('.')[:len] == prefix or ip.split(':')[:len] == prefix:
                prefix_match = True
                break
        # ipv6 with prefix length >64
        else:
            if ip.split(':')[:len] == prefix:
                prefix_match = True
                break
    return prefix_match


def generate_local_ip_aliases_list(local_hosts_csv_path):
    """
    Generates a list, containg the aliases (IPv4, IPv6) addresses of all the hosts in the local network.
    The information is extracted from the local_hosts.csv file

    :param local_hosts_csv_path: Path to the local_hosts.csv file
    """
    df = pd.read_csv(local_hosts_csv_path, sep=';')
    # local_ip_addresses = df['ipv4-1'] + df['ipv4-2']), set(df['ipv4-3']), set(df['ipv6-1']),set(df['ipv6-2']))
    local_ips = [] # each list element is a set of all ip aliases of a machine
    for index, row in df.iterrows():
        local_ips.append({row['ipv4-1'], row['ipv4-2'], row['ipv4-3'], row['ipv6-1'], row['ipv6-2']} - {'-', '?'} )
    return local_ips


def get_local_ip_aliases(target_ip, local_ips_list):
    """
    Checks if target_ip occurs in one of the entries of the aliases list as generated by generate_local_ip_aliases_list()

    :return: aliases or None if no aliases found
    """
    found_aliases = None
    for ip_aliases in local_ips_list:
        if target_ip in ip_aliases:
            found_aliases = list(ip_aliases)
            break
    return found_aliases


def is_internal_or_external_host(mapping_dicts, conn_properties_dict, local_hosts_csv_path, nr_destinations):
    """
    Generating a dictionary that maps the IP adresses occuring in mapping_dicts to the classes internal,
    unknown internal, external

    :param mapping_dicts: Generated by conn_log_extract()
    :param conn_properties_dict: Generated by conn_log_extract()
    :param local_hosts_csv_path: Path to the local_hosts.csv file
    :param nr_destinations: Number of DST IPs
    :return:conn_properties_dict (with IP-class included), ip_to_class_dict (holds IP->class mapping)
    """
    df = pd.read_csv(local_hosts_csv_path, sep=';')
    local_mac_addresses = set(df['mac'])
    local_mac_addresses.discard('?')
    local_ip_addresses = set(df['ipv4-1']).union(set(df['ipv4-2']), set(df['ipv4-3']), set(df['ipv6-1']), set(df['ipv6-2']))
    local_ip_addresses.discard('-')
    local_ip_addresses.discard('?')

    local_ip_prefixes = ['10.7.0.0/16', '151.216.7/24', '10.0.107.0/24', '2a07:1182/32']
    local_ip_prefixes, prefix_lengths = extract_ip_prefixes(local_ip_prefixes)

    ip_to_class_dict = {}

    for conn_id, props in conn_properties_dict.items():
        src_ip, dst_ip = conn_id_to_src_and_dst_ip(conn_id, nr_destinations, mapping_dicts)
        if src_ip in local_ip_addresses:
            src_ip_class = 'internal'
        else:
            if conn_properties_dict[conn_id]['orig_l2_addr'] in local_mac_addresses:
                src_ip_class = 'unknown internal'
            elif match_prefix(src_ip, local_ip_prefixes, prefix_lengths):
                src_ip_class = 'unknown internal'
            else:
                src_ip_class = 'external'
        conn_properties_dict[conn_id]['src_ip_class'] = src_ip_class
        ip_to_class_dict[src_ip] = src_ip_class

        if dst_ip in local_ip_addresses:
            dst_ip_class = 'internal'
        else:
            if conn_properties_dict[conn_id]['resp_l2_addr'] in local_mac_addresses:
                dst_ip_class = 'unknown internal'
            elif match_prefix(dst_ip, local_ip_prefixes, prefix_lengths):
                dst_ip_class = 'unknown internal'
            else:
                dst_ip_class = 'external'
        conn_properties_dict[conn_id]['dst_ip_class'] = dst_ip_class
        ip_to_class_dict[dst_ip] = dst_ip_class

    return conn_properties_dict, ip_to_class_dict


def get_default_gateway_macs(mapping_dicts, local_hosts_csv_path, nr_ips_per_mac_threshold=3):
    """
    Get a list of the MAC addresses that most likely belong to default gateways. The idea is to focus on MAC addresses
    affiliated with many different IPs. These are likely to correspond to gateways.

    :param mapping_dicts: Generated by conn_log_extract()
    :param local_hosts_csv_path: Generated by conn_log_extract()
    :param nr_ips_per_mac_threshold: E.g. if threshold=3 --> a MAC is classified as default gateway if more than 3 IPs assigned
    :return:
    """
    df = pd.read_csv(local_hosts_csv_path, sep=';')
    mac_counter = Counter()
    all_assigned_macs = set()
    potential_gateway_macs = set()

    for index, row in df.iterrows():
        local_ips = [row['ipv4-1'], row['ipv4-2'], row['ipv4-3'], row['ipv6-1'], row['ipv6-2']]
        assigned_macs = set()
        # create a list of all mac addresses assigned to the internal ip adresses (contain also gateway macs, as we have
        # probes before and after the gateways)
        for local_ip in local_ips:
            if local_ip=='-' or local_ip=='?':
                continue
            try:
                assigned_macs = assigned_macs.union(mapping_dicts['ip_to_mac_dict'][local_ip])
            except KeyError:
                continue
            else:
                i=0
                # print(index,local_ip,mapping_dicts['ip_to_mac_dict'][local_ip])

        for mac in assigned_macs:
            if  mac in mac_counter:
                mac_counter[mac] += 1
            else:
                mac_counter[mac] = 1
        all_assigned_macs = all_assigned_macs.union(assigned_macs)

    # removing the macs of which we are sure that they correspond to local machines and not to gateways
    # --> potential_gateway_macs contains all macs that are potentially gateway mac addresses
    # mac_counter countains the number of ip addresses that are assigned to a certain mac address
    # if mac_counter is big, the probability is high that it is a gateway
    potential_gateway_macs = potential_gateway_macs.union(all_assigned_macs.difference(df['mac']))

    potential_gateway_macs = [mac for mac in potential_gateway_macs if len(mapping_dicts['mac_to_ip_dict'][mac]) > nr_ips_per_mac_threshold]
    # create list of tuples (mac, #ips assigned to this mac) --> macs with a high number of assigned ip are most likely a gateway
    potential_gateway_macs_and_counts = sorted([(mac, len(mapping_dicts['mac_to_ip_dict'][mac]) ) for mac in potential_gateway_macs], key=lambda x: x[1], reverse=True)

    for tuple in potential_gateway_macs_and_counts:
        print(tuple)

    return potential_gateway_macs


def add_ip_class_to_snort_csv(data_path, ip_class_dict, malicious_ips):
    """
    Adds the IP classes found by is_internal_or_external_host() to the IPs in the snort_.csv alert log-file

    :param data_path: Directory where snort csv file is located
    :param ip_class_dict: Generated by is_internal_or_external_host()
    :param malicious_ips: List of the malicious IPs extracted from Cobalt Strike reports
    """
    print('Add ip classes to snort CSV...')
    alerts_csv = 'alerts.csv'
    df = pd.read_csv(os.path.join(data_path, 'snort', alerts_csv), sep=',')
    df_column_dict = {k: v for v, k in enumerate(df.columns.values)}

    src_ip_classes = []
    dst_ip_classes = []

    malicious = []

    for row in df.values:
        src_ip = row[df_column_dict['src']]
        dst_ip = row[df_column_dict['dst']]
        src_ip_classes.append(ip_class_dict[src_ip])
        dst_ip_classes.append(ip_class_dict[dst_ip])

        if src_ip in malicious_ips and dst_ip in malicious_ips:
            malicious.append('sd')
        elif src_ip in malicious_ips:
            malicious.append('s')
        elif dst_ip in malicious_ips:
            malicious.append('d')
        else:
            malicious.append('-')

    df['src_ip_class'] = src_ip_classes
    df['dst_ip_class'] = dst_ip_classes
    df['malicious'] = malicious

    df.to_csv(os.path.join(data_path, 'snort', 'alerts_with_IP_class.csv'), sep=',')


def host_names_to_ips(hostnames_path, mapping_dicts):
    """
    hostnames_path contains a textfile listing different domainnames.
    This function tries to resolve these domainnames to IPs.

    """
    hostnames = []
    ips = set()
    with open(hostnames_path, 'r') as fp:
        for line in fp:
            hostname = line.strip()
            hostnames.append(hostname)
            if hostname in mapping_dicts['domain_to_ip_dict']:
                print('{} resolved to ip {}'.format(hostname, mapping_dicts['domain_to_ip_dict'][hostname]))
                ips = ips.union(mapping_dicts['domain_to_ip_dict'][hostname])
            else:
                print('No ip found for {}'.format(hostname))
    return hostnames, ips


def get_all_malicious_ips(ips_path, hostnames_path, mapping_dicts):
    """
    This function resolves the domainnames in hostnames_path to IPs, and merges them with the IPs listed in ips_path
    We used this function to generate the list containing all malicious IPs (from Cobalt Strike reports), as used
    for the "host-labelling"

    :param ips_path:
    :param hostnames_path:
    :param mapping_dicts:  Generated by conn_log_extract() (Contains domain->IP mappings)
    :return: List of all resolved IPs and IPs contained in ips_path
    """
    # resolve hostnames to ip addresses
    _, mal_ips = host_names_to_ips(hostnames_path, mapping_dicts)
    # append the already known malicious ip addresses to the list of the resolved adresses
    with open(ips_path, 'r') as fp:
        for line in fp:
            mal_ips.add(line.strip())
    return mal_ips