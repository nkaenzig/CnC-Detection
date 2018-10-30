"""
Master Thesis
Network Monitoring and Attack Detection

bro_parsers.py
Functions used for parsing the Bro .log-files.

@author: Nicolas Kaenzig, D-ITET, ETH Zurich
"""

import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import pickle
import ipcalc
from collections import Counter
import bro_misc


def conn_log_extract(log_name, log_path='./logs/', skip_invalid_checksums=True):
    """
    Function used to parse bro conn.log files and to extract information.
    All src and dst IPs are enumerated, the mappings from  is stored in mapping_dicts.
    Furthermore dictionaries holding MAC to IP mappings (& reverse directions) are created.


    :param log_name: name of the conn.log file
    :param log_path: path of the directory where conn.log is located
    :param skip_invalid_checksums: Set to True to skip conn.log entries which were created by packets with invalid
                                   checksums
    :return: mapping_dicts - IP to number and the reverse mappings, MAC to IP mappings (& reverse)
             conn_bool_matrix - boolean connection matrix: If conn_bool_matrix[n][k], a connection between IP n and IP k
                                has been observed
             conn_properties_dict - Contains different connection properties for all src-dst IP pairs observed
                                     key=connection ID, value=list of connection properties of all connections
                                     with this src-dst IP pair
    """
    conn_path = os.path.join(log_path, log_name)

    conn_fieldnames = 'ts	uid	id.orig_h	id.orig_p	id.resp_h	id.resp_p	proto	service	duration	orig_bytes	' \
                      'resp_bytes	conn_state	local_orig	local_resp	missed_bytes	history	orig_pkts	orig_ip_bytes	' \
                      'resp_pkts	resp_ip_bytes	tunnel_parents	orig_l2_addr	resp_l2_addr'.split()

    ### READ LOG FILE INTO PANDAS DATAFRAME ###
    print("Load CSV into dataframe ...\nCSV path: {}".format(conn_path))
    df = pd.read_csv(conn_path, sep='\t', comment='#', names=conn_fieldnames, dtype=str)
    df = df.dropna(how='all') # to drop comment lines


    ### REMOVE DUPLICATES ###
    # to get counts of src and dst IPs
    df_src = df.drop_duplicates('id.orig_h')[['id.orig_h']]
    nr_sources = df_src.size
    del df_src

    df_dst = df.drop_duplicates('id.resp_h')[['id.resp_h']]
    nr_destinations = df_dst.size
    del df_dst

    ### DICTIONARIES ###
    # Dictionary that maps column name to column index (need that as we are going to convert the df to numpy array)
    df_column_dict =  {k: v for v, k in enumerate(df.columns.values)}

    # Dictionary that maps IP address strings to their assigned number id (and vice versa)
    src_ip_to_nr_dict = {}
    dst_ip_to_nr_dict = {}
    src_nr_to_ip_dict = {}
    dst_nr_to_ip_dict = {}

    mac_to_ip_dict = {}
    ip_to_mac_dict = {}

    conn_bool_matrix = np.zeros((nr_sources,nr_destinations), dtype = 'bool')
    conn_properties_dict = {}           # {}: key=connection ID, value=list of connection properties of all connections
                                        # with this src-dst IP pair



    # summ all up into one dictionary to save later all together into dataframe
    mapping_dicts = {'src_ip_to_nr_dict': src_ip_to_nr_dict, 'dst_ip_to_nr_dict': dst_ip_to_nr_dict,
                     'src_nr_to_ip_dict': src_nr_to_ip_dict, 'dst_nr_to_ip_dict': dst_nr_to_ip_dict,
                     'mac_to_ip_dict': mac_to_ip_dict, 'ip_to_mac_dict': ip_to_mac_dict}

    src_nr = 0
    dst_nr = 0
    next_src_nr = 0
    next_dst_nr = 0

    for df_row in tqdm(df.values, desc='Parsing dataframe rows'):
        if skip_invalid_checksums and (df_row[df_column_dict['history']] == 'C' or df_row[df_column_dict['history']] == 'Cc'
                                       and df_row[df_column_dict['history']]=='OTH'):
            continue
        src_ip = df_row[df_column_dict['id.orig_h']]
        dst_ip = df_row[df_column_dict['id.resp_h']]

        # assign a number to each ip address
        if src_ip in src_ip_to_nr_dict:
            src_nr = src_ip_to_nr_dict[src_ip]
        else:
            # create dictionaries for ip address to ip number mapping
            src_ip_to_nr_dict[src_ip] = next_src_nr
            src_nr_to_ip_dict[next_src_nr] = src_ip
            src_nr = next_src_nr
            next_src_nr += 1

        if dst_ip in dst_ip_to_nr_dict:
            dst_nr = dst_ip_to_nr_dict[dst_ip]
        else:
            # create dictionaries for ip address to ip number mapping
            dst_ip_to_nr_dict[dst_ip] = next_dst_nr
            dst_nr_to_ip_dict[next_dst_nr] = dst_ip
            dst_nr = next_dst_nr
            next_dst_nr += 1

        # set entry in boolean connection to 1 --> indicates that there is/are connection/s between this src-dst pair
        conn_bool_matrix[src_nr, dst_nr] = 1

        src_mac = df_row[df_column_dict['orig_l2_addr']]
        dst_mac = df_row[df_column_dict['resp_l2_addr']]
        
        if src_mac in mac_to_ip_dict:
            mac_to_ip_dict[src_mac].add(src_ip)
        else:
            mac_to_ip_dict[src_mac] = {src_ip}
        if dst_mac in mac_to_ip_dict:
            mac_to_ip_dict[dst_mac].add(dst_ip)
        else:
            mac_to_ip_dict[dst_mac] = {dst_ip}

        if src_ip in ip_to_mac_dict:
            ip_to_mac_dict[src_ip].add(src_mac)
        else:
            ip_to_mac_dict[src_ip] = {src_mac}
        if dst_ip in ip_to_mac_dict:
            ip_to_mac_dict[dst_ip].add(dst_mac)
        else:
            ip_to_mac_dict[dst_ip] = {src_mac}

        conn_id = src_nr * nr_destinations + dst_nr

        # check if there is already a connection with this src-dst pair in the list, if yes append it there
        if conn_id in conn_properties_dict:
            for key in conn_properties_dict[conn_id]:
                # if df_row[df_column_dict[key]] != '-' and df_row[df_column_dict[key]] not in conn_properties_dict[conn_id][key]:
                if key == 'nr_connections':
                    conn_properties_dict[conn_id][key] += 1
                elif key == 'conn_state_counts':
                    if df_row[df_column_dict['conn_state']] in conn_properties_dict[conn_id][key]:
                        conn_properties_dict[conn_id][key][df_row[df_column_dict['conn_state']]] += 1
                    else:
                        conn_properties_dict[conn_id][key][df_row[df_column_dict['conn_state']]] = 1
                elif df_row[df_column_dict[key]] == '-':
                    continue
                elif key == 'duration':
                    duration = df_row[df_column_dict['duration']]
                    if conn_properties_dict[conn_id][key]['min_duration'] == '-':
                        conn_properties_dict[conn_id][key]['min_duration'] = duration
                        conn_properties_dict[conn_id][key]['max_duration'] = duration
                    elif float(duration) < float(conn_properties_dict[conn_id][key]['min_duration']):
                        conn_properties_dict[conn_id][key]['min_duration'] = duration
                    elif float(duration) > float(conn_properties_dict[conn_id][key]['max_duration']):
                        conn_properties_dict[conn_id][key]['max_duration'] = duration
                elif df_row[df_column_dict[key]] != '-':
                    conn_properties_dict[conn_id][key].add(df_row[df_column_dict[key]])
        else:
            conn_properties_dict[conn_id] = {'proto': {df_row[df_column_dict['proto']]}, 'service': {df_row[df_column_dict['service']]},
                                             'duration': {'min_duration': df_row[df_column_dict['duration']], 'max_duration': df_row[df_column_dict['duration']]},
                                             'id.resp_p': {df_row[df_column_dict['id.resp_p']]}, 'orig_l2_addr': {df_row[df_column_dict['orig_l2_addr']]},
                                             'resp_l2_addr': {df_row[df_column_dict['resp_l2_addr']]}, 'conn_state_counts': Counter({df_row[df_column_dict['conn_state']]: 1}),
                                             'nr_connections': 1}

    return mapping_dicts, conn_bool_matrix, conn_properties_dict


def dhcp_log_extract_subnets(log_name, log_path='./logs/'):
    """
    Parses Bro dhcp.log file to find different subnets based on dhcp protocol header information.

    :param log_name: name of the dhcp.log file
    :param log_path: path of the directory where dhcp.log is located
    :return: a list of the different subnetmasks observed in Bro dhcp.log
    """
    dhcp_path = os.path.join(log_path, log_name)

    dhcp_fieldnames = 'ts	uid	id.orig_h	id.orig_p	id.resp_h	id.resp_p	mac	assigned_ip	lease_time	trans_id	subnet_mask'.split()

    ### READ LOG FILE INTO PANDAS DATAFRAME ###
    print("Load CSV into dataframe ...\nCSV path: {}".format(dhcp_path))
    df = pd.read_csv(dhcp_path, sep='\t', comment='#', names=dhcp_fieldnames, dtype=str)
    df = df.dropna(how='all') # to drop comment

    # Dictionary that maps column name to column index (need that as we are going to convert the df to numpy array)
    df_column_dict = {k: v for v, k in enumerate(df.columns.values)}

    subnets = {}

    for df_row in tqdm(df.values, desc='Parsing dataframe rows'):
        current_client_ip = df_row[df_column_dict['id.orig_h']]
        current_subnetmask = df_row[df_column_dict['subnet_mask']]

        addr = ipcalc.IP(current_client_ip, mask=current_subnetmask)
        subnet_with_cidr = str(addr.guess_network())
        if subnet_with_cidr not in subnets:
            subnets.add(subnet_with_cidr)

    return subnets


def dhcp_log_extract_mac(log_name, log_path='./logs/'):
    """
    Writes a list of all MAC addresses observed in Bro dhcp.log to ./dhcp_macs.csv

    :param log_name: name of the dhcp.log file
    :param log_path: path of the directory where dhcp.log is located
    """
    dhcp_path = os.path.join(log_path, log_name)

    dhcp_fieldnames = 'ts	uid	id.orig_h	id.orig_p	id.resp_h	id.resp_p	mac	assigned_ip	lease_time	trans_id	subnet_mask'.split()

    ### READ LOG FILE INTO PANDAS DATAFRAME ###
    print("Load CSV into dataframe ...\nCSV path: {}".format(dhcp_path))
    df = pd.read_csv(dhcp_path, sep='\t', comment='#', names=dhcp_fieldnames, dtype=str)
    df = df.dropna(how='all') # to drop comment

    # Dictionary that maps column name to column index (need that as we are going to convert the df to numpy array)
    df_column_dict = {k: v for v, k in enumerate(df.columns.values)}

    with open("dhcp_macs.csv", "w") as fp:
        for df_row in tqdm(df.values, desc='Parsing dataframe rows'):
            current_client_ip = df_row[df_column_dict['id.orig_h']]
            current_assigned_ip = df_row[df_column_dict['assigned_ip']]
            current_mac = df_row[df_column_dict['mac']]

            if current_assigned_ip.split('.')[:3] == ['10', '7', '2'] or current_assigned_ip.split(':')[:4] == ['2a07', '1182', '7', '2']:
                fp.write('{};{};{}\n'.format(current_client_ip,current_assigned_ip,current_mac))


def extract_domains(log_name, log_path='./logs/'):
    """
    Generates a mapping from domain to IPs. Works either with dns.log (Query field), http.log (Host field),
    or sslanalyzer.log (domain name from TLS Hello - this .log is generated with sslAnalyzer.bro, written by Roland Meier)

    :param log_name: name of the Bro .log file to be used. Options: dns.log, http.log, sslanalyzer.log
    :param log_path: path of the directory where dhcp.log is located
    :return:
    """
    path = os.path.join(log_path, log_name)

    fieldnames = None
    empty_domain = '-'
    host_name_key = ''
    host_ip_key = ''

    if 'ssl' in path:
        fieldnames = 'ts	uid	evt	is_orig	sAddr	dAddr	sPort	dPort	server_name	size'.split()
        empty_domain = '?'
        host_name_key = 'server_name'
        host_ip_key = 'dAddr'
    elif 'dns' in path:
        fieldnames = 'ts	uid	id.orig_h	id.orig_p	id.resp_h	id.resp_p	proto	trans_id	rtt	query	qclass	' \
                     'qclass_name	qtype	qtype_name	rcode	rcode_name	AA	TC	RD	RA	Z	answers	TTLs	rejected'.split()
        host_name_key = 'query'
        host_ip_key = 'answers'
    elif 'http' in path:
        fieldnames = 'ts	uid	id.orig_h	id.orig_p	id.resp_h	id.resp_p	trans_depth	method	host	uri	referrer' \
                     '	version	user_agent	request_body_len	response_body_len	status_code	status_msg	info_code	' \
                     'info_msg	tags	username	password	proxied	orig_fuids	orig_filenames	orig_mime_types	resp_fuids	' \
                     'resp_filenames	resp_mime_types'.split()
        host_name_key = 'host'
        host_ip_key = 'id.resp_h'

    ### READ LOG FILE INTO PANDAS DATAFRAME ###
    print("Load CSV into dataframe ...\nCSV path: {}".format(path))
    df = pd.read_csv(path, sep='\t', comment='#', names=fieldnames, dtype=str)
    df = df.dropna(how='all') # to drop comment

    # Dictionary that maps column name to column index (need that as we are going to convert the df to numpy array)
    df_column_dict = {k: v for v, k in enumerate(df.columns.values)}
    changing_domains = {}

    domain_to_ip_dict = {}

    for df_row in tqdm(df.values, desc='Parsing dataframe rows'):
        # for DNS we are only interested in type A and AAAA query types
        if host_name_key == 'query':
            if df_row[df_column_dict['qtype_name']] != 'A' and df_row[df_column_dict['qtype_name']] != 'AAAA':
                continue
            current_host_name = df_row[df_column_dict[host_name_key]].strip()
            current_host_ip = df_row[df_column_dict[host_ip_key]].strip().split(',')
            if current_host_ip == ['-']: continue
            # note: sometimes DNS server answers not with an IP but with a domain name
            current_host_ip = [ip for ip in current_host_ip if bro_misc.check_if_valid_IP(ip)]
        else:
            current_host_name = str(df_row[df_column_dict[host_name_key]]).strip()
            current_host_ip = df_row[df_column_dict[host_ip_key]].strip()

        if current_host_name == empty_domain:
            continue
        elif current_host_name in domain_to_ip_dict:
            if host_name_key == 'query':
                domain_to_ip_dict[current_host_name].union(set(current_host_ip))
            else:
                domain_to_ip_dict[current_host_name].add(current_host_ip)
            if current_host_name in changing_domains:
                changing_domains[current_host_name] += 1
            else:
                changing_domains[current_host_name] = 1
        else:
            domain_to_ip_dict[current_host_name] = set(current_host_ip)

    return domain_to_ip_dict, changing_domains