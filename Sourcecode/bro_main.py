"""
Master Thesis
Network Monitoring and Attack Detection

bro_main.py
Main script used for analysing the Bro .log-files.

Main functionalities:
- Domain-name to IP resolution
- Gathering flow information accumulated by src-dst IP pairs
- Labelling the src-IP and dst-IP as internal, unknown internal & external

@author: Nicolas Kaenzig, D-ITET, ETH Zurich
"""

import sys
sys.path.append('/home/nicolas/.local/lib/python3.5/site-packages')
import bro_parsers as ex
import time
import os
import pickle
import numpy as np
import bro_misc
import json


data_path = './data/'
extracted_path = './extracted/ls17/'
cs_path = './cobalt_strike/'

log_path = '/mnt/data/nicolas/bro_logs/ls17/'
conn_log_name = 'conn.log'
http_log_name = 'http.log'
dns_log_name = 'dns.log'
dhcp_log_name = 'dhcp.log'
sslanalyzer_log_name = 'sslanalyzer.log'

local_hosts_csv = 'local_hosts6.csv'


def main():
    if os.path.exists(os.path.join(extracted_path, 'mapping_dicts.pickle')) and \
            os.path.exists(os.path.join(extracted_path, 'conn_bool_matrix.pickle')) and \
            os.path.exists(os.path.join(extracted_path, 'conn_properties.pickle')):
    # if os.path.exists(os.path.join(extracted_path, 'mapping_dicts.pickle')):

        print('Load pickle files found in {} ...'.format(extracted_path))
        with open(os.path.join(extracted_path, 'mapping_dicts.pickle'), 'rb') as fp:
            mapping_dicts = pickle.load(fp)
        with open(os.path.join(extracted_path, 'conn_bool_matrix.pickle'), 'rb') as fp:
            conn_bool_matrix = pickle.load(fp)
        with open(os.path.join(extracted_path, 'conn_properties.pickle'), 'rb') as fp:
            conn_properties_dict = pickle.load(fp)
    else:
        print('Extract data from {} ...'.format(conn_log_name))
        mapping_dicts, conn_bool_matrix, conn_properties_dict = ex.conn_log_extract(log_name=conn_log_name, log_path=log_path,
                                                                                    skip_invalid_checksums=True)
        mapping_dicts = {}

        print('Loading domain names ...')
        ssl_domain_to_ip_dict, ssl_c_doms = ex.extract_domains(log_name=sslanalyzer_log_name, log_path=log_path)
        http_domain_to_ip_dict, http_c_doms = ex.extract_domains(log_name=http_log_name, log_path=log_path)
        dns_domain_to_ip_dict, dns_c_doms = ex.extract_domains(log_name=dns_log_name, log_path=log_path)

        domain_to_ip_dict = bro_misc.merge_domain_dicts(ssl_domain_to_ip_dict, dns_domain_to_ip_dict)
        domain_to_ip_dict = bro_misc.merge_domain_dicts(http_domain_to_ip_dict, domain_to_ip_dict)
        ip_to_domain_dict = bro_misc.create_ip_to_domain_dict(domain_to_ip_dict)

        mapping_dicts['domain_to_ip_dict'] = domain_to_ip_dict
        mapping_dicts['ip_to_domain_dict'] = ip_to_domain_dict

        print("Saving extracted data to {} ...".format(extracted_path))
        if not os.path.exists(extracted_path):
            os.makedirs(extracted_path)

        with open(os.path.join(extracted_path, 'mapping_dicts.pickle'), 'wb') as fp:
            pickle.dump(mapping_dicts, fp)
        with open(os.path.join(extracted_path, 'conn_bool_matrix.pickle'), 'wb') as fp:
            pickle.dump(conn_bool_matrix, fp)
        with open(os.path.join(extracted_path, 'conn_properties.pickle'), 'wb') as fp:
            pickle.dump(conn_properties_dict, fp)

    nr_destinations = conn_bool_matrix.shape[1]

    bro_misc.get_default_gateway_macs(mapping_dicts, os.path.join('./data', local_hosts_csv))

    ### Create .json file holding all malicious IP addresses listed in the cobalt strike reports
    mal_ips = bro_misc.get_all_malicious_ips(os.path.join(cs_path, 'indicatorsofcompromise_IPs.txt'),
                                             os.path.join(cs_path, 'indicatorsofcompromise_domains.txt'), mapping_dicts)
    mal_ips_dict = {"malicious_ips" : list(mal_ips)}
    with open(os.path.join(extracted_path, 'malicious_ips.json'), 'w') as fp1:
        print('Saving malicious ip dictionary to {} ...'.format(extracted_path))
        json.dump(mal_ips_dict, fp1)

    ### Write a .json file holding mappings from IP to whether it is external or internal
    _, ip_class_dict = bro_misc.is_internal_or_external_host(mapping_dicts, conn_properties_dict,
                                                             os.path.join(data_path, local_hosts_csv), nr_destinations)
    with open(os.path.join(extracted_path, 'ip_classes.json'), 'w') as fp2:
        print('Saving ip-classes dictionary to {} ...'.format(extracted_path))
        json.dump(ip_class_dict, fp2)

    ### Write the extracted connection properties to a .csv file
    bro_misc.write_properties_to_csv(conn_properties_dict, mapping_dicts, nr_destinations, extracted_path)
    bro_misc.add_ip_class_to_snort_csv(data_path, extracted_path, conn_properties_dict, nr_destinations,
                                       mapping_dicts, ip_class_dict, mal_ips)

    ### MISC
    # bro_misc.check_multiple_ips_per_mac(mapping_dicts, split=False)
    # bro_misc.check_connections_within_subnet(conn_bool_matrix, mapping_dicts, conn_properties_dict)
    # subnets = ex.dhcp_log_extract_subnets(log_name=dhcp_log_name, log_path=log_path)


if __name__ == "__main__":
    start = time.time()

    main()

    end = time.time()
    print('Execution took {} seconds'.format(end - start))