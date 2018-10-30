"""
Master Thesis
Network Monitoring and Attack Detection

elasticsearch_indexing.py
Functions used to index the pcap (.json), Bro and Snort data into Elasticsearch


@author: Nicolas Kaenzig, D-ITET, ETH Zurich
"""

from elasticsearch import Elasticsearch
from elasticsearch import helpers
from elasticsearch import exceptions
import os
import time
from collections import deque
import sys
import jsonlines
import json
import csv
import datetime
import bro_misc
import pickle
import hex_to_pcap


DOCS_PER_FILE = 500000
MAX_DOC_SIZE = 50000000
THREADS = 4
CHUNK_SIZE = 500

current_id = 0 # global variable used for doc-ID during indexing


def timegm_ms(dt):
    """Function to calculate Unix timestamp in [ms] from GMT. - adapted from calendar.timegm()"""
    EPOCH = 1970
    _EPOCH_ORD = datetime.date(EPOCH, 1, 1).toordinal()

    year, month, day, hour, minute, second, millisecond = dt.timetuple()[:7]
    days = datetime.date(year, month, 1).toordinal() - _EPOCH_ORD + day - 1
    hours = days*24 + hour
    minutes = hours*60 + minute
    seconds = minutes*60 + second
    milliseconds = seconds*1000 + millisecond
    return milliseconds


def snort_generator(snort_fp, index_name):
    """
    Generator function, to generate the Snort documents for indexing in .json format

    :param snort_fp: Filepointer to snorts opened .csv file
    :param index_name: Name of the created snort index
    """
    csv_fields = snort_fp.readline()[:-1].split(',')
    reader = csv.DictReader(snort_fp, csv_fields, delimiter=',')
    id = 0
    for row in reader:
        dt = datetime.datetime.strptime(row['timestamp'], "%m/%d/%y-%H:%M:%S.%f ")
        row['timestamp'] = str(timegm_ms(dt))

        # rename ip fields to match the ip field name of the other indices
        row['src_ip'] = row['src']
        row['dst_ip'] = row['dst']
        del row['src']
        del row['dst']

        row['src_port'] = row['srcport']
        row['dst_port'] = row['dstport']
        del row['srcport']
        del row['dstport']

        source = json.dumps(row)
        action = {
            '_op_type': 'index',
            '_index': index_name,
            '_type': "snort",
            '_id': id,
            '_source': source
        }
        id += 1
        yield action
    print('{} documents indexed'.format(id))


def bro_generator(bro_fp, index_name):
    """
    Generator function, to generate the Bro documents for indexing in .json format

    :param bro_fp: Filepointer to Bro's opened .csv file
    :param index_name: Name of the created Bro index
    """
    for line in bro_fp:
        if line.startswith('#fields'):
            csv_fields = line.split('\t')[1:]
            break

    reader = csv.DictReader(bro_fp, csv_fields, delimiter='\t')
    id = 0
    try:
        for row in reader:

            if row['ts'].startswith('#'):
                continue
            row['ts'] = str(round(float(row['ts'])*1000))
            row['timestamp'] = row['ts']
            del row['ts']

            if 'id' in row:
                row['uid'] = row['id']
                del row['id']
            try:
                if row['duration'] == '-':
                    # del row['duration']
                    row['duration'] = 0
            except KeyError:
                pass
            try:
                if row['id.orig_p'] != '-' or row['id.resp_p'] != '-':
                    row['src_port'] = row['id.orig_p']
                    row['dst_port'] = row['id.resp_p']
                del row['id.orig_p']
                del row['id.resp_p']
            except KeyError:
                pass
            try:
                if row['id.orig_h'] != '-' or row['id.resp_h'] != '-':
                    row['src_ip'] = row['id.orig_h']
                    row['dst_ip'] = row['id.resp_h']
                del row['id.orig_h']
                del row['id.resp_h']
            except KeyError:
                pass

            source = json.dumps(row)
            action = {
                '_op_type': 'index',
                '_index': index_name,
                '_type': "bro",
                '_id': id,
                '_source': source
            }
            id += 1
            yield action
    except:
        print('Some error in bro_generator() occured')
        pass
    if id==0:
        raise IndexError('Zero documents indexed')
    print('{} documents indexed'.format(id))


def ndjson_generator(ndjson_fp, index_name):
    """
    Generator function, to generate the Bro documents from a .ndjson file (used for PCAP indexing)

    :param ndjson_fp: Filepointer opened .ndjson file
    :param index_name: Name of the index
    """
    global current_id
    count = 0
    index_line = ''
    for line in ndjson_fp:
        if line.startswith('{"index'):
            index_line = line
            continue
        else:
            action = {
                '_op_type': 'index',
                '_index': index_name,
                '_type': 'pcap_file',
                '_id': current_id,
                'pipeline': 'rename_pcap_ips',
                '_source': line
            }

            current_id += 1
            count += 1

            yield action
    print('{} documents indexed'.format(count))


def benchmark(es, json_path):
    """
    Helper function used to find the optimal number of threads for parallel_bulk() indexing

    :param es: Elasticsearch instance
    :param json_path: Path of a .ndjson file to be indexed during the benchmark
    """
    nr_threads = [1,2,4,8,16,32,64]

    for threads in nr_threads:
        with open(os.path.join(json_path, json_file), 'r') as fp:
            start = time.time()
            deque(helpers.parallel_bulk(es, ndjson_generator(fp), thread_count=threads), maxlen=0)
            end = time.time()
            es.indices.delete(index='packets-ls17')
            print('B {} threads: {}s'.format(threads, end - start))


def create_index(es, index_name):
    """
    Function that creates a new index named index_name

    :param es: Elasticsearch instance
    :param index_name: Name of the new index
    """
    # since we are running locally, use one shard and no replicas
    request_body = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0
        }
    }
    print("creating '%s' index..." % (index_name))
    res = es.indices.create(index=index_name, body=request_body)


def index_single_file(es, index_name, file_path, doc_generator):
    """
    Function to index a single file into elasticsearch

    :param es: Elasticsearch instance
    :param index_name: Name of the target index
    :param file_path: Path of the file to be indexed
    :param doc_generator: Generator function for the chosen file type (.ndjson, bro, snort)
    """
    with open(file_path, 'r') as fp:
        print('Indexing file {} to index {}  ...'.format(os.path.basename(file_path), index_name))
        start = time.time()
        deque(helpers.parallel_bulk(es, doc_generator(fp, index_name), thread_count=THREADS, chunk_size=CHUNK_SIZE), maxlen=0)
        end = time.time()
        print('Took: {}s\n'.format(end - start))


def index_directory(es, index_name, pcap_path, json_path, sort_paths=False):
    """
    Function to index multiple .ndjson files in a directory (used for PCAP indexing)

    :param es: Elasticsearch instance
    :param index_name: Name of the target index
    :param pcap_path: Directory that holds the original .pcap files
    :param json_path: Directory that holds the .ndjson files
    :param sort_paths: If True, sort the list of filenames in order to process them in a fixed order
    :return:
    """
    def last_8chars(x):
        return (x[-13:-5])

    print('Index name: {}\nData path: {}\n'.format(index_name, pcap_path))
    create_index(es, index_name)
    global current_id
    filename_ids_dict = dict()

    paths = os.listdir(json_path)
    if sort_paths:
        paths = sorted(paths, key=last_8chars)
    start = time.time()
    for file in paths:
        if file.endswith(".json"):
            start_id = current_id
            index_single_file(es, index_name, os.path.join(json_path, file), ndjson_generator)
            end_id = current_id-1
            filename_ids_dict[file] = (start_id,end_id)
    end = time.time()
    print('Indexing directory took: {}s\n'.format(end - start))

    dict_path = os.path.join(pcap_path, index_name + '_filename_ids_dict.json')
    with open(dict_path, 'w') as fp:
        print('Saving filename to document-IDs dictionary in {} ...'.format(dict_path))
        json.dump(filename_ids_dict, fp)


def index_bro(es, bro_log_path, index_pattern, index_postfix):
    """
    Function used to create the Bro index

    :param es: Elasticsearch instance
    :param bro_log_path: Path to the directory holding the bro log-files
    :param index_pattern: Set e.g. to 'conn', to only index the conn.log file
    :param index_postfix: Index will be named bro-conn-index_postfix
    """
    files = os.listdir(bro_log_path)
    for file in files:
        if index_pattern not in file:
            continue
        elif file.endswith(".log"):
            index_name = 'bro-' + os.path.splitext(file)[0] + index_postfix
            create_index(es, index_name)
            index_single_file(es, index_name, os.path.join(bro_log_path, file), bro_generator)
        # deque(helpers.parallel_bulk(es, bro_generator(fp, index_name), thread_count=THREADS, chunk_size=CHUNK_SIZE), maxlen=0)


def index_snort(es, index_name, snort_alerts_path):
    """
    Function used to create the Bro index

    :param es: Elasticsearch instance
    :param index_name: Name of the new index
    :param snort_alerts_path: Path of the snort .csv file
    """
    create_index(es, index_name)
    index_single_file(es, index_name, snort_alerts_path, snort_generator)


def find_max_doc_size(file_path):
    """
    Helperfunction to find the maximum document size in a .ndjson file

    :param file_path: Path to the .ndjson file
    """
    with open(file_path, 'r') as fp:
        id = 0
        max_doc_size = 0
        min_doc_size = float('Inf')
        index_line = None
        for nr, line in enumerate(fp):
            if line.startswith('{"index'):
                index_line = line
                continue
            else:
                # pkt = json.loads(line)
                size = sys.getsizeof(line)

                if size > max_doc_size:
                    max_doc_size = size
                if size < min_doc_size:
                    min_doc_size = size

        print('{} documents'.format(id))
        print('max {}\nmin {}'.format(max_doc_size, min_doc_size))


def delete_indices(es, pattern):
    """
    Function to delete all ES indexes with <pattern> in their name
    """
    for index in es.indices.get('*'):
        if pattern in index:
            es.indices.delete(index=index)
            print('Index {} deleted ...'.format(index))


def label_ip_class(es, index_pattern, doc_type, ip_classes_path, src_ip_field, dst_ip_field):
    """
    Function to label the IP classes (internal, unknown internal, external) in the documents of all indexes that
    have <index_pattern> in their name.

    :param es: Elasticsearch instance
    :param index_pattern: see above
    :param doc_type: Document type of index
    :param ip_classes_path: Path to the .json file holding the ip->class mappings, as generated in bro_main.py
    :param src_ip_field: name of the src-IP field
    :param dst_ip_field: name of the dst-IP field
    """
    with open(ip_classes_path, 'r') as fp:
        ip_classes_dict = json.load(fp)
        internal_ips = []
        unknown_internal_ips = []
        external_ips = []

        for ip, ip_class in ip_classes_dict.items():
            if ip_class=='internal':
                internal_ips.append(ip)
            elif ip_class=='unknown internal':
                unknown_internal_ips.append(ip)
            elif ip_class=='external':
                external_ips.append(ip)

        for index_name in es.indices.get('*'):
            if index_pattern in index_name:
                print('Updating index {}'.format(index_name))

                src_unknown_internal_query = {
                    "script": {
                        "inline": "ctx._source.ip_class='unknown internal'",
                        "lang": "painless"
                    },
                    "query": {
                        "terms": {src_ip_field: unknown_internal_ips}
                    }
                }

                dst_unknown_internal_query = {
                    "script": {
                        "inline": "ctx._source.ip_class='unknown internal'",
                        "lang": "painless"
                    },
                    "query": {
                        "terms": {dst_ip_field: unknown_internal_ips}
                    }
                }

                print(es.update_by_query(body=src_unknown_internal_query, doc_type=doc_type, index=index_name, slices=3, request_timeout=11000))
                print(es.update_by_query(body=dst_unknown_internal_query, doc_type=doc_type, index=index_name, slices=3, request_timeout=11000))


def label_malicious_ips(es, index_pattern, doc_type, malicious_ips_path, src_ip_field, dst_ip_field):
    """
    Function to perform "host-labelling" in the documents of all indexes that have <index_pattern> in their name.

    :param es: Elasticsearch instance
    :param index_pattern: see above
    :param doc_type: Document type of index
    :param malicious_ips_path: Path to the .json file holding all malicious IPs
    :param src_ip_field: name of the src-IP field
    :param dst_ip_field: name of the dst-IP field
    """
    # note: once the update_by_query task is initiated, stopping the python script won't cancel this task
    # to cancel the task, see doc: https://www.elastic.co/guide/en/elasticsearch/reference/current/docs-update-by-query.html

    with open(malicious_ips_path, 'r') as fp:
            malicious_ips = json.load(fp)['malicious_ips']

    for index_name in es.indices.get('*'):
        if index_pattern in index_name:
            print('Updating index {}'.format(index_name))

            src_query = {
                "script": {
                    "inline": "ctx._source.malicious_host='src'",
                    "lang": "painless"
                },
                "query": {
                    "terms" : { src_ip_field : malicious_ips}
                }
            }

            dst_query = {
                "script": {
                    "inline": "ctx._source.malicious_host='dst'",
                    "lang": "painless"
                },
                "query": {
                    "terms" : { dst_ip_field : malicious_ips}
                }
            }

            print(es.update_by_query(body=src_query, doc_type=doc_type, index=index_name, slices=1, request_timeout=11000))
            print(es.update_by_query(body=dst_query, doc_type=doc_type, index=index_name, slices=1, request_timeout=11000))


def label_malicious_sessions(es, index_name, doc_type, malicious_sessions_path, local_hosts_path, mapping_dicts_path,
                             src_ip_field, dst_ip_field):
    """
    Function to perform "session-labelling" in the documents of all indexes that have <index_pattern> in their name.

    :param es: Elasticsearch instance
    :param index_pattern: see above
    :param doc_type: Document type of index
    :param malicious_sessions_path: Path to the malicious_sessions.json file as generated by cs_report_parser.py
    :param local_hosts_path: Path to the local_hosts.csv file
    :param mapping_dicts_path: Path to the mapping_dicts.pickle file as generated in bro_main.py
    :param src_ip_field: name of the src-IP field
    :param dst_ip_field: name of the dst-IP field
    """
    year = '2017'
    updated = 0
    with open(mapping_dicts_path, 'rb') as fp:
        mapping_dicts = pickle.load(fp)

    no_matches = []
    local_ips_list = bro_misc.generate_local_ip_aliases_list(local_hosts_path)
    with open(malicious_sessions_path, 'r') as fp:
        malicious_sessions = json.load(fp)

    for pid, sessions in malicious_sessions.items():
        for session in sessions:
            print('Labeling session {} ...'.format(pid))
            # check if start_time is same as end_time --> this means that there is an initial beacon listed in activity.pdf
            # but no activities are listed  the corresponding session in the activities section of opnotes.pdf
            if session['start_time']==session['end_time']:
                continue

            # set datetime format to match supported format of elasticsearch yyyy-MM-dd HH:mm:ss
            start_time = '{}-{}-{} {}:00'.format(year, session['start_date'].split('/')[1], session['start_date'].split('/')[0], session['start_time'])
            end_time = '{}-{}-{} {}:00'.format(year, session['end_date'].split('/')[1], session['end_date'].split('/')[0], session['end_time'])
            end_time_same_day = '{}-{}-{} {}:00'.format(year, session['start_date'].split('/')[1], session['start_date'].split('/')[0], session['end_time_same_day'])

            src_ip_aliases = bro_misc.get_local_ip_aliases(session['src_ip'], local_ips_list)

            malicious_ips = set()
            malicious_hostnames = set()
            for host in session['hosts']:
                if bro_misc.check_if_valid_IP(host):
                    malicious_ips.add(host)
                elif host in mapping_dicts['domain_to_ip_dict']:
                    malicious_ips = malicious_ips.union(mapping_dicts['domain_to_ip_dict'][host])
                    malicious_hostnames.add(host)
                else:
                    # print('host {} in session {} could not be resolved to an ip'.format(host, pid))
                    malicious_hostnames.add(host)
            if malicious_ips == set():
                continue
            else:
                malicious_ips = list(malicious_ips)

                query = {
                    "script": {
                        "source": "if (ctx._source.containsKey(\"malicious_session\")) { ctx._source.malicious_session.add(params.pid); } else { ctx._source.malicious_session = [params.pid]; }",
                        "params": {"pid": pid},
                        "lang": "painless"
                    },
                    "query": {
                        "bool": {
                            "must": [
                                {
                                    "range": {
                                        "timestamp": {
                                            "gte": start_time,
                                            "lte": end_time,
                                            "format": "yyyy-MM-dd HH:mm:ss"
                                            # "time_zone": "+01:00"
                                    }
                                }
                                },
                                {
                                    "bool": {
                                        "should": [
                                            {"terms": {src_ip_field: malicious_ips}},
                                            {"terms": {dst_ip_field: malicious_ips}}
                                        ]
                                    }
                                },
                                {
                                    "bool": {
                                        "should": [
                                            {"terms": {src_ip_field: src_ip_aliases}},
                                            {"terms": {dst_ip_field: src_ip_aliases}}
                                        ]
                                    }
                                }
                            ]
                        }
                    }
                }

                res = es.update_by_query(body=query, doc_type=doc_type, index=index_name, slices=1, request_timeout=11000, conflicts='proceed')
                print(res)
                if res['updated']==0:
                    no_matches.append(pid)
                updated += res['updated']
    print('No matches found for pids {}'.format(no_matches))
    print('{} documents updated in total'.format(updated))


def generate_malicious_sessions_stats(es, index_name, local_hosts_path, malicious_sessions_path, mapping_dict_path, src_ip_field, dst_ip_field):
    """
    Function to generate some statistics about the session-labelling:
    - Which IPs conributed how much to the labels?
    - Which domainnames in opnotes.pdf that are resolved?

    :param es: Elasticsearch instance
    :param index_name: Name of the index
    :param malicious_sessions_path: Path to the malicious_sessions.json file as generated by cs_report_parser.py
    :param local_hosts_path: Path to the local_hosts.csv file
    :param mapping_dicts_path: Path to the mapping_dicts.pickle file as generated in bro_main.py
    :param src_ip_field: name of the src-IP field
    :param dst_ip_field: name of the dst-IP field
    """
    year = '2017'
    with open(mapping_dict_path, 'rb') as fp:
        mapping_dicts = pickle.load(fp)

    local_ips_list = bro_misc.generate_local_ip_aliases_list(local_hosts_path)
    with open(malicious_sessions_path, 'r') as fp:
        malicious_sessions = json.load(fp)

    all_query = {
        "query": {
            "match_all": {}
        }
    }

    res = es.search(index=index_name, body=all_query)
    nr_connections = res['hits']['total']

    mal_host_to_ip_dict = {}
    hits = []
    for pid, sessions in malicious_sessions.items():
        for session in sessions:
            # check if start_time is same as end_time --> this means that there is an initial beacon listed in activity.pdf
            # but no activities are listed  the corresponding session in the activities section of opnotes.pdf
            if session['start_time']==session['end_time']:
                continue

            # set datetime format to match supported format of elasticsearch yyyy-MM-dd HH:mm:ss
            start_time = '{}-{}-{} {}:00'.format(year, session['start_date'].split('/')[1], session['start_date'].split('/')[0], session['start_time'])
            end_time = '{}-{}-{} {}:00'.format(year, session['end_date'].split('/')[1], session['end_date'].split('/')[0], session['end_time'])
            end_time_same_day = '{}-{}-{} {}:00'.format(year, session['start_date'].split('/')[1], session['start_date'].split('/')[0], session['end_time_same_day'])

            src_ip_aliases = bro_misc.get_local_ip_aliases(session['src_ip'], local_ips_list)
            # print('{} - {}'.format(session['src_ip'], src_ip_aliases))

            malicious_ips = set()
            malicious_hostnames = set()
            for host in session['hosts']:
                if bro_misc.check_if_valid_IP(host):
                    malicious_ips.add(host)
                elif host in mapping_dicts['domain_to_ip_dict']:
                    malicious_ips = malicious_ips.union(mapping_dicts['domain_to_ip_dict'][host])
                    malicious_hostnames.add(host)
                    mal_host_to_ip_dict[host] = mapping_dicts['domain_to_ip_dict'][host]
                else:
                    # print('host {} in session {} could not be resolved to an ip'.format(host, pid))
                    pass
            if malicious_ips == set():
                continue
            else:
                malicious_ips = list(malicious_ips)

                query = {
                    "query": {
                        "bool": {
                            "must": [
                                {
                                    "range": {
                                        "timestamp": {
                                            "gte": start_time,
                                            "lte": end_time,
                                            "format": "yyyy-MM-dd HH:mm:ss"
                                            # "time_zone": "+01:00"
                                    }
                                }
                                },
                                {
                                    "bool": {
                                        "should": [
                                            {"terms": {src_ip_field: malicious_ips}},
                                            {"terms": {dst_ip_field: malicious_ips}}
                                        ]
                                    }
                                },
                                {
                                    "bool": {
                                        "should": [
                                            {"terms": {src_ip_field: src_ip_aliases}},
                                            {"terms": {dst_ip_field: src_ip_aliases}}
                                        ]
                                    }
                                }
                            ]
                        }
                    }
                }

                res = es.search(index=index_name, body=query)
                if res['hits']['total'] > 0:
                    hits.append((malicious_ips, res['hits']['total'], malicious_hostnames))
                    # print('{} - {} hits'.format(mal_ip, res['hits']['total']))

    hits = sorted(hits, key=lambda x: x[1], reverse=True)

    for ips, nr_hits, hosts in hits:
        if hosts == set():
            print('{}% - {} - {}'.format(nr_hits * 100 / nr_connections, nr_hits, ips))
        else:
            print('{}% - {} - {} - {}'.format(nr_hits * 100 / nr_connections, nr_hits, ips, hosts))

    # check what on a session basis, to which IPs of the listed domainnames in opnotes.pdf are resolved
    print("\nIPs of the listed domainnames in opnotes.pdf that are resolved:")
    print(mal_host_to_ip_dict)


def generate_malicious_ip_stats(es, index_name, malicious_ips_path, mal_host_names_path, mapping_dicts_path):
    """
    Function to generate some statistics about the host-labelling:

    :param es: Elasticsearch instance
    :param index_name: Name of the index
    :param malicious_ips_path: Path to the file holding the malicious IPs
    :param mal_host_names_path: Path to the file holding the malicious domain-names
    :param mapping_dicts_path: Path to the mapping_dicts.pickle file generated in bro_main.py
    """
    with open(malicious_ips_path, 'r') as fp:
        malicious_ips = json.load(fp)['malicious_ips']

    with open(mapping_dicts_path, 'rb') as fp:
        mapping_dicts = pickle.load(fp)


    # In the created list malicious_ips, hostnames listed in indicatorsofcompromise.pdf were resolved to ip addresses
    # using http host fields, dns A records and tls hello..
    # so theoretically it is possible that malicious_ips lists a legit ip-address of an external server (e.g. attacker, performed
    # dns cache poisoning for youtube.ex (a domain listed in the .pdf)... so after this point in time connections
    # to youtube.ex will be directed to the attackers server. However, before this attack google.com was mapped to the real legit IP.
    # so here we generate a dictionary to see which hostnames got assigned to which IP addresses during the mapping process
    # using that we can check if legit hostnames are really assigned to a malicious ip or/and to the real one
    # ..when later performing the labelling, we should only label the malicious IPs!
    with open(mal_host_names_path, 'r') as fp:
        mal_ip_to_host_dict = {}
        for line in fp:
            hostname = line.strip()
            if hostname in mapping_dicts['domain_to_ip_dict']:
                for ip in mapping_dicts['domain_to_ip_dict'][hostname]:
                    if ip in mal_ip_to_host_dict:
                        mal_ip_to_host_dict[ip].append(hostname)
                    else:
                        mal_ip_to_host_dict[ip] = [hostname]
    # malicious_ips = list(mal_ip_to_host_dict.keys())

    all_query = {
        "query": {
            "match_all": {}
        }
    }

    res = es.search(index=index_name, body=all_query)
    nr_connections = res['hits']['total']

    hits = []
    for mal_ip in malicious_ips:

        query = {
            "query" : {
                "bool": {
                    "should": [
                        {"match_phrase": {'src_ip': mal_ip}},
                        {"match_phrase": {'dst_ip': mal_ip}}
                    ]
                }
            }
        }

        res = es.search(index=index_name, body=query)
        if res['hits']['total'] > 0:
            hits.append((mal_ip, res['hits']['total']))
            # print('{} - {} hits'.format(mal_ip, res['hits']['total']))

    hits = sorted(hits, key=lambda x: x[1], reverse=True)

    tot_hits = 0
    for ip,nr_hits in hits:
        if ip in mal_ip_to_host_dict:
            print('{}% - {} - {} - IP resolved from {}'.format(nr_hits*100/nr_connections, nr_hits, ip, mal_ip_to_host_dict[ip]))
        else:
            print('{}% - {} - {}'.format(nr_hits * 100 / nr_connections, nr_hits, ip))
        tot_hits += nr_hits

    print('{} ({}%) connections out of total {} correspond to malicious IPs'.format(tot_hits, tot_hits*100/nr_connections, nr_connections))


if __name__ == "__main__":
    es = Elasticsearch(timeout=100)
    # es = Elasticsearch(timeout=100, http_auth=('elastic', 'r6IlsZYJa1KXjY0UetNR'))

    bro_path = '/mnt/data/nicolas/bro_logs/ls17'
    snort_alerts_path = '/home/nicolas/workspace/Network_Analyzer/data/alerts_full.csv'

    data_path = './data'
    extracted_path = './extracted/ls18'
    cs_path = './cobalt_strike/'

    pcap_path = '/mnt/data/LockedShields/BT07_CH_traffic/500tsd_all-traffic24-ordered'
    json_path = os.path.join(pcap_path, 'ndjson')
    duplicate_path = os.path.join(pcap_path, 'duplicates')

    local_hosts_path = os.path.join(data_path, 'local_hosts6.csv')
    malicious_ips_path = os.path.join(extracted_path, 'malicious_ips.json')
    malicious_sessions_path = os.path.join(extracted_path, 'malicious_sessions.json')
    mapping_dicts_path = os.path.join(extracted_path, 'mapping_dicts.pickle')


    ### CREATE PCAP, BRO AND SNORT INDEXES ###
    # index_single_file(es, os.path.join(json_path, json_file), 0)
    index_directory(es, 'packets-ls17', pcap_path, json_path, sort_paths=True)
    # delete_indices(es, 'packets-ls17')

    # index_bro(es, bro_path, index_pattern='', index_postfix='-ls17')
    index_bro(es, bro_path, index_pattern='conn', index_postfix='-ls17')
    # delete_indices(es, 'bro')

    index_snort(es, 'snort-ls17', snort_alerts_path)
    # delete_indices(es, 'snort')


    ### LABEL MALICIOUS FLOWS (COBALT STRIKE)
    # | "session-labelling"
    label_malicious_sessions(es, 'bro-conn-ls17', 'bro', malicious_sessions_path, local_hosts_path, mapping_dicts_path, 'src_ip', 'dst_ip')
    label_malicious_sessions(es, 'snort-ls17', 'snort', malicious_sessions_path, local_hosts_path, mapping_dicts_path, 'src_ip', 'dst_ip')

    # | "host-labelling"
    label_malicious_ips(es, 'snort-ls17', 'snort', malicious_ips_path, 'src_ip', 'dst_ip')
    label_malicious_ips(es, 'bro-conn-ls17', 'bro', malicious_ips_path, 'src_ip', 'dst_ip')

    ### FUNCTIONS TO GENERATE STATISTICS ###
    generate_malicious_ip_stats(es, 'bro-conn-ls17', malicious_ips_path, os.path.join(cs_path, 'indicatorsofcompromise_domains.txt'))
    generate_malicious_sessions_stats(es, 'bro-conn-ls17', local_hosts_path, malicious_sessions_path, mapping_dicts_path, 'src_ip', 'dst_ip')
