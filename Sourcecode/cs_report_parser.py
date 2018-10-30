"""
Master Thesis
Network Monitoring and Attack Detection

cs_report_parser.py
Functions used for parsing the Cobalt Strike Operation Notes report to extract the information needed for the "session-labelling"


@author: Nicolas Kaenzig, D-ITET, ETH Zurich
"""

import os
import collections
from random import randint
import fnmatch
import re
import json

extracted_path = './extracted/'

date_regex = re.compile('\d+/\d+')
init_regex = re.compile('initial')


def is_port(str):
    """
    Check if str contains a valid port number
    Returns True if str contains a valid port number
    """
    try:
        if int(str) >=0 and int(str) <=65535:
            return True
        else:
            return False
    except ValueError:
        return False


def read_opnotes_file(log_path, opnotes_file):
    """
    Function to parse Cobalt Strike's Operation Notes report.
    Returns a dictionary malicious_sessions that contains the start- and endtimes and the source and destination IPs
    for all sessions listed in the report

    :param log_path: Path where the CS reports are located
    :param opnotes_file: Filename of the Operation notes report
    :return: See description above
    """
    malicious_sessions = {}
    read_time = False
    pid_index = 0 # if there are pids used for different sessions, use this to index the sessions (happens only for pid=7604)
    pid_index_dict = {}
    with open(os.path.join(log_path, opnotes_file), "r") as fp:
        lines = fp.readlines()
        current_client_ip = 0
        for i, line in enumerate(lines):
            # skip comments and empty lines
            if line[0] == '#' or line[0] == '\n' or line == '\f\n':
                continue
            line_split = line.split()
            if line_split[0] == 'Zone:':
                current_client_ip = lines[i-1].split()[0]
                read_time = False
            elif line_split[0] == 'PID:':
                pid = line_split[1]
                if pid not in malicious_sessions:
                    malicious_sessions[pid] = [{'src_ip': current_client_ip}]
                else:
                    # in case the same PID is used for different sessions - happens only one for the LS17 dataset (pid=7604)
                    malicious_sessions[pid].append({'src_ip': current_client_ip})
                    pid_index += 1
            elif line_split[0] == 'hosts':
                if ',' not in lines[i+1]:
                    extract_hosts(lines, i, malicious_sessions[pid][pid_index], 'IPs')
                else:
                    extract_hosts(lines, i, malicious_sessions[pid][pid_index], 'domains')
            elif line_split[0] == 'Activity':
                read_time = True
            elif date_regex.match(line_split[0]) != None and read_time:
                read_start_and_end_times(lines, i, malicious_sessions[pid][pid_index])
                read_time = False
                if pid_index > 0:
                    pid_index_dict[pid] = pid_index
                pid_index = 0

    return malicious_sessions, pid_index_dict


def read_activities_file(log_path, activities_file, malicious_sessions):
    """
    Function to parse Cobalt Strike's Activity report.
    Sometimes in the activities .pdf the initial beacon is listed earlier than the first activity in the opnotes .pdf (usually ~1-2min)
    in these cases overwrite the start times.

    :param log_path: Path where the CS reports are located
    :param activities_file: Filename of the Activity report
    :param malicious_sessions: Generated by read_opnotes_file()
    :return: malicious_sessions dict with adapted start-times for the sessions
    """
    with open(os.path.join(log_path, activities_file), "r") as fp:
        lines = fp.readlines()
        pids = set()
        pid_index = 0
        index_7604 = 0 # TODO: currently this is hard coded for pid=7604, as it's the only pid which is assigned to multiple (2) sessions
        for i, line in enumerate(lines):
            if 'initial beacon' in line:
                line_split = line.split()
                pid = line_split[-3]
                if pid == '7604':
                    malicious_sessions[pid][index_7604]['start_time'] = line_split[1]
                    index_7604 += 1
                    continue
                elif pid in pids:
                    continue
                else:
                    # for sessions that have no activities listed in opnotes
                    if 'end_time' not in malicious_sessions[pid][pid_index]:
                        malicious_sessions[pid][pid_index]['start_time'] = line_split[1]
                        malicious_sessions[pid][pid_index]['end_time'] = line_split[1]
                        malicious_sessions[pid][pid_index]['start_date'] = line_split[0]
                        malicious_sessions[pid][pid_index]['end_date'] = line_split[0]
                    else:
                        if malicious_sessions[pid][pid_index]['start_date'] == line_split[0]:
                            # print('pid {} - {} --> {}'.format(pid, malicious_sessions[pid][pid_index]['start_time'], min(line_split[1], malicious_sessions[pid][pid_index]['start_time'])))
                            # set start_time to the minimum of the two times (initial beacon, first activity in opnotes)
                            malicious_sessions[pid][pid_index]['start_time'] = min(line_split[1], malicious_sessions[pid][pid_index]['start_time'])
                pids.add(pid)
    return malicious_sessions


def extract_hosts(lines, host_index, dict, mode):
    """
    Helper function used to extract host-names and IP addresses in the Opnotes Report

    :param lines: Lines to process
    :param host_index: Start-index for parsing
    :param dict: malicious_sessions dictionary
    :param mode: In the communication path section in the report are either listed IPs or domains --> set the mode with
                 this parameter ('IPs' or 'domains')
    """
    dict['hosts'] = []
    dict['ports'] = []
    dict['services'] = []

    index = host_index + 1

    if mode=='IPs':
        line = lines[index]
        while line != '\n':
            if ',' not in line:
                dict['hosts'].append(line.split()[0])
                dict['ports'].append(line.split()[1])
                dict['services'].append(line.split()[2])
                index += 1
                line = lines[index]
            else:
                for h in line.split(', '):
                    if len(h.split()) < 2:
                        dict['hosts'].append(h)
                    else:
                        tmp = h.split()
                        dict['hosts'].append(tmp[0])
                        dict['ports'].append(tmp[1])
                        dict['services'].append(tmp[2])
                index += 1
                line = lines[index]

    elif mode=='domains':
        # only one line of hostnames --> hosts and port number are separated by a space character
        if lines[index+1] == '\n':
            tmp = lines[index].split(' ')
            k = 0
            for h in tmp:
                dict['hosts'].append(h.strip(','))
                k = k+1
                if h[-1] != ',':
                    break
            for h in tmp[k:]:
                dict['ports'].append(h.strip(','))
                k = k+1
                if h[-1] != ',':
                    break
            for h in tmp[k:]:
                dict['services'].append(h)
                k = k+1
                if h[-1] != ',':
                    break
        # multiple lines of hostnames --> hosts and port number are separated by a ',' character with NO SPACE after
        else:
            # process first line
            tmp = lines[index].split(',')
            dict['hosts'].append(tmp[0])

            k = 1
            for h in tmp[k:]:
                # after ',' split, hostnames start with a space char, the first entry not starting with ' ' must thus be a port
                if h[0] != ' ':
                    break
                dict['hosts'].append(h.strip())
                k = k+1
            if len(tmp[k:]) > 1:
                tmp2 = tmp[k:]
                for nr, h in enumerate(tmp[k:]):
                    if len(h.split()) > 1:
                        dict['ports'].append(h.split()[0])
                        dict['services'].append(h.split()[1].strip(','))
                        dict['services'] += [x.strip() for x in tmp2[nr+1:]]
                        break
                    else:
                        dict['ports'].append(h)
            else:
                dict['ports'].append(tmp[k:][0].split()[0])
                dict['services'].append(tmp[k:][0].split()[1].strip(','))

            # process following lines (they contain only hostnames separated with ','
            index += 1
            line = lines[index]
            while line != '\n':
                tmp = line.split(',')
                for h in tmp:
                    dict['hosts'].append(h.strip())

                index += 1
                line = lines[index]


def read_start_and_end_times(lines, index, dict):
    """
    Helper function used to extract start- and endtimes of the sessions listed in the Opnotes Report

    :param lines: Lines to process
    :param index: Start-index for parsing
    :param dict: malicious_sessions dictionary
    """
    dict['start_date'] = lines[index].split()[0]
    dict['end_date'] = dict['start_date']
    dict['start_time'] = lines[index].split()[1]
    dict['end_time'] =  dict['start_time']
    dict['end_time_same_day'] = None
    start_date = dict['start_date']
    last_time = None
    last_date = None

    for i, line in enumerate(lines[index:]):
        if line[0] == '#' or line[0] == '\n' or line == '\f\n':
            continue
        line_split = line.split()
        try:
            # check if first word is a date xx/xx
            if date_regex.match(line_split[0]) != None:
                # if the date is next day, this means there is a big gap with no activity... -> save the last time of same day too
                if start_date != line_split[0] and dict['end_time_same_day']==None:
                    dict['end_time_same_day'] = last_time
                    # break
                last_time = line_split[1]
                last_date = line_split[0]
            elif line_split[0] == 'Zone:' or i==len(lines[index:])-1: # zone section or eof
                dict['end_time'] = last_time
                dict['end_date'] = last_date
                break
        except IndexError:
            continue
    if dict['end_time_same_day']==None:
        dict['end_time_same_day'] = dict['end_time']


def get_infection_times(malicious_sessions):
    """
    This function generates a mapping: IP->start-time, meaning it lists for each IP address the time when the first
    cobalt strike session was opened --> This is the time when the machine got compromised

    :param malicious_sessions: Generated by read_opnotes_file()
    :return: IP->time mapping (dictionary)
    """
    infected_times_dict = {}
    for pid, session_dict in malicious_sessions.items():
        if session_dict[0]['src_ip'] not in infected_times_dict:
            infected_times_dict[session_dict[0]['src_ip']] = (session_dict[0]['start_time'], session_dict[0]['start_date'])
        else:
            # if there are multiple sessions for a IP, take the earliest point in time
            if check_if_time_smaller(session_dict[0]['start_date'], session_dict[0]['start_time'], infected_times_dict[session_dict[0]['src_ip']][1], infected_times_dict[session_dict[0]['src_ip']][0]):
                infected_times_dict[session_dict[0]['src_ip']] = (session_dict[0]['start_time'], session_dict[0]['start_date'])
    return infected_times_dict


def get_malicious_hosts(malicious_sessions):
    """
    Get a list of all malicious hosts listed in the dict malicious_sessions

    :param malicious_sessions: Generated by read_opnotes_file()
    """
    malicious_hosts = set()
    for pid, session_dict in malicious_sessions.items():
        malicious_hosts = malicious_hosts.union(set(session_dict[0]['hosts']))


def check_if_time_smaller(date1, time1, date2, time2):
    """
    Helper function for time comparison...
    Returns True if (time1, date1) is earlier in time than (time2, date2)
    """
    if date1.split('/')[0] < date2.split('/')[0]:
        return True
    else:
        t1 = time1.split(':')[0] * 60 + time1.split(':')[1]
        t2 = time2.split(':')[0] * 60 + time2.split(':')[1]
        if t1 < t2:
            return True
    return False


if __name__ == "__main__":
    log_path = './cobalt_strike/'
    opnotes_file = 'team7CH_opnotes_proc.txt'
    activities_file = 'team7CH_activity.txt'

    malicious_sessions, pid_index_dict = read_opnotes_file(log_path, opnotes_file)
    # sometimes in the activities .pdf the initial beacon is listed earlier than the first activity in the opnotes .pdf (usually ~1-2min)
    # in these cases overwrite the start times
    malicious_sessions = read_activities_file(log_path, activities_file, malicious_sessions)

    infection_times = get_infection_times(malicious_sessions)
    mal_hosts = get_malicious_hosts(malicious_sessions)

    with open(os.path.join(extracted_path, 'malicious_sessions.json'), 'w') as fp1, open(os.path.join(extracted_path, 'infection_times.json'), 'w') as fp2:
        print('Saving malicious sessions dictionary and infection times dictionary to {} ...'.format(extracted_path))
        json.dump(malicious_sessions, fp1)
        json.dump(infection_times, fp2)
