"""
Master Thesis
Network Monitoring and Attack Detection

pcap_functions.py
This module contains different functions to process pcap files:
- PCAP subsampling (we used this to simulate packet losses during capturing)
- Remove duplicate packets from PCAP files
- Count the number of packets in a PCAP file


@author: Nicolas Kaenzig, D-ITET, ETH Zurich
"""

import os
from tqdm import tqdm
import dpkt
from ipaddress import ip_address
import hashlib
import sys
import struct
import csv
import time
import numpy as np
from joblib import Parallel, delayed


def pcap_subsampling(pcap_path, out_directory, subsampling_fraction):
    """
    PCAP subsampling (we used this to simulate packet losses during capturing)

    :param pcap_path:
    :param out_directory:
    :param subsampling_fraction:
    """
    out_path = os.path.join(out_directory, os.path.splitext(os.path.basename(pcap_path))[0] + '-sub{}.pcap'.format(subsampling_fraction))

    tot_packets = 0
    skipped = 0

    with open(pcap_path, 'rb') as fp_r, open(out_path, 'wb') as fp_w:
        reader = dpkt.pcap.Reader(fp_r)
        writer = dpkt.pcap.Writer(fp_w)

        print('Writing to {} ...'.format(out_path))
        for ts, buf in tqdm(reader):
            save = np.random.choice([0, 1], size=(1,), p=[1-subsampling_fraction, subsampling_fraction])[0]
            if save:
                writer.writepkt(buf, ts)
                skipped += 1

            tot_packets += 1

        print('{} ({}%) of {} packets skipped'.format(skipped, skipped*100/nr_packets, tot_packets))


def count_number_of_packets(pcap_path):
    """
    Simple function to count the number of packets in a PCAP file

    :param pcap_path: Path of the PCAP file
    """
    print('Counting nr. of packets in {}'.format(pcap_path))
    tot_packets = 0

    with open(pcap_path, 'rb') as fp_r:
        reader = dpkt.pcap.Reader(fp_r)

        for ts, buf in tqdm(reader):
            tot_packets += 1

    print('Tot. number of packets: {}'.format(tot_packets))
    

def remove_duplicates_from_pcap(pcap_path, out_path, window_size=20):
    """
    Function to remove duplicates with invalid checksum from a PCAP-file
    Parses the PCAP using a sliding window of size 20 (default), where the last 20 packets at each point are stored
    and checked for duplicates. The duplicate checking is performed by first setting the checksum fields to 0, and then
    calucate and compare MD5 hashes of the packet buffers.

    :param pcap_path: Path of the PCAP file
    :param out_path: Output path of the new PCAP without duplicates
    """
    hashes = [-1]*window_size
    bufs = [-1]*window_size
    checksums = [-1]*window_size
    to_write = [False]*window_size
    idx = 0
    nr_duplicates = 0
    # idea use i%5 to index a numpy array of size 5 --> sliding window
    with open(pcap_path, 'rb') as fp1, open(os.path.join(out_path,'modified.pcap'), 'wb') as fp2:
        reader = dpkt.pcap.Reader(fp1)
        writer = dpkt.pcap.Writer(fp2)
        for ts, buf in tqdm(reader):
            eth = dpkt.ethernet.Ethernet(buf)
            ip = eth.data
            proto = ip.data

            src = ip.src

            # check if not TCP or UDP
            if not (ip.p == 6 or ip.p == 17):
                writer.writepkt(buf, ts)
                continue
            else:
                # set_tcp_checksum_to_zero(bytes(ip.data))
                orig_checksum = proto.sum   # checksum read from pcap
                calc_checksum, tcp_buf = calculate_and_set_checksum(ip)

                window_index = idx%window_size
                next_window_index = (window_index + 1)%window_size

                # as we move window 'forward', we need to write out the oldest buffer that is going to be
                # replaced by a new one in the next iteration
                if to_write[window_index]:
                    writer.writepkt(bufs[window_index][0], bufs[window_index][1])
                bufs[window_index] = (buf,ts)
                to_write[window_index] = True
                checksums[window_index] = orig_checksum

                m = hashlib.md5()
                m.update(tcp_buf)
                hashes[window_index] = m.hexdigest()

                for k, hash in enumerate(hashes):
                    if k == window_index or hash==-1:
                        continue

                    if hashes[window_index] == hash:
                        # current packet has invalid checksum
                        if orig_checksum != calc_checksum:
                            to_write[window_index] = False
                            nr_duplicates += 1
                        # the other duplicate packet in the window has invalid checksum
                        elif checksums[k] != calc_checksum:
                            to_write[k] = False
                            nr_duplicates += 1

                        # print ('frame nr {} is a duplicate'.format(idx))

                idx += 1

    print('{} duplicates skipped'.format(nr_duplicates))


def set_tcp_checksum_to_zero(tcp_buf):
    """
    This function sets the TCP checksum field to zero

    :param tcp_buf: Raw packet buffer, as loaded by the dpkt library
    """
    ### TCP HEADER FORMAT ###
    #  0                   1                   2                   3
    #  0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
    # +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    # |          Source Port          |       Destination Port        |
    # +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    # |                        Sequence Number                        |
    # +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    # |                    Acknowledgment Number                      |
    # +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    # |  Data |           |U|A|P|R|S|F|                               |
    # | Offset| Reserved  |R|C|S|S|Y|I|            Window             |
    # |       |           |G|K|H|T|N|N|                               |
    # +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    # |           Checksum            |         Urgent Pointer        |
    # +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    # |                    Options                    |    Padding    |
    # +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    # |                             data                              |
    # +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+


    header_types = '>HHIIBBHHH'
    tcp_header_len = 20

    # set cheksum fields to 0
    tcp_buf = bytearray(tcp_buf)
    # tcp checksum: header bytes 17 & 18 --> index with 16 & 17
    tcp_buf[16] = 0
    tcp_buf[17] = 0
    tcp_buf = bytes(tcp_buf)

    # dpkt.ethernet.Ethernet(buf) sets the checksum somewhere to zero --> tcp_buf holds zero checksum
    # but: later the checksum somewhere gets loaded...
    # but: in ipv6 case --> the checksum is already loaded here
    struct.unpack(header_types, tcp_buf[:tcp_header_len])
    k = 0
    for byte in tcp_buf:
        i = hex(byte)
        k+=1


def calculate_and_set_checksum(ip):
    """
    This function calculates tcp, udp and ip checksums and sets the corresponding packet field to this value

    :param ip: ip objects, loaded using dpkt library
    :return: checksum, packet buffer with set checksum
    """
    if ip.v==4:
        IP_MF = 0x2000
        IP_OFFMASK = 0x1fff
        # set checksum field to zero
        ip.data.sum = 0

        if (ip.p == 6 or ip.p == 17) and (ip.off & (IP_MF | IP_OFFMASK)) == 0 and \
                isinstance(ip.data, dpkt.Packet) and ip.data.sum == 0:
            # Set zeroed TCP and UDP checksums for non-fragments.
            p = bytes(ip.data)
            s = dpkt.struct.pack('>4s4sxBH', ip.src, ip.dst,
                                 ip.p, len(p))
            s = dpkt.in_cksum_add(0, s)
            s = dpkt.in_cksum_add(s, p)
            # ip.data.sum = dpkt.in_cksum_done(s)
            sum = dpkt.in_cksum_done(s)
            if ip.p == 17 and sum == 0:
                sum = 0xffff  # RFC 768
                # XXX - skdada transports which don't need the pseudoheader
            return sum, p
    elif ip.v==6:
        # XXX - set TCP, UDP, and ICMPv6 checksums
        # if (ip.p == 6 or ip.p == 17 or ip.p == 58) and ip.data.sum == None:
        if (ip.p == 6 or ip.p == 17): # 6: TCP, 7: UDP
            # ip.data.sum = None
            # set checksum field to zero
            ip.data = bytearray(bytes(ip.data))
            if ip.p == 6:
                ip.data[16] = 0
                ip.data[17] = 0
            elif ip.p == 17:
                ip.data[6] = 0
                ip.data[7] = 0
            p = bytes(ip.data)
            s = dpkt.struct.pack('>16s16sxBH', ip.src, ip.dst, ip.nxt, len(p))
            s = dpkt.in_cksum_add(0, s)
            s = dpkt.in_cksum_add(s, p)
            try:
                return dpkt.in_cksum_done(s), p
            except AttributeError:
                return None
    else:
        print('no IP?')
        return None


if __name__ == "__main__":

    data_path18 = '/mnt/data/LockedShields18/LS18/all'
    out_path18 = os.path.join(data_path18, 'subsampled')
    pcap_file18 = 'ls18-all-traffic24.pcap'
    pcap_file18 = 'ls18-all-traffic24-snap96.pcap'

    ### SIMULATING PACKET LOSS ###
    # | Caculate 9 subsampled pcaps in parallel
    sub_fractions = [round(x * 0.1, 1) for x in range(1, 11)]
    Parallel(n_jobs=9)(
        delayed(pcap_subsampling)(os.path.join(data_path18, pcap_file18), out_path18, sub_frac) for sub_frac in sub_fractions)

    ### REMOVING PCAP DUPLICATES ###
    remove_duplicates_from_pcap(os.path.join(data_path18, pcap_file18), data_path18)
