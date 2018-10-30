"""
Master Thesis
Network Monitoring and Attack Detection

ml_data.py
This file contains data such as list of feature names or mappings that we used for the machine learning part of the
thesis.

@author: Nicolas Kaenzig, D-ITET, ETH Zurich
"""


################################################
### DATA USED FOR LOCKED SHIELDS EXPERIMENTS ###
################################################

### KDD FEATURES ###

# KDD featurenames
kdd_all_features = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment',
                    'urgent', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
                    'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
                    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
                    'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate']

# Names of categorical features in KDD
kdd_categorical_features = ['protocol_type', 'service', 'flag']

# KDD mapping from nominal to numerical values for KDD's categorical features
kdd_categorical_feature_mapping = {'protocol_type' : {'tcp': 0, 'udp': 1, 'icmp' : 2},
                                   'service': {'other': 0, 'private': 1, 'ecr_i': 2, 'urp_i': 3, 'urh_i': 4, 'red_i': 5, 'eco_i': 6,
                                                'tim_i': 7, 'oth_i': 8, 'domain_u': 9, 'tftp_u': 10, 'ntp_u': 11, 'IRC': 12, 'X11': 13,
                                                'Z39_50': 14, 'aol': 15, 'auth': 16, 'bgp': 17, 'courier': 18, 'csnet_ns': 19, 'ctf': 20,
                                                'daytime': 21, 'discard': 22, 'domain': 23, 'echo': 24, 'efs': 25, 'exec': 26, 'finger': 27,
                                                'ftp': 28, 'ftp_data': 29, 'gopher': 30, 'harvest': 31, 'hostnames': 32, 'http': 33, 'http_2784': 34,
                                                'http_443': 35, 'http_8001': 36, 'icmp': 37, 'imap4': 38, 'iso_tsap': 39, 'klogin': 40, 'kshell': 41,
                                                'ldap': 42, 'link': 43, 'login': 44, 'mtp': 45, 'name': 46, 'netbios_dgm': 47, 'netbios_ns': 48,
                                                'netbios_ssn': 49, 'netstat': 50, 'nnsp': 51, 'nntp': 52, 'pm_dump': 53, 'pop_2': 54, 'pop_3': 55,
                                                'printer': 56, 'remote_job': 57, 'rje': 58, 'shell': 59, 'smtp': 60, 'sql_net': 61, 'ssh': 62,
                                                'sunrpc': 63, 'supdup': 64, 'systat': 65, 'telnet': 66, 'time': 67, 'uucp': 68, 'uucp_path': 69,
                                                'vmnet': 70, 'whois': 71},
                                   'flag': {'SF': 0, 'S0': 1, 'S1': 2, 'S2': 3, 'S3': 4, 'REJ': 5, 'RSTOS0': 6, 'RSTO': 7,
                                            'RSTR': 8, 'SH': 9, 'RSTRH': 10, 'SHR': 11, 'OTH': 12}
                                   }


# KDD set without the features with zero correlation to the labels
kdd_without_zerocor_features = ['src_bytes', 'dst_bytes', 'protocol_type', 'service', 'flag',
                     'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
                    'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
                    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
                    'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_srv_rerror_rate']


### FLOWMETER FEATURES ###

# FlowMeter CSV headernames/featurenames
flowmeter_csv_header_names = ['Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts', 'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Fwd Pkt Len Max',
                'Fwd Pkt Len Min', 'Fwd Pkt Len Mean', 'Fwd Pkt Len Std', 'Bwd Pkt Len Max', 'Bwd Pkt Len Min', 'Bwd Pkt Len Mean',
                'Bwd Pkt Len Std', 'Flow Byts/s', 'Flow Pkts/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
                'Fwd IAT Tot', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Tot', 'Bwd IAT Mean',
                'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags',
                'Fwd Header Len', 'Bwd Header Len', 'Fwd Pkts/s', 'Bwd Pkts/s', 'Pkt Len Min', 'Pkt Len Max', 'Pkt Len Mean',
                'Pkt Len Std', 'Pkt Len Var', 'FIN Flag Cnt', 'SYN Flag Cnt', 'RST Flag Cnt', 'PSH Flag Cnt', 'ACK Flag Cnt',
                'URG Flag Cnt', 'CWR Flag Count', 'ECE Flag Cnt', 'Down/Up Ratio', 'Pkt Size Avg', 'Fwd Seg Size Avg',
                'Bwd Seg Size Avg', 'Fwd Byts/b Avg', 'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg', 'Bwd Byts/b Avg', 'Bwd Pkts/b Avg',
                'Bwd Blk Rate Avg', 'Subflow Fwd Pkts', 'Subflow Fwd Byts', 'Subflow Bwd Pkts', 'Subflow Bwd Byts', 'Init Fwd Win Byts',
                'Init Bwd Win Byts', 'Fwd Act Data Pkts', 'Fwd Seg Size Min', 'Active Mean', 'Active Std', 'Active Max',
                'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min', 'Label']


# features that are constant or always zero
constant_features = ['Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 'URG Flag Cnt', 'Fwd Byts/b Avg', 'Fwd Pkts/b Avg',
                 'Fwd Blk Rate Avg', 'Bwd Byts/b Avg', 'Bwd Pkts/b Avg', 'Bwd Blk Rate Avg']

# with zero correlation to the labels
zerocor_features = ['Fwd PSH Flags', 'CWR Flag Count', 'Active Std', 'Idle Std', 'Flow IAT Min', 'Fwd IAT Min', 'Bwd IAT Min']


ls_all_features_ordered_by_importance = ['Idle Max', 'ECE Flag Cnt', 'Pkt Len Std', 'Fwd Act Data Pkts', 'Bwd IAT Tot', 'Bwd Pkt Len Mean', 'Init Bwd Win Byts',
              'Idle Min', 'TotLen Fwd Pkts', 'Fwd Pkt Len Max', 'Fwd Seg Size Avg', 'RST Flag Cnt', 'Flow Byts/s', 'Subflow Bwd Byts',
              'Pkt Len Var', 'Bwd Pkts/s', 'Fwd Header Len', 'Active Min', 'Pkt Size Avg', 'Bwd Pkt Len Max', 'Active Max',
              'Bwd IAT Std', 'Bwd IAT Max', 'Fwd Pkt Len Min', 'Bwd Seg Size Avg', 'ACK Flag Cnt', 'Tot Fwd Pkts', 'Pkt Len Max',
                                         'Flow IAT Std', 'Fwd Pkt Len Mean', 'Down/Up Ratio', 'Fwd IAT Std', 'Flow Duration', 'TotLen Bwd Pkts', 'Fwd IAT Mean',
                                         'Idle Mean', 'Pkt Len Mean', 'Pkt Len Min', 'Flow IAT Max', 'Fwd IAT Tot', 'Subflow Bwd Pkts', 'Subflow Fwd Byts',
                                         'Bwd Header Len', 'Tot Bwd Pkts', 'Fwd Pkt Len Std', 'Fwd Seg Size Min', 'Bwd Pkt Len Std', 'Bwd IAT Mean', 'Active Mean',
                                         'SYN Flag Cnt', 'Init Fwd Win Byts', 'FIN Flag Cnt', 'Bwd Pkt Len Min', 'Flow Pkts/s', 'PSH Flag Cnt', 'Fwd IAT Max',
                                         'Flow IAT Mean', 'Subflow Fwd Pkts']


# Top 10 features
RF_top10 = ['Active Mean', 'SYN Flag Cnt', 'Init Fwd Win Byts', 'FIN Flag Cnt', 'Bwd Pkt Len Min', 'Flow Pkts/s', 'PSH Flag Cnt',
            'Fwd IAT Max', 'Flow IAT Mean', 'Subflow Fwd Pkts']

# top 10 features without SYN, PSH Count features, and with the new Protocol and Dst Int Ext features
RF_top10_noSYN_noPSH_withProtocol_intExt = ['Active Mean', 'Init Fwd Win Byts', 'FIN Flag Cnt', 'Bwd Pkt Len Min', 'Flow Pkts/s',
            'Fwd IAT Max', 'Flow IAT Mean', 'Subflow Fwd Pkts', 'Protocol', 'dstIntExt']

# top 20 features without SYN, PSH Count features, and with the new Protocol and Dst Int Ext features
RF_top20_noSYN_noPSH_withProtocol_intExt = ['Protocol', 'dstIntExt', 'Flow IAT Max', 'Fwd IAT Tot', 'Subflow Bwd Pkts', 'Subflow Fwd Byts',
              'Bwd Header Len', 'Tot Bwd Pkts', 'Fwd Pkt Len Std', 'Fwd Seg Size Min', 'Bwd Pkt Len Std', 'Bwd IAT Mean', 'Active Mean',
              'Init Fwd Win Byts', 'FIN Flag Cnt', 'Bwd Pkt Len Min', 'Flow Pkts/s', 'Fwd IAT Max', 'Flow IAT Mean', 'Subflow Fwd Pkts']

## SETS FOR EXPERIMENTS WHERE RF MODEL WAS TRAINED EXCLUDING EACH ONE OF THE TOP 10 FEATURES

RF_top9_nosubF = ['Active Mean', 'SYN Flag Cnt', 'Init Fwd Win Byts', 'FIN Flag Cnt', 'Bwd Pkt Len Min', 'Flow Pkts/s', 'PSH Flag Cnt',
            'Fwd IAT Max', 'Flow IAT Mean']

RF_top8_noIAT = ['Active Mean', 'SYN Flag Cnt', 'Init Fwd Win Byts', 'FIN Flag Cnt', 'Bwd Pkt Len Min', 'Flow Pkts/s', 'PSH Flag Cnt', 'Subflow Fwd Pkts']

RF_top9_noPSH = ['Active Mean', 'SYN Flag Cnt', 'Init Fwd Win Byts', 'FIN Flag Cnt', 'Bwd Pkt Len Min', 'Flow Pkts/s',
            'Fwd IAT Max', 'Flow IAT Mean', 'Subflow Fwd Pkts']

RF_top9_no_pkts = ['Active Mean', 'SYN Flag Cnt', 'Init Fwd Win Byts', 'FIN Flag Cnt', 'Bwd Pkt Len Min', 'PSH Flag Cnt',
            'Fwd IAT Max', 'Flow IAT Mean', 'Subflow Fwd Pkts']

RF_top9_no_pktlenmin = ['Active Mean', 'SYN Flag Cnt', 'Init Fwd Win Byts', 'FIN Flag Cnt', 'Flow Pkts/s', 'PSH Flag Cnt',
            'Fwd IAT Max', 'Flow IAT Mean', 'Subflow Fwd Pkts']

RF_top9_noFIN = ['Active Mean', 'SYN Flag Cnt', 'Init Fwd Win Byts', 'Bwd Pkt Len Min', 'Flow Pkts/s', 'PSH Flag Cnt',
            'Fwd IAT Max', 'Flow IAT Mean', 'Subflow Fwd Pkts']

RF_top9_noTCPwin = ['Active Mean', 'SYN Flag Cnt', 'FIN Flag Cnt', 'Bwd Pkt Len Min', 'Flow Pkts/s', 'PSH Flag Cnt',
            'Fwd IAT Max', 'Flow IAT Mean', 'Subflow Fwd Pkts']

RF_top9_noSYN = ['Active Mean', 'Init Fwd Win Byts', 'FIN Flag Cnt', 'Bwd Pkt Len Min', 'Flow Pkts/s', 'PSH Flag Cnt',
            'Fwd IAT Max', 'Flow IAT Mean', 'Subflow Fwd Pkts']

RF_top9_noactive = ['SYN Flag Cnt', 'Init Fwd Win Byts', 'FIN Flag Cnt', 'Bwd Pkt Len Min', 'Flow Pkts/s', 'PSH Flag Cnt',
            'Fwd IAT Max', 'Flow IAT Mean', 'Subflow Fwd Pkts']



## VALUES FOR SIMULATING THE ATTACKS

Flow_Pkts_attack_values = [16129.032258064517, 16000.0, 666666.6666666665, 1000000.0, 16260.162601626012, 500000.0, 16393.442622950817,
                           15873.015873015871, 400000.0, 16528.92561983471, 2000000.0, 333333.33333333326, 16666.666666666668,
                           15748.031496062993, 285714.2857142857, 16806.722689075632, 250000.0, 15625.0, 8064.5161290322585,
                           16949.15254237288, 222222.22222222228]

SYN_Flag_Cnt_attack_values = [0, 6, 4, 1]


Fwd_IAT_Max_attack_values = [2.0, 3.0, 4.0, 1.0, 123.0, 124.0, 122.0, 121.0, 5.0, 125.0, 120.0, 6.0, 126.0, 119.0, 7.0,
                             118.0, 127.0, 8.0, 117.0]

Flow_IAT_Mean_attack_values = [2.0, 3.0, 4.0, 1.0]

PSH_Flag_Cnt_attack_values = [0, 5, 6]

Subflow_Fwd_Pkts_attack_values = [1, 2, 4, 6]

Bwd_Pkt_Len_Min_attack_values = [28.0, 27.0, 89.0, 34.0, 32.0, 107.0, 38.0, 31.0]

Init_Fwd_Win_Byts_attack_values = [14600, 14480, 42780, 514]

Active_Mean_attack_values = [2.0, 3.0, 4.0, 1.0, 123.0, 124.0, 122.0]

all_attacks_dict = {'Flow Pkts/s': Flow_Pkts_attack_values, 'SYN Flag Cnt': SYN_Flag_Cnt_attack_values, 'Fwd IAT Max': Fwd_IAT_Max_attack_values,
                    'Flow IAT Mean': Flow_IAT_Mean_attack_values, 'PSH Flag Cnt': PSH_Flag_Cnt_attack_values, 'Subflow Fwd Pkts': Subflow_Fwd_Pkts_attack_values,
                    'Bwd Pkt Len Min': Bwd_Pkt_Len_Min_attack_values, 'Init Fwd Win Byts': Init_Fwd_Win_Byts_attack_values, 'Active Mean': Active_Mean_attack_values}


############################################
### DATA USED FOR CICIDS2017 EXPERIMENTS ###
############################################

# mappings from nominal labels to numbers
label_mapping_dict = {'Label': {'DDoS': 1, 'DoS slowloris': 5, 'SSH-Patator': 10, 'FTP-Patator': 6, 'Heartbleed': 7,
                                'BENIGN': 0, 'Infiltration': 8, 'DoS Hulk': 3, 'Web Attack - XSS': 13, 'Web Attack - Sql Injection': 12,
                                'DoS GoldenEye': 2, 'Web Attack - Brute Force': 11, 'DoS Slowhttptest': 4, 'PortScan': 9, 'Bot': 14}}

# mappings from nominal labels to numbers (binary encoding)
label_binary_mapping_dict = {'Label': {'DDoS': 1, 'DoS slowloris': 1, 'SSH-Patator': 1, 'FTP-Patator': 1, 'Heartbleed': 1,
                                'BENIGN': 0, 'Infiltration': 1, 'DoS Hulk': 1, 'Web Attack - XSS': 1,  'Web Attack - Sql Injection': 1,
                                'DoS GoldenEye': 1, 'Web Attack - Brute Force': 1, 'DoS Slowhttptest': 1, 'PortScan': 1, 'Bot': 1}}

# headernames of the CICIDS2017 CSV files
cic17_csv_header_names = ['Destination Port', 'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets', 'Total Length of Fwd Packets',
                 'Total Length of Bwd Packets', 'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean',
                 'Fwd Packet Length Std', 'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean', 'Bwd Packet Length Std',
                 'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Total',
                 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std',
                 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 'Fwd Header Length',
                 'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s', 'Min Packet Length', 'Max Packet Length', 'Packet Length Mean',
                 'Packet Length Std', 'Packet Length Variance', 'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count',
                 'ACK Flag Count', 'URG Flag Count', 'CWE Flag Count', 'ECE Flag Count', 'Down/Up Ratio', 'Average Packet Size',
                 'Avg Fwd Segment Size', 'Avg Bwd Segment Size', 'Fwd Header Length', 'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk',
                 'Fwd Avg Bulk Rate', 'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate', 'Subflow Fwd Packets',
                 'Subflow Fwd Bytes', 'Subflow Bwd Packets', 'Subflow Bwd Bytes', 'Init_Win_bytes_forward', 'Init_Win_bytes_backward',
                 'act_data_pkt_fwd', 'min_seg_size_forward', 'Active Mean', 'Active Std', 'Active Max', 'Active Min', 'Idle Mean',
                 'Idle Std', 'Idle Max', 'Idle Min', 'Label']

# Features with zero correlation to labels
cic17_zero_corr_features = ['ECE Flag Count', 'RST Flag Count', 'CWE Flag Count', 'Fwd URG Flags', 'Bwd Avg Bytes/Bulk',
                      'Bwd Avg Packets/Bulk', 'Bwd URG Flags', 'Fwd Avg Bytes/Bulk', 'Fwd Avg Bulk Rate', 'Bwd PSH Flags',
                      'Fwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate']

# List of features after removing the features that have zero correlation with the labels and without the destination port feature
cic17_without_zero_features_noDstPort = ['Fwd Packet Length Min', 'Active Max', 'Active Mean', 'Fwd Packet Length Max', 'FIN Flag Count',
                                   'Avg Fwd Segment Size', 'Active Min', 'Bwd Header Length', 'Active Std', 'Packet Length Variance',
                                   'Fwd Packet Length Mean', 'Idle Mean', 'Bwd Packet Length Std', 'PSH Flag Count', 'Bwd Packet Length Mean',
                                   'Packet Length Std', 'Bwd IAT Std', 'Subflow Fwd Packets', 'Fwd Packet Length Std', 'Subflow Bwd Packets',
                                         'Flow IAT Mean', 'Bwd IAT Max', 'Fwd Header Length', 'Bwd IAT Min', 'Max Packet Length', 'Fwd IAT Min',
                                         'Fwd IAT Std', 'Fwd IAT Mean', 'min_seg_size_forward', 'Total Length of Fwd Packets', 'Flow IAT Std',
                                         'Avg Bwd Segment Size', 'Flow IAT Max', 'Down/Up Ratio', 'Idle Max', 'Subflow Fwd Bytes',
                                         'Init_Win_bytes_forward', 'Average Packet Size', 'Idle Min', 'Packet Length Mean', 'Fwd Packets/s',
                                         'Fwd PSH Flags', 'SYN Flag Count', 'Flow Packets/s', 'ACK Flag Count', 'Total Fwd Packets',
                                         'Bwd IAT Mean', 'Min Packet Length', 'Bwd Packet Length Max', 'Idle Std', 'Bwd Packet Length Min',
                                         'Flow IAT Min', 'Total Length of Bwd Packets', 'Fwd IAT Max', 'Init_Win_bytes_backward',
                                         'act_data_pkt_fwd', 'Flow Duration', 'Flow Bytes/s', 'Fwd IAT Total', 'Bwd IAT Total',
                                         'Total Backward Packets', 'Subflow Bwd Bytes', 'Bwd Packets/s', 'URG Flag Count']


## Lists of features ordered by importance for each attack type
dos_hulk = ['Min Packet Length', 'act_data_pkt_fwd', 'Idle Std', 'Bwd IAT Std', 'Active Max', 'Bwd IAT Max', 'Active Min', 'Idle Mean', 'Active Mean', 'Bwd Packet Length Min', 'Active Std', 'Total Length of Fwd Packets', 'Idle Max', 'Down/Up Ratio', 'Total Backward Packets', 'Subflow Fwd Bytes', 'Idle Min', 'Fwd Packet Length Mean', 'Bwd Packet Length Max', 'Bwd IAT Total', 'Bwd IAT Mean', 'Max Packet Length', 'Flow IAT Std', 'Bwd Packet Length Mean', 'Subflow Bwd Packets', 'Packet Length Std', 'Avg Bwd Segment Size', 'Bwd Header Length', 'min_seg_size_forward', 'Fwd IAT Std', 'Total Fwd Packets', 'PSH Flag Count', 'ACK Flag Count', 'Fwd IAT Min', 'Flow Packets/s', 'Flow IAT Mean', 'Bwd IAT Min', 'Subflow Fwd Packets', 'Avg Fwd Segment Size', 'Flow IAT Max', 'Total Length of Bwd Packets', 'Packet Length Variance', 'Subflow Bwd Bytes', 'Fwd IAT Total', 'Fwd Packet Length Std', 'Fwd IAT Max', 'URG Flag Count', 'Packet Length Mean', 'Fwd Packets/s', 'Average Packet Size', 'Flow IAT Min', 'Fwd PSH Flags', 'Flow Bytes/s', 'Init_Win_bytes_backward', 'Fwd Packet Length Max', 'Flow Duration', 'Fwd IAT Mean', 'SYN Flag Count', 'Fwd Packet Length Min', 'FIN Flag Count', 'Bwd Packet Length Std', 'Init_Win_bytes_forward', 'Fwd Header Length', 'Bwd Packets/s']

dos_goldeneye = ['PSH Flag Count', 'Bwd Packet Length Min', 'Bwd IAT Total', 'Fwd PSH Flags', 'Min Packet Length', 'Down/Up Ratio', 'Bwd IAT Std', 'Fwd Packet Length Min', 'Total Fwd Packets', 'FIN Flag Count', 'act_data_pkt_fwd', 'Bwd Packet Length Mean', 'Subflow Fwd Packets', 'SYN Flag Count', 'Idle Mean', 'Avg Bwd Segment Size', 'Fwd IAT Std', 'Idle Std', 'Bwd Packets/s', 'ACK Flag Count', 'Packet Length Variance', 'Bwd IAT Min', 'Total Length of Fwd Packets', 'Bwd Header Length', 'Subflow Fwd Bytes', 'Fwd Packet Length Std', 'Active Mean', 'Fwd Header Length', 'Total Length of Bwd Packets', 'Packet Length Std', 'Avg Fwd Segment Size', 'Fwd Packet Length Max', 'Flow Packets/s', 'Bwd Packet Length Max', 'Bwd IAT Max', 'Active Max', 'Fwd IAT Mean', 'Fwd Packet Length Mean', 'Fwd IAT Min', 'Flow IAT Std', 'Active Std', 'Active Min', 'Fwd IAT Total', 'Flow Duration', 'Subflow Bwd Packets', 'Average Packet Size', 'Flow IAT Max', 'URG Flag Count', 'Flow Bytes/s', 'Init_Win_bytes_backward', 'Subflow Bwd Bytes', 'Fwd IAT Max', 'Max Packet Length', 'min_seg_size_forward', 'Fwd Packets/s', 'Total Backward Packets', 'Packet Length Mean', 'Bwd IAT Mean', 'Idle Max', 'Init_Win_bytes_forward', 'Idle Min', 'Flow IAT Mean', 'Flow IAT Min', 'Bwd Packet Length Std']

ddos = ['PSH Flag Count', 'Total Fwd Packets', 'min_seg_size_forward', 'Min Packet Length', 'Bwd IAT Max', 'Fwd Packet Length Mean', 'Active Std', 'act_data_pkt_fwd', 'Max Packet Length', 'Fwd PSH Flags', 'Bwd IAT Std', 'Idle Std', 'Fwd Header Length', 'SYN Flag Count', 'ACK Flag Count', 'Active Min', 'Down/Up Ratio', 'Fwd Packet Length Min', 'Bwd Packet Length Std', 'Active Mean', 'Subflow Fwd Packets', 'Bwd Packet Length Mean', 'Bwd IAT Min', 'Bwd IAT Total', 'Avg Fwd Segment Size', 'Packet Length Variance', 'Packet Length Mean', 'Active Max', 'Packet Length Std', 'Bwd Packet Length Max', 'Flow IAT Mean', 'Fwd Packet Length Std', 'Subflow Bwd Bytes', 'Idle Mean', 'Total Length of Bwd Packets', 'Bwd IAT Mean', 'FIN Flag Count', 'Fwd IAT Total', 'Flow Duration', 'Fwd IAT Max', 'Flow IAT Std', 'URG Flag Count', 'Subflow Bwd Packets', 'Flow Packets/s', 'Idle Min', 'Avg Bwd Segment Size', 'Fwd IAT Mean', 'Flow Bytes/s', 'Flow IAT Max', 'Idle Max', 'Total Backward Packets', 'Flow IAT Min', 'Bwd Header Length', 'Fwd Packets/s', 'Fwd IAT Std', 'Bwd Packets/s', 'Bwd Packet Length Min', 'Init_Win_bytes_backward', 'Fwd IAT Min', 'Average Packet Size', 'Init_Win_bytes_forward', 'Total Length of Fwd Packets', 'Subflow Fwd Bytes', 'Fwd Packet Length Max']

dos_slowloris = ['Min Packet Length', 'Bwd Packet Length Min', 'PSH Flag Count', 'Idle Mean', 'Flow IAT Std', 'Total Backward Packets', 'FIN Flag Count', 'Idle Std', 'act_data_pkt_fwd', 'ACK Flag Count', 'Fwd Packet Length Min', 'Active Mean', 'Total Fwd Packets', 'Fwd IAT Total', 'Subflow Bwd Packets', 'Packet Length Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd Packet Length Mean', 'URG Flag Count', 'Avg Fwd Segment Size', 'Max Packet Length', 'Down/Up Ratio', 'Idle Min', 'Flow IAT Max', 'Fwd Packet Length Std', 'Fwd IAT Mean', 'Flow Packets/s', 'Average Packet Size', 'Active Min', 'Packet Length Variance', 'Idle Max', 'Fwd Packet Length Max', 'Flow Bytes/s', 'SYN Flag Count', 'Avg Bwd Segment Size', 'Bwd Packets/s', 'Active Std', 'Fwd IAT Min', 'Subflow Fwd Packets', 'Bwd Packet Length Mean', 'Packet Length Std', 'Flow Duration', 'Bwd IAT Mean', 'Bwd Packet Length Max', 'Init_Win_bytes_backward', 'Bwd IAT Std', 'Init_Win_bytes_forward', 'Active Max', 'Bwd IAT Min', 'Bwd IAT Total', 'Fwd PSH Flags', 'Bwd Header Length', 'Bwd Packet Length Std', 'Fwd Packets/s', 'Subflow Fwd Bytes', 'Flow IAT Mean', 'Fwd Header Length', 'Total Length of Bwd Packets', 'Total Length of Fwd Packets', 'min_seg_size_forward', 'Subflow Bwd Bytes', 'Bwd IAT Max', 'Flow IAT Min']

dos_slowhttp = ['Idle Mean', 'Idle Std', 'FIN Flag Count', 'SYN Flag Count', 'Total Backward Packets', 'Subflow Bwd Packets', 'ACK Flag Count', 'act_data_pkt_fwd', 'Fwd Packet Length Std', 'PSH Flag Count', 'Idle Max', 'Bwd Header Length', 'Bwd IAT Max', 'Subflow Fwd Packets', 'Fwd IAT Total', 'Bwd Packets/s', 'Bwd IAT Total', 'Flow Bytes/s', 'Flow IAT Max', 'Packet Length Variance', 'Fwd PSH Flags', 'Total Length of Bwd Packets', 'Bwd Packet Length Min', 'Bwd Packet Length Mean', 'Fwd IAT Max', 'Fwd Packet Length Mean', 'Down/Up Ratio', 'Average Packet Size', 'Avg Bwd Segment Size', 'Packet Length Mean', 'Packet Length Std', 'Fwd IAT Mean', 'Idle Min', 'min_seg_size_forward', 'Max Packet Length', 'Bwd Packet Length Max', 'Flow IAT Mean', 'Fwd IAT Std', 'Active Max', 'URG Flag Count', 'Subflow Bwd Bytes', 'Min Packet Length', 'Flow Packets/s', 'Fwd Packet Length Max', 'Flow Duration', 'Avg Fwd Segment Size', 'Fwd Packets/s', 'Bwd IAT Mean', 'Init_Win_bytes_backward', 'Total Fwd Packets', 'Bwd Packet Length Std', 'Subflow Fwd Bytes', 'Fwd Header Length', 'Active Std', 'Active Mean', 'Bwd IAT Min', 'Init_Win_bytes_forward', 'Fwd IAT Min', 'Flow IAT Min', 'Total Length of Fwd Packets', 'Fwd Packet Length Min', 'Flow IAT Std', 'Bwd IAT Std', 'Active Min']

botnet = ['Min Packet Length', 'Idle Mean', 'Idle Max', 'Bwd Packet Length Min', 'Idle Min', 'Fwd Header Length', 'min_seg_size_forward', 'Active Std', 'Bwd Packet Length Std', 'Fwd Packet Length Min', 'Fwd Packet Length Std', 'Idle Std', 'Active Max', 'Fwd PSH Flags', 'Subflow Bwd Packets', 'Total Fwd Packets', 'FIN Flag Count', 'Active Min', 'Subflow Fwd Packets', 'Total Length of Fwd Packets', 'SYN Flag Count', 'ACK Flag Count', 'Total Backward Packets', 'Active Mean', 'Bwd Packet Length Max', 'Fwd Packet Length Max', 'Subflow Fwd Bytes', 'Packet Length Std', 'Packet Length Variance', 'Bwd IAT Total', 'Total Length of Bwd Packets', 'Fwd IAT Total', 'Subflow Bwd Bytes', 'Bwd IAT Mean', 'Fwd Packet Length Mean', 'Flow Packets/s', 'Fwd IAT Mean', 'act_data_pkt_fwd', 'Fwd IAT Min', 'Max Packet Length', 'URG Flag Count', 'Fwd Packets/s', 'Bwd IAT Max', 'Fwd IAT Max', 'Flow IAT Mean', 'Bwd Header Length', 'Fwd IAT Std', 'Flow Duration', 'Down/Up Ratio', 'Flow IAT Std', 'Flow Bytes/s', 'Bwd Packet Length Mean', 'Flow IAT Min', 'Avg Fwd Segment Size', 'Flow IAT Max', 'Avg Bwd Segment Size', 'Bwd IAT Min', 'PSH Flag Count', 'Packet Length Mean', 'Bwd IAT Std', 'Init_Win_bytes_backward', 'Init_Win_bytes_forward', 'Bwd Packets/s', 'Average Packet Size']

web_xss =['Bwd IAT Total', 'Bwd IAT Max', 'PSH Flag Count', 'Min Packet Length', 'Idle Mean', 'Bwd IAT Mean', 'act_data_pkt_fwd', 'Total Fwd Packets', 'Idle Max', 'Fwd Header Length', 'Flow IAT Std', 'Subflow Bwd Packets', 'Bwd Packet Length Min', 'Bwd IAT Std', 'Idle Min', 'Avg Bwd Segment Size', 'Active Std', 'URG Flag Count', 'Fwd Packet Length Std', 'Fwd Packet Length Min', 'Total Backward Packets', 'Subflow Bwd Bytes', 'Idle Std', 'Subflow Fwd Packets', 'Flow Bytes/s', 'Fwd PSH Flags', 'Total Length of Bwd Packets', 'Active Max', 'FIN Flag Count', 'Active Min', 'min_seg_size_forward', 'Down/Up Ratio', 'Packet Length Std', 'SYN Flag Count', 'Packet Length Variance', 'Bwd Packet Length Mean', 'Active Mean', 'ACK Flag Count', 'Flow IAT Max', 'Packet Length Mean', 'Flow Packets/s', 'Flow Duration', 'Max Packet Length', 'Fwd IAT Std', 'Flow IAT Mean', 'Bwd Packet Length Max', 'Bwd Header Length', 'Average Packet Size', 'Fwd Packet Length Mean', 'Fwd IAT Max', 'Bwd Packets/s', 'Flow IAT Min', 'Avg Fwd Segment Size', 'Bwd IAT Min', 'Bwd Packet Length Std', 'Fwd IAT Total', 'Total Length of Fwd Packets', 'Fwd Packet Length Max', 'Fwd Packets/s', 'Fwd IAT Mean', 'Subflow Fwd Bytes', 'Init_Win_bytes_forward', 'Fwd IAT Min', 'Init_Win_bytes_backward']

web_sql = ['PSH Flag Count', 'Min Packet Length', 'Fwd Header Length', 'act_data_pkt_fwd', 'Idle Mean', 'Flow IAT Mean', 'Idle Max', 'Init_Win_bytes_forward', 'Bwd IAT Std', 'Subflow Bwd Packets', 'Bwd IAT Mean', 'Flow IAT Min', 'Flow IAT Std', 'min_seg_size_forward', 'Bwd Packet Length Min', 'Max Packet Length', 'Idle Min', 'Total Fwd Packets', 'Total Backward Packets', 'Fwd Packet Length Mean', 'Avg Bwd Segment Size', 'Active Std', 'Fwd Packet Length Min', 'Flow IAT Max', 'Fwd IAT Min', 'Idle Std', 'Bwd Packet Length Mean', 'Active Max', 'Total Length of Fwd Packets', 'Subflow Fwd Packets', 'Average Packet Size', 'Avg Fwd Segment Size', 'Fwd PSH Flags', 'Down/Up Ratio', 'FIN Flag Count', 'Active Min', 'Packet Length Std', 'Packet Length Variance', 'Fwd Packets/s', 'Packet Length Mean', 'SYN Flag Count', 'ACK Flag Count', 'Subflow Fwd Bytes', 'Total Length of Bwd Packets', 'URG Flag Count', 'Active Mean', 'Flow Bytes/s', 'Fwd Packet Length Max', 'Bwd Packet Length Max', 'Bwd Header Length', 'Fwd Packet Length Std', 'Fwd IAT Mean', 'Fwd IAT Max', 'Fwd IAT Std', 'Flow Packets/s', 'Subflow Bwd Bytes', 'Bwd IAT Total', 'Bwd Packets/s', 'Bwd IAT Max', 'Fwd IAT Total', 'Init_Win_bytes_backward', 'Flow Duration', 'Bwd IAT Min', 'Bwd Packet Length Std']

web_bruteforce = ['Bwd IAT Total', 'PSH Flag Count', 'Min Packet Length', 'Flow IAT Mean', 'Idle Mean', 'Fwd IAT Mean', 'min_seg_size_forward', 'Idle Max', 'Bwd IAT Max', 'Subflow Bwd Packets', 'Bwd Packet Length Min', 'Active Std', 'Total Backward Packets', 'Fwd Packet Length Min', 'Idle Std', 'Idle Min', 'Bwd IAT Std', 'Total Fwd Packets', 'Total Length of Bwd Packets', 'Active Max', 'URG Flag Count', 'Fwd PSH Flags', 'Fwd Packet Length Std', 'FIN Flag Count', 'Bwd IAT Mean', 'Fwd IAT Max', 'Bwd Header Length', 'Active Min', 'SYN Flag Count', 'act_data_pkt_fwd', 'Active Mean', 'ACK Flag Count', 'Packet Length Std', 'Down/Up Ratio', 'Flow IAT Std', 'Packet Length Variance', 'Fwd IAT Std', 'Fwd Packet Length Max', 'Average Packet Size', 'Packet Length Mean', 'Bwd Packet Length Mean', 'Fwd IAT Total', 'Subflow Bwd Bytes', 'Flow IAT Max', 'Avg Fwd Segment Size', 'Bwd Packet Length Std', 'Max Packet Length', 'Flow Duration', 'Bwd Packet Length Max', 'Avg Bwd Segment Size', 'Flow Packets/s', 'Bwd IAT Min', 'Fwd Packet Length Mean', 'Fwd Header Length', 'Init_Win_bytes_forward', 'Bwd Packets/s', 'Subflow Fwd Packets', 'Fwd Packets/s', 'Total Length of Fwd Packets', 'Flow Bytes/s', 'Flow IAT Min', 'Subflow Fwd Bytes', 'Fwd IAT Min', 'Init_Win_bytes_backward']

ssh_patator = ['PSH Flag Count', 'Idle Mean', 'Min Packet Length', 'Idle Max', 'Total Backward Packets', 'Bwd Packet Length Min', 'Subflow Bwd Packets', 'Active Std', 'Idle Min', 'Fwd Packet Length Std', 'Idle Std', 'Active Max', 'Fwd Packet Length Min', 'Down/Up Ratio', 'FIN Flag Count', 'Active Mean', 'Fwd PSH Flags', 'ACK Flag Count', 'act_data_pkt_fwd', 'Avg Bwd Segment Size', 'Avg Fwd Segment Size', 'SYN Flag Count', 'Fwd Packet Length Mean', 'Total Length of Fwd Packets', 'Fwd IAT Std', 'Bwd Packet Length Mean', 'Packet Length Variance', 'Packet Length Std', 'Flow IAT Min', 'Bwd Packet Length Max', 'Bwd IAT Total', 'Subflow Fwd Packets', 'Active Min', 'Bwd IAT Mean', 'Fwd IAT Mean', 'Fwd IAT Min', 'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Fwd IAT Max', 'Bwd IAT Max', 'Bwd IAT Std', 'Bwd Packet Length Std', 'Packet Length Mean', 'Fwd Header Length', 'Max Packet Length', 'Flow IAT Std', 'Fwd Packets/s', 'Subflow Fwd Bytes', 'Total Fwd Packets', 'Flow Duration', 'Fwd IAT Total', 'Bwd Packets/s', 'Bwd IAT Min', 'Average Packet Size', 'Flow IAT Max', 'Fwd Packet Length Max', 'min_seg_size_forward', 'Init_Win_bytes_forward', 'URG Flag Count', 'Total Length of Bwd Packets', 'Bwd Header Length', 'Subflow Bwd Bytes', 'Init_Win_bytes_backward']

ftp_patator = ['Bwd IAT Total', 'PSH Flag Count', 'Min Packet Length', 'act_data_pkt_fwd', 'Bwd IAT Max', 'Idle Mean', 'Idle Max', 'Bwd IAT Std', 'Fwd IAT Mean', 'Subflow Bwd Packets', 'Bwd IAT Mean', 'Bwd Packet Length Min', 'Total Fwd Packets', 'Bwd Packet Length Std', 'Idle Min', 'Active Std', 'Average Packet Size', 'Flow IAT Std', 'Fwd Packet Length Min', 'Total Backward Packets', 'Idle Std', 'Bwd Packet Length Mean', 'Fwd Packet Length Std', 'Active Max', 'Avg Bwd Segment Size', 'FIN Flag Count', 'Subflow Fwd Packets', 'Active Mean', 'Active Min', 'URG Flag Count', 'Bwd IAT Min', 'Down/Up Ratio', 'Fwd IAT Min', 'ACK Flag Count', 'Packet Length Std', 'Flow Packets/s', 'Bwd Packet Length Max', 'Flow Duration', 'Total Length of Bwd Packets', 'Fwd IAT Max', 'Flow IAT Mean', 'Packet Length Mean', 'Fwd IAT Total', 'Packet Length Variance', 'Fwd IAT Std', 'min_seg_size_forward', 'Init_Win_bytes_backward', 'Bwd Packets/s', 'Flow IAT Max', 'Flow Bytes/s', 'Fwd Header Length', 'Flow IAT Min', 'Fwd Packets/s', 'Init_Win_bytes_forward', 'Fwd Packet Length Max', 'SYN Flag Count', 'Subflow Fwd Bytes', 'Fwd Packet Length Mean', 'Total Length of Fwd Packets', 'Fwd PSH Flags', 'Subflow Bwd Bytes', 'Avg Fwd Segment Size', 'Bwd Header Length', 'Max Packet Length']

portscan = ['Max Packet Length', 'Avg Bwd Segment Size', 'act_data_pkt_fwd', 'Idle Max', 'min_seg_size_forward', 'Active Std', 'Total Length of Bwd Packets', 'Idle Mean', 'Fwd Packet Length Min', 'URG Flag Count', 'Idle Std', 'Bwd IAT Mean', 'Bwd Packet Length Mean', 'Bwd IAT Std', 'Fwd PSH Flags', 'Bwd Packet Length Min', 'Down/Up Ratio', 'Subflow Bwd Bytes', 'FIN Flag Count', 'SYN Flag Count', 'Fwd Packet Length Std', 'Active Min', 'Average Packet Size', 'Bwd IAT Max', 'Min Packet Length', 'Active Mean', 'Active Max', 'Packet Length Std', 'Subflow Fwd Packets', 'ACK Flag Count', 'Fwd Packet Length Mean', 'Bwd Packets/s', 'Bwd Packet Length Max', 'Subflow Bwd Packets', 'Total Fwd Packets', 'Flow IAT Std', 'Flow IAT Min', 'Bwd IAT Total', 'Bwd Header Length', 'Flow IAT Max', 'Packet Length Variance', 'Total Backward Packets', 'Flow Packets/s', 'Fwd Packet Length Max', 'Fwd IAT Std', 'Avg Fwd Segment Size', 'Fwd IAT Min', 'Packet Length Mean', 'Fwd IAT Max', 'Init_Win_bytes_backward', 'Init_Win_bytes_forward', 'Flow Duration', 'Fwd Packets/s', 'Bwd Packet Length Std', 'Flow IAT Mean', 'Fwd IAT Mean', 'Bwd IAT Min', 'Fwd Header Length', 'Idle Min', 'PSH Flag Count', 'Fwd IAT Total', 'Flow Bytes/s', 'Subflow Fwd Bytes', 'Total Length of Fwd Packets']

infiltration = ['Bwd IAT Total', 'Flow Packets/s', 'PSH Flag Count', 'Min Packet Length', 'Init_Win_bytes_backward', 'Fwd Header Length', 'act_data_pkt_fwd', 'Idle Max', 'min_seg_size_forward', 'Total Fwd Packets', 'Flow IAT Min', 'Bwd Packets/s', 'Init_Win_bytes_forward', 'Fwd Packet Length Min', 'Flow IAT Mean', 'Bwd IAT Max', 'Subflow Bwd Packets', 'Bwd IAT Mean', 'Bwd Packet Length Min', 'Bwd Packet Length Std', 'Fwd Packet Length Mean', 'Active Std', 'Max Packet Length', 'Fwd Packet Length Std', 'Down/Up Ratio', 'FIN Flag Count', 'Avg Fwd Segment Size', 'Fwd IAT Std', 'Total Length of Bwd Packets', 'Packet Length Std', 'Subflow Fwd Packets', 'Total Backward Packets', 'Average Packet Size', 'Packet Length Variance', 'Fwd Packets/s', 'Idle Min', 'Subflow Bwd Bytes', 'ACK Flag Count', 'Fwd IAT Mean', 'Fwd PSH Flags', 'URG Flag Count', 'Idle Std', 'Idle Mean', 'Bwd Header Length', 'Flow Bytes/s', 'Flow IAT Max', 'Flow IAT Std', 'Active Max', 'Fwd IAT Max', 'Fwd Packet Length Max', 'Bwd IAT Std', 'Fwd IAT Total', 'Flow Duration', 'Avg Bwd Segment Size', 'Active Mean', 'Packet Length Mean', 'SYN Flag Count', 'Bwd IAT Min', 'Active Min', 'Bwd Packet Length Mean', 'Bwd Packet Length Max', 'Subflow Fwd Bytes', 'Fwd IAT Min', 'Total Length of Fwd Packets']

heartbleed = ['Bwd IAT Total', 'Bwd IAT Max', 'PSH Flag Count', 'Flow IAT Mean', 'Flow IAT Min', 'Flow Packets/s', 'Fwd IAT Max', 'Init_Win_bytes_backward', 'Fwd IAT Mean', 'Fwd Header Length', 'Min Packet Length', 'Subflow Bwd Packets', 'act_data_pkt_fwd', 'Idle Mean', 'Bwd IAT Mean', 'Flow IAT Std', 'Idle Max', 'Init_Win_bytes_forward', 'Bwd Packets/s', 'Bwd IAT Std', 'min_seg_size_forward', 'Fwd Packet Length Mean', 'Max Packet Length', 'Total Fwd Packets', 'Bwd Packet Length Min', 'Bwd Packet Length Std', 'Total Backward Packets', 'Total Length of Bwd Packets', 'Idle Min', 'Active Std', 'Subflow Fwd Packets', 'URG Flag Count', 'Fwd Packet Length Min', 'Subflow Bwd Bytes', 'Fwd Packet Length Std', 'Flow IAT Max', 'Avg Fwd Segment Size', 'Average Packet Size', 'Idle Std', 'Fwd IAT Min', 'Flow Duration', 'Flow Bytes/s', 'Active Max', 'Fwd IAT Total', 'Total Length of Fwd Packets', 'Fwd Packets/s', 'Fwd PSH Flags', 'Fwd Packet Length Max', 'Down/Up Ratio', 'FIN Flag Count', 'Active Min', 'Packet Length Std', 'Packet Length Variance', 'Packet Length Mean', 'Fwd IAT Std', 'Bwd Header Length', 'Bwd IAT Min', 'Subflow Fwd Bytes', 'SYN Flag Count', 'Active Mean', 'ACK Flag Count', 'Bwd Packet Length Max', 'Bwd Packet Length Mean', 'Avg Bwd Segment Size']


all_attack_featurelists = {'dos_hulk': dos_hulk, 'dos_goldeneye': dos_goldeneye, 'ddos': ddos, 'dos_slowloris': dos_slowloris, 'dos_slowhttp': dos_slowhttp,
                           'botnet': botnet, 'web_xss': web_xss, 'web_sql': web_sql, 'web_bruteforce': web_bruteforce, 'ssh_patator': ssh_patator,
                            'ftp_patator': ftp_patator, 'portscan': portscan, 'infiltration': infiltration, 'heartbleed': heartbleed}

dos_attacks_featurelists = {'dos_hulk': dos_hulk, 'dos_goldeneye': dos_goldeneye, 'ddos': ddos, 'dos_slowloris': dos_slowloris, 'dos_slowhttp': dos_slowhttp}


# Color map used for T-SNE plots
# Vega color palette from https://github.com/vega/vega/wiki/Scales#scale-range-literals
color_map = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5',
             '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f', '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf']


# Featurenames in the CICIDS2017 datasets are different than in the CSVs generated by the FlowMeter version we used
# for Locked Shields... Create a mapping between these names
cic17_to_ls18_headernames_dict = {'Flow Duration': 'Flow Duration', 'Total Fwd Packets': 'Tot Fwd Pkts',
                                  'Total Backward Packets': 'Tot Bwd Pkts', 'Total Length of Fwd Packets': 'TotLen Fwd Pkts',
                                  'Total Length of Bwd Packets': 'TotLen Bwd Pkts', 'Fwd Packet Length Max': 'Fwd Pkt Len Max',
                                  'Fwd Packet Length Min': 'Fwd Pkt Len Min', 'Fwd Packet Length Mean': 'Fwd Pkt Len Mean',
                                  'Fwd Packet Length Std': 'Fwd Pkt Len Std', 'Bwd Packet Length Max': 'Bwd Pkt Len Max',
                                  'Bwd Packet Length Min': 'Bwd Pkt Len Min', 'Bwd Packet Length Mean': 'Bwd Pkt Len Mean',
                                  'Bwd Packet Length Std': 'Bwd Pkt Len Std', 'Flow Bytes/s': 'Flow Byts/s',
                                  'Flow Packets/s': 'Flow Pkts/s', 'Flow IAT Mean': 'Flow IAT Mean', 'Flow IAT Std': 'Flow IAT Std',
                                  'Flow IAT Max': 'Flow IAT Max', 'Flow IAT Min': 'Flow IAT Min', 'Fwd IAT Total': 'Fwd IAT Tot',
                                  'Fwd IAT Mean': 'Fwd IAT Mean', 'Fwd IAT Std': 'Fwd IAT Std', 'Fwd IAT Max': 'Fwd IAT Max',
                                  'Fwd IAT Min': 'Fwd IAT Min', 'Bwd IAT Total': 'Bwd IAT Tot', 'Bwd IAT Mean': 'Bwd IAT Mean',
                                  'Bwd IAT Std': 'Bwd IAT Std', 'Bwd IAT Max': 'Bwd IAT Max', 'Bwd IAT Min': 'Bwd IAT Min',
                                  'Fwd PSH Flags': 'Fwd PSH Flags', 'Bwd PSH Flags': 'Bwd PSH Flags', 'Fwd URG Flags': 'Fwd URG Flags',
                                  'Bwd URG Flags': 'Bwd URG Flags', 'Fwd Header Length': 'Fwd Header Len',
                                  'Bwd Header Length': 'Bwd Header Len', 'Fwd Packets/s': 'Fwd Pkts/s', 'Bwd Packets/s': 'Bwd Pkts/s',
                                  'Min Packet Length': 'Pkt Len Min', 'Max Packet Length': 'Pkt Len Max',
                                  'Packet Length Mean': 'Pkt Len Mean', 'Packet Length Std': 'Pkt Len Std',
                                  'Packet Length Variance': 'Pkt Len Var', 'FIN Flag Count': 'FIN Flag Cnt',
                                  'SYN Flag Count': 'SYN Flag Cnt', 'RST Flag Count': 'RST Flag Cnt', 'PSH Flag Count': 'PSH Flag Cnt',
                                  'ACK Flag Count': 'ACK Flag Cnt', 'URG Flag Count': 'URG Flag Cnt', 'CWE Flag Count': 'CWR Flag Count',
                                  'ECE Flag Count': 'ECE Flag Cnt', 'Down/Up Ratio': 'Down/Up Ratio', 'Average Packet Size': 'Pkt Size Avg',
                                  'Avg Fwd Segment Size': 'Fwd Seg Size Avg', 'Avg Bwd Segment Size': 'Bwd Seg Size Avg',
                                  'Fwd Avg Bytes/Bulk': 'Fwd Byts/b Avg', 'Fwd Avg Packets/Bulk': 'Fwd Pkts/b Avg',
                                  'Fwd Avg Bulk Rate': 'Fwd Blk Rate Avg', 'Bwd Avg Bytes/Bulk': 'Bwd Byts/b Avg',
                                  'Bwd Avg Packets/Bulk': 'Bwd Pkts/b Avg', 'Bwd Avg Bulk Rate': 'Bwd Blk Rate Avg',
                                  'Subflow Fwd Packets': 'Subflow Fwd Pkts', 'Subflow Fwd Bytes': 'Subflow Fwd Byts',
                                  'Subflow Bwd Packets': 'Subflow Bwd Pkts', 'Subflow Bwd Bytes': 'Subflow Bwd Byts',
                                  'Init_Win_bytes_forward': 'Init Fwd Win Byts', 'Init_Win_bytes_backward': 'Init Bwd Win Byts',
                                  'act_data_pkt_fwd': 'Fwd Act Data Pkts', 'min_seg_size_forward': 'Fwd Seg Size Min',
                                  'Active Mean': 'Active Mean', 'Active Std': 'Active Std', 'Active Max': 'Active Max',
                                  'Active Min': 'Active Min', 'Idle Mean': 'Idle Mean', 'Idle Std': 'Idle Std', 'Idle Max': 'Idle Max',
                                  'Idle Min': 'Idle Min', 'Label': 'Label'}