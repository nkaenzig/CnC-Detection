
@load base/protocols/conn
@load base/protocols/ssl
@load base/bif/plugins/Bro_X509.events.bif.bro

module SSLAnalyzer;

export {
    redef SSL::disable_analyzer_after_detection = F;
    redef enum Log::ID += {LOG};
    
    #global foo_ports: set[port] = { 8080/tcp } &redef;
    #redef dpd_config += { [Analyzer::ANALYZER_FTP] = [$ports = foo_ports] };
   
    type EncryptedData: record {
        ts: time &log;
        uid: string &log;
        evt: string &log;
        is_orig: bool &log &optional;
        sAddr: addr &log &optional;
        dAddr: addr &log &optional;
        sPort: port &log &optional;
        dPort: port &log &optional;
        server_name: string &log &optional;
        size: count &log &optional;
    };
}

event bro_init() {
    Log::create_stream(LOG,[$columns=EncryptedData]);

    local rec: SSLAnalyzer::EncryptedData = [$ts=network_time(), $uid="", $evt="bro_init"];
	#Log::write(SSLAnalyzer::LOG, rec);
}

event connection_state_remove(c:connection) {
    #print network_time();
}

event tcp_packet (c: connection, is_orig: bool, flags: string, seq: count, ack: count, len: count, payload: string) {
    #print c$id$resp_p;
}

event ssl_encrypted_data(c:connection, is_orig:bool, content_type:count, length:count) &priority=5 {
    #print c$id$orig_p+"a";
    #print(fmt("ssl_encrypted_data \t %s %s %s %s" , c$start_time, c$id$orig_p, c$ssl$server_name, length));
    #Log::write(SSLAnalyzer::LOG, [$ts=c$start_time, $sPort = c$id$orig_p, $server_name=c$ssl$server_name]);
    
    
    local rec: SSLAnalyzer::EncryptedData;
    local sn: string = "?";

    if(c$ssl?$server_name)
        sn=c$ssl$server_name;
    
    if(content_type == 23) { #application data
    	rec = [$ts=network_time(), $uid=c$uid, $evt="ssl_encrypted_data", $is_orig = is_orig, $sAddr=c$id$orig_h, $dAddr=c$id$resp_h, $sPort=c$id$orig_p, $dPort=c$id$resp_p, $server_name=sn, $size=length];

    Log::write(SSLAnalyzer::LOG, rec);
	#print "ssl_encrypted_data";
    }
}

event ssl_extension_server_name(c:connection, is_orig:bool, names: string_vec) {
	local snames: string = "";

	for (i in names)
	snames = fmt("%s %s",snames,names[i]);

	local rec: SSLAnalyzer::EncryptedData = [$ts=network_time(), $uid=c$uid, $evt="ssl_extension_server_name", $is_orig = is_orig, $sAddr=c$id$orig_h, $dAddr=c$id$resp_h, $sPort=c$id$orig_p, $dPort=c$id$resp_p, $server_name=snames];
	Log::write(SSLAnalyzer::LOG, rec);
}


event ssl_client_hello(c: connection, version: count, possible_ts: time, client_random: string, session_id: string, ciphers: index_vec) {
	local sn: string = "?";

    	if(c$ssl?$server_name)
        	sn=c$ssl$server_name;	

	local rec: SSLAnalyzer::EncryptedData = [$ts=network_time(), $uid=c$uid, $evt="ssl_client_hello", $is_orig = T, $sAddr=c$id$orig_h, $dAddr=c$id$resp_h, $sPort=c$id$orig_p, $dPort=c$id$resp_p, $server_name=sn];
	Log::write(SSLAnalyzer::LOG, rec);
}

event ssl_server_hello(c: connection, version: count, possible_ts: time, server_random: string, session_id: string, cipher: count, comp_method: count) {
	local sn: string = "?";

    	if(c$ssl?$server_name)
        	sn=c$ssl$server_name;	

	local rec: SSLAnalyzer::EncryptedData = [$ts=network_time(), $uid=c$uid, $evt="ssl_server_hello", $is_orig = F, $sAddr=c$id$orig_h, $dAddr=c$id$resp_h, $sPort=c$id$orig_p, $dPort=c$id$resp_p, $server_name=sn];
	Log::write(SSLAnalyzer::LOG, rec);
}

event bro_done() {
    #Log::write(SSLAnalyzer::LOG, [$ts="", $sPort = "", $server_name="a"]);
}

