package cic.cs.unb.ca.jnetpcap;

import cic.cs.unb.ca.jnetpcap.worker.FlowGenListener;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Set;

import java.util.Iterator;
import java.util.Map;

public class FlowGenerator {
    public static final Logger logger = LoggerFactory.getLogger(FlowGenerator.class);

    //total 85 colums
	/*public static final String timeBasedHeader = "Flow ID, Source IP, Source Port, Destination IP, Destination Port, Protocol, "
			+ "Timestamp, Flow Duration, Total Fwd Packets, Total Backward Packets,"
			+ "Total Length of Fwd Packets, Total Length of Bwd Packets, "
			+ "Fwd Packet Length Max, Fwd Packet Length Min, Fwd Packet Length Mean, Fwd Packet Length Std,"
			+ "Bwd Packet Length Max, Bwd Packet Length Min, Bwd Packet Length Mean, Bwd Packet Length Std,"
			+ "Flow Bytes/s, Flow Packets/s, Flow IAT Mean, Flow IAT Std, Flow IAT Max, Flow IAT Min,"
			+ "Fwd IAT Total, Fwd IAT Mean, Fwd IAT Std, Fwd IAT Max, Fwd IAT Min,"
			+ "Bwd IAT Total, Bwd IAT Mean, Bwd IAT Std, Bwd IAT Max, Bwd IAT Min,"
			+ "Fwd PSH Flags, Bwd PSH Flags, Fwd URG Flags, Bwd URG Flags, Fwd Header Length, Bwd Header Length,"
			+ "Fwd Packets/s, Bwd Packets/s, Min Packet Length, Max Packet Length, Packet Length Mean, Packet Length Std, Packet Length Variance,"
			+ "FIN Flag Count, SYN Flag Count, RST Flag Count, PSH Flag Count, ACK Flag Count, URG Flag Count, "
			+ "CWE Flag Count, ECE Flag Count, Down/Up Ratio, Average Packet Size, Avg Fwd Segment Size, Avg Bwd Segment Size, Fwd Header Length,"
			+ "Fwd Avg Bytes/Bulk, Fwd Avg Packets/Bulk, Fwd Avg Bulk Rate, Bwd Avg Bytes/Bulk, Bwd Avg Packets/Bulk,"
			+ "Bwd Avg Bulk Rate,"
			+ "Subflow Fwd Packets, Subflow Fwd Bytes, Subflow Bwd Packets, Subflow Bwd Bytes,"
			+ "Init_Win_bytes_forward, Init_Win_bytes_backward, act_data_pkt_fwd, min_seg_size_forward,"
			+ "Active Mean, Active Std, Active Max, Active Min,"
			+ "Idle Mean, Idle Std, Idle Max, Idle Min, Label";*/

	//40/86
	private FlowGenListener mListener;
	private HashMap<String,BasicFlow> currentFlows;
	private HashMap<Integer,BasicFlow> finishedFlows;
	private HashMap<String,ArrayList> IPAddresses;

	private boolean bidirectional;
	private boolean singlePktFlows;
	private long    flowTimeOut;
	private long    flowActivityTimeOut;
	private int     finishedFlowCount;
	private int     timeoutFlowCount;
	private String[] intSubnets = null;

	public FlowGenerator(boolean bidirectional, long flowTimeout, long activityTimeout, boolean singlePktFlows, String subnets) {
		super();
		this.bidirectional = bidirectional;
		this.flowTimeOut = flowTimeout;
		this.flowActivityTimeOut = activityTimeout;
		this.singlePktFlows = singlePktFlows;
		if(!subnets.isEmpty()){
			this.intSubnets = subnets.split(",");
		}
		init();
	}		
	
	private void init(){
		currentFlows = new HashMap<>();
		finishedFlows = new HashMap<>();
		IPAddresses = new HashMap<>();
		finishedFlowCount = 0;
		timeoutFlowCount = 0;

	}

	public void close_timed_out_flows(BasicPacketInfo packet){
		long currentTimestamp = packet.getTimeStamp();
		long oldTimestamp;

		System.out.println("Remove timed out flows");

		Map<String,BasicFlow> map = this.currentFlows;

		for(Iterator<Map.Entry<String,BasicFlow>> it = map.entrySet().iterator(); it.hasNext(); ) {
			Map.Entry<String, BasicFlow> entry = it.next();
//			System.out.println(entry.getKey() + "/" + entry.getValue());
			BasicFlow flow = entry.getValue();
			oldTimestamp = flow.getFlowStartTime();

			if ((currentTimestamp - oldTimestamp) > this.flowTimeOut) {
//				System.out.println("Flow removed");
				if(this.singlePktFlows || flow.packetCount()>1){
                    finishedFlows.put(getFlowCount(), flow);
                    timeoutFlowCount += 1;
                }
				it.remove();
			}
		}

	}

	public void addFlowListener(FlowGenListener listener) {
		mListener = listener;
	}

    public void addPacket(BasicPacketInfo packet){
        if(packet == null) {
            return;
        }
        
    	BasicFlow   flow;
    	long        currentTimestamp = packet.getTimeStamp();


    	/*Problem 1: finishedFlows list continuously growing
    	* Solution: Flush the list periodically - say all 5000 flows - to CSV, and then clear finishedFlows list/

		/*Problem 2: currentFlows list continuously growing
		* Reason: After exceeding the timeout, a flow should be removed from the currentFlows list and added to the finishedFlows,
		* However, the timeout is ONLY checked when we see a packet that generates the same flowId (srcIP-dstIP-srcP-dst_P-prot) as the timedout packet..
		* Surely, there are some ports (mostly source ports) that we only see once in a long time --> these flows will stay forever in currentFLows, when
		* no FIN packet closed the flow...
		* Solution: parse periodically through the currentFlows... check if there are timedout flows, if yes add them to finishedFLows and then remove them from currentFlows*/
    	if(this.currentFlows.containsKey(packet.getFlowId())){
    		flow = currentFlows.get(packet.getFlowId());
    		// Flow finished due flowtimeout: 
    		// 1.- we move the flow to finished flow list
    		// 2.- we eliminate the flow from the current flow list
    		// 3.- we create a new flow with the packet-in-process
    		if((currentTimestamp -flow.getFlowStartTime())>flowTimeOut){
				if(this.singlePktFlows || flow.packetCount()>1){
					if (mListener != null) {
						mListener.onFlowGenerated(flow);
					}else{
						if(this.singlePktFlows || flow.packetCount()>1){
							finishedFlows.put(getFlowCount(), flow);
						}
					}
				}

				//flow.endActiveIdleTime(currentTimestamp,this.flowActivityTimeOut, this.flowTimeOut, false);

				timeoutFlowCount++;
    			currentFlows.remove(packet.getFlowId());    			
    			currentFlows.put(packet.getFlowId(), new BasicFlow(bidirectional,packet,flow.getSrc(),flow.getDst(),flow.getSrcPort(),flow.getDstPort(),intSubnets));
    			
    			int cfsize = currentFlows.size();
    			if(cfsize%50==0) {
    				logger.debug("Timeout current has {} flow",cfsize);
    	    	}
    			
        	// Flow finished due FIN flag (tcp only):
    		// 1.- we add the packet-in-process to the flow (it is the last packet)
        	// 2.- we move the flow to finished flow list
        	// 3.- we eliminate the flow from the current flow list   	
    		}else if(packet.hasFlagFIN()){
    	    	logger.debug("FlagFIN current has {} flow",currentFlows.size());
    	    	flow.addPacket(packet);
                if (mListener != null) {
                    mListener.onFlowGenerated(flow);
                } else {
					if(this.singlePktFlows || flow.packetCount()>1){
						finishedFlows.put(getFlowCount(), flow);
					}
				}
                currentFlows.remove(packet.getFlowId());
    		}else{
    			flow.updateActiveIdleTime(currentTimestamp,this.flowActivityTimeOut);
    			flow.addPacket(packet);
    			currentFlows.put(packet.getFlowId(), flow);
    		}
    	}else{
    		currentFlows.put(packet.getFlowId(), new BasicFlow(bidirectional,packet,intSubnets));
    	}
    }

	public int dumpFinishedFlows(String path, String filename, boolean writeHeader, boolean writeLastFlows){
		BasicFlow flow;

		try {
			//total = finishedFlows.size()+currentFlows.size(); becasue there are 0 packet BasicFlow in the currentFlows
			File csvFile = new File(path+filename);
			if(writeHeader){
				// If there exists already a csv with this name, delete it
				boolean deletedExistingFile = Files.deleteIfExists((new File(path+filename)).toPath());
				if(deletedExistingFile){
                    System.out.println("Deleted existing .csv");
                }
				Files.write(csvFile.toPath(), (FlowFeature.getHeader()+"\n").getBytes());
			}

			FileOutputStream output = new FileOutputStream(csvFile, true);
			logger.debug("dumpLabeledFlow: ", path + filename);

			Set<Integer> fkeys = this.finishedFlows.keySet();
			for(Integer key:fkeys){
				flow = this.finishedFlows.get(key);
				// Ignore flows that are generated after the first FIN packet (tcp)
				if((flow.getFlagCount("SYN")>0 && flow.getProtocol()==6) || flow.getProtocol()!=6){
					output.write((flow.dumpFlowBasedFeaturesEx() + "\n").getBytes());
				}
			}
			// clear the list to free up memory
			this.finishedFlows.clear();

			if(writeLastFlows){
				Set<String> ckeys = currentFlows.keySet();
				for(String key:ckeys){
					flow = currentFlows.get(key);
					if(this.singlePktFlows || flow.packetCount()>1){
                        if((flow.getFlagCount("SYN")>0 && flow.getProtocol()==6) || flow.getProtocol()!=6){
                            output.write((flow.dumpFlowBasedFeaturesEx() + "\n").getBytes());
                            finishedFlowCount++;
                            timeoutFlowCount++;
                        }
					}
				}
			}
			output.flush();
			output.close();
		} catch (IOException e) {
			e.printStackTrace();
		}

		System.out.println("Appended finished flows to CSV. Flows finished: " + Integer.toString(finishedFlowCount));

		return finishedFlowCount;
	}


	public void dumpFlowBasedFeatures(String path, String filename,String header){
    	
		BasicFlow   flow;
    	try {
    		System.out.println("TOTAL Flows: "+(finishedFlows.size()+currentFlows.size()));
    		FileOutputStream output = new FileOutputStream(new File(path+filename));    
    		
    		output.write((header+"\n").getBytes());
    		Set<Integer> fkeys = finishedFlows.keySet();    		
			for(Integer key:fkeys){
	    		flow = finishedFlows.get(key);
	    		if(flow.packetCount()>1)				
	    			output.write((flow.dumpFlowBasedFeaturesEx()+"\n").getBytes());
			}
			Set<String> ckeys = currentFlows.keySet();   		
			for(String key:ckeys){
	    		flow = currentFlows.get(key);
	    		if(flow.packetCount()>1)				
	    			output.write((flow.dumpFlowBasedFeaturesEx()+"\n").getBytes());
			}			
			
			output.flush();
			output.close();			
		} catch (IOException e) {
			e.printStackTrace();
		}
    }

    public int dumpLabeledFlowBasedFeatures(String path, String filename,String header){
    	BasicFlow   flow;
    	int         total = 0;
    	int   zeroPkt = 0;

    	try {
    		//total = finishedFlows.size()+currentFlows.size(); becasue there are 0 packet BasicFlow in the currentFlows

    		FileOutputStream output = new FileOutputStream(new File(path+filename));
			logger.debug("dumpLabeledFlow: ", path + filename);
    		output.write((header+"\n").getBytes());
    		Set<Integer> fkeys = finishedFlows.keySet();    		
			for(Integer key:fkeys){
	    		flow = finishedFlows.get(key);
                if (flow.packetCount() > 1) {
                    output.write((flow.dumpFlowBasedFeaturesEx() + "\n").getBytes());
                    total++;
                } else {
                    zeroPkt++;
                }
            }
            logger.debug("dumpLabeledFlow finishedFlows -> {},{}",zeroPkt,total);

            Set<String> ckeys = currentFlows.keySet();
			for(String key:ckeys){
	    		flow = currentFlows.get(key);
	    		if(flow.packetCount()>1) {
                    output.write((flow.dumpFlowBasedFeaturesEx() + "\n").getBytes());
                    total++;
                }else{
                    zeroPkt++;
                }

			}
            logger.debug("dumpLabeledFlow total(include current) -> {},{}",zeroPkt,total);
            output.flush();
            output.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

        return total;
    }       
    
    private int getFlowCount(){
    	this.finishedFlowCount++;
    	return this.finishedFlowCount;
    }
}
