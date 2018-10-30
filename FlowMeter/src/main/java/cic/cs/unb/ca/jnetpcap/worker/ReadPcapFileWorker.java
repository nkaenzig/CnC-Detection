package cic.cs.unb.ca.jnetpcap.worker;

import cic.cs.unb.ca.jnetpcap.BasicPacketInfo;
import cic.cs.unb.ca.jnetpcap.FlowFeature;
import cic.cs.unb.ca.jnetpcap.FlowGenerator;
import cic.cs.unb.ca.jnetpcap.PacketReader;
import org.jnetpcap.Pcap;
import org.jnetpcap.PcapHeader;
import org.jnetpcap.nio.JBuffer;
import org.jnetpcap.packet.PcapPacket;
import org.jnetpcap.protocol.lan.Ethernet;
import org.jnetpcap.protocol.network.Ip4;
import org.jnetpcap.protocol.network.Ip6;
import org.jnetpcap.nio.JMemory;
import org.apache.commons.io.FilenameUtils;
import org.jnetpcap.PcapClosedException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import swing.common.SwingUtils;

import javax.swing.*;
import java.io.File;
import java.io.FileNotFoundException;
import java.nio.file.Files;
import java.io.IOException;
import java.nio.file.StandardOpenOption;
import java.text.DateFormat;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.*;

import static cic.cs.unb.ca.Sys.FILE_SEP;


public class ReadPcapFileWorker extends SwingWorker<List<String>,String>{

    public static final Logger logger = LoggerFactory.getLogger(ReadPcapFileWorker.class);
    public static final String PROPERTY_FILE_CNT = "file_count";
    public static final String PROPERTY_CUR_FILE = "file_current";
    private static final String DividingLine = "----------------------------------------------------------------------------";

    private PacketReader    packetReader;
    private BasicPacketInfo basicPacket = null;
    private FlowGenerator   flowGen; //15000 useconds = 15ms.///////////////8
    private long flowTimeout;
    private long activityTimeout;
    private String intSubnets;
    private Date startTime = null;
    private Date endTime = null;
    private boolean readIP6 = false;
    private boolean readIP4 = true;
    private int     totalFlows = 0;
    
    private File pcapPath;
    private String outPutDirectory;
    private List<String> chunks;

    public ReadPcapFileWorker(File inputFile, String outPutDir) {
        super();
        pcapPath = inputFile;
        outPutDirectory = outPutDir;
        chunks = new ArrayList<>();

        if(!outPutDirectory.endsWith(FILE_SEP)) {
            outPutDirectory = outPutDirectory + FILE_SEP;
        }
        flowTimeout = 120000000L;
        activityTimeout = 5000000L;
    }

    public ReadPcapFileWorker(File inputFile, String outPutDir, long param1, long param2, String param3, String param4, String param5) {
        super();
        pcapPath = inputFile;
        outPutDirectory = outPutDir;
        chunks = new ArrayList<>();

        if(!outPutDirectory.endsWith(FILE_SEP)) {
            outPutDirectory = outPutDirectory + FILE_SEP;
        }
        flowTimeout = param1;
        activityTimeout = param2;
        intSubnets = param5;

        try {
            if (!param3.equals("")) {
                DateFormat format = new SimpleDateFormat("h:mm dd/MM/yy", Locale.ENGLISH);
                startTime = format.parse(param3);
//                System.out.println(startTime);
            }
            if (!param4.equals("")) {
                DateFormat format = new SimpleDateFormat("h:mm dd/MM/yy", Locale.ENGLISH);
                endTime = format.parse(param4);
//                System.out.println(endTime);
            }
        } catch (ParseException e) {
            e.printStackTrace();
        }
    }

    @Override
    protected List<String> doInBackground() {

        if (pcapPath.isDirectory()) {
            readPcapDir(pcapPath,outPutDirectory);
        } else {

            if (!SwingUtils.isPcapFile(pcapPath)) {
                publish("Please select pcap file!");
                publish("");
            } else {
                publish("CICFlowMeter received 1 pcap file");
                publish("");
                publish("");
                readPcapFile(pcapPath.getPath(), outPutDirectory);
            }
        }
        chunks.clear();
        chunks.add("");
        chunks.add(DividingLine);
        chunks.add(String.format("TOTAL FLOWS GENERATED :%s", totalFlows));
        chunks.add(DividingLine);
        publish(chunks.toArray( new String[chunks.size()]));

        return chunks;
    }

    @Override
    protected void done() {
        super.done();
    }

    @Override
    protected void process(List<String> chunks) {
        super.process(chunks);
        firePropertyChange("progress","",chunks);
    }

    private Date getTimestampOfFirstPacket(File inputPath) {
        StringBuilder errbuf = new StringBuilder();
        Pcap pcap = Pcap.openOffline(inputPath.toString(), errbuf);

        if (pcap == null) {
            System.err.printf("Error while opening file for capture: " + errbuf.toString());
            return new Date();
        }

        Ip4 ipv4 = new Ip4();
        Ip6 ipv6 = new Ip6();

        PcapHeader hdr = new PcapHeader(JMemory.POINTER);
        JBuffer buf = new JBuffer(JMemory.POINTER);

        if(pcap.nextEx(hdr,buf) == Pcap.NEXT_EX_OK) {
            PcapPacket packet = new PcapPacket(hdr, buf);
            packet.scan(Ethernet.ID);
            pcap.close();
            return new Date(packet.getCaptureHeader().timestampInMillis());
        } else {
            pcap.close();
            return new Date();
        }


    }
    
    private void readPcapDir(File inputPath, String outPath) {
        if(inputPath==null||outPath==null) {
            return;
        }

        //File[] pcapFiles = inputPath.listFiles(file -> file.getName().toLowerCase().endsWith("pcap"));
        File[] pcapFiles = inputPath.listFiles(file -> SwingUtils.isPcapFile(file));
        try {
            File processed = new File(outPath+"processed_files.log"); // contains a list of all .pcaps we have already processed in a past run
            List<String> processedFiles = new ArrayList<String>();

            if (processed.createNewFile()){
                System.out.println("New processed_files.log created");
            } else {
//                System.out.println("File already exists.");
                Scanner scanner = new Scanner(processed);
                while (scanner.hasNext()) {
                    processedFiles.add(scanner.next());
                }
                scanner.close();
                }

            int file_cnt = pcapFiles.length;
            logger.debug("CICFlowMeter found :{} pcap files", file_cnt);
            publish(String.format("CICFlowMeter found :%s pcap files", file_cnt));
            publish("");
            publish("");

            for(int i=0;i<file_cnt;i++) {
                File file = pcapFiles[i];

                if (file.isDirectory()) {
                    continue;
                } else if (processedFiles.contains(file.toString())) {
                    continue;
                }
                Date firstTimeStamp = getTimestampOfFirstPacket(file);
                if (startTime != null) {
                    if (startTime.after(firstTimeStamp)) {
                        continue;
                    }
                }
                if (endTime != null) {
                    if (endTime.before(firstTimeStamp)) {
                        continue;
                    }
                }
                firePropertyChange(PROPERTY_CUR_FILE, "", String.format("Reading %s ...", file.getName()));
                firePropertyChange(PROPERTY_FILE_CNT, file_cnt, i + 1);//begin with 1
                String pcap_path = file.getPath();
                readPcapFile(pcap_path, outPath);

                Files.write(processed.toPath(), (pcap_path + "\n").getBytes(), StandardOpenOption.APPEND);

            }
        } catch (IOException e1) {
            e1.printStackTrace();
        }

    }
    
    private void readPcapFile(String inputFile, String outPath) {

        if(inputFile==null ||outPath==null ) {
            return;
        }
        
        String fullname = FilenameUtils.getName(inputFile);

        boolean singlePktFlows = false;
        flowGen = new FlowGenerator(true,flowTimeout, activityTimeout, singlePktFlows, intSubnets);
        packetReader = new PacketReader(inputFile,readIP4,readIP6);
        publish(String.format("Working on... %s",inputFile));
        logger.debug("Working on... {}",inputFile);

        int nValid=0;
        int nTotal=0;
        int nDiscarded = 0;
        long start = System.currentTimeMillis();
//        int flush_period = 1000000;
        int flush_period = 500000;
        boolean writeHeader = true;
        while(true) {
            try{
                basicPacket = packetReader.nextPacket();
                nTotal++;
                if(basicPacket!=null){
                    flowGen.addPacket(basicPacket);
                    nValid++;

                    if(nValid%flush_period == 0){
                        flowGen.close_timed_out_flows(basicPacket);
                        flowGen.dumpFinishedFlows(outPath, fullname+"_Flow.csv", writeHeader, false);
                        System.out.println("Total processed packets: " + Integer.toString(nTotal));
                        System.out.println("Valid processed packets: " + Integer.toString(nValid));
                        writeHeader = false; // only write header the first time
                    }
                }else{
                    nDiscarded++;
                }
            }catch(PcapClosedException e){
                System.out.println(e);
                System.out.println(".Pcap closed");
                break;
            }catch (Exception e) {
                e.printStackTrace();
//                System.out.println(e);
                break;
            }
        }
        long end = System.currentTimeMillis();
        chunks.clear();
        chunks.add(String.format("Done! in %d seconds",((end-start)/1000)));
        chunks.add(String.format("\t Total packets: %d",nTotal));
        chunks.add(String.format("\t Valid packets: %d",nValid));
        chunks.add(String.format("\t Ignored packets:%d %d ", nDiscarded,(nTotal-nValid)));
        chunks.add(String.format("PCAP duration %d seconds",((packetReader.getLastPacket()-packetReader.getFirstPacket())/1000)));
        chunks.add(DividingLine);
//        int singleTotal = flowGen.dumpLabeledFlowBasedFeatures(outPath, fullname+"_Flow.csv", FlowFeature.getHeader());
        int singleTotal = flowGen.dumpFinishedFlows(outPath, fullname+"_Flow.csv", writeHeader, true);
        chunks.add(String.format("Number of Flows: %d",singleTotal));
        chunks.add("");
        publish(chunks.toArray( new String[chunks.size()]));
        totalFlows += singleTotal;

        logger.debug("{} is done,Total {}",inputFile,singleTotal);
    }
}
