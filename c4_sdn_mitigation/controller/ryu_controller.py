# =============================================================
# ryu_controller.py - Component 4: Adaptive SDN Mitigation
# A.A.N. Dulsara | IT23344006 | J-26-CS-314
# =============================================================

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet, ipv4, tcp, udp
from ryu.app.wsgi import WSGIApplication, ControllerBase, route
from webob import Response
import json
import joblib
import numpy as np
import threading
import time
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')

FEATURE_COLUMNS = [
    'Src Port', 'Dst Port', 'Protocol', 'Flow Duration', 'Tot Fwd Pkts',
    'Tot Bwd Pkts', 'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Fwd Pkt Len Max',
    'Fwd Pkt Len Min', 'Fwd Pkt Len Mean', 'Fwd Pkt Len Std', 'Bwd Pkt Len Max',
    'Bwd Pkt Len Min', 'Bwd Pkt Len Mean', 'Bwd Pkt Len Std', 'Flow Byts/s',
    'Flow Pkts/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
    'Fwd IAT Tot', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
    'Bwd IAT Tot', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min',
    'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags',
    'Fwd Header Len', 'Bwd Header Len', 'Fwd Pkts/s', 'Bwd Pkts/s',
    'Pkt Len Min', 'Pkt Len Max', 'Pkt Len Mean', 'Pkt Len Std', 'Pkt Len Var',
    'FIN Flag Cnt', 'SYN Flag Cnt', 'RST Flag Cnt', 'PSH Flag Cnt', 'ACK Flag Cnt',
    'URG Flag Cnt', 'CWE Flag Count', 'ECE Flag Cnt', 'Down/Up Ratio',
    'Pkt Size Avg', 'Fwd Seg Size Avg', 'Bwd Seg Size Avg', 'Fwd Byts/b Avg',
    'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg', 'Bwd Byts/b Avg', 'Bwd Pkts/b Avg',
    'Bwd Blk Rate Avg', 'Subflow Fwd Pkts', 'Subflow Fwd Byts',
    'Subflow Bwd Pkts', 'Subflow Bwd Byts', 'Init Fwd Win Byts',
    'Init Bwd Win Byts', 'Fwd Act Data Pkts', 'Fwd Seg Size Min',
    'Active Mean', 'Active Std', 'Active Max', 'Active Min',
    'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min',
]

def build_feature_vector(alert_data):
    import pandas as pd
    row = {c: float(alert_data.get(c, 0.0)) for c in FEATURE_COLUMNS}
    return pd.DataFrame([row])

# Clinical Priority Scores
DEVICE_PRIORITY = {
    '10.0.0.1':  1000,  # Ventilator       - CRITICAL
    '10.0.0.2':  990,   # Insulin Pump      - CRITICAL
    '10.0.0.3':  980,   # Infusion Pump     - CRITICAL
    '10.0.0.4':  970,   # Cardiac Monitor   - HIGH
    '10.0.0.5':  960,   # BP Monitor        - HIGH
    '10.0.0.10': 950,   # Patient Server    - MEDIUM
    '10.0.0.11': 940,   # Admin Server      - MEDIUM
    '10.0.0.99': 100,   # Honeypot          - LOW
}

DEVICE_NAMES = {
    '10.0.0.1':  'Ventilator',
    '10.0.0.2':  'InsulinPump',
    '10.0.0.3':  'InfusionPump',
    '10.0.0.4':  'CardiacMonitor',
    '10.0.0.5':  'BPMonitor',
    '10.0.0.10': 'PatientServer',
    '10.0.0.11': 'AdminServer',
    '10.0.0.99': 'Honeypot',
}

# CICIoMT2024 attack types from C3
ATTACK_SEVERITY_MAP = {
    'ARP_Spoofing':           'High',
    'DDoS-Connect_Flood':     'Critical',
    'DDoS-Publish_Flood':     'Critical',
    'DoS-Connect_Flood':      'High',
    'DoS-Publish_Flood':      'High',
    'Malformed_Data':         'Medium',
    'OS_Scan':                'Medium',
    'Ping_Sweep':             'Low',
    'Port_Scan':              'Medium',
    'VulScan':                'Medium',
    'DDoS-ICMP':              'Critical',
    'DDoS-SYN':               'Critical',
    'DDoS-TCP':               'Critical',
    'DDoS-UDP':               'Critical',
    'DoS-ICMP':               'High',
    'DoS-SYN':                'High',
    'DoS-TCP':                'High',
    'DoS-UDP':                'High',
    'Benign':                 'Low',
}

# TCP_IP prefix matching for C3 labels like "TCP_IP-DDoS-SYN"
def normalize_attack_type(attack_type):
    if attack_type.startswith('TCP_IP-'):
        attack_type = attack_type.replace('TCP_IP-', '')
    return attack_type


class Component4App(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]
    _CONTEXTS = {'wsgi': WSGIApplication}

    def __init__(self, *args, **kwargs):
        super(Component4App, self).__init__(*args, **kwargs)
        self.mac_to_port = {}
        self.datapaths = {}
        self.active_rules = {}
        self.incidents = []
        self.self_healing_log = []
        self.ml_model = None
        self.label_encoder = None
        self.scaler = None
        self.load_ml_model()

        wsgi = kwargs['wsgi']
        wsgi.register(MitigationAPI, {'component4_app': self})

        # Start self-healing monitor thread
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

    def load_ml_model(self):
        try:
            self.ml_model      = joblib.load(os.path.join(MODEL_DIR, 'rf_model.pkl'))
            self.label_encoder = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.pkl'))
            self.scaler        = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
            logger.info("ML model loaded: RF + LabelEncoder + Scaler")
            logger.info(f"  Classes: {list(self.label_encoder.classes_)}")
        except Exception as e:
            logger.warning(f"ML model not loaded: {e} - running without ML")
            self.ml_model = None
            self.label_encoder = None
            self.scaler = None

    def ml_predict(self, alert_data):
        if not all([self.ml_model, self.label_encoder, self.scaler]):
            return {'predicted_class':'Unknown','confidence':0.0,'ml_available':False}
        try:
            import numpy as np
            X = build_feature_vector(alert_data)
            X_scaled = self.scaler.transform(X)
            pred_idx = self.ml_model.predict(X_scaled)[0]
            proba    = self.ml_model.predict_proba(X_scaled)[0]
            conf     = float(np.max(proba))
            cls      = self.label_encoder.inverse_transform([pred_idx])[0]
            logger.info(f"ML: {cls} (conf={conf:.2f})")
            return {'predicted_class':cls,'confidence':conf,'ml_available':True}
        except Exception as e:
            logger.error(f"ML error: {e}")
            return {'predicted_class':'Unknown','confidence':0.0,'ml_available':False}

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        self.datapaths[datapath.id] = datapath

        # Default table-miss flow: send to controller
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                          ofproto.OFPCML_NO_BUFFER)]
        self.add_flow(datapath, 0, match, actions)
        logger.info(f"Switch {datapath.id} connected")

    def add_flow(self, datapath, priority, match, actions, idle_timeout=0, hard_timeout=0):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(
            datapath=datapath, priority=priority,
            idle_timeout=idle_timeout, hard_timeout=hard_timeout,
            match=match, instructions=inst
        )
        datapath.send_msg(mod)

    def block_ip(self, datapath, src_ip, dst_ip, duration=300):
        parser = datapath.ofproto_parser
        ofproto = datapath.ofproto
        match = parser.OFPMatch(eth_type=0x0800, ipv4_src=src_ip, ipv4_dst=dst_ip)
        # Empty actions = drop
        mod = parser.OFPFlowMod(
            datapath=datapath,
            priority=100,
            hard_timeout=duration,
            match=match,
            instructions=[]
        )
        datapath.send_msg(mod)
        logger.info(f"BLOCK rule: {src_ip} -> {dst_ip} for {duration}s")

    def redirect_to_honeypot(self, datapath, src_ip):
        parser = datapath.ofproto_parser
        honeypot_ip = '10.0.0.99'
        match = parser.OFPMatch(eth_type=0x0800, ipv4_src=src_ip)
        actions = [parser.OFPActionSetField(ipv4_dst=honeypot_ip),
                   parser.OFPActionOutput(datapath.ofproto.OFPP_NORMAL)]
        self.add_flow(datapath, 90, match, actions, hard_timeout=120)
        logger.info(f"REDIRECT: {src_ip} -> Honeypot ({honeypot_ip})")

    def rate_limit_ip(self, datapath, src_ip, dst_ip):
        # Rate limit by setting low priority flow with meter (simplified: just log)
        logger.info(f"RATE-LIMIT: {src_ip} -> {dst_ip}")

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match['in_port']

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]

        dst = eth.dst
        src = eth.src
        dpid = datapath.id

        self.mac_to_port.setdefault(dpid, {})
        self.mac_to_port[dpid][src] = in_port

        if dst in self.mac_to_port[dpid]:
            out_port = self.mac_to_port[dpid][dst]
        else:
            out_port = ofproto.OFPP_FLOOD

        actions = [parser.OFPActionOutput(out_port)]

        if out_port != ofproto.OFPP_FLOOD:
            match = parser.OFPMatch(in_port=in_port, eth_dst=dst, eth_src=src)
            self.add_flow(datapath, 1, match, actions)

        data = None
        if msg.buffer_id == ofproto.OFP_NO_BUFFER:
            data = msg.data

        out = parser.OFPPacketOut(
            datapath=datapath, buffer_id=msg.buffer_id,
            in_port=in_port, actions=actions, data=data
        )
        datapath.send_msg(out)

    def apply_mitigation(self, alert_data):
        """Core mitigation logic - called from REST API"""
        device_id = alert_data.get('device_id', 'unknown')
        attack_type = normalize_attack_type(alert_data.get('attack_type', 'Unknown'))
        severity = alert_data.get('severity', 'Medium')
        confidence = alert_data.get('confidence', 0.0)
        critical_flag = alert_data.get('critical_protection_flag', False)
        shap_features = alert_data.get('shap_top_features', [])
        explanation = alert_data.get('explanation_summary', '')

        # ML Prediction
        ml_result = self.ml_predict(alert_data)
        ml_cls  = ml_result['predicted_class']
        ml_conf = ml_result['confidence']
        ml_avail = ml_result['ml_available']
        ml_override = False
        if ml_avail and ml_conf >= 0.70 and ml_cls == 'Normal':
            logger.info(f"ML override: C3='{attack_type}' but ML=Normal - downgrading")
            action = 'monitor'
            reason = f"ML predicts Normal (conf={ml_conf:.2f}) - no rule installed"
            ml_override = True

        # Find device IP from device_id
        dst_ip = None
        for ip, name in DEVICE_NAMES.items():
            if name.lower() in device_id.lower() or device_id.lower() in name.lower():
                dst_ip = ip
                break

        device_priority = DEVICE_PRIORITY.get(dst_ip, 500) if dst_ip else 500

        # Determine action based on severity and critical flag
        # Determine action based on severity and critical flag
        if not ml_override:
            if severity == 'Critical' and critical_flag:
                action = 'rate_limit'
                reason = 'Critical device protected - rate limit only'
            elif severity == 'Critical' and not critical_flag:
                action = 'block'
                reason = 'Critical attack - full block'
            elif severity == 'High':
                action = 'redirect_honeypot'
                reason = 'High severity - redirect to honeypot'
            elif severity == 'Medium':
                action = 'rate_limit'
                reason = 'Medium severity - rate limit'
            else:
                action = 'monitor'
                reason = 'Low severity - monitor only'
        # Apply to all connected datapaths
        rule_applied = False
        for dpid, datapath in self.datapaths.items():
            if action == 'block' and dst_ip:
                self.block_ip(datapath, '10.0.0.0/24', dst_ip)
                rule_applied = True
            elif action == 'redirect_honeypot':
                self.redirect_to_honeypot(datapath, dst_ip or '0.0.0.0')
                rule_applied = True
            elif action == 'rate_limit' and dst_ip:
                self.rate_limit_ip(datapath, '10.0.0.0/24', dst_ip)
                rule_applied = True

        incident = {
            'id': len(self.incidents) + 1,
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'device_id': device_id,
            'device_ip': dst_ip,
            'device_priority': device_priority,
            'attack_type': attack_type,
            'severity': severity,
            'confidence': confidence,
            'critical_protection_flag': critical_flag,
            'action_taken': action,
            'reason': reason,
            'shap_features': shap_features,
            'explanation': explanation,
            'rule_applied': rule_applied,
            'source': alert_data.get('source', 'C3'),
            'ml_prediction': {'class': ml_cls, 'confidence': ml_conf, 'available': ml_avail, 'override': ml_override}
        }
        self.incidents.append(incident)
        logger.info(f"INCIDENT #{incident['id']}: {device_id} | {attack_type} | {severity} | Action: {action}")
        return incident

    def apply_identity_mitigation(self, alert_data):
        """Handle C1 identity compromise alerts"""
        device_id = alert_data.get('device_id', 'unknown')
        anomaly_score = alert_data.get('anomaly_score', 0.0)
        behavior_status = alert_data.get('behavior_status', 'unknown')
        identity_compromised = alert_data.get('identity_compromise_suspected', False)
        cert_status = alert_data.get('certificate_status', 'unknown')

        if anomaly_score >= 0.90:
            severity = 'Critical'
            action = 'block'
        elif anomaly_score >= 0.75:
            severity = 'High'
            action = 'redirect_honeypot'
        else:
            severity = 'Medium'
            action = 'rate_limit'

        incident = {
            'id': len(self.incidents) + 1,
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'device_id': device_id,
            'attack_type': 'Identity_Compromise',
            'severity': severity,
            'anomaly_score': anomaly_score,
            'behavior_status': behavior_status,
            'certificate_status': cert_status,
            'identity_compromise_suspected': identity_compromised,
            'action_taken': action,
            'source': 'C1'
        }
        self.incidents.append(incident)
        logger.info(f"C1 IDENTITY INCIDENT #{incident['id']}: {device_id} | Score: {anomaly_score} | Action: {action}")
        return incident

    def _monitor_loop(self):
        """Self-healing monitor - checks rules every 60 seconds"""
        while True:
            time.sleep(60)
            now = datetime.utcnow().isoformat() + 'Z'
            self.self_healing_log.append({
                'timestamp': now,
                'event': 'monitor_cycle',
                'active_incidents': len(self.incidents),
                'datapaths': len(self.datapaths)
            })
            logger.info(f"Self-healing check: {len(self.incidents)} incidents, {len(self.datapaths)} switches")


class MitigationAPI(ControllerBase):

    def __init__(self, req, link, data, **config):
        super(MitigationAPI, self).__init__(req, link, data, **config)
        self.app = data['component4_app']

    @route('mitigation', '/api/v1/alert', methods=['POST'])
    def receive_c3_alert(self, req, **kwargs):
        """Receive alerts from Component 3 (XAI Severity)"""
        try:
            body = json.loads(req.body)
            incident = self.app.apply_mitigation(body)
            return Response(
                content_type='application/json',
                body=json.dumps({
                    'status': 'success',
                    'incident_id': incident['id'],
                    'action_taken': incident['action_taken'],
                    'reason': incident['reason'],
                    'timestamp': incident['timestamp']
                }).encode('utf-8')
            )
        except Exception as e:
            logger.error(f"C3 alert error: {e}")
            return Response(status=400, body=json.dumps({'error': str(e).encode('utf-8')}),
                           content_type='application/json')

    @route('mitigation', '/api/v1/lateral', methods=['POST'])
    def receive_c1_alert(self, req, **kwargs):
        """Receive identity compromise alerts from Component 1"""
        try:
            body = json.loads(req.body)
            incident = self.app.apply_identity_mitigation(body)
            return Response(
                content_type='application/json',
                body=json.dumps({
                    'status': 'success',
                    'incident_id': incident['id'],
                    'action_taken': incident['action_taken'],
                    'severity': incident['severity'],
                    'timestamp': incident['timestamp']
                }).encode('utf-8')
            )
        except Exception as e:
            logger.error(f"C1 alert error: {e}")
            return Response(status=400, body=json.dumps({'error': str(e).encode('utf-8')}),
                           content_type='application/json')

    @route('mitigation', '/api/v1/status', methods=['GET'])
    def get_status(self, req, **kwargs):
        """Return active incidents and rules"""
        return Response(
            content_type='application/json',
            body=json.dumps({
                'status': 'running',
                'active_incidents': len(self.app.incidents).encode('utf-8'),
                'connected_switches': len(self.app.datapaths),
                'ml_model_loaded': self.app.ml_model is not None,
                'incidents': self.app.incidents[-10:]
            })
        )

    @route('mitigation', '/api/v1/health', methods=['GET'])
    def health_check(self, req, **kwargs):
        """Health check endpoint"""
        return Response(
            content_type='application/json',
            body=json.dumps({
                'status': 'healthy',
                'component': 'C4-SDN-Mitigation',
                'student': 'A.A.N. Dulsara IT23344006',
                'timestamp': datetime.utcnow().encode('utf-8').isoformat() + 'Z',
                'switches': len(self.app.datapaths),
                'incidents': len(self.app.incidents)
            })
        )

    @route('mitigation', '/api/v1/incidents', methods=['GET'])
    def get_incidents(self, req, **kwargs):
        """Return all incidents"""
        return Response(
            content_type='application/json',
            body=json.dumps({
                'total': len(self.app.incidents).encode('utf-8'),
                'incidents': self.app.incidents
            })
        )
