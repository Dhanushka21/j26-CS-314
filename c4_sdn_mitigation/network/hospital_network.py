# =============================================================
# hospital_network.py - Component 4: IoMT Hospital Topology
# A.A.N. Dulsara | IT23344006 | J-26-CS-314
# =============================================================

from mininet.net import Mininet
from mininet.node import Controller, RemoteController, OVSKernelSwitch
from mininet.cli import CLI
from mininet.log import setLogLevel, info
from mininet.link import TCLink
import time

def create_hospital_network():
    setLogLevel('info')

    net = Mininet(
        controller=RemoteController,
        switch=OVSKernelSwitch,
        link=TCLink,
        autoSetMacs=True
    )

    info('*** Adding Ryu Remote Controller\n')
    c0 = net.addController(
        'c0',
        controller=RemoteController,
        ip='127.0.0.1',
        port=6633
    )

    info('*** Adding OpenFlow Switch\n')
    s1 = net.addSwitch('s1', protocols='OpenFlow13')

    info('*** Adding IoMT Hospital Devices\n')
    # Critical devices (P=1000-980)
    venti          = net.addHost('venti',     ip='10.0.0.1/24',  mac='00:00:00:00:00:01')
    insulin_p      = net.addHost('insulin_p',      ip='10.0.0.2/24',  mac='00:00:00:00:00:02')
    infusion_p     = net.addHost('infusion_p',     ip='10.0.0.3/24',  mac='00:00:00:00:00:03')

    # High priority devices (P=970-960)
    cardiac        = net.addHost('cardiac',    ip='10.0.0.4/24',  mac='00:00:00:00:00:04')
    bp_mon         = net.addHost('bp_mon',     ip='10.0.0.5/24',  mac='00:00:00:00:00:05')

    # Medium priority devices (P=950-940)
    patsrv    = net.addHost('patsrv',    ip='10.0.0.10/24', mac='00:00:00:00:00:0a')
    admsrv      = net.addHost('admsrv',      ip='10.0.0.11/24', mac='00:00:00:00:00:0b')

    # Honeypot (P=100)
    honey          = net.addHost('honey',       ip='10.0.0.99/24', mac='00:00:00:00:00:63')

    info('*** Creating links\n')
    net.addLink(venti,     s1, bw=10)
    net.addLink(insulin_p,   s1, bw=10)
    net.addLink(infusion_p,  s1, bw=10)
    net.addLink(cardiac,    s1, bw=10)
    net.addLink(bp_mon,     s1, bw=10)
    net.addLink(patsrv, s1, bw=100)
    net.addLink(admsrv,   s1, bw=100)
    net.addLink(honey,       s1, bw=10)

    info('*** Starting network\n')
    net.build()
    c0.start()
    s1.start([c0])

    info('*** Waiting for controller connection\n')
    time.sleep(3)

    info('*** Network ready\n')
    info('*** Devices:\n')
    info('    Ventilator:     10.0.0.1  (Priority 1000)\n')
    info('    Insulin Pump:   10.0.0.2  (Priority 990)\n')
    info('    Infusion Pump:  10.0.0.3  (Priority 980)\n')
    info('    Cardiac Mon:    10.0.0.4  (Priority 970)\n')
    info('    BP Monitor:     10.0.0.5  (Priority 960)\n')
    info('    Patient Server: 10.0.0.10 (Priority 950)\n')
    info('    Admin Server:   10.0.0.11 (Priority 940)\n')
    info('    Honeypot:       10.0.0.99 (Priority 100)\n')

    CLI(net)
    net.stop()

if __name__ == '__main__':
    create_hospital_network()
