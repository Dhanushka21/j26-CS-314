"""
C1 Function 1 - feature extraction.

Reads a PCAP, groups packets into fixed time windows per device, and emits the
45 flow features listed in contracts/features.json (exact order).

Device identity:
  - filename gives the device label (CICIoMT2024 profiling captures)
  - source MAC is recorded alongside so filename labels can be cross-checked

The five derived features (Magnitue, Radius, Covariance, Variance, Weight)
follow the CICIoT2023/CICIoMT2024 definitions as published; exact formulas are
not documented in features.json, so these are best-effort and flagged in
FEATURE_NOTES below for confirmation with C2's owner.
"""
from __future__ import annotations

import json
import math
import re
import statistics as st
from collections import Counter
from pathlib import Path

from scapy.all import ARP, ICMP, IP, TCP, UDP, Ether, PcapReader

_CONTRACTS = Path(__file__).resolve().parents[1] / "contracts"
FEATURES: list[str] = json.loads((_CONTRACTS / "features.json").read_text())["features"]
N_FEATURES = len(FEATURES)

FEATURE_NOTES = {
    "Magnitue": "sqrt(mean(inbound sizes) + mean(outbound sizes)) - CICIoT2023 definition",
    "Radius": "sqrt(var(inbound sizes) + var(outbound sizes))",
    "Covariance": "covariance between inbound and outbound size series (truncated to equal length)",
    "Variance": "var(inbound sizes) / var(outbound sizes), 0 when denominator is 0",
    "Weight": "count(inbound) * count(outbound)",
}

# well-known ports -> contract feature name
_PORT_FEATURES = {
    80: "HTTP", 8080: "HTTP",
    443: "HTTPS", 8443: "HTTPS",
    53: "DNS",
    23: "Telnet",
    25: "SMTP", 587: "SMTP",
    22: "SSH",
    6667: "IRC", 6697: "IRC",
    67: "DHCP", 68: "DHCP",
}


def device_from_filename(path: str | Path) -> str:
    """Device label from a CICIoMT2024 profiling filename.

    Blink_Camera_LAN_MIC.pcap -> Blink_Camera
    Multifunctional_Pager_WAN_APP.pcap -> Multifunctional_Pager
    SenseU_LAN_EMERGENCY.pcap -> SenseU
    """
    stem = Path(path).stem
    stem = re.sub(r"\s*\(\d+\)$", "", stem)  # strip " (1)" duplicates
    # activity suffixes appended by the capture tooling
    stem = re.sub(
        r"_(LAN|WAN)_[A-Z]+$|_(Power|Idle)$|_(MIC|WATCH|PHOTO|RECORDING|PRECORDING|APP|PHYSICAL|EMERGENCY)$",
        "",
        stem,
    )
    return stem


def activity_from_filename(path: str | Path) -> str:
    stem = Path(path).stem
    m = re.search(r"_(LAN|WAN)_([A-Z]+)$|_(Power|Idle)$", stem)
    return m.group(0).lstrip("_") if m else "unknown"


class _Window:
    """Accumulates packets for one time window and computes the 45 features."""

    __slots__ = ("t_start", "t_last", "sizes", "hdr_lens", "iats", "protos",
                 "flags", "proto_flags", "inbound", "outbound", "n", "src_macs")

    def __init__(self, t_start: float):
        self.t_start = t_start
        self.t_last = t_start
        self.sizes: list[int] = []
        self.hdr_lens: list[int] = []
        self.iats: list[float] = []
        self.protos: Counter = Counter()
        self.flags: Counter = Counter()
        self.proto_flags: set[str] = set()
        self.inbound: list[int] = []
        self.outbound: list[int] = []
        self.n = 0
        self.src_macs: Counter = Counter()

    def add(self, pkt, local_prefix: str = "192.168.") -> None:
        t = float(pkt.time)
        if self.n:
            self.iats.append(t - self.t_last)
        self.t_last = t
        self.n += 1
        size = len(pkt)
        self.sizes.append(size)

        if pkt.haslayer(Ether):
            self.src_macs[pkt[Ether].src] += 1

        hdr = 14  # ethernet
        if pkt.haslayer(IP):
            ip = pkt[IP]
            hdr += ip.ihl * 4 if ip.ihl else 20
            self.protos[ip.proto] += 1
            self.proto_flags.add("IPv")
            # direction relative to the local subnet
            if str(ip.src).startswith(local_prefix):
                self.outbound.append(size)
            else:
                self.inbound.append(size)
            if ip.proto == 1:
                self.proto_flags.add("ICMP")
            elif ip.proto == 2:
                self.proto_flags.add("IGMP")
        elif pkt.haslayer(ARP):
            self.proto_flags.add("ARP")
            self.outbound.append(size)
        else:
            self.proto_flags.add("LLC")
            self.outbound.append(size)

        if pkt.haslayer(TCP):
            tcp = pkt[TCP]
            hdr += tcp.dataofs * 4 if tcp.dataofs else 20
            self.proto_flags.add("TCP")
            f = int(tcp.flags)
            for bit, name in ((0x01, "fin"), (0x02, "syn"), (0x04, "rst"),
                              (0x08, "psh"), (0x10, "ack"), (0x40, "ece"), (0x80, "cwr")):
                if f & bit:
                    self.flags[name] += 1
            for port in (tcp.sport, tcp.dport):
                if port in _PORT_FEATURES:
                    self.proto_flags.add(_PORT_FEATURES[port])
        elif pkt.haslayer(UDP):
            hdr += 8
            self.proto_flags.add("UDP")
            udp = pkt[UDP]
            for port in (udp.sport, udp.dport):
                if port in _PORT_FEATURES:
                    self.proto_flags.add(_PORT_FEATURES[port])

        self.hdr_lens.append(hdr)

    # -- feature computation -------------------------------------------------
    def vector(self) -> list[float]:
        n = self.n
        dur = max(self.t_last - self.t_start, 1e-9)
        sizes = self.sizes or [0]

        mean_sz = st.mean(sizes)
        std_sz = st.pstdev(sizes) if len(sizes) > 1 else 0.0
        var_in = st.pvariance(self.inbound) if len(self.inbound) > 1 else 0.0
        var_out = st.pvariance(self.outbound) if len(self.outbound) > 1 else 0.0
        mean_in = st.mean(self.inbound) if self.inbound else 0.0
        mean_out = st.mean(self.outbound) if self.outbound else 0.0

        k = min(len(self.inbound), len(self.outbound))
        if k > 1:
            a, b = self.inbound[:k], self.outbound[:k]
            ma, mb = st.mean(a), st.mean(b)
            cov = sum((x - ma) * (y - mb) for x, y in zip(a, b)) / k
        else:
            cov = 0.0

        proto_type = self.protos.most_common(1)[0][0] if self.protos else 0.0
        has = lambda name: 1.0 if name in self.proto_flags else 0.0

        values = {
            "Header_Length": float(sum(self.hdr_lens)),
            "Protocol Type": float(proto_type),
            "Duration": float(dur),
            "Rate": n / dur,
            "Srate": len(self.outbound) / dur,
            "Drate": len(self.inbound) / dur,
            "fin_flag_number": 1.0 if self.flags["fin"] else 0.0,
            "syn_flag_number": 1.0 if self.flags["syn"] else 0.0,
            "rst_flag_number": 1.0 if self.flags["rst"] else 0.0,
            "psh_flag_number": 1.0 if self.flags["psh"] else 0.0,
            "ack_flag_number": 1.0 if self.flags["ack"] else 0.0,
            "ece_flag_number": 1.0 if self.flags["ece"] else 0.0,
            "cwr_flag_number": 1.0 if self.flags["cwr"] else 0.0,
            "ack_count": float(self.flags["ack"]),
            "syn_count": float(self.flags["syn"]),
            "fin_count": float(self.flags["fin"]),
            "rst_count": float(self.flags["rst"]),
            "HTTP": has("HTTP"), "HTTPS": has("HTTPS"), "DNS": has("DNS"),
            "Telnet": has("Telnet"), "SMTP": has("SMTP"), "SSH": has("SSH"),
            "IRC": has("IRC"), "TCP": has("TCP"), "UDP": has("UDP"),
            "DHCP": has("DHCP"), "ARP": has("ARP"), "ICMP": has("ICMP"),
            "IGMP": has("IGMP"), "IPv": has("IPv"), "LLC": has("LLC"),
            "Tot sum": float(sum(sizes)),
            "Min": float(min(sizes)),
            "Max": float(max(sizes)),
            "AVG": float(mean_sz),
            "Std": float(std_sz),
            "Tot size": float(mean_sz),
            "IAT": float(st.mean(self.iats)) if self.iats else 0.0,
            "Number": float(n),
            "Magnitue": math.sqrt(max(mean_in + mean_out, 0.0)),
            "Radius": math.sqrt(max(var_in + var_out, 0.0)),
            "Covariance": float(cov),
            "Variance": float(var_in / var_out) if var_out else 0.0,
            "Weight": float(len(self.inbound) * len(self.outbound)),
        }
        return [values[name] for name in FEATURES]

    def dominant_mac(self) -> str:
        return self.src_macs.most_common(1)[0][0] if self.src_macs else ""


def extract(pcap_path: str | Path, window_s: float = 30.0, stride_s: float = 15.0,
            min_packets: int = 5, limit: int | None = None) -> list[dict]:
    """Sliding-window feature extraction over one PCAP.

    Returns a list of row dicts: device, activity, t_start, n_packets,
    src_mac, and the 45 features keyed by contract name.
    """
    device = device_from_filename(pcap_path)
    activity = activity_from_filename(pcap_path)

    packets: list = []
    with PcapReader(str(pcap_path)) as reader:
        for i, pkt in enumerate(reader):
            packets.append(pkt)
            if limit and i + 1 >= limit:
                break
    if not packets:
        return []

    t0 = float(packets[0].time)
    t_end = float(packets[-1].time)
    rows: list[dict] = []
    start = t0
    idx = 0
    while start < t_end:
        stop = start + window_s
        win = _Window(start)
        j = idx
        while j < len(packets) and float(packets[j].time) < stop:
            if float(packets[j].time) >= start:
                win.add(packets[j])
            j += 1
        if win.n >= min_packets:
            row = {
                "device": device,
                "activity": activity,
                "t_start": start - t0,
                "n_packets": win.n,
                "src_mac": win.dominant_mac(),
            }
            row.update(dict(zip(FEATURES, win.vector())))
            rows.append(row)
        start += stride_s
        while idx < len(packets) and float(packets[idx].time) < start:
            idx += 1
    return rows
