"""Severity taxonomy and the life-critical safety override.

The attack names below were checked against contracts/classes.json (a test enforces it).
The attack -> severity table is a PROPOSAL. Agree it with Shanthasiri and your supervisor.
"""
from typing import Optional, Tuple

from . import config
from .contract_loader import DEVICES

ATTACK_SEVERITY = {
    # Volumetric floods that can take a device or broker offline
    "TCP_IP-DDoS-ICMP": "Critical",
    "TCP_IP-DDoS-SYN": "Critical",
    "TCP_IP-DDoS-TCP": "Critical",
    "TCP_IP-DDoS-UDP": "Critical",
    "MQTT-DDoS-Connect_Flood": "Critical",
    "MQTT-DDoS-Publish_Flood": "Critical",
    # Single-source denial of service, spoofing, malformed data
    "TCP_IP-DoS-ICMP": "High",
    "TCP_IP-DoS-SYN": "High",
    "TCP_IP-DoS-TCP": "High",
    "TCP_IP-DoS-UDP": "High",
    "MQTT-DoS-Connect_Flood": "High",
    "MQTT-DoS-Publish_Flood": "High",
    "MQTT-Malformed_Data": "High",
    "ARP_Spoofing": "High",
    # Reconnaissance
    "Recon-VulScan": "Medium",
    "Recon-OS_Scan": "Medium",
    "Recon-Port_Scan": "Low",
    "Recon-Ping_Sweep": "Low",
}

SEVERITY_LEVELS = ["Low", "Medium", "High", "Critical"]
UNKNOWN_ATTACK_DEFAULT = "Medium"


def device_is_life_critical(device_id: str) -> bool:
    """Source of truth is contracts/devices.json (ventilator, insulin pump, infusion pump)."""
    dev = DEVICES.get(device_id)
    if dev is None:
        return config.UNKNOWN_DEVICE_IS_CRITICAL
    return bool(dev["life_critical"])


def rule_severity(attack_type: str, confidence: Optional[float] = None,
                  escalate_conf: Optional[float] = None) -> str:
    sev = ATTACK_SEVERITY.get(attack_type, UNKNOWN_ATTACK_DEFAULT)
    if escalate_conf is not None and confidence is not None and sev == "High" and confidence >= escalate_conf:
        sev = "Critical"
    return sev


def apply_override(severity: str, life_critical: bool) -> Tuple[str, bool]:
    """The fail-safe. Runs AFTER any model and cannot be skipped.
    Returns (final_severity, critical_protection_flag)."""
    if life_critical:
        return "Critical", True
    return severity, False
