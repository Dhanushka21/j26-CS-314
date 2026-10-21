# Component 4: Adaptive SDN Mitigation with Self-Healing for IoMT Networks

**Student:** A.A.N. Dulsara | IT23344006
**Group:** J-26-CS-314 | SLIIT IT4010 Final Year Project

## Overview

Component 4 implements an intelligent SDN mitigation engine that receives security alerts from Component 3, classifies attacks using a trained ML model (Random Forest, 92% accuracy), and enforces real-time OpenFlow rules on a Mininet hospital IoMT network.

## ML Model Performance

| Model | Accuracy | F1 | CV Score | Gap |
|-------|----------|-----|---------|-----|
| Random Forest | 92.03% | 94.92% | 88.06% | 3.98% |
| Decision Tree | 85.74% | 90.28% | 82.78% | 2.96% |
| KNN | 97.78% | 98.05% | 92.38% | 5.40% |

Selected: Random Forest (best generalization)

## Hospital IoMT Topology

| Device | IP | Priority |
|--------|-----|----------|
| Ventilator | 10.0.0.1 | 1000 Critical |
| Insulin Pump | 10.0.0.2 | 990 Critical |
| Infusion Pump | 10.0.0.3 | 980 Critical |
| Cardiac Monitor | 10.0.0.4 | 970 High |
| BP Monitor | 10.0.0.5 | 960 High |
| Patient Server | 10.0.0.10 | 950 Medium |
| Admin Server | 10.0.0.11 | 940 Medium |
| Honeypot | 10.0.0.99 | 100 Trap |

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /api/v1/health | Health check |
| GET | /api/v1/status | Controller status |
| GET | /api/v1/incidents | All incidents |
| POST | /api/v1/alert | Receive C3 alert |
| POST | /api/v1/lateral | Receive C1 alert |

## Quick Start

Terminal 1 - Ryu Controller:
  source ~/ryu-venv/bin/activate && cd ~/component4
  ryu-manager --wsapi-port 8080 controller/ryu_controller.py

Terminal 2 - Mininet Network:
  sudo python3 network/hospital_network.py

Terminal 3 - Run Tests:
  python3 api/test_api.py

## Dataset

InSDN Dataset (Elsayed et al., 2020, IEEE Access)
343,889 records, 8 attack classes, 79 features
