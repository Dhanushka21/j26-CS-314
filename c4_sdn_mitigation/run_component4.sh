#!/bin/bash
# =============================================================
# run_component4.sh - Startup Script for Component 4
# A.A.N. Dulsara | IT23344006 | J-26-CS-314
# =============================================================

echo "============================================"
echo " Component 4: Adaptive SDN Mitigation"
echo " A.A.N. Dulsara | IT23344006"
echo "============================================"

# Activate virtual environment
source ~/ryu-venv/bin/activate

# Clean up any previous Mininet state
echo "[*] Cleaning up previous state..."
sudo mn -c 2>/dev/null

# Start Ryu controller in background
echo "[*] Starting Ryu SDN Controller..."
cd ~/component4
ryu-manager --wsapi-port 8080 controller/ryu_controller.py &
RYU_PID=$!
echo "[*] Ryu started (PID: $RYU_PID)"

# Wait for Ryu to be ready
sleep 4

# Start Mininet network
echo "[*] Starting Hospital IoMT Network..."
sudo python3 ~/j26-CS-314/c4_sdn_mitigation/network/hospital_network.py

# Cleanup on exit
kill $RYU_PID 2>/dev/null
echo "[*] Component 4 stopped."
