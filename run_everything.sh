#!/bin/bash

# =========================
# Setup
# =========================
run "Setting up the file structure..." python3 setup_project.py
run "Installing Requirements..." pip3 install -r requirements.txt
run "Verifying Requirements Installation..." python3 verify_installation.py

echo "Starting pipeline..."

# =========================
# Phase 1
# =========================
run "" python3 run_phase1ab.py
beep

run "" python3 verify_phase1ab.py
beep

run "" python3 run_phase1c.py
beep

run "" python3 verify_phase1c.py
beep

run "" python3 run_phase1d.py
beep

run "" python3 verify_phase1d.py
beep

speak "Phase 1 complete"

# =========================
# Phase 2
# =========================
run "" python3 run_phase2a.py
beep

run "" python3 verify_phase2a.py
beep

run "" python3 run_phase2b.py
beep

run "" python3 verify_phase2b.py
beep

run "" python3 run_phase2c.py
beep

run "" python3 verify_phase2c.py
beep

run "" python3 run_phase2d.py
beep

run "" python3 verify_phase2d.py
beep

run "" python3 run_phase2e.py
beep

speak "Phase 2 complete"

# =========================
# Phase 3
# =========================
run "" python3 run_phase3a.py
beep

run "" python3 verify_phase3a.py
beep

run "" python3 run_phase3b.py
beep

run "" python3 verify_phase3b.py
beep

run "" python3 run_phase3c.py
beep

run "" python3 verify_phase3c.py
beep

speak "Phase 3 complete"

# =========================
# Phase 4+
# =========================
run "" python3 run_phase4.py
beep

run "" python3 verify_phase4.py
beep

speak "Phase 4 and 5 complete"

run "" python3 run_phase6.py
beep
speak "Phase 6 complete"

run "" python3 run_phase7.py
beep
speak "Phase 7 complete"

run "" python3 run_phase8.py
beep
speak "Phase 8 complete"

run "" python3 run_phase9.py
beep
speak "Phase 9 complete"

run "" python3 run_phase10.py
beep
speak "Phase 10 complete"

run "" python3 run_phase11.py
beep
speak "Phase 11 complete"

speak "Pipeline complete"

# =========================
# UI Prompt
# =========================
read -p "Do you want to launch the UI (run_ui.py)? (y/n): " choice

if [[ "$choice" == "y" ]]; then
    echo "Launching UI..."
    speak "Launching user interface"
    python3 run_ui.py
else
    echo "Skipping UI..."
    speak "Exiting without launching user interface"
fi

echo "Done."
read -p "Press enter to exit..."

# =========================
# FUNCTIONS
# =========================

run() {
    msg=$1
    shift

    if [[ "$msg" != "" ]]; then
        echo "$msg"
    fi

    "$@"
    if [[ $? -ne 0 ]]; then
        echo ""
        echo "ERROR: Step failed - stopping pipeline."
        beep_error
        speak "Error occurred. Pipeline stopped."
        exit 1
    fi
}

beep() {
    printf "\a"
}

beep_error() {
    printf "\a\a\a"
}

speak() {
    say "$1"
}
