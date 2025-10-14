# controller_client.py (hardened ssh + explicit remote command logging)
import subprocess
import threading
import time
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
log_path = os.path.join(BASE_DIR, "controller_output.log")
os.makedirs(os.path.dirname(log_path), exist_ok=True)
sys.stdout = open(log_path, "w", buffering=1)
sys.stderr = sys.stdout

print("✅ controller_client.py started")
print("cwd:", BASE_DIR)

ALL_CLIENTS = [
    "node1-1.grid.orbit-lab.org",
    "node1-2.grid.orbit-lab.org",
    "node1-20.grid.orbit-lab.org",
    "node3-1.grid.orbit-lab.org", 
    #"node1-8.grid.orbit-lab.org",
    "node1-10.grid.orbit-lab.org",
    "node2-1.grid.orbit-lab.org",
    "node2-2.grid.orbit-lab.org",
    "node2-3.grid.orbit-lab.org",
    "node2-4.grid.orbit-lab.org",
    "node2-6.grid.orbit-lab.org",
    "node2-20.grid.orbit-lab.org",
    "node2-8.grid.orbit-lab.org",
    "node2-12.grid.orbit-lab.org",
    "node2-13.grid.orbit-lab.org",
    "node2-14.grid.orbit-lab.org",
    "node2-15.grid.orbit-lab.org",
    "node2-16.grid.orbit-lab.org",
    "node2-17.grid.orbit-lab.org",
    "node13-1.grid.orbit-lab.org",
]

MAX_RETRIES = 3
WAIT_BETWEEN_RUNS = 10
WAIT_BETWEEN_RETRIES = 10

SCRIPT_PATH = "/l2h/CIFAR10L2H/run_client.py"
VENV_PYTHON = "/root/pytorch/bin/python3"

EXPERT_HOST = "node21-2.grid.orbit-lab.org"
EXPERT_PORT = 7000  # keep 7000

def stream_output(node, process):
    try:
        for line in process.stdout:
            print(f"[{node}] {line.strip()}")
    except Exception as e:
        print(f"[{node}] ❗ stream error: {e}")
    finally:
        try:
            process.stdout.close()
        except Exception:
            pass

NUM_TESTS_TOTAL = len(ALL_CLIENTS)  # 19

for n in range(1, NUM_TESTS_TOTAL + 1):
    nodes_to_use = ALL_CLIENTS[:n]
    print(f"\n🚀 Running with {n} client(s): {nodes_to_use}")

    # MINIMAL: give slower nodes more time to reach the barrier
    start_at = int(time.time()) + 30  # was +10
    print(f"⏱  Barrier start_at (epoch): {start_at}")

    for attempt in range(1, MAX_RETRIES + 1):
        print(f"🔁 Attempt {attempt} for {n} clients")
        processes, threads = [], []
        success = True

        for i, node in enumerate(nodes_to_use):
            primary_flag = "--primary_client" if i == 0 else ""
            client_count_flag = f"--num_clients {n}" if i == 0 else ""

            remote_exec = (
                f"cd /l2h/CIFAR10L2H && "
                f"exec {VENV_PYTHON} -u {SCRIPT_PATH} "
                f"--host {EXPERT_HOST} --port {EXPERT_PORT} "
                f"--start_at {start_at} --num_tests_total {NUM_TESTS_TOTAL} "
                f"{primary_flag} {client_count_flag}"
            ).strip()

            print(f"[{node}] REMOTE CMD: {remote_exec}")

            cmd_list = [
                "ssh",
                "-o", "BatchMode=yes",
                "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null",
                "-o", "ConnectTimeout=8",
                "-o", "ServerAliveInterval=5",
                "-o", "ServerAliveCountMax=2",
                f"root@{node}",
                "bash", "-lc", remote_exec
            ]

            print(f"[{node}] ▶ Launching clientV3.py")
            p = subprocess.Popen(cmd_list, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            t = threading.Thread(target=stream_output, args=(node, p), daemon=True)
            t.start()
            processes.append((node, p))
            threads.append((node, t))

        for node, thread in threads:
            thread.join()

        for node, proc in processes:
            ret = proc.wait()
            if ret != 0:
                print(f"❌ {node} exited with code {ret}")
                success = False

        if success:
            print(f"✅ Run with {n} client(s) complete.\n")
            break
        elif attempt == MAX_RETRIES:
            print(f"🚨 Giving up after {MAX_RETRIES} attempts for {n} clients\n")
        else:
            print(f"🔁 Retrying {n} clients after delay...\n")
            time.sleep(WAIT_BETWEEN_RETRIES)

    time.sleep(WAIT_BETWEEN_RUNS)



'''
# controller_client.py (hardened ssh + explicit remote command logging)
import subprocess
import threading
import time
import sys
import os

# Log to a file next to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
log_path = os.path.join(BASE_DIR, "controller_output.log")
os.makedirs(os.path.dirname(log_path), exist_ok=True)
sys.stdout = open(log_path, "w", buffering=1)
sys.stderr = sys.stdout

print("✅ controller_client.py started")
print("cwd:", BASE_DIR)

ALL_CLIENTS = [
    "node1-1.grid.orbit-lab.org",
    "node1-2.grid.orbit-lab.org",
    "node1-4.grid.orbit-lab.org",
    "node1-8.grid.orbit-lab.org",
    "node1-10.grid.orbit-lab.org",
    "node2-1.grid.orbit-lab.org",
    "node2-2.grid.orbit-lab.org",
    "node2-3.grid.orbit-lab.org",
    "node2-4.grid.orbit-lab.org",
    "node2-6.grid.orbit-lab.org",
    "node2-7.grid.orbit-lab.org",
    "node2-8.grid.orbit-lab.org",
    "node2-12.grid.orbit-lab.org",
    "node2-13.grid.orbit-lab.org",
    "node2-14.grid.orbit-lab.org",
    "node2-15.grid.orbit-lab.org",
    "node2-16.grid.orbit-lab.org",
    "node2-17.grid.orbit-lab.org",
    "node2-18.grid.orbit-lab.org",
]

MAX_RETRIES = 3
WAIT_BETWEEN_RUNS = 10
WAIT_BETWEEN_RETRIES = 10

SCRIPT_PATH = "/l2h/CIFAR10L2H/clientV3.py"
VENV_PYTHON = "/root/pytorch/bin/python3"

EXPERT_HOST = "node21-2.grid.orbit-lab.org"
EXPERT_PORT = 7000

def stream_output(node, process):
    try:
        for line in process.stdout:
            print(f"[{node}] {line.strip()}")
    except Exception as e:
        print(f"[{node}] ❗ stream error: {e}")
    finally:
        try:
            process.stdout.close()
        except Exception:
            pass

NUM_TESTS_TOTAL = len(ALL_CLIENTS)  # 19

for n in range(1, NUM_TESTS_TOTAL + 1):
    nodes_to_use = ALL_CLIENTS[:n]
    print(f"\n🚀 Running with {n} client(s): {nodes_to_use}")

    start_at = int(time.time()) + 10  # sync start ~10s ahead

    for attempt in range(1, MAX_RETRIES + 1):
        print(f"🔁 Attempt {attempt} for {n} clients")
        processes, threads = [], []
        success = True

        for i, node in enumerate(nodes_to_use):
            primary_flag = "--primary_client" if i == 0 else ""
            client_count_flag = f"--num_clients {n}" if i == 0 else ""

            remote_exec = (
                f"cd /l2h/CIFAR10L2H && "
                f"exec {VENV_PYTHON} -u {SCRIPT_PATH} "
                f"--host {EXPERT_HOST} --port {EXPERT_PORT} "
                f"--start_at {start_at} --num_tests_total {NUM_TESTS_TOTAL} "
                f"{primary_flag} {client_count_flag}"
            ).strip()

            print(f"[{node}] REMOTE CMD: {remote_exec}")

            cmd_list = [
                "ssh",
                "-o", "BatchMode=yes",
                "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null",
                "-o", "ConnectTimeout=8",
                "-o", "ServerAliveInterval=5",
                "-o", "ServerAliveCountMax=2",
                f"root@{node}",
                "bash", "-lc", remote_exec
            ]

            print(f"[{node}] ▶ Launching clientV3.py")
            p = subprocess.Popen(cmd_list, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            t = threading.Thread(target=stream_output, args=(node, p), daemon=True)
            t.start()
            processes.append((node, p))
            threads.append((node, t))

        for node, thread in threads:
            thread.join()

        for node, proc in processes:
            ret = proc.wait()
            if ret != 0:
                print(f"❌ {node} exited with code {ret}")
                success = False

        if success:
            print(f"✅ Run with {n} client(s) complete.\n")
            break
        elif attempt == MAX_RETRIES:
            print(f"🚨 Giving up after {MAX_RETRIES} attempts for {n} clients\n")
        else:
            print(f"🔁 Retrying {n} clients after delay...\n")
            time.sleep(WAIT_BETWEEN_RETRIES)

    time.sleep(WAIT_BETWEEN_RUNS)
    '''




