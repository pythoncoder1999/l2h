# controller_client.py (hardened ssh + explicit remote command logging)
import subprocess
import threading
import time
import sys
import os
import socket

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
    "node2-18.grid.orbit-lab.org",
]

MAX_RETRIES = 3
WAIT_BETWEEN_RUNS = 10
WAIT_BETWEEN_RETRIES = 10

SCRIPT_PATH = "/l2h/CIFAR10L2H/run_client.py"
VENV_PYTHON = "/root/pytorch/bin/python3"

EXPERT_HOST = "node21-2.grid.orbit-lab.org"
EXPERT_PORT = 7000  # keep 7000
# Barrier server (controller-hosted)
BARRIER_PORT_BASE = 5500  # stable port for the per-run barrier


class BarrierServer:
    def __init__(self, host: str, port: int, expected: int, timeout: int = 300):
        self.host = host
        self.port = port
        self.expected = expected
        self.timeout = timeout
        self._srv = None
        self._conns = []

    def __enter__(self):
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind((self.host, self.port))
        self._srv.listen(1024)
        print(f"[barrier] Listening on {self.host}:{self.port} for {self.expected} clients")
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            for c in self._conns:
                try: c.close()
                except: pass
        finally:
            try: self._srv.close()
            except: pass

    def wait_and_release(self):
        self._srv.settimeout(1.0)
        start = time.time()
        while len(self._conns) < self.expected:
            if time.time() - start > self.timeout:
                raise TimeoutError(f"[barrier] Timeout waiting for {self.expected} clients (got {len(self._conns)})")
            try:
                c, addr = self._srv.accept()
                c.settimeout(10.0)
                data = c.recv(64)
                if not data or b"READY" not in data.upper():
                    c.close()
                else:
                    self._conns.append(c)
                    print(f"[barrier] +1 ready ({len(self._conns)}/{self.expected}) from {addr}")
            except socket.timeout:
                pass

        # All ready → broadcast GO
        msg = b"GO\n"
        for c in self._conns:
            try: c.sendall(msg)
            except: pass
        print("[barrier] GO sent to all")



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

    # Controller’s FQDN will be reachable by clients over SSH network
    barrier_host = subprocess.check_output(["hostname", "-f"], text=True).strip()
    barrier_port = BARRIER_PORT_BASE  # keep stable; change to +n if you want distinct ports per run

    # Soft fallback epoch start if a client can’t hit the barrier
    start_in = 10
    start_at = int(time.time()) + start_in
    print(f"⏱  Soft fallback start_at (epoch): {start_at}")
    print(f"⛓️  Barrier at {barrier_host}:{barrier_port} expecting {n} clients")

    with BarrierServer(host=barrier_host, port=barrier_port, expected=n, timeout=300) as barrier:
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
                    f"--barrier_host {barrier_host} --barrier_port {barrier_port} "
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

                print(f"[{node}] ▶ Launching run_client.py")
                p = subprocess.Popen(cmd_list, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                t = threading.Thread(target=stream_output, args=(node, p), daemon=True)
                t.start()
                processes.append((node, p))
                threads.append((node, t))

            # Wait until all clients report READY, then send GO
            try:
                barrier.wait_and_release()
            except Exception as e:
                print(f"🚨 Barrier failed: {e}")
                success = False

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




