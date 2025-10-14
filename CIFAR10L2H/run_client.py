# clientV3.py (client-side normalization; expert RTT includes serialization; latencies exclude rejector & normalization)
import json
import os, sys, io, socket, struct, time, argparse, pickle
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import Subset, DataLoader

# ── Force outputs into this repo dir ─────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

# ── Headless plotting backend (for savefig) ──────────────────────────────────
import matplotlib
matplotlib.use("Agg")

# ── Args ────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--primary_client', action='store_true')
parser.add_argument('--num_clients', type=int, default=None)
parser.add_argument('--num_tests_total', type=int, default=None)
parser.add_argument('--host', type=str, default='node21-2.grid.orbit-lab.org')
parser.add_argument('--port', type=int, default=7000)  # Go proxy
parser.add_argument('--start_at', type=int, default=None)  # epoch seconds barrier
parser.add_argument('--conn_timeout', type=float, default=30.0)
parser.add_argument('--retries', type=int, default=3)
parser.add_argument('--thresholds', type=str, default='0.325')  # comma-separated
parser.add_argument('--barrier_host', type=str, default=None)
parser.add_argument('--barrier_port', type=int, default=None)
args = parser.parse_args()

print("[clientV3] starting")
print("[clientV3] cwd:", os.getcwd())
print("[clientV3] args:", vars(args))

# ── Files / device ──────────────────────────────────────────────────────────
REJECTOR_PATH = 'RejectorModels/rejector_70.pt'
CLIENT_MODEL_PATH = 'ClientModels/client_model70.pt'
device = "cuda" if torch.cuda.is_available() else "cpu"
print("[clientV3] device:", device)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
# ── Models ──────────────────────────────────────────────────────────────────
class Client(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*8*8, 128), nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

class Rejector(nn.Module):
    def __init__(self, hidden=256, drop=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(3 * 32 * 32, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Linear(hidden, 2)
        )
    def forward(self, x):
        x = x.view(x.size(0), -1)  # [B, 3072]
        return self.block(x)

# ── Data (deterministic 100/class subset = 1000 imgs) ───────────────────────
print("[clientV3] preparing CIFAR-10 test set (may download once)...")
try:
    raw_testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=T.ToTensor()
    )
except Exception as e:
    print("[clientV3] CIFAR download failed, retrying with download=False:", e, flush=True)
    raw_testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=False, transform=T.ToTensor()
    )

targets = np.array(raw_testset.targets)
desired_per_class = 100
rng = np.random.RandomState(42)
class_indices = {cls: np.where(targets == cls)[0] for cls in np.unique(targets)}
chosen_indices = []
for cls, idxs in class_indices.items():
    if len(idxs) < desired_per_class:
        raise ValueError(f"Not enough samples in class {cls}")
    chosen_indices.extend(rng.choice(idxs, size=desired_per_class, replace=False))
stratified_subset = Subset(raw_testset, chosen_indices)
testloader = DataLoader(stratified_subset, batch_size=1, shuffle=False)
print("[clientV3] dataset ready:", len(stratified_subset), "images")

# ── Load checkpoints ────────────────────────────────────────────────────────
client_model = Client().to(device)
client_model.load_state_dict(torch.load(CLIENT_MODEL_PATH, map_location=device))
client_model.eval()
print("[clientV3] loaded client model:", CLIENT_MODEL_PATH)

rejector = Rejector().to(device)
rejector.load_state_dict(torch.load(REJECTOR_PATH, map_location=device))
rejector.eval()
print("[clientV3] loaded rejector:", REJECTOR_PATH)

# ── Net helpers ─────────────────────────────────────────────────────────────
def recvall(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise RuntimeError("Connection closed prematurely")
        buf.extend(chunk)
    return bytes(buf)

def request_batch_expert(batch_cpu, rejector_logits, host, port, timeout, retries):
    """Serializes batch_cpu (already normalized), sends with rejector logits. Returns (logits, T2, T3, client_count, server_cps)."""
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            with socket.create_connection((host, port), timeout=timeout) as sock:
                sock.settimeout(timeout)
                try: sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                except OSError: pass
                try: sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                except OSError: pass

                # Command
                sock.sendall(b'B')

                # Serialize CPU tensor (INCLUDED in RTT)
                buf = io.BytesIO()
                torch.save(batch_cpu, buf)
                batch_data = buf.getvalue()

                # Rejector logits as explicit little-endian float32 (INCLUDED in RTT packing)
                rej_data = rejector_logits.detach().cpu().numpy().astype('<f4').tobytes()

                # Send lengths + payload
                sock.sendall(struct.pack('>II', len(batch_data), len(rej_data)))
                sock.sendall(batch_data + rej_data)

                # Read response
                raw = recvall(sock, 4)
                plen = int.from_bytes(raw, 'big')
                pdata = recvall(sock, plen)

            # Unpack
            T2 = struct.unpack('>d', pdata[:8])[0]
            T3 = struct.unpack('>d', pdata[8:16])[0]
            client_count = struct.unpack('>I', pdata[16:20])[0]
            server_cps = struct.unpack('>f', pdata[20:24])[0]
            logits_data = pdata[24:]

            # Logits payload is float32 (little-endian)
            vals = struct.unpack('<' + 'f' * (len(logits_data)//4), logits_data)
            logits = torch.tensor(vals, dtype=torch.float32).view(batch_cpu.size(0), 10)
            return logits, T2, T3, client_count, server_cps

        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(0.5 * attempt)
            else:
                print(f"[clientV3] expert request failed after {retries} tries: {e}", flush=True)
                return None, None, None, None, None



def barrier_wait(host: str, port: int, soft_start_at: int | None, post_go_sleep: float = 0.0):
    """Connect to controller barrier; send READY, wait for GO. If barrier unreachable, fall back to start_at."""
    if host and port:
        try:
            with socket.create_connection((host, port), timeout=60.0) as s:
                s.settimeout(120.0)
                s.sendall(b"READY\n")
                data = s.recv(16)
                if b"GO" in data:
                    if post_go_sleep > 0:
                        time.sleep(post_go_sleep)
                    print("[run_client] barrier released (GO).")
                    return
        except Exception as e:
            print(f"[run_client] barrier conn failed: {e} (falling back)", flush=True)

    # Fallback: old epoch-style if provided
    if soft_start_at:
        now = int(time.time())
        sleep_for = soft_start_at - now
        if sleep_for > 0:
            print(f"[run_client] fallback wait {sleep_for}s (start_at)", flush=True)
            time.sleep(sleep_for)


# ── Main evaluation ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Sync start across clients

    # ---- WARM-UP (excluded from timing) ----
    with torch.inference_mode():
        dummy = torch.zeros(1, 3, 32, 32, device=device)
        _ = rejector(dummy.view(1, -1))
        _ = client_model(dummy)
    print("[clientV3] prep complete; waiting on barrier...", flush=True)

    # ---- BARRIER (true rendezvous) ----
    barrier_wait(args.barrier_host, args.barrier_port, args.start_at, post_go_sleep=0.2)

    
    client_request_count = 0
    client_start_time = time.time()
    host = args.host
    port = args.port

    # CIFAR-10 normalization (client applies for BOTH paths)
    norm = T.Normalize((0.4914, 0.4822, 0.4465),
                       (0.2023, 0.1994, 0.2010))

    thresholds = [0.325]#[float(x) for x in args.thresholds.split(',')] if args.thresholds else [0.0]
    iterations = 5
    client_cps = 0.0

    stats = {
        "client_accs": [], "expert_accs": [], "accs": [],
        "latencies": [], "latencies2": [],
        "clientt": [], "servert": [],
        "e2c": [], "c2e": [], "rtt": [],
        "expert_t": [], "client_t": [], "rej_time": [],
        "cps": [], "eps": [],
        "overall_latency": []
    }

    for threshold in thresholds:
        client_correct = client_total = 0
        expert_correct = expert_total = 0
        correct = total = 0
        total_latency = 0.0
        server_calls = client_calls = 0
        sum_c2e = sum_e2c = sum_rtt = sum_et = sum_ct = 0.0

        print(f"\n[clientV3] Threshold {threshold:.2f}", flush=True)
        for _ in range(iterations):
            for imgs, labels in testloader:
                label = labels.item()

                # ---- Rejector (excluded from latency) ----
                img_gpu = imgs.to(device)
                t_rej_start = time.perf_counter()
                with torch.inference_mode():
                    r_logits = rejector(img_gpu.view(1, -1))
                    prob = torch.softmax(r_logits, dim=1)[0, 1]
                t_rej_end = time.perf_counter()
                rej_time = max(0.0, min(0.02, t_rej_end - t_rej_start))
                stats["rej_time"].append(rej_time)

                if prob >= threshold:
                    # ── Expert path ──
                    # Normalize on CLIENT (excluded from RTT)
                    img_cpu_norm = norm(imgs.clone())

                    # Start RTT BEFORE serialization so RTT includes it
                    t0 = time.perf_counter()
                    logits, T2, T3, client_count, server_cps = request_batch_expert(
                        img_cpu_norm, r_logits[0], host, port, args.conn_timeout, args.retries
                    )
                    t4 = time.perf_counter()

                    if logits is None:
                        continue

                    et  = max(0.0, T3 - T2)      # server compute only
                    rtt = max(0.0, t4 - t0)      # includes serialization + wire + deser + forward
                    net = max(0.0, rtt - et)
                    c2e = e2c = net / 2.0

                    total_latency += rtt
                    sum_c2e += c2e
                    sum_e2c += e2c
                    sum_et  += et
                    sum_rtt += rtt
                    server_calls += 1

                    # client CPS (requests per second to expert)
                    client_request_count += 1
                    now = time.time()
                    elapsed = now - client_start_time
                    if elapsed >= 1.0:
                        client_cps = client_request_count / elapsed
                        client_request_count = 0
                        client_start_time = now
                    stats["cps"].append(client_cps)
                    stats["eps"].append(server_cps)

                    pred = logits.argmax(dim=1).item()
                    expert_total += 1
                    expert_correct += (pred == label)
                    total += 1
                    correct += (pred == label)

                else:
                    # ── Client path ──
                    # Normalize BEFORE timing so ct = forward only
                    img_norm_gpu = norm(img_gpu)

                    t_client_start = time.perf_counter()
                    with torch.inference_mode():
                        logits = client_model(img_norm_gpu)
                    t_client_end = time.perf_counter()

                    ct = t_client_end - t_client_start
                    sum_ct += ct
                    client_calls += 1

                    pred = logits.argmax(dim=1).item()
                    client_total += 1
                    client_correct += (pred == label)
                    total += 1
                    correct += (pred == label)

        # Stats per threshold
        acc = correct / total if total else 0.0
        client_acc = (client_correct / client_total) if client_total else 0.0
        expert_acc = (expert_correct / expert_total) if expert_total else 0.0
        stats["accs"].append(acc)
        stats["client_accs"].append(client_acc)
        stats["expert_accs"].append(expert_acc)

        avg_lat  = total_latency / server_calls if server_calls else 0.0                # expert RTT (avg over expert calls)
        avg_lat2 = total_latency / total if total else 0.0                              # expert RTT spread over all samples
        stats["latencies"].append(avg_lat)
        stats["latencies2"].append(avg_lat2)
        stats["c2e"].append(sum_c2e / server_calls if server_calls else 0.0)
        stats["e2c"].append(sum_e2c / server_calls if server_calls else 0.0)
        stats["rtt"].append(sum_rtt / server_calls if server_calls else 0.0)
        stats["expert_t"].append(sum_et / server_calls if server_calls else 0.0)
        stats["client_t"].append(sum_ct / client_calls if client_calls else 0.0)
        stats["clientt"].append(client_calls)
        stats["servert"].append(server_calls)

        # overall average latency across all points (expert RTT + client forward only)
        overall_avg = (sum_rtt + sum_ct) / total if total else 0.0
        stats["overall_latency"].append(overall_avg)

        print(f"[clientV3] Acc: {acc*100:.2f}% | Avg server RTT: {avg_lat:.4f}s | Overall avg: {overall_avg:.4f}s", flush=True)
        time.sleep(1)

    print(f"[clientV3] client CPS samples: {len(stats['cps'])} | server CPS samples: {len(stats['eps'])}", flush=True)

    # ── Primary writes histogram slot + ALWAYS plots ────────────────────────
    if args.primary_client and args.num_clients:
        PICKLE_PATH = "list_pickle.pkl"
        NUM_TESTS = int(args.num_tests_total or 19)

        # Load/create and normalize length
        if os.path.exists(PICKLE_PATH):
            with open(PICKLE_PATH, "rb") as f:
                latency_list = pickle.load(f)
            if not isinstance(latency_list, list) or len(latency_list) != NUM_TESTS:
                latency_list = [-1.0] * NUM_TESTS
        else:
            latency_list = [-1.0] * NUM_TESTS

        fill_index = args.num_clients - 1
        if 0 <= fill_index < NUM_TESTS:
            avg_rtt = stats["rtt"][-1] if stats["rtt"] else -1.0
            latency_list[fill_index] = float(avg_rtt)
            print(f"[clientV3] saved RTT at idx {fill_index}: {avg_rtt:.4f}")
        else:
            print(f"[clientV3] WARNING invalid index {fill_index} for NUM_TESTS {NUM_TESTS}")

        with open(PICKLE_PATH, "wb") as f:
            pickle.dump(latency_list, f)
        print(f"[clientV3] wrote {PICKLE_PATH} -> {os.path.abspath(PICKLE_PATH)}")

        # Overall latency (client+expert) per number of clients
        OVERALL_PICKLE = "overall_latency.pkl"
        if os.path.exists(OVERALL_PICKLE):
            with open(OVERALL_PICKLE, "rb") as f:
                overall_list = pickle.load(f)
            if not isinstance(overall_list, list) or len(overall_list) != NUM_TESTS:
                overall_list = [-1.0] * NUM_TESTS
        else:
            overall_list = [-1.0] * NUM_TESTS

        overall_val = stats["overall_latency"][-1] if stats.get("overall_latency") else -1.0
        if 0 <= fill_index < NUM_TESTS:
            overall_list[fill_index] = float(overall_val)

        with open(OVERALL_PICKLE, "wb") as f:
            pickle.dump(overall_list, f)
        print(f"[clientV3] wrote {OVERALL_PICKLE} -> {os.path.abspath(OVERALL_PICKLE)}")

        # Total expert calls per test (all clients)
        CALLS_PICKLE = "expert_calls.pkl"
        if os.path.exists(CALLS_PICKLE):
            with open(CALLS_PICKLE, "rb") as f:
                calls_list = pickle.load(f)
            if not isinstance(calls_list, list) or len(calls_list) != NUM_TESTS:
                calls_list = [-1.0] * NUM_TESTS
        else:
            calls_list = [-1.0] * NUM_TESTS

        per_client_calls = stats["servert"][-1] if stats.get("servert") else 0
        total_calls = int(per_client_calls * args.num_clients) if args.num_clients else 0
        if 0 <= fill_index < NUM_TESTS:
            calls_list[fill_index] = total_calls

        with open(CALLS_PICKLE, "wb") as f:
            pickle.dump(calls_list, f)
        print(f"[clientV3] wrote {CALLS_PICKLE} -> {os.path.abspath(CALLS_PICKLE)}")

        # ── Only draw/save PNGs on the FINAL run (max client count) ───────────
        FINAL_RUN = (int(args.num_clients) == int(args.num_tests_total or 19))
        if FINAL_RUN:
            import matplotlib.pyplot as plt
            # RTT vs clients (expert-only)
            x_vals = [i + 1 for i in range(NUM_TESTS)]
            y_vals = [v if isinstance(v, (int, float)) and v > 0 else 0.0 for v in latency_list]
            plt.figure(figsize=(10, 5))
            plt.bar(x_vals, y_vals)
            plt.xlabel("Number of Clients")
            plt.ylabel("Latency (s)")
            plt.title("Number of Clients vs Expert RTT")
            plt.tight_layout()
            out_path = os.path.join(BASE_DIR, "Client_vs_Latency_Histogram.png")
            plt.savefig(out_path)
            print(f"[clientV3] histogram saved: {out_path}", flush=True)

            # Overall avg latency vs clients
            x_vals = [i + 1 for i in range(NUM_TESTS)]
            y_vals = [v if isinstance(v, (int, float)) and v > 0 else 0.0 for v in overall_list]
            plt.figure(figsize=(10, 5))
            plt.bar(x_vals, y_vals)
            plt.xlabel("Number of Clients")
            plt.ylabel("Overall Avg Latency (s)")
            plt.title("Number of Clients vs Overall Average Latency (Expert RTT + Client Forward)")
            plt.tight_layout()
            out_overall = os.path.join(BASE_DIR, "Clients_vs_OverallLatency_Histogram.png")
            plt.savefig(out_overall)
            print(f"[clientV3] overall latency histogram saved: {out_overall}", flush=True)

            # Total expert calls vs clients
            x_vals = [i + 1 for i in range(NUM_TESTS)]
            y_vals = [int(v) if isinstance(v, (int, float)) and v >= 0 else 0 for v in calls_list]
            plt.figure(figsize=(10, 5))
            plt.bar(x_vals, y_vals)
            plt.xlabel("Number of Clients")
            plt.ylabel("Total Expert Calls")
            plt.title("Number of Clients vs Total Expert Calls")
            plt.tight_layout()
            out_calls = os.path.join(BASE_DIR, "Clients_vs_ExpertCalls_Histogram.png")
            plt.savefig(out_calls)
            print(f"[clientV3] expert calls histogram saved: {out_calls}", flush=True)

            # Accuracy vs threshold (simple bar)
            try:
                x_lbls = [f"{t:.3f}" for t in thresholds]
                y_vals = [float(a) for a in stats["accs"]]
                plt.figure(figsize=(10, 5))
                plt.bar(x_lbls, y_vals)
                plt.xlabel("Threshold")
                plt.ylabel("Overall Accuracy")
                plt.title("Overall Accuracy vs Threshold")
                plt.ylim(0.0, 1.0)
                plt.tight_layout()
                acc_png = os.path.join(BASE_DIR, "Accuracy_vs_Threshold.png")
                plt.savefig(acc_png)
                print(f"[clientV3] accuracy bar saved: {acc_png}")
            except Exception as e:
                print(f"[clientV3] WARNING: failed to write accuracy PNG: {e}", flush=True)

            plt.close('all')
