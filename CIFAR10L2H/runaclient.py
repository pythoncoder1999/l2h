#2 logits
'''python client_single_2log.py \
  --host node21-2.grid.orbit-lab.org \
  --port 7000 \
  --thresholds 0.3225 \
  --iterations 5 \
  --save_dir out_2log \
  --client_model ClientModels/client_model70.pt \
  --rejector_model RejectorModels/rejector_70.pt


python client_single_3log.py \
  --host node21-2.grid.orbit-lab.org \
  --port 7000 \
  --thresholds 0.3225 \
  --iterations 5 \
  --save_dir out_3log \
  --client_model ClientModels/client_model70.pt \
  --rejector_model RejectorModels/rejector_70_3l.pt

'''

import json
import os, io, socket, struct, time, argparse, pickle
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
import matplotlib.pyplot as plt  # safe with Agg

# ── Args ────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Single-client CIFAR-10 client↔expert evaluation")
parser.add_argument('--host', type=str, default='node21-2.grid.orbit-lab.org', help='Expert proxy host')
parser.add_argument('--port', type=int, default=7000, help='Expert proxy port')
parser.add_argument('--conn_timeout', type=float, default=30.0, help='Socket connect/read timeout (s)')
parser.add_argument('--retries', type=int, default=3, help='Retries for expert requests')
parser.add_argument('--thresholds', type=str, default='0.3225', help='Comma-separated thresholds (e.g., "0.25,0.33,0.40")')
parser.add_argument('--iterations', type=int, default=5, help='How many passes over the 1000-image subset')
parser.add_argument('--save_dir', type=str, default='single_run_out', help='Where to save outputs')
parser.add_argument('--client_model', type=str, default='ClientModels/client_model70.pt')
parser.add_argument('--rejector_model', type=str, default='RejectorModels/rejector_70.pt')
args = parser.parse_args()

print("[client_single] starting")
print("[client_single] cwd:", os.getcwd())
print("[client_single] args:", vars(args))

os.makedirs(args.save_dir, exist_ok=True)

# ── Device ──────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print("[client_single] device:", device)

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
    def forward(self, x): return self.classifier(self.features(x))

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
        x = x.view(x.size(0), -1)
        return self.block(x)

# ── Data (deterministic 100/class subset = 1000 imgs) ───────────────────────
print("[client_single] preparing CIFAR-10 test set (may download once)...")
try:
    raw_testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=T.ToTensor()
    )
except Exception as e:
    print("[client_single] CIFAR download failed, retrying with download=False:", e, flush=True)
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
print("[client_single] dataset ready:", len(stratified_subset), "images")

# ── Load checkpoints ────────────────────────────────────────────────────────
client_model = Client().to(device)
client_model.load_state_dict(torch.load(args.client_model, map_location=device))
client_model.eval()
print("[client_single] loaded client model:", args.client_model)

rejector = Rejector().to(device)
rejector.load_state_dict(torch.load(args.rejector_model, map_location=device))
rejector.eval()
print("[client_single] loaded rejector:", args.rejector_model)

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
    """Serialize batch_cpu (already normalized) + rejector logits; return (logits, T2, T3, client_count, server_cps)."""
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

            vals = struct.unpack('<' + 'f' * (len(logits_data)//4), logits_data)
            logits = torch.tensor(vals, dtype=torch.float32).view(batch_cpu.size(0), 10)
            return logits, T2, T3, client_count, server_cps

        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(0.5 * attempt)
            else:
                print(f"[client_single] expert request failed after {retries} tries: {e}", flush=True)
                return None, None, None, None, None

# ── Evaluation (single client) ──────────────────────────────────────────────
def run_once():
    host, port = args.host, args.port
    thresholds = [float(x) for x in args.thresholds.split(',')] if args.thresholds else [0.0]
    iterations = int(args.iterations)

    # CIFAR-10 normalization
    norm = T.Normalize((0.4914, 0.4822, 0.4465),
                       (0.2023, 0.1994, 0.2010))

    stats = {
        "client_accs": [], "expert_accs": [], "accs": [],
        "latencies": [], "latencies2": [],
        "clientt": [], "servert": [],
        "e2c": [], "c2e": [], "rtt": [],
        "expert_t": [], "client_t": [], "rej_time": [],
        "overall_latency": []
    }

    for threshold in thresholds:
        # ----- per-threshold tracking (2-logit script) -----
        # --- track rejector logits (2-logit) ---
        rej_logit0_list = []   # raw rejector logit for class 0
        rej_logit1_list = []   # raw rejector logit for class 1

        client_correct = client_total = 0
        expert_correct = expert_total = 0
        correct = total = 0
        total_latency = 0.0
        server_calls = client_calls = 0
        sum_c2e = sum_e2c = sum_rtt = sum_et = sum_ct = 0.0

        print(f"\n[client_single] Threshold {threshold:.3f}", flush=True)
        for _ in range(iterations):
            for imgs, labels in testloader:
                label = labels.item()

                # Rejector (excluded from latency)
                img_gpu = imgs.to(device)
                t_rej_start = time.perf_counter()
                with torch.inference_mode():
                    r_logits = rejector(img_gpu.view(1, -1))
                    prob = torch.softmax(r_logits, dim=1)[0, 1]
                    # collect raw rejector logits
                    rej_logit0_list.append(float(r_logits[0, 0].item()))
                    rej_logit1_list.append(float(r_logits[0, 1].item()))
                t_rej_end = time.perf_counter()
                rej_time = max(0.0, min(0.02, t_rej_end - t_rej_start))
                stats["rej_time"].append(rej_time)

                if prob >= threshold:
                    # Expert path
                    img_cpu_norm = norm(imgs.clone())
                    t0 = time.perf_counter()
                    logits, T2, T3, client_count, server_cps = request_batch_expert(
                        img_cpu_norm, r_logits[0], host, port, args.conn_timeout, args.retries
                    )
                    t4 = time.perf_counter()
                    if logits is None:
                        continue

                    et  = max(0.0, T3 - T2)      # server compute only
                    rtt = max(0.0, t4 - t0)      # full round-trip
                    net = max(0.0, rtt - et)
                    c2e = e2c = net / 2.0

                    total_latency += rtt
                    sum_c2e += c2e
                    sum_e2c += e2c
                    sum_et  += et
                    sum_rtt += rtt
                    server_calls += 1

                    pred = logits.argmax(dim=1).item()
                    expert_total += 1
                    expert_correct += (pred == label)
                    total += 1
                    correct += (pred == label)

                else:
                    # Client path
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

        # Per-threshold stats
        acc = correct / total if total else 0.0
        client_acc = (client_correct / client_total) if client_total else 0.0
        expert_acc = (expert_correct / expert_total) if expert_total else 0.0
        stats["accs"].append(acc)
        stats["client_accs"].append(client_acc)
        stats["expert_accs"].append(expert_acc)

        avg_lat  = total_latency / server_calls if server_calls else 0.0          # expert RTT avg (expert calls only)
        avg_lat2 = total_latency / total if total else 0.0                        # spread over all samples
        stats["latencies"].append(avg_lat)
        stats["latencies2"].append(avg_lat2)
        stats["c2e"].append(sum_c2e / server_calls if server_calls else 0.0)
        stats["e2c"].append(sum_e2c / server_calls if server_calls else 0.0)
        stats["rtt"].append(sum_rtt / server_calls if server_calls else 0.0)
        stats["expert_t"].append(sum_et / server_calls if server_calls else 0.0)
        stats["client_t"].append(sum_ct / client_calls if client_calls else 0.0)
        stats["clientt"].append(client_calls)
        stats["servert"].append(server_calls)

        overall_avg = (sum_rtt + sum_ct) / total if total else 0.0
        stats["overall_latency"].append(overall_avg)

        print(f"[client_single] Acc: {acc*100:.2f}% | Avg server RTT: {avg_lat:.4f}s | Overall avg: {overall_avg:.4f}s", flush=True)
        # ----- Save per-threshold summary (counts, latency, accuracy) -----
        summary_json = {
            "threshold": float(threshold),
            "expert_calls": int(server_calls),
            "client_calls": int(client_calls),
            "avg_expert_RTT": float(stats["rtt"][-1]) if stats["rtt"] else 0.0,
            "avg_expert_compute": float(stats["expert_t"][-1]) if stats["expert_t"] else 0.0,
            "avg_client_time": float(stats["client_t"][-1]) if stats["client_t"] else 0.0,
            "avg_overall_latency": float(overall_avg),
            "accuracy": float(acc),
            "client_acc": float(client_acc),
            "expert_acc": float(expert_acc),
            "rej_logit0_mean": float(np.mean(rej_logit0_list)) if rej_logit0_list else None,
            "rej_logit1_mean": float(np.mean(rej_logit1_list)) if rej_logit1_list else None,
        }
        with open(os.path.join(args.save_dir, f"summary_{threshold:.3f}.json"), "w") as f:
            json.dump(summary_json, f, indent=2)

        # ----- Plot rejector raw logit distributions (2-logit) -----
        try:
            xlbl = f"{threshold:.3f}"

            if rej_logit0_list:
                plt.figure(figsize=(7,4))
                plt.hist(rej_logit0_list, bins=40)
                plt.xlabel("Rejector raw logit: class 0")
                plt.ylabel("Count")
                plt.title(f"Rejector logit[0] distribution @ {xlbl}")
                plt.tight_layout()
                plt.savefig(os.path.join(args.save_dir, f"rejector_logit0_{xlbl}.png"))
                plt.close()

            if rej_logit1_list:
                plt.figure(figsize=(7,4))
                plt.hist(rej_logit1_list, bins=40)
                plt.xlabel("Rejector raw logit: class 1")
                plt.ylabel("Count")
                plt.title(f"Rejector logit[1] distribution @ {xlbl}")
                plt.tight_layout()
                plt.savefig(os.path.join(args.save_dir, f"rejector_logit1_{xlbl}.png"))
                plt.close()
        except Exception as e:
            print(f"[client_single] WARNING: rejector logit plot failed: {e}", flush=True)


    # Save stats
    out_json = os.path.join(args.save_dir, "single_stats.json")
    with open(out_json, "w") as f:
        json.dump({"thresholds": [float(x) for x in args.thresholds.split(',')],
                   "iterations": iterations,
                   "stats": stats}, f, indent=2)
    print(f"[client_single] wrote {out_json}")

    # Quick plots (single client)
    try:
        x_lbls = [f"{t:.3f}" for t in [float(x) for x in args.thresholds.split(',')]]
        # Accuracy
        plt.figure(figsize=(8,4))
        plt.bar(x_lbls, [float(a) for a in stats["accs"]])
        plt.xlabel("Threshold")
        plt.ylabel("Overall Accuracy")
        plt.title("Overall Accuracy vs Threshold (single client)")
        plt.ylim(0.0, 1.0)
        plt.tight_layout()
        acc_png = os.path.join(args.save_dir, "Accuracy_vs_Threshold.png")
        plt.savefig(acc_png)
        print(f"[client_single] saved {acc_png}")
        # RTT (expert only)
        plt.figure(figsize=(8,4))
        plt.bar(x_lbls, [float(a) for a in stats["rtt"]])
        plt.xlabel("Threshold")
        plt.ylabel("Avg Expert RTT (s)")
        plt.title("Avg Expert RTT vs Threshold (single client)")
        plt.tight_layout()
        rtt_png = os.path.join(args.save_dir, "RTT_vs_Threshold.png")
        plt.savefig(rtt_png)
        print(f"[client_single] saved {rtt_png}")
        plt.close('all')
    except Exception as e:
        print(f"[client_single] WARNING: plotting failed: {e}", flush=True)

if __name__ == "__main__":
    run_once()




'''

import json
import os, io, socket, struct, time, argparse, pickle
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
import matplotlib.pyplot as plt  # safe with Agg

# ── Args ────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Single-client CIFAR-10 client↔expert evaluation")
parser.add_argument('--host', type=str, default='node21-2.grid.orbit-lab.org', help='Expert proxy host')
parser.add_argument('--port', type=int, default=7000, help='Expert proxy port')
parser.add_argument('--conn_timeout', type=float, default=30.0, help='Socket connect/read timeout (s)')
parser.add_argument('--retries', type=int, default=3, help='Retries for expert requests')
parser.add_argument('--thresholds', type=str, default='0.3225', help='Comma-separated thresholds (e.g., "0.25,0.33,0.40")')
parser.add_argument('--iterations', type=int, default=5, help='How many passes over the 1000-image subset')
parser.add_argument('--save_dir', type=str, default='single_run_out', help='Where to save outputs')
parser.add_argument('--client_model', type=str, default='ClientModels/client_model70.pt')
parser.add_argument('--rejector_model', type=str, default='RejectorModels/rejector_70_3l.pt')
args = parser.parse_args()

print("[client_single] starting")
print("[client_single] cwd:", os.getcwd())
print("[client_single] args:", vars(args))

os.makedirs(args.save_dir, exist_ok=True)

# ── Device ──────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print("[client_single] device:", device)

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
    def forward(self, x): return self.classifier(self.features(x))

class Rejector(nn.Module):
    def __init__(self, hidden=256, drop=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(3 * 32 * 32, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Linear(hidden, 3)
        )
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.block(x)

# ── Data (deterministic 100/class subset = 1000 imgs) ───────────────────────
print("[client_single] preparing CIFAR-10 test set (may download once)...")
try:
    raw_testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=T.ToTensor()
    )
except Exception as e:
    print("[client_single] CIFAR download failed, retrying with download=False:", e, flush=True)
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
print("[client_single] dataset ready:", len(stratified_subset), "images")

# ── Load checkpoints ────────────────────────────────────────────────────────
client_model = Client().to(device)
client_model.load_state_dict(torch.load(args.client_model, map_location=device))
client_model.eval()
print("[client_single] loaded client model:", args.client_model)

rejector = Rejector().to(device)
rejector.load_state_dict(torch.load(args.rejector_model, map_location=device))
rejector.eval()
print("[client_single] loaded rejector:", args.rejector_model)

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
    """Serialize batch_cpu (already normalized) + rejector logits; return (logits, T2, T3, client_count, server_cps)."""
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

            vals = struct.unpack('<' + 'f' * (len(logits_data)//4), logits_data)
            logits = torch.tensor(vals, dtype=torch.float32).view(batch_cpu.size(0), 10)
            return logits, T2, T3, client_count, server_cps

        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(0.5 * attempt)
            else:
                print(f"[client_single] expert request failed after {retries} tries: {e}", flush=True)
                return None, None, None, None, None

# ── Evaluation (single client) ──────────────────────────────────────────────
def run_once():
    host, port = args.host, args.port
    thresholds = [float(x) for x in args.thresholds.split(',')] if args.thresholds else [0.0]
    iterations = int(args.iterations)

    # CIFAR-10 normalization
    norm = T.Normalize((0.4914, 0.4822, 0.4465),
                       (0.2023, 0.1994, 0.2010))

    stats = {
        "client_accs": [], "expert_accs": [], "accs": [],
        "latencies": [], "latencies2": [],
        "clientt": [], "servert": [],
        "e2c": [], "c2e": [], "rtt": [],
        "expert_t": [], "client_t": [], "rej_time": [],
        "overall_latency": []
    }

    for threshold in thresholds:
        # --- Track rejector logits (3-logit) ---
        rej_logit0_list = []
        rej_logit1_list = []
        rej_logit2_list = []
        client_correct = client_total = 0
        expert_correct = expert_total = 0
        correct = total = 0
        total_latency = 0.0
        server_calls = client_calls = 0
        sum_c2e = sum_e2c = sum_rtt = sum_et = sum_ct = 0.0

        print(f"\n[client_single] Threshold {threshold:.3f}", flush=True)
        for _ in range(iterations):
            for imgs, labels in testloader:
                label = labels.item()

                # Rejector (excluded from latency)
                img_gpu = imgs.to(device)
                t_rej_start = time.perf_counter()
                with torch.inference_mode():
                    r_logits = rejector(img_gpu.view(1, -1))
                    probs = torch.softmax(r_logits, dim=1)
                    prob1 = probs[0,0]
                    prob2 = probs[0,1]
                    prob3 = probs[0,2]
                    rej_logit0_list.append(float(r_logits[0, 0].item()))
                    rej_logit1_list.append(float(r_logits[0, 1].item()))
                    rej_logit2_list.append(float(r_logits[0, 2].item()))
                t_rej_end = time.perf_counter()
                rej_time = max(0.0, min(0.02, t_rej_end - t_rej_start))
                stats["rej_time"].append(rej_time)

                if prob2 > max(0.5+threshold,prob1):
                    # Expert path
                    img_cpu_norm = norm(imgs.clone())
                    t0 = time.perf_counter()
                    logits, T2, T3, client_count, server_cps = request_batch_expert(
                        img_cpu_norm, r_logits[0], host, port, args.conn_timeout, args.retries
                    )
                    t4 = time.perf_counter()
                    if logits is None:
                        continue

                    et  = max(0.0, T3 - T2)      # server compute only
                    rtt = max(0.0, t4 - t0)      # full round-trip
                    net = max(0.0, rtt - et)
                    c2e = e2c = net / 2.0

                    total_latency += rtt
                    sum_c2e += c2e
                    sum_e2c += e2c
                    sum_et  += et
                    sum_rtt += rtt
                    server_calls += 1

                    pred = logits.argmax(dim=1).item()
                    expert_total += 1
                    expert_correct += (pred == label)
                    total += 1
                    correct += (pred == label)

                elif prob1 > max(0.5+threshold,prob2) or ((prob2 <= 0.5+threshold) and (prob1 <= 0.5+threshold)):
                    # Client path
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
                else:
                    raise Exception("Bad Move")
        # Per-threshold stats
        acc = correct / total if total else 0.0
        client_acc = (client_correct / client_total) if client_total else 0.0
        expert_acc = (expert_correct / expert_total) if expert_total else 0.0
        stats["accs"].append(acc)
        stats["client_accs"].append(client_acc)
        stats["expert_accs"].append(expert_acc)

        avg_lat  = total_latency / server_calls if server_calls else 0.0          # expert RTT avg (expert calls only)
        avg_lat2 = total_latency / total if total else 0.0                        # spread over all samples
        stats["latencies"].append(avg_lat)
        stats["latencies2"].append(avg_lat2)
        stats["c2e"].append(sum_c2e / server_calls if server_calls else 0.0)
        stats["e2c"].append(sum_e2c / server_calls if server_calls else 0.0)
        stats["rtt"].append(sum_rtt / server_calls if server_calls else 0.0)
        stats["expert_t"].append(sum_et / server_calls if server_calls else 0.0)
        stats["client_t"].append(sum_ct / client_calls if client_calls else 0.0)
        stats["clientt"].append(client_calls)
        stats["servert"].append(server_calls)

        overall_avg = (sum_rtt + sum_ct) / total if total else 0.0
        stats["overall_latency"].append(overall_avg)

        print(f"[client_single] Acc: {acc*100:.2f}% | Avg server RTT: {avg_lat:.4f}s | Overall avg: {overall_avg:.4f}s", flush=True)


        # ----- Save per-threshold summary (counts, latency, accuracy) -----
        summary_json = {
            "threshold": float(threshold),
            "expert_calls": int(server_calls),
            "client_calls": int(client_calls),
            "avg_expert_RTT": float(stats["rtt"][-1]) if stats["rtt"] else 0.0,
            "avg_expert_compute": float(stats["expert_t"][-1]) if stats["expert_t"] else 0.0,
            "avg_client_time": float(stats["client_t"][-1]) if stats["client_t"] else 0.0,
            "avg_overall_latency": float(overall_avg),
            "accuracy": float(acc),
            "client_acc": float(client_acc),
            "expert_acc": float(expert_acc),
            "rej_logit0_mean": float(np.mean(rej_logit0_list)) if rej_logit0_list else None,
            "rej_logit1_mean": float(np.mean(rej_logit1_list)) if rej_logit1_list else None,
            "rej_logit2_mean": float(np.mean(rej_logit2_list)) if rej_logit2_list else None,
        }
        with open(os.path.join(args.save_dir, f"summary_{threshold:.3f}.json"), "w") as f:
            json.dump(summary_json, f, indent=2)

        # ----- Plot rejector raw logit distributions (3-logit) -----
        try:
            xlbl = f"{threshold:.3f}"

            if rej_logit0_list:
                plt.figure(figsize=(7,4))
                plt.hist(rej_logit0_list, bins=40)
                plt.xlabel("Rejector raw logit: class 0")
                plt.ylabel("Count")
                plt.title(f"Rejector logit[0] distribution @ {xlbl}")
                plt.tight_layout()
                plt.savefig(os.path.join(args.save_dir, f"rejector_logit0_{xlbl}.png"))
                plt.close()

            if rej_logit1_list:
                plt.figure(figsize=(7,4))
                plt.hist(rej_logit1_list, bins=40)
                plt.xlabel("Rejector raw logit: class 1")
                plt.ylabel("Count")
                plt.title(f"Rejector logit[1] distribution @ {xlbl}")
                plt.tight_layout()
                plt.savefig(os.path.join(args.save_dir, f"rejector_logit1_{xlbl}.png"))
                plt.close()

            if rej_logit2_list:
                plt.figure(figsize=(7,4))
                plt.hist(rej_logit2_list, bins=40)
                plt.xlabel("Rejector raw logit: class 2")
                plt.ylabel("Count")
                plt.title(f"Rejector logit[2] distribution @ {xlbl}")
                plt.tight_layout()
                plt.savefig(os.path.join(args.save_dir, f"rejector_logit2_{xlbl}.png"))
                plt.close()
        except Exception as e:
            print(f"[client_single] WARNING: rejector logit plot failed: {e}", flush=True)




    # Save stats
    out_json = os.path.join(args.save_dir, "single_stats.json")
    with open(out_json, "w") as f:
        json.dump({"thresholds": [float(x) for x in args.thresholds.split(',')],
                   "iterations": iterations,
                   "stats": stats}, f, indent=2)
    print(f"[client_single] wrote {out_json}")

    # Quick plots (single client)
    try:
        x_lbls = [f"{t:.3f}" for t in [float(x) for x in args.thresholds.split(',')]]
        # Accuracy
        plt.figure(figsize=(8,4))
        plt.bar(x_lbls, [float(a) for a in stats["accs"]])
        plt.xlabel("Threshold")
        plt.ylabel("Overall Accuracy")
        plt.title("Overall Accuracy vs Threshold (single client)")
        plt.ylim(0.0, 1.0)
        plt.tight_layout()
        acc_png = os.path.join(args.save_dir, "Accuracy_vs_Threshold.png")
        plt.savefig(acc_png)
        print(f"[client_single] saved {acc_png}")
        # RTT (expert only)
        plt.figure(figsize=(8,4))
        plt.bar(x_lbls, [float(a) for a in stats["rtt"]])
        plt.xlabel("Threshold")
        plt.ylabel("Avg Expert RTT (s)")
        plt.title("Avg Expert RTT vs Threshold (single client)")
        plt.tight_layout()
        rtt_png = os.path.join(args.save_dir, "RTT_vs_Threshold.png")
        plt.savefig(rtt_png)
        print(f"[client_single] saved {rtt_png}")
        plt.close('all')
    except Exception as e:
        print(f"[client_single] WARNING: plotting failed: {e}", flush=True)

if __name__ == "__main__":
    run_once()



'''
