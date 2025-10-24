#2 logits + noise probe (NEW FILE)

import json
import os, io, socket, struct, time, argparse, pickle
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import Subset, DataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # safe with Agg

# ----------------- ARGS -----------------
parser = argparse.ArgumentParser(description="Single-client CIFAR-10 client↔expert evaluation (with optional noise)")
parser.add_argument('--host', type=str, default='node21-2.grid.orbit-lab.org')
parser.add_argument('--port', type=int, default=7000)
parser.add_argument('--conn_timeout', type=float, default=30.0)
parser.add_argument('--retries', type=int, default=3)
parser.add_argument('--thresholds', type=str, default='0.3225')
parser.add_argument('--iterations', type=int, default=5)
parser.add_argument('--save_dir', type=str, default='single_run_out')
parser.add_argument('--client_model', type=str, default='ClientModels/client_model70.pt')
parser.add_argument('--rejector_model', type=str, default='RejectorModels/rejector_70.pt')

# NEW (noise controls)
parser.add_argument('--noise_fraction', type=float, default=0.25, help='fraction of subset to noise')
parser.add_argument('--noise_std', type=float, default=0.35, help='Gaussian std in [0,1] image space')
parser.add_argument('--noise_seed', type=int, default=4242, help='which images to noise (reproducible)')
parser.add_argument('--noise_where', type=str, choices=['all','expert','client'], default='all',
                    help='apply noise to: all paths, only expert path, or only client path')
args = parser.parse_args()

print("[client_single_noise] starting")
print("[client_single_noise] cwd:", os.getcwd())
print("[client_single_noise] args:", vars(args))

os.makedirs(args.save_dir, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("[client_single_noise] device:", device)

# ----------------- MODELS -----------------
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

# ----------------- DATA -----------------
print("[client_single_noise] preparing CIFAR-10 test set...")
try:
    raw_testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=T.ToTensor()
    )
except Exception as e:
    print("[client_single_noise] download=False fallback:", e, flush=True)
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
chosen_indices = np.array(chosen_indices, dtype=int)

# Decide which images will be noised (fixed across runs)
noise_rng = np.random.RandomState(args.noise_seed)
num_noised = int(len(chosen_indices) * max(0.0, min(1.0, args.noise_fraction)))
noised_mask = np.zeros(len(chosen_indices), dtype=bool)
if num_noised > 0:
    noised_mask[noise_rng.choice(len(chosen_indices), size=num_noised, replace=False)] = True
print(f"[client_single_noise] will noise {noised_mask.sum()} / {len(chosen_indices)} images")

stratified_subset = Subset(raw_testset, chosen_indices.tolist())
testloader = DataLoader(stratified_subset, batch_size=1, shuffle=False)
print("[client_single_noise] dataset size:", len(stratified_subset))

# ----------------- LOAD CKPTS -----------------
client_model = Client().to(device)
client_model.load_state_dict(torch.load(args.client_model, map_location=device))
client_model.eval()
print("[client_single_noise] loaded client:", args.client_model)

rejector = Rejector().to(device)
rejector.load_state_dict(torch.load(args.rejector_model, map_location=device))
rejector.eval()
print("[client_single_noise] loaded rejector:", args.rejector_model)

# ----------------- NET HELPERS -----------------
def recvall(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise RuntimeError("Connection closed prematurely")
        buf.extend(chunk)
    return bytes(buf)

def request_batch_expert(batch_cpu, rejector_logits, host, port, timeout, retries):
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            with socket.create_connection((host, port), timeout=timeout) as sock:
                sock.settimeout(timeout)
                try: sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                except OSError: pass
                try: sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                except OSError: pass

                sock.sendall(b'B')

                buf = io.BytesIO()
                torch.save(batch_cpu, buf)
                batch_data = buf.getvalue()
                rej_data = rejector_logits.detach().cpu().numpy().astype('<f4').tobytes()

                sock.sendall(struct.pack('>II', len(batch_data), len(rej_data)))
                sock.sendall(batch_data + rej_data)

                raw = recvall(sock, 4)
                plen = int.from_bytes(raw, 'big')
                pdata = recvall(sock, plen)

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
                print(f"[client_single_noise] expert request failed after retries: {e}", flush=True)
                return None, None, None, None, None

@torch.no_grad()
def add_gaussian_noise01(x, std):
    if std <= 0: return x
    noisy = x + std * torch.randn_like(x)
    return torch.clamp(noisy, 0.0, 1.0)

# ----------------- EVAL -----------------
def run_once():
    host, port = args.host, args.port
    thresholds = [float(x) for x in args.thresholds.split(',')] if args.thresholds else [0.0]
    iterations = int(args.iterations)

    norm = T.Normalize((0.4914, 0.4822, 0.4465),
                       (0.2023, 0.1994, 0.2010))

    stats = {
        "client_accs": [], "expert_accs": [], "accs": [],
        "latencies": [], "latencies2": [],
        "clientt": [], "servert": [],
        "e2c": [], "c2e": [], "rtt": [],
        "expert_t": [], "client_t": [], "rej_time": [], "overall_latency": []
    }

    # NEW: per-sample capture for expert/client confidence
    probe_rows = []

    for threshold in thresholds:
        client_correct = client_total = 0
        expert_correct = expert_total = 0
        correct = total = 0
        total_latency = 0.0
        server_calls = client_calls = 0
        sum_c2e = sum_e2c = sum_rtt = sum_et = sum_ct = 0.0

        print(f"\n[client_single_noise] Threshold {threshold:.3f}", flush=True)

        for _ in range(iterations):
            idx_ptr = 0
            for imgs, labels in testloader:
                label = labels.item()
                is_noisy = bool(noised_mask[idx_ptr])
                idx_ptr += 1

                img_clean = imgs  # in [0,1]

                # Rejector sees noise only if noise_where == 'all'
                rej_input = img_clean
                if is_noisy and args.noise_where == 'all':
                    rej_input = add_gaussian_noise01(rej_input, args.noise_std)

                img_gpu = rej_input.to(device)
                t_rej_start = time.perf_counter()
                with torch.inference_mode():
                    r_logits = rejector(img_gpu.view(1, -1))
                    prob = torch.softmax(r_logits, dim=1)[0, 1]
                t_rej_end = time.perf_counter()
                rej_time = max(0.0, min(0.02, t_rej_end - t_rej_start))
                stats["rej_time"].append(rej_time)

                if prob >= threshold:
                    # EXPERT path
                    x = img_clean
                    if is_noisy and args.noise_where in ('all','expert'):
                        x = add_gaussian_noise01(x, args.noise_std)

                    x_norm_cpu = T.Compose([norm])(x.clone())
                    t0 = time.perf_counter()
                    logits, T2, T3, client_count, server_cps = request_batch_expert(
                        x_norm_cpu, r_logits[0], host, port, args.conn_timeout, args.retries
                    )
                    t4 = time.perf_counter()
                    if logits is None:
                        continue

                    et  = max(0.0, T3 - T2)
                    rtt = max(0.0, t4 - t0)
                    net = max(0.0, rtt - et)
                    c2e = e2c = net / 2.0

                    total_latency += rtt
                    sum_c2e += c2e
                    sum_e2c += e2c
                    sum_et  += et
                    sum_rtt += rtt
                    server_calls += 1

                    pred = int(logits.argmax(dim=1).item())
                    expert_total += 1
                    expert_correct += int(pred == label)
                    total += 1
                    correct += int(pred == label)

                    with torch.no_grad():
                        top1 = float(logits.max(dim=1).values.item())
                        pmax = float(torch.softmax(logits, dim=1).max(dim=1).values.item())
                        top2vals, _ = logits.topk(2, dim=1)
                        margin = float((top2vals[0,0] - top2vals[0,1]).item())
                    probe_rows.append({
                        "routed":"expert","is_noisy":int(is_noisy),
                        "label":int(label),"pred":pred,"correct":int(pred==label),
                        "top1_logit":top1,"softmax_pmax":pmax,"logit_margin":margin,
                        "rtt":rtt,"server_time":et
                    })

                else:
                    # CLIENT path
                    x = img_clean
                    if is_noisy and args.noise_where in ('all','client'):
                        x = add_gaussian_noise01(x, args.noise_std)

                    x_norm_gpu = norm(x.to(device))
                    t_cs = time.perf_counter()
                    with torch.inference_mode():
                        logits = client_model(x_norm_gpu)
                    t_ce = time.perf_counter()
                    ct = t_ce - t_cs
                    sum_ct += ct
                    client_calls += 1

                    pred = int(logits.argmax(dim=1).item())
                    client_total += 1
                    client_correct += int(pred == label)
                    total += 1
                    correct += int(pred == label)

                    with torch.no_grad():
                        top1 = float(logits.max(dim=1).values.item())
                        pmax = float(torch.softmax(logits, dim=1).max(dim=1).values.item())
                        top2vals, _ = logits.topk(2, dim=1)
                        margin = float((top2vals[0,0] - top2vals[0,1]).item())
                    probe_rows.append({
                        "routed":"client","is_noisy":int(is_noisy),
                        "label":int(label),"pred":pred,"correct":int(pred==label),
                        "top1_logit":top1,"softmax_pmax":pmax,"logit_margin":margin,
                        "client_time":ct
                    })

        acc = correct / total if total else 0.0
        client_acc = (client_correct / client_total) if client_total else 0.0
        expert_acc = (expert_correct / expert_total) if expert_total else 0.0
        stats["accs"].append(acc)
        stats["client_accs"].append(client_acc)
        stats["expert_accs"].append(expert_acc)

        avg_lat  = total_latency / server_calls if server_calls else 0.0
        avg_lat2 = total_latency / total if total else 0.0
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

        print(f"[client_single_noise] Acc: {acc*100:.2f}% | Avg server RTT: {avg_lat:.4f}s | Overall avg: {overall_avg:.4f}s", flush=True)

    out_json = os.path.join(args.save_dir, "single_stats.json")
    with open(out_json, "w") as f:
        json.dump({"thresholds": [float(x) for x in args.thresholds.split(',')],
                   "iterations": iterations,
                   "stats": stats}, f, indent=2)
    print(f"[client_single_noise] wrote {out_json}")

    # Save CSV for per-sample analysis
    csv_path = os.path.join(args.save_dir, "probe_logit_confidence.csv")
    try:
        import csv
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "routed","is_noisy","label","pred","correct",
                "top1_logit","softmax_pmax","logit_margin",
                "rtt","server_time","client_time"
            ])
            w.writeheader()
            for row in probe_rows:
                w.writerow(row)
        print(f"[client_single_noise] wrote {csv_path}")
    except Exception as e:
        print(f"[client_single_noise] WARNING: CSV write failed: {e}")

    # Original plots
    try:
        x_lbls = [f"{t:.3f}" for t in [float(x) for x in args.thresholds.split(',')]]
        plt.figure(figsize=(8,4))
        plt.bar(x_lbls, [float(a) for a in stats["accs"]])
        plt.xlabel("Threshold"); plt.ylabel("Overall Accuracy")
        plt.title("Overall Accuracy vs Threshold (single client)")
        plt.ylim(0.0, 1.0); plt.tight_layout()
        plt.savefig(os.path.join(args.save_dir, "Accuracy_vs_Threshold.png"))

        plt.figure(figsize=(8,4))
        plt.bar(x_lbls, [float(a) for a in stats["rtt"]])
        plt.xlabel("Threshold"); plt.ylabel("Avg Expert RTT (s)")
        plt.title("Avg Expert RTT vs Threshold (single client)")
        plt.tight_layout()
        plt.savefig(os.path.join(args.save_dir, "RTT_vs_Threshold.png"))
        plt.close('all')
    except Exception as e:
        print(f"[client_single_noise] WARNING: plotting failed: {e}", flush=True)

    # NEW: expert confidence clean vs noisy
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        dfe = df[df["routed"]=="expert"]
        if not dfe.empty:
            grp = dfe.groupby("is_noisy").agg(mean_pmax=("softmax_pmax","mean")).reset_index()
            clean = float(grp.loc[grp.is_noisy==0,"mean_pmax"].iloc[0]) if (grp.is_noisy==0).any() else 0.0
            noisy = float(grp.loc[grp.is_noisy==1,"mean_pmax"].iloc[0]) if (grp.is_noisy==1).any() else 0.0
            plt.figure(figsize=(6,4))
            plt.bar(["clean","noisy"], [clean, noisy])
            plt.ylim(0.0, 1.0)
            plt.ylabel("Mean softmax max (expert)")
            plt.title("Expert confidence: clean vs noisy")
            plt.tight_layout()
            plt.savefig(os.path.join(args.save_dir, "Expert_confidence_clean_vs_noisy.png"))
            plt.close('all')
            print(f"[client_single_noise] expert mean pmax clean={clean:.3f} noisy={noisy:.3f}")
        else:
            print("[client_single_noise] no expert samples to plot confidence.")
    except Exception as e:
        print(f"[client_single_noise] WARNING: confidence plot failed: {e}", flush=True)

if __name__ == "__main__":
    run_once()
