# =============================================
# FILE 1: single_client_logits_2logit.py
# =============================================
# Full script: single-client CIFAR-10 with 2-logit rejector
# Adds full logits-based comparison of Expert vs Client quality.
# We judge "badness" by raw logits:
#   - top1_logit (higher is better)
#   - logit_margin = top1_logit - top2_logit (larger margin => more decisive)
# If both Expert and Client are low, both are bad.
# Expert is "better" than Client when Expert has higher margin by >= delta
# (and optionally higher top1). We log distributions, paired comparisons,
# and per-threshold CSV/JSON of notable cases.

import json
import os, io, socket, struct, time, argparse
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
parser = argparse.ArgumentParser(description="Single-client CIFAR-10 (2-logit rejector) with logits-based analysis")
parser.add_argument('--host', type=str, default='node21-2.grid.orbit-lab.org', help='Expert proxy host')
parser.add_argument('--port', type=int, default=7000, help='Expert proxy port')
parser.add_argument('--conn_timeout', type=float, default=30.0, help='Socket connect/read timeout (s)')
parser.add_argument('--retries', type=int, default=3, help='Retries for expert requests')
parser.add_argument('--thresholds', type=str, default='0.3225', help='Comma-separated thresholds (e.g., "0.25,0.33,0.40")')
parser.add_argument('--iterations', type=int, default=5, help='How many passes over the 1000-image subset')
parser.add_argument('--save_dir', type=str, default='single_run_out_2log', help='Where to save outputs')
parser.add_argument('--client_model', type=str, default='ClientModels/client_model70.pt')
parser.add_argument('--rejector_model', type=str, default='RejectorModels/rejector_70.pt')

# Analysis options
parser.add_argument('--dump_raw', action='store_true', help='Dump per-sample CSVs (paired logits, preds, metrics)')
parser.add_argument('--low_top1', type=float, default=1.0, help='Threshold: top1_logit below this is considered low/bad-ish')
parser.add_argument('--low_margin', type=float, default=0.2, help='Threshold: logit_margin below this is considered low/bad-ish')
parser.add_argument('--delta_margin', type=float, default=0.15, help='Expert considered better if margin_ex - margin_cl >= delta')
parser.add_argument('--delta_top1', type=float, default=0.15, help='(Optional) Also require top1_ex - top1_cl >= delta_top1 when comparing')
parser.add_argument('--require_top1_for_better', action='store_true', help='If set, use both margin and top1 deltas to call Expert better/worse')
parser.add_argument('--probe_both', action='store_true', help='Also query Expert when routed to client to compare logits directly (network load!)')
args = parser.parse_args()

print("[client_single] starting (2-logit)")
print("[client_single] args:", vars(args))

os.makedirs(args.save_dir, exist_ok=True)

# ── Device ──────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"

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
        x = x.view(x.size(0), -1)
        return self.block(x)

# ── Data (deterministic 100/class subset = 1000 imgs) ───────────────────────
print("[client_single] preparing CIFAR-10 test set (may download once)...")
try:
    raw_testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=T.ToTensor()
    )
except Exception as e:
    print("[client_single] CIFAR download failed, retrying with download=False:", e)
    raw_testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=False, transform=T.ToTensor()
    )

np_targets = np.array(raw_testset.targets)
desired_per_class = 100
rng = np.random.RandomState(42)
class_indices = {c: np.where(np_targets == c)[0] for c in np.unique(np_targets)}
chosen = []
for c, idxs in class_indices.items():
    if len(idxs) < desired_per_class:
        raise ValueError(f"Not enough samples in class {c}")
    chosen.extend(rng.choice(idxs, size=desired_per_class, replace=False))
subset = Subset(raw_testset, chosen)
loader = DataLoader(subset, batch_size=1, shuffle=False)

# ── Load checkpoints ────────────────────────────────────────────────────────
client_model = Client().to(device)
client_model.load_state_dict(torch.load(args.client_model, map_location=device))
client_model.eval()

rejector = Rejector().to(device)
rejector.load_state_dict(torch.load(args.rejector_model, map_location=device))
rejector.eval()

# ── Net helpers ─────────────────────────────────────────────────────────────
import socket as pysock

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
            with pysock.create_connection((host, port), timeout=timeout) as sock:
                sock.settimeout(timeout)
                try: sock.setsockopt(pysock.SOL_SOCKET, pysock.SO_KEEPALIVE, 1)
                except OSError: pass
                try: sock.setsockopt(pysock.IPPROTO_TCP, pysock.TCP_NODELAY, 1)
                except OSError: pass
                sock.sendall(b'B')
                buf = io.BytesIO(); torch.save(batch_cpu, buf)
                batch_data = buf.getvalue()
                rej_data = rejector_logits.detach().cpu().numpy().astype('<f4').tobytes()
                sock.sendall(struct.pack('>II', len(batch_data), len(rej_data)))
                sock.sendall(batch_data + rej_data)
                plen = int.from_bytes(recvall(sock, 4), 'big')
                pdata = recvall(sock, plen)
            T2 = struct.unpack('>d', pdata[:8])[0]
            T3 = struct.unpack('>d', pdata[8:16])[0]
            logits_data = pdata[24:]
            vals = struct.unpack('<' + 'f' * (len(logits_data)//4), logits_data)
            logits = torch.tensor(vals, dtype=torch.float32).view(batch_cpu.size(0), 10)
            return logits, T2, T3
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(0.5 * attempt)
            else:
                print(f"[client_single] expert request failed after {retries} tries: {e}")
                return None, None, None

# ── Logit metrics helpers ───────────────────────────────────────────────────
@torch.no_grad()
def top1_top2_logits(logits: torch.Tensor):
    vals, idxs = logits.topk(2, dim=1)
    top1 = vals[0,0].item(); top2 = vals[0,1].item()
    c1 = int(idxs[0,0].item()); c2 = int(idxs[0,1].item())
    return top1, top2, c1, c2, (top1 - top2)

# ── Evaluation ──────────────────────────────────────────────────────────────
def run_once():
    host, port = args.host, args.port
    thresholds = [float(x) for x in args.thresholds.split(',')] if args.thresholds else [0.0]
    iterations = int(args.iterations)

    norm = T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    for thr in thresholds:
        route_client = route_expert = 0
        # Paired logs for CSV
        paired_rows = []
        # Distributions for plots
        ex_marg, cl_marg = [], []
        ex_top1, cl_top1 = [], []
        ex_right = cl_right = 0
        ex_wrong = cl_wrong = 0
        ex_bad_list = []      # expert worse than client by logits
        both_bad_list = []    # both have low logits

        print(f"\n[2log] threshold {thr:.3f}")
        for _ in range(iterations):
            for imgs, labels in loader:
                y = int(labels.item())
                img_gpu = imgs.to(device)
                with torch.inference_mode():
                    r_logits = rejector(img_gpu.view(1, -1))
                    prob = torch.softmax(r_logits, dim=1)[0,1].item()

                # Always compute client locally
                img_norm_gpu = norm(img_gpu)
                with torch.inference_mode():
                    logits_c = client_model(img_norm_gpu)
                c_top1, c_top2, c_arg, c_arg2, c_margin = top1_top2_logits(logits_c)
                cl_top1.append(c_top1); cl_marg.append(c_margin)
                pred_c = int(torch.argmax(logits_c, dim=1).item())
                cl_right += int(pred_c == y); cl_wrong += int(pred_c != y)

                expert_used = (prob >= thr)
                if expert_used or args.probe_both:
                    # Query expert over network
                    img_cpu_norm = norm(imgs.clone())
                    t0 = time.perf_counter()
                    logits_e, T2, T3 = request_batch_expert(img_cpu_norm, r_logits[0], host, port, args.conn_timeout, args.retries)
                    t4 = time.perf_counter()
                    if logits_e is None:
                        continue
                    e_top1, e_top2, e_arg, e_arg2, e_margin = top1_top2_logits(logits_e)
                    ex_top1.append(e_top1); ex_marg.append(e_margin)
                    pred_e = int(torch.argmax(logits_e, dim=1).item())
                    ex_right += int(pred_e == y); ex_wrong += int(pred_e != y)
                else:
                    logits_e = None
                    e_top1 = e_top2 = e_margin = float('nan')
                    e_arg = e_arg2 = -1
                    pred_e = -1

                # Route counts
                if expert_used: route_expert += 1
                else: route_client += 1

                # Badness by logits
                c_low = (c_top1 < args.low_top1) or (c_margin < args.low_margin)
                e_low = (e_top1 < args.low_top1) or (e_margin < args.low_margin) if logits_e is not None else False

                expert_better = False; expert_worse = False
                if logits_e is not None:
                    mm = (e_margin - c_margin) >= args.delta_margin
                    tt = (e_top1 - c_top1) >= args.delta_top1
                    expert_better = (mm and (tt if args.require_top1_for_better else True))
                    ww = (c_margin - e_margin) >= args.delta_margin
                    tt2 = (c_top1 - e_top1) >= args.delta_top1
                    expert_worse = (ww and (tt2 if args.require_top1_for_better else True))

                if e_low and c_low:
                    both_bad_list.append({
                        "label": y,
                        "client_top1": c_top1, "client_margin": c_margin, "client_pred": pred_c,
                        "expert_top1": e_top1, "expert_margin": e_margin, "expert_pred": pred_e,
                        "route": "expert" if expert_used else "client"
                    })
                elif expert_worse:
                    ex_bad_list.append({
                        "label": y,
                        "client_top1": c_top1, "client_margin": c_margin, "client_pred": pred_c,
                        "expert_top1": e_top1, "expert_margin": e_margin, "expert_pred": pred_e,
                        "route": "expert" if expert_used else "client"
                    })

                if args.dump_raw:
                    paired_rows.append({
                        "label": y,
                        "route": "expert" if expert_used else "client",
                        "client_pred": pred_c, "client_top1": c_top1, "client_margin": c_margin,
                        "expert_pred": pred_e, "expert_top1": e_top1, "expert_margin": e_margin
                    })

        # ── Save plots and dumps per-threshold ─────────────────────────────
        xlbl = f"{thr:.3f}"
        # Route counts
        plt.figure(figsize=(6,4)); plt.bar(["client","expert"],[route_client,route_expert])
        plt.title(f"Routes @ {xlbl}"); plt.tight_layout()
        plt.savefig(os.path.join(args.save_dir, f"routes_{xlbl}.png")); plt.close()
        # Margins
        if len(cl_marg)>0:
            plt.figure(figsize=(7,4)); plt.hist(cl_marg, bins=30); plt.title(f"Client logit margin @ {xlbl}")
            plt.tight_layout(); plt.savefig(os.path.join(args.save_dir, f"client_margin_{xlbl}.png")); plt.close()
        if len(ex_marg)>0:
            plt.figure(figsize=(7,4)); plt.hist(ex_marg, bins=30); plt.title(f"Expert logit margin @ {xlbl}")
            plt.tight_layout(); plt.savefig(os.path.join(args.save_dir, f"expert_margin_{xlbl}.png")); plt.close()
        # Top1
        if len(cl_top1)>0:
            plt.figure(figsize=(7,4)); plt.hist(cl_top1, bins=30); plt.title(f"Client top1 logit @ {xlbl}")
            plt.tight_layout(); plt.savefig(os.path.join(args.save_dir, f"client_top1_{xlbl}.png")); plt.close()
        if len(ex_top1)>0:
            plt.figure(figsize=(7,4)); plt.hist(ex_top1, bins=30); plt.title(f"Expert top1 logit @ {xlbl}")
            plt.tight_layout(); plt.savefig(os.path.join(args.save_dir, f"expert_top1_{xlbl}.png")); plt.close()
        # Correctness split bars (aggregate across all iters)
        plt.figure(figsize=(5,4)); plt.bar(["right","wrong"],[cl_right, cl_wrong])
        plt.title(f"Client right vs wrong @ {xlbl}"); plt.tight_layout()
        plt.savefig(os.path.join(args.save_dir, f"client_rightwrong_{xlbl}.png")); plt.close()
        plt.figure(figsize=(5,4)); plt.bar(["right","wrong"],[ex_right, ex_wrong])
        plt.title(f"Expert right vs wrong @ {xlbl}"); plt.tight_layout()
        plt.savefig(os.path.join(args.save_dir, f"expert_rightwrong_{xlbl}.png")); plt.close()
        # Scatter paired margins if we have expert for many
        if len(ex_marg)>0:
            # Only plot pairs of same length: use min
            n = min(len(ex_marg), len(cl_marg))
            plt.figure(figsize=(6,6))
            plt.scatter(cl_marg[:n], ex_marg[:n], s=10, alpha=0.5)
            plt.xlabel('Client margin'); plt.ylabel('Expert margin')
            plt.title(f"Paired margins (first {n}) @ {xlbl}")
            plt.tight_layout(); plt.savefig(os.path.join(args.save_dir, f"paired_margins_{xlbl}.png")); plt.close()

        # JSON dumps
        with open(os.path.join(args.save_dir, f"expert_worse_than_client_{xlbl}.json"), 'w') as f:
            json.dump(ex_bad_list, f, indent=2)
        with open(os.path.join(args.save_dir, f"both_bad_{xlbl}.json"), 'w') as f:
            json.dump(both_bad_list, f, indent=2)
        if args.dump_raw and len(paired_rows)>0:
            import csv
            csvp = os.path.join(args.save_dir, f"paired_logits_{xlbl}.csv")
            keys = list(paired_rows[0].keys())
            with open(csvp, 'w', newline='') as cf:
                w = csv.DictWriter(cf, fieldnames=keys);
                w.writeheader(); w.writerows(paired_rows)
            print("[2log] wrote", csvp)

if __name__ == "__main__":
    run_once()


# =============================================
# FILE 2: single_client_logits_3logit.py
# =============================================
# Full script: single-client CIFAR-10 with 3-logit rejector
# Routing policy (unchanged from your code):
#   probs = softmax(r_logits)
#   If prob2 > max(0.5 + thr, prob1)  -> EXPERT
#   Elif prob1 > max(0.5 + thr, prob2) or ((prob2 <= 0.5 + thr) and (prob1 <= 0.5 + thr)) -> CLIENT
# Same logits-based analysis as the 2-logit script.

import json as _json
import os as _os, io as _io, socket as _socket, struct as _struct, time as _time
import numpy as _np
import torch as _torch
import torch.nn as _nn
import torchvision as _tv
import torchvision.transforms as _T
from torch.utils.data import Subset as _Subset, DataLoader as _DataLoader

_BASE_DIR = _os.path.dirname(_os.path.abspath(__file__))
_os.chdir(_BASE_DIR)

import matplotlib as _mpl
_mpl.use("Agg")
import matplotlib.pyplot as _plt

import argparse as _argparse
_p = _argparse.ArgumentParser(description="Single-client CIFAR-10 (3-logit rejector) with logits-based analysis")
_p.add_argument('--host', type=str, default='node21-2.grid.orbit-lab.org')
_p.add_argument('--port', type=int, default=7000)
_p.add_argument('--conn_timeout', type=float, default=30.0)
_p.add_argument('--retries', type=int, default=3)
_p.add_argument('--thresholds', type=str, default='0.3225')
_p.add_argument('--iterations', type=int, default=5)
_p.add_argument('--save_dir', type=str, default='single_run_out_3log')
_p.add_argument('--client_model', type=str, default='ClientModels/client_model70.pt')
_p.add_argument('--rejector_model', type=str, default='RejectorModels/rejector_70_3l.pt')
# Analysis options
_p.add_argument('--dump_raw', action='store_true')
_p.add_argument('--low_top1', type=float, default=1.0)
_p.add_argument('--low_margin', type=float, default=0.2)
_p.add_argument('--delta_margin', type=float, default=0.15)
_p.add_argument('--delta_top1', type=float, default=0.15)
_p.add_argument('--require_top1_for_better', action='store_true')
_p.add_argument('--probe_both', action='store_true')
_args = _p.parse_args()

_os.makedirs(_args.save_dir, exist_ok=True)
_device = "cuda" if _torch.cuda.is_available() else "cpu"

class _Client(_nn.Module):
    def __init__(self):
        super().__init__()
        self.features = _nn.Sequential(
            _nn.Conv2d(3, 32, 3, padding=1), _nn.ReLU(), _nn.MaxPool2d(2),
            _nn.Conv2d(32, 64, 3, padding=1), _nn.ReLU(), _nn.MaxPool2d(2),
            _nn.Dropout(0.25)
        )
        self.classifier = _nn.Sequential(
            _nn.Flatten(), _nn.Linear(64*8*8, 128), _nn.ReLU(), _nn.Linear(128, 10)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

class _Rejector(_nn.Module):
    def __init__(self, hidden=256, drop=0.2):
        super().__init__()
        self.block = _nn.Sequential(
            _nn.Linear(3 * 32 * 32, hidden),
            _nn.BatchNorm1d(hidden),
            _nn.ReLU(inplace=True),
            _nn.Dropout(drop),
            _nn.Linear(hidden, 3)
        )
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.block(x)

print("[client_single] preparing CIFAR-10 test set (may download once)...")
try:
    _raw = _tv.datasets.CIFAR10(root='./data', train=False, download=True, transform=_T.ToTensor())
except Exception as e:
    print("[client_single] CIFAR download failed, retrying with download=False:", e)
    _raw = _tv.datasets.CIFAR10(root='./data', train=False, download=False, transform=_T.ToTensor())

_np_targets = _np.array(_raw.targets)
_desired_per_class = 100
_rng = _np.random.RandomState(42)
_clsidx = {c: _np.where(_np_targets == c)[0] for c in _np.unique(_np_targets)}
_chosen = []
for c, idxs in _clsidx.items():
    if len(idxs) < _desired_per_class:
        raise ValueError(f"Not enough samples in class {c}")
    _chosen.extend(_rng.choice(idxs, size=_desired_per_class, replace=False))
_subset = _Subset(_raw, _chosen)
_loader = _DataLoader(_subset, batch_size=1, shuffle=False)

_client = _Client().to(_device)
_client.load_state_dict(_torch.load(_args.client_model, map_location=_device))
_client.eval()

_rejector = _Rejector().to(_device)
_rejector.load_state_dict(_torch.load(_args.rejector_model, map_location=_device))
_rejector.eval()

import socket as _pysock

def _recvall(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise RuntimeError("Connection closed prematurely")
        buf.extend(chunk)
    return bytes(buf)

def _request_expert(batch_cpu, rej_logits, host, port, timeout, retries):
    for attempt in range(1, retries+1):
        try:
            with _pysock.create_connection((host, port), timeout=timeout) as sock:
                sock.settimeout(timeout)
                try: sock.setsockopt(_pysock.SOL_SOCKET, _pysock.SO_KEEPALIVE, 1)
                except OSError: pass
                try: sock.setsockopt(_pysock.IPPROTO_TCP, _pysock.TCP_NODELAY, 1)
                except OSError: pass
                sock.sendall(b'B')
                buf = _io.BytesIO(); _torch.save(batch_cpu, buf)
                batch_data = buf.getvalue()
                rej_data = rej_logits.detach().cpu().numpy().astype('<f4').tobytes()
                sock.sendall(_struct.pack('>II', len(batch_data), len(rej_data)))
                sock.sendall(batch_data + rej_data)
                plen = int.from_bytes(_recvall(sock, 4), 'big')
                pdata = _recvall(sock, plen)
            T2 = _struct.unpack('>d', pdata[:8])[0]
            T3 = _struct.unpack('>d', pdata[8:16])[0]
            logits_data = pdata[24:]
            vals = _struct.unpack('<' + 'f' * (len(logits_data)//4), logits_data)
            logits = _torch.tensor(vals, dtype=_torch.float32).view(batch_cpu.size(0), 10)
            return logits, T2, T3
        except Exception as e:
            if attempt < retries:
                _time.sleep(0.5*attempt)
            else:
                print(f"[client_single] expert request failed after {retries} tries: {e}")
                return None, None, None

@_torch.no_grad()
def _tt2(logits: _torch.Tensor):
    vals, idxs = logits.topk(2, dim=1)
    top1 = vals[0,0].item(); top2 = vals[0,1].item()
    c1 = int(idxs[0,0].item()); c2 = int(idxs[0,1].item())
    return top1, top2, c1, c2, (top1 - top2)


def _run():
    host, port = _args.host, _args.port
    thr_list = [float(x) for x in _args.thresholds.split(',')] if _args.thresholds else [0.0]
    iters = int(_args.iterations)
    norm = _T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    for thr in thr_list:
        route_client = route_expert = 0
        paired_rows = []
        ex_marg, cl_marg = [], []
        ex_top1, cl_top1 = [], []
        ex_right = cl_right = 0
        ex_wrong = cl_wrong = 0
        ex_bad_list = []
        both_bad_list = []

        print(f"\n[3log] threshold {thr:.3f}")
        for _ in range(iters):
            for imgs, labels in _loader:
                y = int(labels.item())
                img_gpu = imgs.to(_device)

                with _torch.inference_mode():
                    r_logits = _rejector(img_gpu.view(1, -1))
                    probs = _torch.softmax(r_logits, dim=1)
                    p1, p2, p3 = probs[0,0].item(), probs[0,1].item(), probs[0,2].item()

                # Routing policy (unchanged)
                if p2 > max(0.5 + thr, p1):
                    expert_used = True
                elif p1 > max(0.5 + thr, p2) or ((p2 <= 0.5 + thr) and (p1 <= 0.5 + thr)):
                    expert_used = False
                else:
                    # tie region -> default to client
                    expert_used = False

                # Client logits always (local)
                img_norm_gpu = norm(img_gpu)
                with _torch.inference_mode():
                    logits_c = _client(img_norm_gpu)
                c_top1, c_top2, c_arg, c_arg2, c_margin = _tt2(logits_c)
                cl_top1.append(c_top1); cl_marg.append(c_margin)
                pred_c = int(_torch.argmax(logits_c, dim=1).item())
                cl_right += int(pred_c == y); cl_wrong += int(pred_c != y)

                # Expert logits when routed OR when probing both
                if expert_used or _args.probe_both:
                    img_cpu_norm = norm(imgs.clone())
                    logits_e, T2, T3 = _request_expert(img_cpu_norm, r_logits[0], host, port, _args.conn_timeout, _args.retries)
                    if logits_e is None:
                        continue
                    e_top1, e_top2, e_arg, e_arg2, e_margin = _tt2(logits_e)
                    ex_top1.append(e_top1); ex_marg.append(e_margin)
                    pred_e = int(_torch.argmax(logits_e, dim=1).item())
                    ex_right += int(pred_e == y); ex_wrong += int(pred_e != y)
                else:
                    logits_e = None
                    e_top1 = e_top2 = e_margin = float('nan'); pred_e = -1

                # Count route
                if expert_used: route_expert += 1
                else: route_client += 1

                # Badness: by logits thresholds
                c_low = (c_top1 < _args.low_top1) or (c_margin < _args.low_margin)
                e_low = (e_top1 < _args.low_top1) or (e_margin < _args.low_margin) if logits_e is not None else False

                expert_better = False; expert_worse = False
                if logits_e is not None:
                    mm = (e_margin - c_margin) >= _args.delta_margin
                    tt = (e_top1 - c_top1) >= _args.delta_top1
                    expert_better = (mm and (tt if _args.require_top1_for_better else True))
                    ww = (c_margin - e_margin) >= _args.delta_margin
                    tt2 = (c_top1 - e_top1) >= _args.delta_top1
                    expert_worse = (ww and (tt2 if _args.require_top1_for_better else True))

                if e_low and c_low:
                    both_bad_list.append({
                        "label": y,
                        "client_top1": c_top1, "client_margin": c_margin, "client_pred": pred_c,
                        "expert_top1": e_top1, "expert_margin": e_margin, "expert_pred": pred_e,
                        "route": "expert" if expert_used else "client"
                    })
                elif expert_worse:
                    ex_bad_list.append({
                        "label": y,
                        "client_top1": c_top1, "client_margin": c_margin, "client_pred": pred_c,
                        "expert_top1": e_top1, "expert_margin": e_margin, "expert_pred": pred_e,
                        "route": "expert" if expert_used else "client"
                    })

                if _args.dump_raw:
                    paired_rows.append({
                        "label": y,
                        "route": "expert" if expert_used else "client",
                        "client_pred": pred_c, "client_top1": c_top1, "client_margin": c_margin,
                        "expert_pred": pred_e, "expert_top1": e_top1, "expert_margin": e_margin
                    })

        xlbl = f"{thr:.3f}"
        # Route counts
        _plt.figure(figsize=(6,4)); _plt.bar(["client","expert"],[route_client,route_expert])
        _plt.title(f"Routes @ {xlbl}"); _plt.tight_layout()
        _plt.savefig(_os.path.join(_args.save_dir, f"routes_{xlbl}.png")); _plt.close()
        # Margins
        if len(cl_marg)>0:
            _plt.figure(figsize=(7,4)); _plt.hist(cl_marg, bins=30); _plt.title(f"Client logit margin @ {xlbl}")
            _plt.tight_layout(); _plt.savefig(_os.path.join(_args.save_dir, f"client_margin_{xlbl}.png")); _plt.close()
        if len(ex_marg)>0:
            _plt.figure(figsize=(7,4)); _plt.hist(ex_marg, bins=30); _plt.title(f"Expert logit margin @ {xlbl}")
            _plt.tight_layout(); _plt.savefig(_os.path.join(_args.save_dir, f"expert_margin_{xlbl}.png")); _plt.close()
        # Top1
        if len(cl_top1)>0:
            _plt.figure(figsize=(7,4)); _plt.hist(cl_top1, bins=30); _plt.title(f"Client top1 logit @ {xlbl}")
            _plt.tight_layout(); _plt.savefig(_os.path.join(_args.save_dir, f"client_top1_{xlbl}.png")); _plt.close()
        if len(ex_top1)>0:
            _plt.figure(figsize=(7,4)); _plt.hist(ex_top1, bins=30); _plt.title(f"Expert top1 logit @ {xlbl}")
            _plt.tight_layout(); _plt.savefig(_os.path.join(_args.save_dir, f"expert_top1_{xlbl}.png")); _plt.close()
        # Correctness split bars (aggregate visibility)
        _plt.figure(figsize=(5,4)); _plt.bar(["right","wrong"],[cl_right, cl_wrong])
        _plt.title(f"Client right vs wrong @ {xlbl}"); _plt.tight_layout()
        _plt.savefig(_os.path.join(_args.save_dir, f"client_rightwrong_{xlbl}.png")); _plt.close()
        _plt.figure(figsize=(5,4)); _plt.bar(["right","wrong"],[ex_right, ex_wrong])
        _plt.title(f"Expert right vs wrong @ {xlbl}"); _plt.tight_layout()
        _plt.savefig(_os.path.join(_args.save_dir, f"expert_rightwrong_{xlbl}.png")); _plt.close()
        # Paired scatter (when expert logits exist for many)
        if len(ex_marg)>0:
            n = min(len(ex_marg), len(cl_marg))
            _plt.figure(figsize=(6,6)); _plt.scatter(cl_marg[:n], ex_marg[:n], s=10, alpha=0.5)
            _plt.xlabel('Client margin'); _plt.ylabel('Expert margin')
            _plt.title(f"Paired margins (first {n}) @ {xlbl}")
            _plt.tight_layout(); _plt.savefig(_os.path.join(_args.save_dir, f"paired_margins_{xlbl}.png")); _plt.close()

        with open(_os.path.join(_args.save_dir, f"expert_worse_than_client_{xlbl}.json"), 'w') as f:
            _json.dump(ex_bad_list, f, indent=2)
        with open(_os.path.join(_args.save_dir, f"both_bad_{xlbl}.json"), 'w') as f:
            _json.dump(both_bad_list, f, indent=2)
        if _args.dump_raw and len(paired_rows)>0:
            import csv as _csv
            csvp = _os.path.join(_args.save_dir, f"paired_logits_{xlbl}.csv")
            keys = list(paired_rows[0].keys())
            with open(csvp, 'w', newline='') as cf:
                w = _csv.DictWriter(cf, fieldnames=keys)
                w.writeheader(); w.writerows(paired_rows)
            print("[3log] wrote", csvp)

if __name__ == "__main__":
    _run()
