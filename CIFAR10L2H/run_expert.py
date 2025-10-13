# expertV2.py (expects already-normalized tensors; T2/T3 wrap model forward only)
import io
import socket
import struct
import threading
import time

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: F401

HOST       = '0.0.0.0'
PORT       = 6000
MODEL_PATH = 'expert_model.pt'
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.set_num_threads(1)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

class Expert(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.3),

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.3),

            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.3),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256),         nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, 10),
        )
    def forward(self, x):
        return self.classifier(self.features(x))

expert = Expert().to(DEVICE)
expert.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
expert.eval()

with torch.inference_mode():
    warm = torch.zeros(1, 3, 32, 32, device=DEVICE)
    _ = expert(warm)
print("[Expert] Warm-up done", flush=True)

client_last_seen = {}
request_count    = 0
start_time       = time.time()
calls_per_second = 0.0
state_lock = threading.Lock()

def recvall(conn: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise RuntimeError('Connection closed')
        buf.extend(chunk)
    return bytes(buf)

def handle_batch_expert(conn: socket.socket, addr):
    global request_count, start_time, calls_per_second
    try:
        # ip header from Go proxy
        raw = recvall(conn, 4)
        ip_len = struct.unpack('>I', raw)[0]
        client_ip = recvall(conn, ip_len).decode('utf-8', errors='replace')

        # command
        cmd = recvall(conn, 1)
        if cmd != b'B':
            return

        now = time.time()
        with state_lock:
            client_last_seen[client_ip] = now
            for ip, ts in list(client_last_seen.items()):
                if now - ts > 1.0:
                    del client_last_seen[ip]

            request_count += 1
            elapsed = now - start_time
            if elapsed >= 1.0:
                calls_per_second = request_count / elapsed
                request_count = 0
                start_time = now

        # lengths + payload
        lens = recvall(conn, 8)
        batch_len, rej_len = struct.unpack('>II', lens)
        full_data = recvall(conn, batch_len + rej_len)
        batch_data = full_data[:batch_len]
        rejector_data = full_data[batch_len:]

        # Optional sanity: rejector logits are LE float32
        if rej_len % 4 != 0:
            raise RuntimeError(f"Rejector logits length not multiple of 4: {rej_len}")

        # Load already-normalized tensor sent by client
        batch = torch.load(io.BytesIO(batch_data), map_location=DEVICE)
        if batch.dtype != torch.float32:
            batch = batch.float()

        # Model forward only — T2/T3 wrap inference (deserialization is outside)
        T2 = time.perf_counter()
        with torch.inference_mode():
            logits = expert(batch)
        T3 = time.perf_counter()

        logits_bytes = logits.detach().cpu().numpy().astype('<f4').tobytes()  # LE payload

        with state_lock:
            client_count = len(client_last_seen)
            cps_snapshot = calls_per_second

        resp = (
            struct.pack('>d', T2) +            # BE header
            struct.pack('>d', T3) +
            struct.pack('>I', client_count) +
            struct.pack('>f', cps_snapshot) +
            logits_bytes                         # LE payload
        )
        conn.sendall(len(resp).to_bytes(4, 'big') + resp)

    except Exception as e:
        try:
            print(f"[Expert] Handler error from {addr}: {e}", flush=True)
        except Exception:
            pass
    finally:
        try:
            conn.close()
        except Exception:
            pass

def main():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((HOST, PORT))
        srv.listen(1024)
        print(f"[Expert] Listening on {HOST}:{PORT}", flush=True)

        while True:
            conn, addr = srv.accept()
            try:
                conn.settimeout(30.0)
                try:
                    conn.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                except OSError:
                    pass
                threading.Thread(target=handle_batch_expert, args=(conn, addr), daemon=True).start()
            except Exception as e:
                try:
                    print(f"[Expert] Accept error: {e}", flush=True)
                except Exception:
                    pass
                try:
                    conn.close()
                except Exception:
                    pass

if __name__ == '__main__':
    main()
