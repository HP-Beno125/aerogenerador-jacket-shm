#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Servidor TCP con GUI (Tkinter) para 8 emisores ESP32 + ADXL355 - VERSIÓN CORREGIDA v2.2
MODELO: sockets bloqueantes, un hilo por sensor (SIN asyncio)
CORRECCIONES v2.2 (rendimiento GUI):
  1. Threshold de count_total: 200 → 50 muestras (actualización ~5×/s a 250 Hz)
  2. Log limitado a MAX_LOG_LINES; escritura en batch (una sola operación por ciclo)
  3. _drain_events cada 100 ms en vez de 200 ms
  4. Caché del Treeview: solo llama a tree.item() si el valor cambió
CORRECCIONES v2.1 (conexión):
  5. Timeout 300 s (cubre escalonado sensor 8 = 84 s + margen)
  6. Captura de TimeoutError y BlockingIOError además de socket.timeout
"""

import socket, threading, time, csv, os, glob, statistics, queue, sys, webbrowser, re
from datetime import datetime

try:
    import tkinter as tk
    from tkinter import ttk, messagebox
except Exception:
    print("Tkinter no disponible.")
    sys.exit(1)

_HAVE_PANDAS = False
_HAVE_OPENPYXL = False
try:
    import pandas as pd
    _HAVE_PANDAS = True
except Exception:
    try:
        import openpyxl
        _HAVE_OPENPYXL = True
    except Exception:
        pass

# Máximo de líneas visibles en el log antes de recortar
MAX_LOG_LINES = 250


def now_us() -> int:
    return time.time_ns() // 1_000


class SensorServer:
    def __init__(self, gui_cb, num_sensors=8, stale_ms=50):
        self.gui_cb = gui_cb
        self.num_sensors = num_sensors
        self.stale_ms = int(stale_ms)
        self.host = "0.0.0.0"
        self.port = 5000

        self.srv_sock = None
        self.accept_thread = None
        self.stop_event = threading.Event()

        # Salida en disco
        self.ts_root = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.out_dir = os.path.abspath(f"per_sensor_{self.ts_root}")
        os.makedirs(self.out_dir, exist_ok=True)

        self.trial_no = 0
        self.base_name_user = "1_1_1A"
        self.base_name_current = None

        # CSV global
        self.global_csv_file = None
        self.global_writer = None
        self.global_lock = threading.Lock()

        # Estado compartido
        self.last_val = {sid: {"x": 0.0, "y": 0.0, "z": 0.0, "t": 0.0}
                         for sid in range(1, self.num_sensors + 1)}
        self.last_lock = threading.Lock()

        # Sockets por sensor_id (para enviar comandos)
        self.socks_by_id = {}
        self.socks_lock = threading.Lock()

        # Todos los sockets activos (para broadcast y cierre)
        self.all_socks = set()
        self.all_socks_lock = threading.Lock()

        # Contadores
        self.sample_total = {sid: 0 for sid in range(1, self.num_sensors + 1)}
        self.count_lock = threading.Lock()

        self.seq_last = {}

    # ---------------- Nombres de archivo ----------------
    def _unique_base_name(self):
        base = (self.base_name_user or "1_1_1A").strip()
        while True:
            global_path = os.path.abspath(f"{base}.csv")
            per_sensor_glob = glob.glob(os.path.join(self.out_dir, f"sensor_*_{base}.csv"))
            if not os.path.exists(global_path) and not per_sensor_glob:
                return base
            m = re.match(r"^(.*_)(\d+)(A)$", base, re.IGNORECASE)
            if m:
                prefix, num, suf = m.groups()
                base = f"{prefix}{int(num)+1}{suf}"
            else:
                base = base + "_2A"

    def _open_global_csv(self):
        with self.global_lock:
            try:
                if self.global_csv_file:
                    self.global_csv_file.flush()
                    self.global_csv_file.close()
            except Exception:
                pass

            if not self.base_name_current:
                self.base_name_current = self._unique_base_name()

            path = f"{self.base_name_current}.csv"
            self.global_csv_file = open(path, "a", newline="", encoding="utf-8",
                                        buffering=1 << 20)
            self.global_writer = csv.writer(self.global_csv_file)

            header = []
            for sid in range(1, self.num_sensors + 1):
                header += [f"x{sid}", f"y{sid}", f"z{sid}", f"id{sid}"]
            self.global_writer.writerow(header)

        self.gui_cb("status", f"[TRIAL] Abierto global: {path}")

    def ensure_global_open(self):
        if self.trial_no == 0 or self.global_writer is None:
            if self.trial_no == 0:
                self.trial_no = 1
            if not self.base_name_current:
                self.base_name_current = self._unique_base_name()
            self._open_global_csv()
            self.seq_last[self.trial_no] = {}
            with self.count_lock:
                self.sample_total = {sid: 0 for sid in range(1, self.num_sensors + 1)}

    def new_trial(self):
        self.trial_no += 1
        self.base_name_current = self._unique_base_name()

        with self.last_lock:
            for sid in self.last_val:
                self.last_val[sid] = {"x": 0.0, "y": 0.0, "z": 0.0, "t": 0.0}

        with self.count_lock:
            self.sample_total = {sid: 0 for sid in range(1, self.num_sensors + 1)}
        self.seq_last[self.trial_no] = {}

        self._open_global_csv()
        self.gui_cb("status",
                    f"[TRIAL] Nueva prueba #{self.trial_no:02d} base='{self.base_name_current}' lista.")

    # ---------------- Setters ----------------
    def set_host_port(self, host, port):
        self.host = host
        self.port = int(port)

    def set_stale_ms(self, stale_ms):
        self.stale_ms = int(stale_ms)

    def set_base_name(self, base_name):
        self.base_name_user = (base_name or "1_1_1A").strip()

    # ---------------- Arranque / parada ----------------
    def start(self):
        if self.accept_thread and self.accept_thread.is_alive():
            return
        self.stop_event.clear()
        self.srv_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.srv_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.srv_sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        self.srv_sock.bind((self.host, self.port))
        self.srv_sock.listen(16)
        self.srv_sock.settimeout(1.0)

        self.accept_thread = threading.Thread(target=self._accept_loop, daemon=True)
        self.accept_thread.start()
        self.gui_cb("status", f"Servidor escuchando en {self.host}:{self.port}")

    def _accept_loop(self):
        while not self.stop_event.is_set():
            try:
                conn, addr = self.srv_sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break

            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            conn.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            try:
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 5)
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 3)
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 3)
            except Exception:
                pass

            with self.all_socks_lock:
                self.all_socks.add(conn)
            t = threading.Thread(target=self._handle_client, args=(conn, addr), daemon=True)
            t.start()

    def stop(self):
        self.stop_event.set()
        try:
            if self.srv_sock:
                self.srv_sock.close()
        except Exception:
            pass

        with self.all_socks_lock:
            for s in list(self.all_socks):
                try:
                    s.shutdown(socket.SHUT_RDWR)
                except Exception:
                    pass
                try:
                    s.close()
                except Exception:
                    pass
            self.all_socks.clear()

        with self.socks_lock:
            self.socks_by_id.clear()

        self.gui_cb("status", "Servidor detenido.")

    def close_all(self):
        self.stop()
        with self.global_lock:
            try:
                if self.global_csv_file:
                    self.global_csv_file.flush()
                    self.global_csv_file.close()
            except Exception:
                pass

    # ---------------- Envio de comandos ----------------
    def _send_to_sock(self, sock, data: bytes):
        try:
            sock.sendall(data)
            return True
        except Exception:
            return False

    def broadcast_resync(self):
        with self.all_socks_lock:
            targets = list(self.all_socks)
        for s in targets:
            self._send_to_sock(s, b"RESYNC\n")
        self.gui_cb("status", "RESYNC enviado a todos los emisores.")

    def send_command(self, dest_id: int, cmd: str):
        """Envía comando. Si dest_id=0, envía a todos los sensores conectados."""
        line = (cmd.strip() + "\n").encode()

        if dest_id == 0:
            with self.all_socks_lock:
                targets = list(self.all_socks)
        else:
            with self.socks_lock:
                s = self.socks_by_id.get(dest_id)
                targets = [s] if s else []

        success_count = 0
        for s in targets:
            if self._send_to_sock(s, line):
                success_count += 1

        if dest_id == 0:
            self.gui_cb("status",
                        f">>> '{cmd.strip()}' a TODOS ({success_count}/{len(targets)} enviados)")
        else:
            if success_count > 0:
                self.gui_cb("status", f">>> '{cmd.strip()}' a ID {dest_id}")
            else:
                self.gui_cb("status", f"[ERROR] Sensor {dest_id} no conectado")

    # ---------------- Handshake de sincronizacion ----------------
    def _time_sync(self, conn, rf, rounds=8, timeout=1.0):
        offs = []
        conn.settimeout(timeout)
        for _ in range(rounds):
            try:
                t0 = now_us()
                conn.sendall(b"SYNC_REQ\n")
                line = rf.readline()
                t2 = now_us()
                if not line:
                    continue
                s = line.decode("utf-8", "ignore").strip()
                if not s.startswith("TS "):
                    continue
                t_client = int(s.split()[1])
                off = ((t0 + t2) // 2) - t_client
                offs.append(off)
                conn.sendall(f"OFF {off}\n".encode())
            except Exception:
                continue
        try:
            conn.sendall(b"SYNC_DONE\n")
        except Exception:
            pass
        return (int(statistics.median(offs)) if offs else 0), offs

    # ---------------- Manejador de cliente (un hilo por conexion) ----------------
    def _handle_client(self, conn, addr):
        ip = addr[0]
        self.gui_cb("status", f"+ Conectado {addr}")

        rf = conn.makefile("rb")

        sensor_id = None
        sensor_file = None
        sensor_writer = None
        rows_since_flush = 0

        try:
            off_med, offs = self._time_sync(conn, rf)
            self.gui_cb("status", f"[SYNC] {ip} offset_us={off_med} (muestras={len(offs)})")

            # 300 s: cubre escalonado sensor 8 (84 s) + captura (120 s) + margen
            conn.settimeout(300.0)

            while not self.stop_event.is_set():
                try:
                    raw = rf.readline()
                except (socket.timeout, TimeoutError, BlockingIOError):
                    self.gui_cb("status", f"[TIMEOUT] {ip} esperando datos...")
                    continue
                except Exception as e:
                    self.gui_cb("status", f"[ERR read] {ip}: {e}")
                    break

                if not raw:
                    break

                s = raw.decode("utf-8", "ignore").strip()
                if not s:
                    continue

                if s.startswith(("TS ", "OFF ", "SYNC")):
                    continue

                if s.upper().startswith("HEARTBEAT"):
                    continue

                if s.upper().startswith("ACK "):
                    self.gui_cb("status", f"[ACK] {s}")
                    continue

                if s.upper().startswith("HELLO"):
                    self.gui_cb("status", f"[HELLO] {s}")
                    continue

                if s.upper().startswith("RESYNC"):
                    off_med, offs = self._time_sync(conn, rf)
                    conn.settimeout(300.0)
                    self.gui_cb("status", f"[RESYNC] {ip} offset_us={off_med}")
                    continue

                # Datos: "sid,x,y,z"  o  "sid,seq,t_us,x,y,z"
                parts = s.split(",")
                try:
                    if len(parts) == 4:
                        sid = int(parts[0]); x = float(parts[1]); y = float(parts[2]); z = float(parts[3])
                        seq = -1
                    elif len(parts) == 6:
                        sid = int(parts[0]); seq = int(parts[1])
                        x = float(parts[3]); y = float(parts[4]); z = float(parts[5])
                    else:
                        continue
                except ValueError:
                    continue

                self.ensure_global_open()

                if sensor_id is None:
                    sensor_id = sid
                    with self.socks_lock:
                        self.socks_by_id[sensor_id] = conn
                    path = os.path.join(self.out_dir,
                                        f"sensor_{sensor_id}_{self.base_name_current}.csv")
                    sensor_file = open(path, "a", newline="", encoding="utf-8",
                                       buffering=1 << 16)
                    sensor_writer = csv.writer(sensor_file)
                    sensor_writer.writerow(["timestamp_iso", "client_ip", "sensor_id",
                                            "seq", "x", "y", "z"])
                    self.gui_cb("status", f"[FS] Sensor {sensor_id}: {path}")
                    self.gui_cb("conn", (sensor_id, True))

                if seq >= 0:
                    dtrial = self.seq_last.setdefault(self.trial_no, {})
                    if seq <= dtrial.get(sensor_id, -1):
                        continue
                    dtrial[sensor_id] = seq

                sensor_writer.writerow([datetime.now().isoformat(), ip, sid, seq,
                                        f"{x:.6f}", f"{y:.6f}", f"{z:.6f}"])
                rows_since_flush += 1

                # ── CORRECCIÓN: threshold 200 → 50 ──────────────────────────
                # A 250 Hz, el contador se actualiza ~5 veces/segundo por sensor
                # (antes era cada 0.8 s). Con 8 sensores son 40 eventos/s, bien
                # dentro de lo que el drain de 100 ms puede manejar.
                with self.count_lock:
                    self.sample_total[sensor_id] += 1
                    total = self.sample_total[sensor_id]
                if total % 50 == 0:
                    self.gui_cb("count_total", (sensor_id, total))
                # ────────────────────────────────────────────────────────────

                with self.last_lock:
                    self.last_val[sid] = {"x": x, "y": y, "z": z, "t": time.time()}
                    row = []
                    nowt = time.time()
                    for k in range(1, self.num_sensors + 1):
                        ik = self.last_val[k]
                        stale = (ik["t"] == 0.0) or ((nowt - ik["t"]) * 1000.0 > self.stale_ms)
                        if stale:
                            vx = vy = vz = 0.0
                        else:
                            vx, vy, vz = ik["x"], ik["y"], ik["z"]
                        row += [f"{vx:.4f}", f"{vy:.4f}", f"{vz:.4f}", str(k)]
                with self.global_lock:
                    if self.global_writer:
                        self.global_writer.writerow(row)

                if rows_since_flush >= 500:
                    try:
                        sensor_file.flush()
                    except Exception:
                        pass
                    rows_since_flush = 0

        except Exception as e:
            self.gui_cb("status", f"[ERR] {addr}: {e}")
        finally:
            try:
                if sensor_file:
                    sensor_file.flush()
                    sensor_file.close()
            except Exception:
                pass
            with self.socks_lock:
                if sensor_id and self.socks_by_id.get(sensor_id) is conn:
                    del self.socks_by_id[sensor_id]
            with self.all_socks_lock:
                self.all_socks.discard(conn)
            try:
                rf.close()
            except Exception:
                pass
            try:
                conn.close()
            except Exception:
                pass
            if sensor_id:
                self.gui_cb("conn", (sensor_id, False))
            self.gui_cb("status", f"- Desconectado {addr}")

    def flush_all(self):
        with self.global_lock:
            try:
                if self.global_csv_file:
                    self.global_csv_file.flush()
            except Exception:
                pass


# =============================================================================
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Servidor Sensores 8x (ESP32/ADXL355) - Threaded v2.2")
        self.geometry("1020x660")
        self.resizable(True, True)

        self.event_q = queue.Queue()
        self.server = SensorServer(self._on_server_event)

        # ── Caché del Treeview ───────────────────────────────────────────────
        # Almacena el último valor mostrado por sensor para evitar llamadas
        # redundantes a tree.item(), que son costosas en Tkinter.
        # Formato: {sid: {"conn": "No", "count": 0}}
        self._tree_cache = {sid: {"conn": "No", "count": 0} for sid in range(1, 9)}
        # ─────────────────────────────────────────────────────────────────────

        self._build_ui()
        self.after(100, self._drain_events)   # primer ciclo a 100 ms

    def _build_ui(self):
        pad = {'padx': 8, 'pady': 6}

        top = ttk.Frame(self); top.pack(fill="x", **pad)
        ttk.Label(top, text="Host:").grid(row=0, column=0, sticky="e")
        self.host_var = tk.StringVar(value="0.0.0.0")
        ttk.Entry(top, textvariable=self.host_var, width=18).grid(row=0, column=1, sticky="w")
        ttk.Label(top, text="Puerto:").grid(row=0, column=2, sticky="e")
        self.port_var = tk.StringVar(value="5000")
        ttk.Entry(top, textvariable=self.port_var, width=8).grid(row=0, column=3, sticky="w", padx=(0, 12))
        ttk.Label(top, text="STALE_MS:").grid(row=0, column=4, sticky="e")
        self.stale_var = tk.StringVar(value="50")
        ttk.Entry(top, textvariable=self.stale_var, width=8).grid(row=0, column=5, sticky="w", padx=(0, 12))
        ttk.Label(top, text="Prueba #:").grid(row=0, column=6, sticky="e")
        self.prueba_var = tk.StringVar(value="0")
        ttk.Label(top, textvariable=self.prueba_var, width=6).grid(row=0, column=7, sticky="w")
        ttk.Label(top, text="Archivo base:").grid(row=0, column=8, sticky="e")
        self.base_name_var = tk.StringVar(value="1_1_1A")
        ttk.Entry(top, textvariable=self.base_name_var, width=16).grid(row=0, column=9, sticky="w")

        btns = ttk.Frame(self); btns.pack(fill="x", **pad)
        self.btn_start   = ttk.Button(btns, text="Conectar (Start)",                command=self._start_server)
        self.btn_stop    = ttk.Button(btns, text="Desconectar (Stop)",               command=self._stop_server,  state="disabled")
        self.btn_resync  = ttk.Button(btns, text="Re-sincronizar emisores",          command=self._resync,       state="disabled")
        self.btn_excel   = ttk.Button(btns, text="Guardar a Excel (todas las pruebas)", command=self._save_excel, state="disabled")
        self.btn_open    = ttk.Button(btns, text="Abrir carpeta",                    command=self._open_folder)
        self.btn_start.grid(row=0, column=0, padx=4)
        self.btn_stop.grid(row=0, column=1, padx=4)
        self.btn_resync.grid(row=0, column=2, padx=4)
        self.btn_excel.grid(row=0, column=3, padx=4)
        self.btn_open.grid(row=0, column=4, padx=4)

        cmdfrm = ttk.LabelFrame(self, text="Comandos a emisores"); cmdfrm.pack(fill="x", **pad)
        ttk.Label(cmdfrm, text="Destino:").grid(row=0, column=0, sticky="e")
        self.dest_var = tk.StringVar(value="0 (Todos)")
        opts = ["0 (Todos)"] + [str(i) for i in range(1, 9)]
        self.dest_cb = ttk.Combobox(cmdfrm, values=opts, textvariable=self.dest_var, width=12, state="readonly")
        self.dest_cb.grid(row=0, column=1, sticky="w", padx=(0, 12))
        self.btn_iniciar   = ttk.Button(cmdfrm, text="INICIAR",                command=self._cmd_iniciar,   state="disabled")
        self.btn_reiniciar = ttk.Button(cmdfrm, text="REINICIAR (solo reset)", command=self._cmd_reiniciar, state="disabled")
        self.btn_iniciar.grid(row=0, column=2, padx=4)
        self.btn_reiniciar.grid(row=0, column=3, padx=4)

        self.log = tk.Text(self, height=14); self.log.pack(fill="both", expand=True, **pad)
        self.log.configure(state="disabled")

        tblfrm = ttk.LabelFrame(self, text="Sensores"); tblfrm.pack(fill="x", **pad)
        cols = ("ID", "Conectado", "Muestras")
        self.tree = ttk.Treeview(tblfrm, columns=cols, show="headings", height=8)
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=140 if c != "Muestras" else 200, anchor="center")
        self.tree.pack(fill="x", padx=6, pady=6)
        for sid in range(1, 9):
            self.tree.insert("", "end", iid=str(sid), values=(sid, "No", 0))

        self.statusbar = ttk.Label(self, anchor="w"); self.statusbar.pack(fill="x", padx=8, pady=4)
        self._set_status("Listo. Configure y presione 'Conectar'.")

    # ---------------- Servidor ----------------
    def _start_server(self):
        try:
            self.server.set_host_port(self.host_var.get().strip(), int(self.port_var.get().strip()))
            self.server.set_stale_ms(int(self.stale_var.get().strip()))
            self.server.set_base_name(self.base_name_var.get().strip())
            self.server.start()
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return
        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.btn_resync.config(state="normal")
        self.btn_excel.config(state="normal")
        self.btn_iniciar.config(state="normal")
        self.btn_reiniciar.config(state="normal")
        self._update_prueba_label()

    def _stop_server(self):
        try:
            self.server.close_all()
        except Exception as e:
            self._append_log(f"[Stop] {e}")
        self.btn_start.config(state="normal")
        self.btn_stop.config(state="disabled")
        self.btn_resync.config(state="disabled")
        self.btn_iniciar.config(state="disabled")
        self.btn_reiniciar.config(state="disabled")

    def _resync(self):
        self.server.broadcast_resync()

    def _cmd_iniciar(self):
        self.server.set_base_name(self.base_name_var.get())
        self.server.new_trial()
        self._update_prueba_label()
        dest = self._parse_dest()
        self.server.send_command(dest, "INICIAR")

    def _cmd_reiniciar(self):
        dest = self._parse_dest()
        self.server.send_command(dest, "REINICIAR")

    def _save_excel(self):
        self.server.flush_all()
        made, warn = [], []

        for csv_path in sorted(p for p in glob.glob("*.csv")
                               if not os.path.basename(p).startswith("sensor_")):
            xlsx_path = os.path.splitext(csv_path)[0] + ".xlsx"
            try:
                if _HAVE_PANDAS:
                    pd.read_csv(csv_path).to_excel(xlsx_path, index=False)
                elif _HAVE_OPENPYXL:
                    from openpyxl import Workbook
                    wb = Workbook(); ws = wb.active; ws.title = "raw_join"
                    with open(csv_path, "r", encoding="utf-8") as f:
                        for row in csv.reader(f):
                            ws.append(row)
                    wb.save(xlsx_path)
                else:
                    warn.append("No hay pandas/openpyxl: se mantienen CSV.")
                    break
                made.append(xlsx_path)
            except Exception as e:
                self._append_log(f"[Excel] {csv_path}: {e}")

        for csv_path in sorted(glob.glob(os.path.join(self.server.out_dir, "sensor_*_*.csv"))):
            xlsx_path = os.path.splitext(csv_path)[0] + ".xlsx"
            try:
                if _HAVE_PANDAS:
                    pd.read_csv(csv_path).to_excel(xlsx_path, index=False)
                elif _HAVE_OPENPYXL:
                    from openpyxl import Workbook
                    wb = Workbook(); ws = wb.active; ws.title = "sensor"
                    with open(csv_path, "r", encoding="utf-8") as f:
                        for row in csv.reader(f):
                            ws.append(row)
                    wb.save(xlsx_path)
                else:
                    warn.append("No hay pandas/openpyxl: se mantienen CSV.")
                    break
                made.append(xlsx_path)
            except Exception as e:
                self._append_log(f"[Excel] {csv_path}: {e}")

        if made:
            self._append_log(f"[Excel] Generado: {', '.join(made)}")
            messagebox.showinfo("Excel", "Archivos Excel generados exitosamente.")
        if warn:
            self._append_log(f"[Excel] Aviso: {'; '.join(warn)}")
            messagebox.showwarning("Excel", "\n".join(warn))

    def _open_folder(self):
        path = os.path.abspath(".")
        webbrowser.open(f"file:///{path}".replace("\\", "/"))

    def _parse_dest(self) -> int:
        sel = self.dest_var.get().strip()
        if sel.startswith("0"):
            return 0
        try:
            return int(sel)
        except Exception:
            return 0

    def _update_prueba_label(self):
        self.prueba_var.set(str(self.server.trial_no))

    def _on_server_event(self, etype, payload):
        self.event_q.put((etype, payload))

    # ---------------- Drain optimizado ----------------
    def _drain_events(self):
        """
        Ciclo de 100 ms (antes 200 ms). Mejoras:
        - Acumula líneas de log y las escribe en UNA sola operación al final.
        - Usa _tree_cache para omitir tree.item() cuando el valor no cambió.
        - Recorta el log cuando supera MAX_LOG_LINES.
        """
        log_lines = []
        last_status = None
        processed = 0

        try:
            while processed < 100:
                etype, payload = self.event_q.get_nowait()
                processed += 1

                if etype == "status":
                    log_lines.append(payload)
                    last_status = payload

                elif etype == "count_total":
                    sid, total = payload
                    cache = self._tree_cache.get(sid, {})
                    # Solo actualizar el Treeview si el valor cambió
                    if cache.get("count") != total:
                        self._tree_cache[sid]["count"] = total
                        if str(sid) in self.tree.get_children():
                            vals = list(self.tree.item(str(sid), "values"))
                            vals[1] = "Sí"
                            vals[2] = total
                            self.tree.item(str(sid), values=vals)

                elif etype == "conn":
                    sid, connected = payload
                    conn_str = "Sí" if connected else "No"
                    cache = self._tree_cache.get(sid, {})
                    new_count = cache.get("count", 0) if connected else 0
                    if cache.get("conn") != conn_str or (not connected):
                        self._tree_cache[sid]["conn"]  = conn_str
                        self._tree_cache[sid]["count"] = new_count
                        if str(sid) in self.tree.get_children():
                            vals = list(self.tree.item(str(sid), "values"))
                            vals[1] = conn_str
                            if not connected:
                                vals[2] = 0
                            self.tree.item(str(sid), values=vals)

        except queue.Empty:
            pass

        # ── Escritura batch del log ──────────────────────────────────────────
        if log_lines:
            self.log.configure(state="normal")
            self.log.insert("end", "\n".join(log_lines) + "\n")

            # Recortar si supera el límite (operación barata: una sola delete)
            total_lines = int(self.log.index("end-1c").split(".")[0])
            if total_lines > MAX_LOG_LINES:
                excess = total_lines - MAX_LOG_LINES
                self.log.delete("1.0", f"{excess + 1}.0")

            self.log.see("end")
            self.log.configure(state="disabled")

            if last_status:
                self._set_status(last_status)
        # ────────────────────────────────────────────────────────────────────

        self.after(100, self._drain_events)   # reschedule a 100 ms

    def _append_log(self, text):
        """Escritura directa (para mensajes fuera del drain, ej. Excel)."""
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n")
        total_lines = int(self.log.index("end-1c").split(".")[0])
        if total_lines > MAX_LOG_LINES:
            excess = total_lines - MAX_LOG_LINES
            self.log.delete("1.0", f"{excess + 1}.0")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _set_status(self, txt):
        self.statusbar.config(text=txt)


if __name__ == "__main__":
    app = App()
    app.mainloop()