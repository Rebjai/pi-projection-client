#!/usr/bin/env python3
"""
client/client.py
RPi client that registers to the server, downloads assigned tiles, saves config locally,
applies homography on SHOW and writes processed images for each HDMI output.
It spawns display_worker processes (one per configured display).
"""

import os
import sys
import time
import json
import errno
import signal
import shutil
import requests
import threading
import traceback
from multiprocessing import Process, Queue
from urllib.parse import urlparse
from typing import Dict, Any, List
from display_worker import run_display_worker


import numpy as np
import cv2
import socketio  # python-socketio client

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(BASE_DIR, "config")
LOCAL_CONFIG_PATH = os.path.join(CONFIG_DIR, "client.json")  # local editable config
TILES_DIR = os.path.join(BASE_DIR, "tiles")
DISPLAY_OUT_DIR = os.path.join(BASE_DIR, "display_out")
LOG_PREFIX = "[client]"

# Ensure folders
os.makedirs(CONFIG_DIR, exist_ok=True)
os.makedirs(TILES_DIR, exist_ok=True)
os.makedirs(DISPLAY_OUT_DIR, exist_ok=True)

# Display workers processes
# Key: drm_name, Value: (Process, Queue)
DISPLAY_PROCS = {}
CMD_QUEUES = {}
ACK_QUEUE = Queue()

# --- Helpers: local config ---
def create_template_config(path):
    template = {
        "client_id": "pi-01",
        "server_url": "http://127.0.0.1:5000",   # EDIT this to your server URL
        # assignments will be created/updated by ASSIGN_TILES
        "assignments": [],   # list of {"image": "name.png", "tile_index": 0, "hdmi_output": 0, "file": "path", "downloaded_at": 0}
        "homographies": {}   # "tile_index": [[...],[...],[...]]
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(template, f, indent=2)
    print(LOG_PREFIX, "Template config written to", path)
    print(LOG_PREFIX, "Edit server_url and client_id then restart the client.")
    return template

def load_local_config() -> Dict[str, Any]:
    if not os.path.exists(LOCAL_CONFIG_PATH):
        create_template_config(LOCAL_CONFIG_PATH)
        sys.exit(1)  # exit so user can edit config
    with open(LOCAL_CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_local_config(cfg: Dict[str, Any]):
    with open(LOCAL_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

def ensure_displays_in_config(path="config/client.json"):
    config = load_local_config()
    # If displays not set, try to auto-detect HDMI outputs

    detected = detect_display_outputs()
    if not detected:
        detected = []  # fallback
    config["displays"] = detected
    save_local_config(config)

    return config

# --- Display worker launcher (multiprocess) ---
def start_display_worker_process(drm_name: str) -> Process:
    """Spawn display_worker.py as a separate process for a given DRM display."""
    worker_py = os.path.join(BASE_DIR, "display_worker.py")
    if not os.path.exists(worker_py):
        print(LOG_PREFIX, "display_worker.py not found; create it in same folder as client.py")
        return None
    cmd_queue = Queue()
    proc = Process(
        target=run_display_worker,
        args=(drm_name, cmd_queue, ACK_QUEUE, CLIENT_ID),
        daemon=True
    )
    proc.start()
    DISPLAY_PROCS[drm_name] = proc
    CMD_QUEUES[drm_name] = cmd_queue
    print(LOG_PREFIX, f"Started display worker for {drm_name} (PID {proc.pid})")
    return proc

def _run_display_worker_subprocess(drm_name: str):
    """
    Import display_worker.py and call main() directly.
    display_worker.main will map DRM name -> Xrandr -> SDL index internally.
    """
    import importlib.util

    worker_path = os.path.join(BASE_DIR, "display_worker.py")
    spec = importlib.util.spec_from_file_location("display_worker", worker_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main(drm_name)  # now accepts DRM name directly


# --- Networking / SocketIO ---
sio = socketio.Client(reconnection=True, logger=False, engineio_logger=False)

# We'll fill these after loading config
LOCAL_CFG: Dict[str, Any] = {}
CLIENT_ID = None
SERVER_URL = None

# Lock to avoid concurrent write to config
config_lock = threading.Lock()

def safe_print(*args, **kwargs):
    print(LOG_PREFIX, *args, **kwargs)

# Downloads
def download_file(url: str, out_path: str, timeout: int = 15) -> None:
    safe_print("Downloading", url, "->", out_path)
    tmp = out_path + ".part"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    try:
        with requests.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        os.replace(tmp, out_path)
    except Exception as e:
        safe_print("Download error:", e)
        if os.path.exists(tmp):
            try: os.remove(tmp)
            except: pass
        raise

# SocketIO handlers
@sio.event
def connect():
    safe_print("[connection] Connected to server:", SERVER_URL)
    cfg = load_local_config()
    if not cfg:
        safe_print("Failed loading local config on connect")
        return
    meta  = cfg
    safe_print("Register sent:", CLIENT_ID, meta)
    sio.emit("register", meta)

@sio.on("registered")
def on_registered(data):
    safe_print("Server registered response:", data)

@sio.on("FETCH_TILES")
def on_assign_tiles(payload):
    try:
        safe_print("FETCH_TILES received:", payload.get("filename"))
        handle_assign_tiles(payload)
    except Exception as e:
        safe_print("Error in ASSIGN_TILES:", e, traceback.format_exc())

@sio.on("SHOW")
def on_show(payload):
    try:
        safe_print("SHOW received:", payload)
        handle_show(payload)
    except Exception as e:
        safe_print("Error in SHOW:", e, traceback.format_exc())

@sio.on("CONFIG")
def on_config(data):
    safe_print("CONFIG update received:", data)
    if not isinstance(data, dict):
        safe_print("Invalid CONFIG payload")
        return
    with config_lock:
        cfg = load_local_config()
        if not cfg:
            safe_print("Failed loading local config in CONFIG")
            return
        homographies = data.get("homographies", {})
        if homographies:
            for display_name, hdata in homographies.items():
                points = hdata.get("matrix")
                if not points or len(points) != 4:
                    safe_print(f"Invalid homography points for {display_name}")
                    continue
                cmd_queue = CMD_QUEUES.get(display_name)
                if not cmd_queue:
                    safe_print(f"No worker for display {display_name} to set homography")
                    continue
                cmd_queue.put({
                    "type": "SET_POINTS",
                    "points": points
                })
                safe_print(f"Sent SET_HOMOGRAPHY to {display_name} with points: {points}")
        # update local config
        cfg.update(data)
        save_local_config(cfg)
        LOCAL_CFG = cfg
    safe_print("Local config updated with CONFIG payload")


# presentation socket events
@sio.on("START_PRESENTATION")
def on_start_presentation(data):
    """
    Example payload:
    {
      "images": ["B10_EVAPO", "B11_TEMP", ...],
    }
    """
    print("[client] START_PRESENTATION received", data)
    images = data.get("images", [])
    preloaded_images = preload_images(images)
    # send preload command to each worker
    print(f"[client] Sending PRELOAD_IMAGES to {len(CMD_QUEUES)} display workers")
    for drm_name, cmd_queue in CMD_QUEUES.items():
        print(f"[client] Sending PRELOAD_IMAGES to {drm_name}")
        cmd_queue.put({"type":"PRELOAD_IMAGES",
                       "images": preloaded_images if preloaded_images else []
                      }
        )


    # wait for ACKs
    expected = set(CMD_QUEUES.keys())
    ready = set()
    while ready != expected:
        ack = ACK_QUEUE.get()
        if ack.get("type") == "PRELOAD_DONE":
            ready.add(ack["display"])
            print(f"[client] {ack['display']} ready")
    preload_ok = (preloaded_images is not False) and (len(ready) == len(expected))
    print(f"[client] All displays ready: {preload_ok}")

    # Send ACK back to server
    sio.emit("PRESENTATION_READY", {
        "client_id": CLIENT_ID,
        "ready": preload_ok
    })


@sio.on("SHOW_IMAGE")
def on_show_image(data):
    """
    Example payload:
    {
      "image": "B10_EVAPO.png",
    }
    """
    image = data.get("image")
    if not image:
        print("[client] SHOW_IMAGE missing image in payload", data)
        return
    print("[client] SHOW_IMAGE received:", image)
    img = get_safe_filename_without_extension(image)
    show_image(img)
    

@sio.on("RESUME_PRESENTATION")
def on_resume_presentation(_):
    print("[client] RESUME_PRESENTATION -> resuming slideshow")
    # nothing to do, just listen for next SHOW_IMAGE
    pass



def preload_images(images):
    """
    Ensure all required tiles exist locally in TILES_DIR.
    """
    try:
        print(f"[client] Preloadiiiiiiiiiiiiiiiiiiiiiiiiiing images: {images}")
        loaded_image_list = []
        for img in images:
            safe_filename = get_safe_filename_without_extension(img)
            cfg = LOCAL_CFG
            if not cfg:
                safe_print("Failed loading local config in PRELOAD_IMAGES")
                return False
            for d in cfg.get("displays", []):
                hdmi_output = d.get("name")
                if not hdmi_output:
                    continue
                url = f"{SERVER_URL}/tiles/{safe_filename}/client_{CLIENT_ID}_tile_{hdmi_output}.png"
                local_folder = os.path.join(TILES_DIR, safe_filename)
                os.makedirs(local_folder, exist_ok=True)
                out_path = os.path.join(local_folder, f"{CLIENT_ID}_tile_{hdmi_output}.png")
                if not os.path.exists(out_path):
                    print(f"[client] Downloading tile for {hdmi_output} from {url}")
                    download_file(url, out_path)
            loaded_image_list.append(safe_filename)
        print(f"[client] Preloaded images: {loaded_image_list}")
        return loaded_image_list
    except Exception as e:
        safe_print("Error in PRELOAD_IMAGES:", e, traceback.format_exc())
        return False

def show_image(image):
    """
    Notify each display worker to show the given image (by base name).
    """
    print(f"[client] Displaying tiles for {image} for each display")
    for drm_name, cmd_queue in CMD_QUEUES.items():
        cmd_queue.put(
            {"type":"SHOW_IMAGE",
            "image": image
            }
)


@sio.event
def disconnect():
    safe_print("Disconnected from server")

# --- Handlers ---
def handle_assign_tiles(payload: Dict[str, Any]):
    """
    payload example:
    {
      "filename": "name.png",
    }
    fetch the file from server_url/tiles/safe_filename(filename)/cient_<client_id>_tile_<display_output>.png
    and
    save to TILES_DIR/filename
    """
    print (LOG_PREFIX, "Handling ASSIGN_TILES payload", payload)
    filename = payload.get("filename")
    if not filename:
        safe_print("No filename in ASSIGN_TILES payload")
        return
    safe_filename = get_safe_filename_without_extension(filename)
    print(LOG_PREFIX, "Safe filename:", safe_filename)
    cfg = LOCAL_CFG
    if not cfg:
        safe_print("Failed loading local config in ASSIGN_TILES")
        return
    for d in cfg.get("displays", []):
        hdmi_output = d.get("name")
        if not hdmi_output:
            continue
        url = f"{SERVER_URL}/tiles/{safe_filename}/client_{CLIENT_ID}_tile_{hdmi_output}.png"
        local_folder = os.path.join(TILES_DIR, safe_filename)
        os.makedirs(local_folder, exist_ok=True)
        out_path = os.path.join(local_folder, f"{CLIENT_ID}_tile_{hdmi_output}.png")
        try:
            download_file(url, out_path)
            #end and exit try
        except Exception as e:
            safe_print("Failed downloading tile for hdmi", hdmi_output, "err", e)
            continue

def get_safe_filename_without_extension(fname: str) -> str:
    # Remove any path components and keep only alphanum, dash, underscore, dot
    image_basename = os.path.splitext(os.path.basename(fname))[0]
    safe = image_basename.replace(" ", "_")
    # Remove extension
    return safe
    

def handle_show(payload: Dict[str, Any]):
    """
    On SHOW we:
    - iterate assignments in local config
    - load each assigned tile file
    - apply homography if present
    - write result to DISPLAY_OUT_DIR/output_<hdmi_output>.png
    - display workers will detect and show the images
    """
    print("handling show")
    with config_lock:
        cfg = LOCAL_CFG
        assignments = cfg.get("assignments", [])

    if not assignments:
        safe_print("No assignments to show")
        return

    # For each assignment, create processed image for its hdmi output
    # If multiple assignments map to the same hdmi_output, last one wins (log warning)
    target_images_for_output: Dict[int, str] = {}
    for a in assignments:
        tile_file = a.get("file")
        hdmi_out = str(a.get("hdmi_output"))
        tile_index = a.get("tile_index")
        if not tile_file or not os.path.exists(tile_file):
            safe_print("Missing tile file for assignment:", a)
            continue

        # Read image
        img = cv2.imread(tile_file, cv2.IMREAD_COLOR)
        if img is None:
            safe_print("Failed reading image:", tile_file)
            continue

        # Apply homography if available in local config
        with config_lock:
            homos = cfg.get("homographies", {})
            H = homos.get(str(tile_index))

        if H:
            try:
                H_np = np.array(H, dtype=np.float32)
                h, w = img.shape[:2]
                # We warp to same size as tile (dsize = (w,h))
                warped = cv2.warpPerspective(img, H_np, (w, h))
                out_img = warped
            except Exception as e:
                safe_print("Homography warp failed for tile", tile_index, "err", e)
                out_img = img
        else:
            out_img = img

        # Save result to display out path
        out_fname = f"output_{hdmi_out}.png"
        out_path = os.path.join(DISPLAY_OUT_DIR, out_fname)
        # If multiple tiles map to same hdmi, we simply overwrite (last wins). Could be changed to composite.
        cv2.imwrite(out_path, out_img)
        target_images_for_output[hdmi_out] = out_path
        safe_print(f"Prepared display image for hdmi {hdmi_out} -> {out_path}")

    # Optionally notify display workers by touching a small 'trigger' file or rely on workers watching mtime
    for hdmi, path in target_images_for_output.items():
        try:
            # touch the file to update mtime (already written) - ensures worker picks it up
            os.utime(path, None)
        except Exception:
            pass

    # Send a simple ack
    try:
        sio.emit("DISPLAY_READY", {"client_id": CLIENT_ID, "frame_id": payload.get("frame_id")})
    except:
        pass

# -------------------------

def detect_display_outputs():
    drm_path = "/sys/class/drm"
    outputs = []
    if os.path.exists(drm_path):
        for entry in os.listdir(drm_path):
            status_file = os.path.join(drm_path, entry, "status")
            modes_file = os.path.join(drm_path, entry, "modes")
            if os.path.exists(status_file):
                with open(status_file, "r") as f:
                    status = f.read().strip()

                if status == "connected":
                    resolution = None
                    if os.path.exists(modes_file):
                        try:
                            with open(modes_file, "r") as mf:
                                # First mode is usually current/active
                                first_line = mf.readline().strip()
                                if first_line:
                                    #2560x1440 to {"width": 2560, "height": 1440}
                                    resolution = {}
                                    parts = first_line.split('x')
                                    if len(parts) == 2:
                                        resolution["width"] = int(parts[0])
                                        resolution["height"] = int(parts[1])
                        except Exception:
                            pass

                    outputs.append({
                        "name": entry,
                        "status": status,
                        "resolution": resolution,
                        "active": True
                    })
    print(LOG_PREFIX, "Detected displays:", outputs)
    return outputs


def main():
    global LOCAL_CFG, CLIENT_ID, SERVER_URL, DISPLAY_PROCS

    LOCAL_CFG = ensure_displays_in_config()
    print(LOG_PREFIX, "Local config loaded:", LOCAL_CFG)
    
    CLIENT_ID = LOCAL_CFG.get("client_id")
    SERVER_URL = LOCAL_CFG.get("server_url")
    displays = LOCAL_CFG.get("displays")
    

    if not CLIENT_ID or not SERVER_URL:
        safe_print("client_id or server_url not set in config/client.json — edit the file and restart.")
        sys.exit(1)

    if not displays:
        safe_print("No displays configured/found. Edit config/client.json to set 'displays' or connect HDMI monitor.")
        sys.exit(1)

    safe_print("Starting client", CLIENT_ID, "connecting to", SERVER_URL)

    # spawn display worker processes (one per display index)
    for d in displays:
        name = d["name"]
        p = start_display_worker_process(name)

    print(LOG_PREFIX, f"Spawned {len(DISPLAY_PROCS)} display worker processes.")


    # connect to socketio with reconnects
    # Make sure server_url uses http(s) and Socket.IO default path
    try:
        print(LOG_PREFIX, "Connecting to server...")
        print(LOG_PREFIX, "Connecting to server...")
        print(LOG_PREFIX, "Connecting to server...")
        sio.connect(SERVER_URL)
    except Exception as e:
        safe_print("Initial connect failed:", e)
    # run forever
    try:
        sio.wait()
    except KeyboardInterrupt:
        safe_print("Interrupted, shutting down")
    finally:
        for p in DISPLAY_PROCS:
            try:
                p.terminate()
            except:
                pass

if __name__ == "__main__":
    main()
