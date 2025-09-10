#!/usr/bin/env python3
"""
Display worker: run in separate process per display. It watches a file
DISPLAY_OUT_DIR/output_<display_name>.png and shows it fullscreen on the
given display name resolved to SDL display index using pygame/SDL.

Call: python display_worker.py <display_name>
Or the main() function is invoked by client.process wrapper.
"""

import os
import sys
import time
import subprocess
import pygame
from PIL import Image
import re
from multiprocessing import Queue


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DISPLAY_OUT_DIR = os.path.join(BASE_DIR, "display_out")
os.makedirs(DISPLAY_OUT_DIR, exist_ok=True)


def show_image_fullscreen(surface, screen):
    sw, sh = screen.get_size()
    img_surf = pygame.transform.smoothscale(surface, (sw, sh))
    screen.blit(img_surf, (0, 0))
    pygame.display.flip()

def load_image_as_surface(path):
    pil = Image.open(path).convert("RGB")
    mode = pil.mode
    size = pil.size
    data = pil.tobytes()
    return pygame.image.fromstring(data, size, mode)

def build_drm_xrandr_map():
    """
    Map DRM connector names (cardX-XXX) to Xrandr names (DP-1, HDMI-1, etc.)
    """
    mapping = {}
    try:
        output = subprocess.check_output(["xrandr", "--verbose"], text=True)
        current_x = None
        for line in output.splitlines():
            line = line.strip()
            # Xrandr display line
            m = re.match(r"^(\S+) connected", line)
            if m:
                current_x = m.group(1)
            elif "Connector:" in line and current_x:
                drm_name = line.split()[-1]
                mapping[drm_name] = current_x
                current_x = None
    except Exception as e:
        print(f"[display_worker] Could not build DRM → Xrandr map: {e}")
    return mapping

def get_xrandr_monitors():
    """Parse xrandr --listmonitors and return {name: (w,h)} mapping."""
    monitors = {}
    try:
        output = subprocess.check_output(["xrandr", "--listmonitors"]).decode().splitlines()
        for line in output[1:]:  # skip "Monitors: N"
            parts = line.split()
            if len(parts) >= 4:
                # format: " 0: +HDMI-1 1920/477x1080/268+0+0  HDMI-1"
                res = parts[2].split("x")
                if len(res) == 2:
                    try:
                        w = int(res[0].split("/")[0])
                        h = int(res[1].split("/")[0])
                        name = parts[-1]
                        monitors[name] = (w, h)
                    except ValueError:
                        pass
    except Exception as e:
        print(f"[display_worker] Failed to get xrandr monitors: {e}")
    return monitors


def map_display_name_to_index(display_name: str) -> int:
    """
    Try to resolve xrandr display name (HDMI-1, DP-2, etc.)
    to SDL display index by matching resolution.
    """
    monitors = get_xrandr_monitors()
    print(f"[display_worker] xrandr monitors: {monitors}")
    if display_name not in monitors:
        print(f"[display_worker] WARNING: display '{display_name}' not found in xrandr, defaulting to 0")
        return 0

    target_res = monitors[display_name]

    pygame.display.init()
    num_displays = pygame.display.get_num_displays()
    desktop_sizes = pygame.display.get_desktop_sizes()

    # Try to match resolution
    for i, (w, h) in enumerate(desktop_sizes):
        if (w, h) == target_res:
            return i

    # fallback: first display
    print(f"[display_worker] WARNING: no resolution match for {display_name}, defaulting to 0")
    return 0


class DisplayWorker:
    def __init__(self, drm_name: str, cmd_queue: Queue, ack_queue: Queue, client_id: str):
        """
        assignments: list of display_outputs this worker is responsible for
        """
        self.drm_name = drm_name
        self.cmd_queue = cmd_queue
        self.ack_queue = ack_queue
        self.client_id = client_id
        self.images = {}  # key: img_base, value: pygame.Surface
        self.current_image = None
        self.screen = None
        self.display_index = 0
        
        drm_map = build_drm_xrandr_map()
        x_name = drm_map.get(drm_name, None)
        display_index = map_display_name_to_index(x_name) if x_name else 0
        self.display_index = display_index
        print(f"[{self.drm_name}] mapped to SDL display index {display_index}")
        os.environ["SDL_VIDEO_FULLSCREEN_DISPLAY"] = str(display_index)

    def preload_images(self, images: list[str]):
        """ Preload images into memory from path: ./tiles/<img>/<client_id>_tile_<drm_name>.png
        for fast display later.
        """
        print(f"[worker {self.drm_name}] preloading images: {images}")
        TILES_DIR = os.path.join(BASE_DIR, "tiles")
        for img in images:
            safe = img
            local_folder = os.path.join(TILES_DIR, safe)
            out_path = os.path.join(local_folder, f"{self.client_id}_tile_{self.drm_name}.png")
            if os.path.exists(out_path):
                try:
                    surface = pygame.image.load(out_path)
                    self.images[img] = surface
                    print(f"[worker {self.drm_name}] preloaded {out_path}")
                except Exception as e:
                    print(f"[worker {self.drm_name}] failed to load {out_path}: {e}")
        # notify main process
        print(f"[worker {self.drm_name}] preload done, {len(self.images)} images loaded")
        self.ack_queue.put({"type": "PRELOAD_DONE", "display": self.drm_name})

    def show_image(self, img: str):
        if img not in self.images:
            print(f"[worker {self.drm_name}] image {img} not preloaded")
            return
        surface = self.images[img]
        screen = pygame.display.set_mode(surface.get_size(), pygame.NOFRAME)
        screen.blit(surface, (0, 0))
        pygame.display.flip()
        print(f"[worker {self.drm_name}] showing {img}")

    def run(self):
        """
        Non-blocking event loop to handle SHOW_IMAGE commands from client
        """
        pygame.display.init()
        pygame.event.set_blocked(None)
        info = pygame.display.Info()
        screen_w, screen_h = info.current_w, info.current_h
        screen = pygame.display.set_mode((screen_w, screen_h), pygame.FULLSCREEN)
        pygame.mouse.set_visible(False)
        self.screen = screen

        try:    
            running = True
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
                while not self.cmd_queue.empty():
                    cmd = self.cmd_queue.get()  # blocking wait
                    ctype = cmd.get("type")
                    if ctype == "PRELOAD_IMAGES":
                        self.preload_images(cmd.get("images", []))
                    elif ctype == "SHOW_IMAGE":
                        self.show_image(cmd.get("image"))
                    elif ctype == "STOP":
                        break
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            pygame.quit()


def run_display_worker(drm_name: str, cmd_queue: Queue, ack_queue: Queue, client_id: str):
    worker = DisplayWorker(drm_name, cmd_queue, ack_queue, client_id)
    worker.run()
