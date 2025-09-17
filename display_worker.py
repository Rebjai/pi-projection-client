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
import numpy as np
import cv2


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DISPLAY_OUT_DIR = os.path.join(BASE_DIR, "display_out")
os.makedirs(DISPLAY_OUT_DIR, exist_ok=True)

# ---------------- Homography Utilities ----------------

def get_homography_warp(src_surface, dst_pts, output_size):
    """
    Apply homography warp to a Pygame surface.
    src_surface: Pygame.Surface
    dst_pts: 4 points [(x0,y0), (x1,y1), (x2,y2), (x3,y3)]
    output_size: (width, height) of output surface
    Returns a new Pygame surface with warped image.
    """
    # Convert surface to OpenCV image
    src_array = pygame.surfarray.array3d(src_surface)
    src_array = np.transpose(src_array, (1, 0, 2))  # Pygame -> OpenCV orientation

    h_src, w_src = src_array.shape[:2]
    src_pts = np.float32([[0,0], [w_src-1,0], [w_src-1,h_src-1], [0,h_src-1]])
    dst_pts = np.float32(dst_pts)

    H, _ = cv2.findHomography(src_pts, dst_pts)
    warped = cv2.warpPerspective(src_array, H, output_size)

    # Convert back to Pygame surface
    warped = np.transpose(warped, (1,0,2))
    warped_surf = pygame.surfarray.make_surface(warped)
    return warped_surf

#---------------- Main Display Worker ----------------

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

def get_connected_drm_connectors():
    """
    Return a list of DRM connectors that are currently connected.
    Looks in /sys/class/drm for card*-* entries.
    """
    connectors = []
    try:
        drm_path = "/sys/class/drm"
        for entry in os.listdir(drm_path):
            entry_path = os.path.join(drm_path, entry)
            if os.path.isdir(entry_path):
                status_file = os.path.join(entry_path, "status")
                if os.path.exists(status_file):
                    with open(status_file) as f:
                        status = f.read().strip()
                        if status == "connected":
                            connectors.append(entry)
    except Exception as e:
        print(f"[mapper] Failed to list DRM connectors: {e}")
    return connectors


def build_drm_xrandr_map():
    """
    Automatically map DRM connectors (card*-*) to xrandr names (HDMI-1, DP-1, etc.)
    in order of connection.
    """
    mapping = {}
    try:
        output = subprocess.check_output(["xrandr", "--verbose"], text=True).splitlines()
        drm_connectors = get_connected_drm_connectors()
        # Assign each connected xrandr output to a DRM connector
        for line in output:
            line = line.strip()
            m = re.match(r"^(\S+) connected", line)
            if m and drm_connectors:
                drm_name = drm_connectors.pop(0)
                x_name = m.group(1)
                mapping[drm_name] = x_name
    except Exception as e:
        print(f"[mapper] Failed to build DRM → Xrandr map: {e}")
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
    monitors = get_xrandr_monitors()
    print(f"[display_worker] xrandr monitors: {monitors}")
    if display_name not in monitors:
        print(f"[display_worker] WARNING: display '{display_name}' not found, defaulting to 0")
        return 0

    # Create a stable ordering of displays (by name)
    sorted_names = sorted(monitors.keys())
    display_index = sorted_names.index(display_name)
    print(f"[display_worker] mapped {display_name} -> SDL index {display_index}")
    return display_index



class DisplayWorker:
    def __init__(self, drm_name: str, cmd_queue: Queue, ack_queue: Queue, client_id: str):
        self.drm_name = drm_name
        self.cmd_queue = cmd_queue
        self.ack_queue = ack_queue
        self.client_id = client_id
        self.images = {}  # key: img_base, value: pygame.Surface
        self.current_image = None
        self.screen = None
        self.display_index = 0
        self.points = None  # no homography by default

        drm_map = build_drm_xrandr_map()
        print(f"[{self.drm_name}] DRM to Xrandr map: {drm_map}")
        x_name = drm_map.get(drm_name, None)
        display_index = map_display_name_to_index(x_name) if x_name else 0
        self.display_index = display_index
        print(f"[{self.drm_name}] mapped to SDL display index {display_index}")
        
        self.pygame = pygame
        print(f"[{self.drm_name}] pygame display initialized with {pygame.display.get_num_displays()} displays")

    def set_points(self, points):
        """ Set homography points for warping images.
        scale normalized points [0..1] to screen size.
        points:  points: [[0.3635831750541259, 0.17705501044122854], [0.7709439058282997, 0.10253079110435967], [0.7782326664186925, 0.8002758705145345], [0.379268278858621, 0.7869399019440089]]
        """
        if not points or len(points) != 4:
            print(f"[worker {self.drm_name}] invalid homography points: {points}")
            self.points = None
            return
        if not self.screen:
            print(f"[worker {self.drm_name}] screen not initialized, cannot set points")
            self.points = None
            return
        sw, sh = self.screen.get_size()
        scaled_points = []
        for p in points:
            if len(p) != 2:
                print(f"[worker {self.drm_name}] invalid point: {p}")
                self.points = None
                return
            x = int(p[0] * sw)
            y = int(p[1] * sh)
            scaled_points.append((x, y))
        self.points = scaled_points
        print(f"[worker {self.drm_name}] set homography points: {self.points}")

    def preload_images(self, images: list[str]):
        print(f"[worker {self.drm_name}] preloading images: {images}")
        TILES_DIR = os.path.join(BASE_DIR, "tiles")
        for img in images:
            local_folder = os.path.join(TILES_DIR, img)
            out_path = os.path.join(local_folder, f"{self.client_id}_tile_{self.drm_name}.png")
            if os.path.exists(out_path):
                try:
                    surface = pygame.image.load(out_path).convert()
                    self.images[img] = surface
                    print(f"[worker {self.drm_name}] preloaded {out_path}, size={surface.get_size()}")
                except Exception as e:
                    print(f"[worker {self.drm_name}] failed to load {out_path}: {e}")
        print(f"[worker {self.drm_name}] preload done, {len(self.images)} images loaded")
        self.ack_queue.put({"type": "PRELOAD_DONE", "display": self.drm_name})

    def show_image(self, img: str, homography_pts=None):
        """
        Show preloaded image fullscreen. If homography_pts is given (4 points),
        the image will be warped to that quadrilateral.
        """
        if img not in self.images:
            print(f"[worker {self.drm_name}] image {img} not preloaded")
            return

        surface = self.images[img]

        sw, sh = self.screen.get_size()
        if homography_pts and len(homography_pts) == 4:
            warped_surf = get_homography_warp(surface, homography_pts, (sw, sh))
            self.screen.blit(warped_surf, (0,0))
        else:
            # scale to fullscreen preserving aspect ratio
            self._blit_fullscreen(surface)

        pygame.display.flip()
        self.current_image = img
        print(f"[worker {self.drm_name}] showing image '{img}'")
        self.ack_queue.put({"type": "SHOW_DONE", "display": self.drm_name, "image": img})

    def get_monitor_offsets(self):
        """
        Calculate the X offset for each display based on previous monitors.
        """
        sizes = pygame.display.get_desktop_sizes()
        x_offset = sum(s[0] for s in sizes[:self.display_index])
        return x_offset-x_offset*.1, 0

    def _blit_fullscreen(self, surface):
        if not self.screen:
            print(f"[worker {self.drm_name}] WARNING: screen not initialized")
            return
        sw, sh = self.screen.get_size()
        iw, ih = surface.get_size()
        scale = min(sw / iw, sh / ih)
        nw, nh = int(iw * scale), int(ih * scale)
        x = (sw - nw) // 2
        y = (sh - nh) // 2
        img_surf = pygame.transform.smoothscale(surface, (nw, nh))
        self.screen.fill((0, 0, 0))
        self.screen.blit(img_surf, (x, y))
        pygame.display.flip()
        print(f"[worker {self.drm_name}] displayed image at {nw}x{nh} on screen {sw}x{sh}")

    def run(self):
        print(f"[{self.drm_name}] DisplayWorker starting on display index {self.display_index}")
        # Set environment variable to select the display
        os.environ["SDL_VIDEO_FULLSCREEN_DISPLAY"] = str(self.display_index)
        print(f"[{self.drm_name}] set SDL_VIDEO_FULLSCREEN_DISPLAY={self.display_index}")

        pygame.display.init()

        num_displays = pygame.display.get_num_displays()
        print(f"[{self.drm_name}] pygame sees {num_displays} displays")

        if self.display_index >= num_displays:
            print(f"[{self.drm_name}] ERROR: display index {self.display_index} out of range!")
            return

        # Query the resolution of the target display
        disp_bounds = pygame.display.get_desktop_sizes()[self.display_index]
        print(f"[{self.drm_name}] display {self.display_index} bounds: {disp_bounds}")
        sw, sh = disp_bounds
        print(f"[{self.drm_name}] desktop size for index {self.display_index}: {sw}x{sh}")

        # Open the window on that display
        if self.display_index == 0:
            flags = pygame.FULLSCREEN
        else:
            flags = pygame.NOFRAME
            x, y = self.get_monitor_offsets()
            os.environ["SDL_VIDEO_WINDOW_POS"] = f"{x},{y}"

        # Create the window
        self.screen = pygame.display.set_mode((sw, sh), flags, display=self.display_index)
        print(f"[{self.drm_name}] initialized on display {self.display_index} -- size={self.screen.get_size()}, flags={flags}")

        print(
            f"[{self.drm_name}] initialized on display {self.display_index} "
            f"-- pygame screen size: {self.screen.get_size()}"
            f", flags={flags}"
        )

        pygame.mouse.set_visible(False)
        points = None # no homography by default
        # points = [(100,100), (2400,50), (2500,1300), (50,1400)] # example homography points large
        # points = [(0,0), (sw,0), (sw,sh), (0,sh)] # full screen quad
        # points = [(100,100), (sw-100,50), (sw-50,sh-50), (50,sh-100)] # inset quad
        # points = [(50,50), (sw-50,50), (sw-50,sh-50), (50,sh-50)] # inset quad

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.VIDEOEXPOSE:
                    # redraw current image if screen cleared
                    if self.current_image:
                        self._blit_fullscreen(self.images[self.current_image])

            while not self.cmd_queue.empty():
                cmd = self.cmd_queue.get()
                ctype = cmd.get("type")
                if ctype == "PRELOAD_IMAGES":
                    self.preload_images(cmd.get("images", []))
                elif ctype == "SHOW_IMAGE":
                    self.show_image(cmd.get("image"), homography_pts=self.points)
                elif ctype == "SET_POINTS":
                    self.set_points(cmd.get("points"))
                elif ctype == "STOP":
                    running = False
            time.sleep(0.05)

        pygame.quit()


def run_display_worker(drm_name: str, cmd_queue: Queue, ack_queue: Queue, client_id: str):
    worker = DisplayWorker(drm_name, cmd_queue, ack_queue, client_id)
    worker.run()
