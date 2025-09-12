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
        print(f"[{self.drm_name}] using SDL_VIDEO_FULLSCREEN_DISPLAY={os.environ['SDL_VIDEO_FULLSCREEN_DISPLAY']}")

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
        pygame.display.init()
        info = pygame.display.Info()
        print(f"[worker {self.drm_name}] pygame display info: {info}")
        sw, sh = info.current_w, info.current_h
        # Use SCALED + NOFRAME instead of FULLSCREEN to survive alt-tab
        self.screen = pygame.display.set_mode((sw, sh), pygame.FULLSCREEN)
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
                    self.show_image(cmd.get("image"), homography_pts=points)
                elif ctype == "STOP":
                    running = False
            time.sleep(0.05)

        pygame.quit()


def run_display_worker(drm_name: str, cmd_queue: Queue, ack_queue: Queue, client_id: str):
    worker = DisplayWorker(drm_name, cmd_queue, ack_queue, client_id)
    worker.run()
