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


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DISPLAY_OUT_DIR = os.path.join(BASE_DIR, "display_out")
os.makedirs(DISPLAY_OUT_DIR, exist_ok=True)


def show_image_fullscreen(surface, screen):
    """Blit surface to screen scaled to screen size and flip."""
    sw, sh = screen.get_size()
    img_surf = pygame.transform.smoothscale(surface, (sw, sh))
    screen.blit(img_surf, (0, 0))
    pygame.display.flip()


def load_image_as_surface(path):
    """Load image via PIL then convert to pygame surface (RGB)."""
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


def main(display_name: str):
    """
    Monitor DISPLAY_OUT_DIR/output_<display_name>.png and show fullscreen on given display.
    """
    print(f"[display_worker] Starting for display '{display_name}'")
    drm_map = build_drm_xrandr_map()
    x_name = drm_map.get(display_name, None)
    display_index = map_display_name_to_index(x_name) if x_name else 0
    os.environ["SDL_VIDEO_FULLSCREEN_DISPLAY"] = str(display_index)

    # Init pygame video only
    pygame.display.init()
    pygame.event.set_blocked(None)
    info = pygame.display.Info()
    screen_w, screen_h = info.current_w, info.current_h
    screen = pygame.display.set_mode((screen_w, screen_h), pygame.FULLSCREEN)
    pygame.mouse.set_visible(False)

    target_fname = os.path.join(DISPLAY_OUT_DIR, f"output_{display_name}.png")
    last_mtime = 0

    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
            if os.path.exists(target_fname):
                mtime = os.path.getmtime(target_fname)
                if mtime != last_mtime:
                    try:
                        surf = load_image_as_surface(target_fname)
                        show_image_fullscreen(surf, screen)
                        last_mtime = mtime
                    except Exception as e:
                        print("[display_worker] failed to load/show", target_fname, e)
            time.sleep(0.25)
    except KeyboardInterrupt:
        pass
    finally:
        pygame.quit()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python display_worker.py <display_name>")
        sys.exit(1)
    main(str(sys.argv[1]))
