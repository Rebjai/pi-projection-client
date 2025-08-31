# client.py
import socketio, os
from PIL import Image
import io, binascii

sio = socketio.Client()

CLIENT_ID = "pi1"
TILE_FOLDER = "tiles"

os.makedirs(TILE_FOLDER, exist_ok=True)

@sio.event
def connect():
    print("Connected to server")
    sio.emit("register", {"client_id": CLIENT_ID})

@sio.on("tile")
def tile(data):
    filename = data["filename"]
    img_bytes = binascii.unhexlify(data["data"])
    path = os.path.join(TILE_FOLDER, filename)
    with open(path, "wb") as f:
        f.write(img_bytes)
    print(f"Saved {filename}")
    # Quick display test with Pillow
    Image.open(path).show()

sio.connect("http://127.0.0.1:5000")
sio.wait()
