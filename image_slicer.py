#image slicer
from PIL import Image
import os
import json

UPLOAD_FOLDER = 'uploads'

def slice_all():
    """
    Launch slicing for all images in uploads, using config from config/clent.json to take the rectangle coordinates to slice
    the resulting images are saved in tiles folder with name <original_image_name>_<display_name>.png
    """
    with open('config/client.json') as f:
        config = json.load(f)
    
    for image_name in os.listdir(UPLOAD_FOLDER):
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(UPLOAD_FOLDER, image_name)
            img = Image.open(image_path)
            img_width, img_height = img.size
            
            for assignment in config.get('assignments', []):
                display_name = assignment['display_output']
                rect = assignment['rect']
                
                # Calculate pixel coordinates
                left = int(rect['x'] * img_width / config['client_canvas_size']['width'])
                upper = int(rect['y'] * img_height / config['client_canvas_size']['height'])
                right = int((rect['x'] + rect['w']) * img_width / config['client_canvas_size']['width'])
                lower = int((rect['y'] + rect['h']) * img_height / config['client_canvas_size']['height'])
                
                # Crop and save the image
                cropped_img = img.crop((left, upper, right, lower))
                safe_filename = get_safe_filename_without_extension(image_name)
                os.makedirs('tiles', exist_ok=True)
                output_filename = f"{safe_filename}_{display_name}.png"
                output_path = os.path.join('tiles', output_filename)
                cropped_img.save(output_path)
                print(f"Sliced image saved to {output_path}")
    print("All images processed.")

def get_safe_filename_without_extension(fname: str) -> str:
    # Remove any path components and keep only alphanum, dash, underscore, dot
    image_basename = os.path.splitext(os.path.basename(fname))[0]
    safe = image_basename.replace(" ", "_")
    # Remove extension
    return safe



if __name__ == "__main__":
    slice_all()
