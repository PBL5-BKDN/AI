from PIL import Image
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial

output_folder = 'masks_temp'
input_folder = 'masks'
os.makedirs(output_folder, exist_ok=True)

def resize_image(filename, input_folder, output_folder):
    if not filename.lower().endswith('.png'):
        return
    input_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, filename)

    try:
        with Image.open(input_path) as img:
            img_resized = img.resize((640, 640), Image.Resampling.LANCZOS)
            img_resized.save(output_path)
            #print(f'Resized: {filename}')
    except Exception as e:
        print(f'Error processing {filename}: {e}')

image_files = os.listdir(input_folder)

if __name__ == "__main__":
    with ThreadPoolExecutor() as executor:
        func = partial(resize_image, input_folder=input_folder, output_folder=output_folder)
        executor.map(func, image_files)