import os
from PIL import Image
from tqdm import tqdm

root = './datasets'
for name in tqdm(os.listdir(root)):
    dirpath = os.path.join(root, name)
    if not name.endswith('.gz'):
        for filename in os.listdir(dirpath):
            filepath = os.path.join(dirpath, filename)
            if filepath.endswith('.ppm'):
                image = Image.open(filepath)
                image.save(filepath.replace('.ppm', '.jpg'))
