"""Visualize low-frequency, high-frequency, and Canny edge processing."""
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageFilter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'DejaVu Sans'  # Use English labels instead
import os

def gaussian_blur(img, kernel_size=21):
    """Apply Gaussian blur to PIL image."""
    return img.filter(ImageFilter.GaussianBlur(radius=kernel_size//3))

def high_frequency(img, kernel_size=21):
    """High frequency = original - blurred."""
    blurred = gaussian_blur(img, kernel_size)
    return Image.fromarray(np.clip(np.array(img, dtype=int) - np.array(blurred, dtype=int) + 128, 0, 255).astype(np.uint8))

def canny_edge(img):
    """Canny edge detection."""
    import cv2
    gray = np.array(img.convert('L'))
    edges = cv2.Canny(gray, 50, 150)
    return Image.fromarray(edges)

# Test on a few images
data_dir = './data'
import glob

sample_paths = []

# Prioritize high-resolution datasets
# Oxford Pets (high res, animals with texture/patterns)
pets_dir = os.path.join(data_dir, 'oxford-iiit-pet', 'images')
if os.path.exists(pets_dir):
    jpg_files = sorted(glob.glob(os.path.join(pets_dir, '*.jpg')))[:2]
    for p in jpg_files:
        img = Image.open(p).resize((448, 448))
        sample_paths.append(('Oxford Pets', img))

# Food-101 (high res, fine-grained texture)
food_dir = os.path.join(data_dir, 'food-101', 'images')
if os.path.exists(food_dir):
    subdirs = sorted(os.listdir(food_dir))[:2]
    for s in subdirs:
        jpgs = sorted(glob.glob(os.path.join(food_dir, s, '*.jpg')))
        if jpgs:
            img = Image.open(jpgs[0]).resize((448, 448))
            sample_paths.append(('Food-101', img))

# Fallback to CIFAR-100 (low res, only if nothing else)
if not sample_paths:
    cifar_dir = os.path.join(data_dir, 'cifar-100-python')
    if os.path.exists(cifar_dir):
        import pickle
        with open(os.path.join(cifar_dir, 'train'), 'rb') as f:
            batch = pickle.load(f, encoding='bytes')
        for i in range(2):
            img_arr = batch[b'data'][i].reshape(3, 32, 32).transpose(1, 2, 0)
            sample_paths.append(('CIFAR-100', Image.fromarray(img_arr)))

sample_paths = sample_paths[:2]  # 2 rows, 4 cols each
n_rows = len(sample_paths)
fig, axes = plt.subplots(n_rows, 4, figsize=(16, 5*n_rows))
if n_rows == 1:
    axes = axes.reshape(1, -1)

methods = [
    ('Original\n(all frequencies)', lambda x: x),
    ('Low Frequency\n(blur k=21)', lambda x: gaussian_blur(x, 21)),
    ('High Frequency\n(original - blurred)', lambda x: high_frequency(x, 21)),
    ('Canny Edges', lambda x: canny_edge(x)),
]

for row, (name, img) in enumerate(sample_paths):
    for col, (method_name, method_fn) in enumerate(methods):
        ax = axes[row, col]
        result = method_fn(img)
        ax.imshow(np.array(result), cmap='gray' if col == 3 else None)
        ax.set_title(f'{name}\n{method_name}', fontsize=11)
        ax.axis('off')

plt.tight_layout()
os.makedirs('docs', exist_ok=True)
plt.savefig('docs/frequency_demo.png', dpi=150, bbox_inches='tight')
print("Saved to docs/frequency_demo.png")
plt.close()
