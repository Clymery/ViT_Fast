"""Visualize all experiment types across all datasets."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter, ImageOps
import os, glob, random, cv2

# Setup
data_dir = '/data/ypxia/Workspace/miss_patch/data'
docs_dir = 'docs'
os.makedirs(docs_dir, exist_ok=True)

# Find sample images from each dataset
samples = []

# Oxford Pets (high-res animals)
pets_dir = os.path.join(data_dir, 'oxford-iiit-pet', 'images')
if os.path.exists(pets_dir):
    files = sorted(glob.glob(os.path.join(pets_dir, '*.jpg')))
    random.seed(42)
    f = random.choice(files)
    img = Image.open(f)
    name = os.path.basename(f).rsplit('_', 1)[0]
    samples.append(('Oxford Pets\n(cat/dog breed)', img, 93.81))

# Food-101 (fine-grained dishes)
food_dir = os.path.join(data_dir, 'food-101', 'images')
if os.path.exists(food_dir):
    subdirs = sorted(os.listdir(food_dir))
    random.seed(43)
    sd = random.choice(subdirs)
    files = sorted(glob.glob(os.path.join(food_dir, sd, '*.jpg')))
    if files:
        img = Image.open(files[0])
        name = sd.replace('_', ' ')
        samples.append((f'Food-101\n({name})', img, 91.37))

# DTD (texture)
dtd_dir = os.path.join(data_dir, 'dtd', 'dtd', 'images')
if os.path.exists(dtd_dir):
    cats = sorted(os.listdir(dtd_dir))
    random.seed(44)
    c = random.choice(cats)
    files = sorted(glob.glob(os.path.join(dtd_dir, c, '*.jpg')))
    if files:
        img = Image.open(files[0])
        samples.append((f'DTD\n({c})', img, 80.85))

# CIFAR-100 (32x32 native)
cifar_dir = os.path.join(data_dir, 'cifar-100-python')
if os.path.exists(cifar_dir):
    import pickle
    with open(os.path.join(cifar_dir, 'train'), 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    cifar_labels = [
        'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
        'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
        'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
        'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
        'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
        'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
        'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
        'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
        'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
        'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
        'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
        'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
        'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
        'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
    ]
    random.seed(45)
    idx = random.randint(0, 50000)
    arr = batch[b'data'][idx].reshape(3, 32, 32).transpose(1, 2, 0)
    label = cifar_labels[batch[b'fine_labels'][idx]]
    img = Image.fromarray(arr)
    samples.append((f'CIFAR-100\n({label}, 32x32 native)', img, 91.69))

# Process each sample into a standard 224x224 base
def preprocess_to_224(img):
    """Same as training preprocessing: short edge -> 255, center crop -> 224."""
    short = min(img.size)
    scale = 255 / short
    new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
    img_resized = img.resize(new_size, Image.BILINEAR)
    left = (new_size[0] - 224) // 2
    top = (new_size[1] - 224) // 2
    return img_resized.crop((left, top, left + 224, top + 224))

def process_funcs():
    """Define all processing methods."""
    return [
        ('Original\n224x224', lambda img: img),
        ('168x168\n(100 patches)', lambda img: img.resize((168, 168), Image.BILINEAR)),
        ('112x112\n(49 patches)', lambda img: img.resize((112, 112), Image.BILINEAR)),
        ('80x80\n(25 patches)', lambda img: img.resize((80, 80), Image.BILINEAR)),
        ('Grayscale', lambda img: ImageOps.grayscale(img).convert('RGB')),
        ('Blur k=15\n(low freq)', lambda img: img.filter(ImageFilter.GaussianBlur(15))),
        ('Canny Edges\n(structure)', lambda img: canny_on_pil(img)),
        ('2-bit Color\n(4 colors)', lambda img: posterize(img, 2)),
    ]

def canny_on_pil(pil_img):
    arr = np.array(pil_img)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return Image.fromarray(np.stack([edges] * 3, axis=-1))

def posterize(pil_img, bits):
    arr = np.array(pil_img)
    arr = (arr >> (8 - bits)) << (8 - bits)
    return Image.fromarray(arr)

# Create figure
methods = process_funcs()
n_rows = len(samples)
n_cols = len(methods) + 1  # +1 for dataset info column

fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.8, n_rows * 2.5))

# Column headers
for col in range(1, n_cols):
    col_idx = col - 1
    axes[0, col].set_title(methods[col_idx][0], fontsize=9, fontweight='bold')

for row, (dataset_name, img, baseline_acc) in enumerate(samples):
    img_224 = preprocess_to_224(img)

    # First column: dataset info + sample image
    ax = axes[row, 0]
    # Show a small version of the original
    thumb = img_224.copy()
    ax.imshow(np.array(thumb))
    ax.set_ylabel(dataset_name, fontsize=9, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    # Add baseline info
    if row == 0:
        ax.set_title('Original\nimage', fontsize=9)

    # Remaining columns: each processing method
    for col, (_, func) in enumerate(methods):
        ax = axes[row, col + 1]
        result = func(img_224)
        ax.imshow(np.array(result), cmap='gray' if 'Canny' in methods[col][0] else None)
        ax.set_xticks([])
        ax.set_yticks([])

plt.tight_layout()
plt.savefig(os.path.join(docs_dir, 'all_experiments_guide.png'), dpi=180, bbox_inches='tight')
print(f"Saved to {docs_dir}/all_experiments_guide.png")

# Also create a second figure: resolution sweep only, all datasets in one row
fig2, axes2 = plt.subplots(len(samples), 6, figsize=(16, 2.5 * len(samples)))
res_methods = [
    ('224x224\n196 patches', lambda x: x),
    ('168x168\n100 patches', lambda x: x.resize((168, 168), Image.BILINEAR)),
    ('112x112\n49 patches', lambda x: x.resize((112, 112), Image.BILINEAR)),
    ('80x80\n25 patches', lambda x: x.resize((80, 80), Image.BILINEAR)),
    ('64x64\n16 patches', lambda x: x.resize((64, 64), Image.BILINEAR)),
    ('48x48\n9 patches', lambda x: x.resize((48, 48), Image.BILINEAR)),
]

for row, (dataset_name, img, _) in enumerate(samples):
    img_224 = preprocess_to_224(img)
    for col, (label, func) in enumerate(res_methods):
        ax = axes2[row, col]
        result = func(img_224)
        ax.imshow(np.array(result))
        ax.set_xticks([])
        ax.set_yticks([])
        if row == 0:
            ax.set_title(label, fontsize=10, fontweight='bold')
        if col == 0:
            ax.set_ylabel(dataset_name, fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(docs_dir, 'resolution_sweep_guide.png'), dpi=180, bbox_inches='tight')
print(f"Saved to {docs_dir}/resolution_sweep_guide.png")
