"""
Train ViT-B/16 on grayscale images.
Tests how much classification relies on color vs shape/texture.

Usage:
  python test_grayscale_train.py --dataset oxford_pets --gpu 3
  python test_grayscale_train.py --dataset food101 --gpu 4
"""
import torch, torch.nn as nn, time, os, sys, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets import get_oxford_pets_loader, get_food101_loader, get_dtd_loader, get_cifar100_loader
import timm
from torchvision import transforms

DATASETS = {
    'cifar100':    ('get_cifar100_loader',  100, 100, 91.69),
    'oxford_pets': ('get_oxford_pets_loader', 37, 100, 93.81),
    'food101':     ('get_food101_loader',   101,  30, 91.37),
    'dtd':         ('get_dtd_loader',        47, 100, 80.85),
}

def make_grayscale_loader(dataset_name, batch_size=64, num_workers=4):
    """Create dataloader with Grayscale transform applied."""
    from torchvision import datasets as tv_datasets
    from torch.utils.data import DataLoader

    # Standard normalization for each dataset
    norms = {
        'cifar100': ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        'oxford_pets': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        'food101': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        'dtd': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    }
    mean, std = norms[dataset_name]

    # Grayscale: PIL -> L mode (1 channel) -> replicate to 3 channels
    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.Grayscale(num_output_channels=3),  # 1ch -> 3ch (same gray in all)
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    if dataset_name == 'cifar100':
        train = tv_datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        test = tv_datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
        n_cls = 100
    elif dataset_name == 'oxford_pets':
        from torchvision.datasets import OxfordIIITPet
        full = OxfordIIITPet(root='./data', split='trainval', download=True, transform=transform_train)
        test = OxfordIIITPet(root='./data', split='test', download=True, transform=transform_test)
        n_val = int(len(full) * 0.2)
        n_tr = len(full) - n_val
        gen = torch.Generator().manual_seed(42)
        train, val = torch.utils.data.random_split(full, [n_tr, n_val], generator=gen)
        n_cls = 37
    elif dataset_name == 'food101':
        from torchvision.datasets import Food101
        train = Food101(root='./data', split='train', download=True, transform=transform_train)
        test = Food101(root='./data', split='test', download=True, transform=transform_test)
        n_cls = 101
    elif dataset_name == 'dtd':
        from torchvision.datasets import DTD
        train = DTD(root='./data', split='train', download=True, transform=transform_train)
        test = DTD(root='./data', split='test', download=True, transform=transform_test)
        n_cls = 47

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader, n_cls


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0; correct = 0; total = 0
    optimizer.zero_grad()
    for i, (images, targets) in enumerate(loader):
        images, targets = images.to(device), targets.to(device)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
        _, pred = logits.max(1)
        total += targets.size(0)
        correct += pred.eq(targets).sum().item()
    return total_loss / len(loader), 100. * correct / total

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0; correct = 0; total = 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        logits = model(images)
        loss = criterion(logits, targets)
        total_loss += loss.item()
        _, pred = logits.max(1)
        total += targets.size(0)
        correct += pred.eq(targets).sum().item()
    return total_loss / len(loader), 100. * correct / total

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=list(DATASETS.keys()))
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    args = parser.parse_args()

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    loader_name, n_cls, epochs, baseline_acc = DATASETS[args.dataset]

    print(f'Device: {device}', flush=True)
    print(f'Dataset: {args.dataset}', flush=True)
    print(f'Mode: GRAYSCALE (RGB->Gray->3ch)', flush=True)
    print(f'Batch: {args.batch_size}, Epochs: {epochs}', flush=True)
    print(f'Baseline Acc: {baseline_acc}%', flush=True)

    train_loader, test_loader, n_cls = make_grayscale_loader(args.dataset, args.batch_size)
    print(f'Train: {len(train_loader.dataset)}, Test: {len(test_loader.dataset)}, Classes: {n_cls}', flush=True)

    model = timm.create_model('vit_base_patch16_224.augreg_in21k', pretrained=True, num_classes=n_cls)
    model = model.to(device)
    print(f'Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M', flush=True)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0
    save_dir = f'checkpoints/{args.dataset}_vit_b16_grayscale'
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(epochs):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        t = time.time() - t0
        print(f'Epoch {epoch+1}/{epochs} ({t:.0f}s)  Train: {train_acc:.2f}%  Test: {test_acc:.2f}%', flush=True)
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'test_acc': test_acc},
                       f'{save_dir}/best_model.pth')

    print(f'\nBest Test Acc: {best_acc:.2f}% (Baseline: {baseline_acc}%, Diff: {best_acc-baseline_acc:+.2f}%)', flush=True)
    print(f'Grayscale vs RGB on {args.dataset}: color impact = {best_acc-baseline_acc:+.2f}%', flush=True)

if __name__ == '__main__':
    main()
