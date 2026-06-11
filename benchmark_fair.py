"""Fair benchmark: all models, same conditions."""
import torch, time, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets import get_cifar100_loader
from models import create_model
import timm

device = 'cuda:3'
# Create dataloaders for each input size
_, loader_224, _ = get_cifar100_loader(batch_size=128, data_dir='./data', num_workers=4, image_size=224)
_, loader_168, _ = get_cifar100_loader(batch_size=128, data_dir='./data', num_workers=4, image_size=168)
_, loader_112, _ = get_cifar100_loader(batch_size=128, data_dir='./data', num_workers=4, image_size=112)

def bench(model, dl, nb=50):
    model.eval().to(device)
    for im,_ in dl: _ = model(im.to(device)); break
    torch.cuda.synchronize()
    t0=time.time(); n=0
    for i,(im,_) in enumerate(dl):
        if i>=nb: break
        _ = model(im.to(device)); n+=im.size(0)
    torch.cuda.synchronize()
    e=time.time()-t0
    return n/e, e/n*1000

def acc(model, dl):
    model.eval(); c=t=0
    with torch.no_grad():
        for im,lb in dl:
            im,lb=im.to(device),lb.to(device)
            _,p=model(im).max(1); t+=lb.size(0); c+=p.eq(lb).sum().item()
    return 100.*c/t

models = []

print('1/5 Baseline...', flush=True)
m = timm.create_model('vit_base_patch16_224.augreg_in21k', pretrained=True, num_classes=100)
ckpt = torch.load('checkpoints/cifar100_vit_b16_ft/best_model.pth', map_location='cpu', weights_only=True)
m.load_state_dict(ckpt.get('model_state_dict', ckpt))
models.append(('Baseline (196)', 196, m, loader_224))

print('2/5 MAE keep 75%...', flush=True)
m = create_model('mae_patch_selection_vit_b16', num_classes=100, keep_ratio=0.75, pretrained=True,
                 decoder_embed_dim=512, decoder_depth=4)
ckpt = torch.load('checkpoints/cifar100_mae_patchsel_b16_keep75_distill/best_model.pth', map_location='cpu')
m.load_state_dict(ckpt['model_state_dict'])
models.append(('MAE+蒸馏(75%)', 147, m, loader_224))

print('3/5 MAE keep 50%...', flush=True)
m = create_model('mae_patch_selection_vit_b16', num_classes=100, keep_ratio=0.5, pretrained=True,
                 decoder_embed_dim=512, decoder_depth=4)
ckpt = torch.load('checkpoints/cifar100_mae_patchsel_b16_keep50/best_model.pth', map_location='cpu')
m.load_state_dict(ckpt['model_state_dict'])
models.append(('MAE+蒸馏(50%)', 98, m, loader_224))

print('4/5 Downsample 168...', flush=True)
m = timm.create_model('vit_base_patch16_224.augreg_in21k', pretrained=True, num_classes=100, img_size=168)
ckpt = torch.load('checkpoints/cifar100_vit_b16_img168/best_model.pth', map_location='cpu', weights_only=True)
m.load_state_dict(ckpt['model_state_dict'])
models.append(('降采样168x168', 100, m, loader_168))

print('5/5 Downsample 112...', flush=True)
m = timm.create_model('vit_base_patch16_224.augreg_in21k', pretrained=True, num_classes=100, img_size=112)
ckpt = torch.load('checkpoints/cifar100_vit_b16_img112/best_model.pth', map_location='cpu', weights_only=True)
m.load_state_dict(ckpt['model_state_dict'])
models.append(('降采样112x112', 49, m, loader_112))

print(f'\n{"Model":<25} {"Patches":>8} {"Acc":>8} {"Throughput":>12} {"Latency":>8}', flush=True)
print('-'*65, flush=True)
for name, p, m, loader in models:
    t, l = bench(m, loader)
    a = acc(m, loader)
    print(f'{name:<25} {p:>8} {a:>7.2f}% {t:>10.0f}/s {l:>7.3f}ms', flush=True)

print(f'\nGPU:{device} batch=128 warmup+50batches', flush=True)
