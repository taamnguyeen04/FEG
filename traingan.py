import os
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision.datasets import ImageFolder
from torchvision.transforms import Resize, ToTensor, Compose, Normalize
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import Generator, Discriminator
from dataset import Affectnet

def save_checkpoint(filepath, epoch, step, generator, discriminator, g_optimizer, d_optimizer, g_loss):
    # print(f"save {filepath}", filepath)
    checkpoint = {
        'epoch': epoch,
        'step' : step,
        'g_loss': g_loss.item(),
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'g_optimizer_state_dict': g_optimizer.state_dict(),
        'd_optimizer_state_dict': d_optimizer.state_dict()
    }
    torch.save(checkpoint, filepath)

def load_checkpoint(filepath, generator, discriminator, g_optimizer, d_optimizer, best_g_loss, device):
    last_path = os.path.join(filepath, "last_model.pt")
    best_path = os.path.join(filepath, "best_model.pt")
    if os.path.isfile(last_path):
        print(f"Loading checkpoint '{last_path}'")
        checkpoint = torch.load(last_path, map_location=device)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Loaded checkpoint '{last_path}' (epoch {checkpoint['epoch']}, step {checkpoint['step']})")
    else:
        print(f"No checkpoint found at '{last_path}'")
        start_epoch = 0
    if os.path.isfile(best_path):
        checkpoint = torch.load(best_path, map_location=device)
        best_g_loss = checkpoint['g_loss']
    else:
        best_g_loss = float('inf')
    return start_epoch


def gradient_penalty(device, y, x):
    weight = torch.ones(y.size()).to(device)
    dydx = torch.autograd.grad(outputs=y,
                               inputs=x,
                               grad_outputs=weight,
                               retain_graph=True,
                               create_graph=True,
                               only_inputs=True)[0]

    dydx = dydx.view(dydx.size(0), -1)
    dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
    return torch.mean((dydx_l2norm-1)**2)
def label2onehot(labels, dim):
    batch_size = labels.size(0)
    out = torch.zeros(batch_size, dim)
    out[np.arange(batch_size), labels.long()] = 1
    return out

def create_labels(c_org, c_dim=5):
    c_trg_list = []
    for i in range(c_dim):
        c_trg = label2onehot(torch.ones(c_org.size(0)) * i, c_dim)
        c_trg_list.append(c_trg)
    return c_trg_list

def cout(a):
    print("****************************")
    print(a)
    print(type(a))
    print("****************************")

def train():
    batch_size = 4
    lr = 1e-4
    num_epochs = 100
    image_channels = 3
    c_dim = 11
    image_size = 224
    g_conv_dim = 64
    d_conv_dim = 64
    repeat_num = 6
    n_critic = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # labels = ["angry", "disgust", "fear","happy", "neutral", "sad", "surprise"]

    log_dir = "runs/exp"
    model_path = "/home/tam/Desktop/pythonProject1/FEG"
    out_path = "out"

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
        os.makedirs(log_dir, exist_ok=True)
    else:
        os.makedirs(log_dir, exist_ok=True)
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
        os.makedirs(out_path, exist_ok=True)
    else:
        os.makedirs(out_path, exist_ok=True)

    print(device)
    best_d_loss = float('inf')
    best_g_loss = float('inf')
    g_loss = None
    d_loss = None
    writer = SummaryWriter(log_dir)


    transform = Compose([
        Resize((image_size, image_size)),
        ToTensor(),
        Normalize(mean=[0.5402, 0.4410, 0.3938], std=[0.2914, 0.2657, 0.2609]),#tìm mean std
    ])
    # Data loader
    # train_dataset = ImageFolder(root='/home/tam/Desktop/pythonProject1/archive/AffectNet/data', transform=transform)
    train_dataset = Affectnet(root="/home/tam/Desktop/pythonProject1/data/AffectNet", is_train=True, transform=transform)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=8,
        shuffle=True,
        drop_last=True
    )

    val_dataset = Affectnet(root="/home/tam/Desktop/pythonProject1/data/AffectNet", is_train=False, transform=transform)
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=8,
        shuffle=True,
        drop_last=True
    )

    # Models
    G = Generator(g_conv_dim, c_dim, repeat_num).to(device)
    D = Discriminator(image_size, d_conv_dim, c_dim, repeat_num).to(device)

    # Optimizer
    g_optimizer = torch.optim.Adam(params=G.parameters(), lr=lr, betas=(0.5, 0.99))
    d_optimizer = torch.optim.Adam(params=D.parameters(), lr=lr, betas=(0.5, 0.99))


    # Load the best model if it exists
    start_epoch = load_checkpoint(model_path, G, D, g_optimizer, d_optimizer, best_g_loss, device)

    x_fixed, c_org = next(iter(val_dataloader))
    x_fixed = x_fixed.to(device)
    c_fixed_list = create_labels(c_org, c_dim)
    c_fixed_list = torch.stack(c_fixed_list).to(device)

    try:
        for epoch in range(start_epoch, num_epochs):
            for i, (x_real, label_org) in enumerate(train_dataloader):
                rand_idx = torch.randperm(label_org.size(0))
                label_trg = label_org[rand_idx]

                c_org = label2onehot(label_org, c_dim).to(device)
                c_trg = label2onehot(label_trg, c_dim).to(device)

                x_real = x_real.to(device)
                label_org = label_org.to(device)
                label_trg = label_trg.to(device)


                # Train Discriminator
                out_src, out_cls = D(x_real)
                d_loss_real = -torch.mean(out_src)
                d_loss_cls = F.cross_entropy(out_cls, label_org)

                # Compute loss with fake images
                x_fake = G(x_real, c_trg)
                out_src, out_cls = D(x_fake.detach())
                d_loss_fake = torch.mean(out_src)

                # Compute loss for gradient penalty.
                alpha = torch.rand(x_real.size(0), 1, 1, 1).to(device)
                x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
                out_src, _ = D(x_hat)
                d_loss_gp = gradient_penalty(device, out_src, x_hat)

                # Compute total loss
                d_loss = d_loss_real + d_loss_fake + d_loss_cls + 10 * d_loss_gp
                g_optimizer.zero_grad()
                d_optimizer.zero_grad()
                d_loss.backward()
                d_optimizer.step()

                # Logging.
                loss = {}
                loss['D/loss_total'] = d_loss.item()
                loss['D/loss_real'] = d_loss_real.item()
                loss['D/loss_fake'] = d_loss_fake.item()
                loss['D/loss_cls'] = d_loss_cls.item()
                loss['D/loss_gp'] = d_loss_gp.item()

                # Train Genrator
                if i % n_critic == 0:
                    # Original-to-target domain.
                    x_fake = G(x_real, c_trg)
                    out_src, out_cls = D(x_fake)
                    g_loss_fake = -torch.mean(out_src)
                    g_loss_cls = F.cross_entropy(out_cls, label_trg)

                    # Target-to-original domain.
                    x_reconst = G(x_fake, c_org)
                    g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))

                    # Backward and optimize.
                    g_loss = g_loss_fake + 10 * g_loss_rec + g_loss_cls
                    g_optimizer.zero_grad()
                    d_optimizer.zero_grad()
                    g_loss.backward()
                    g_optimizer.step()

                    # Logging.
                    loss['G/loss_total'] = g_loss.item()
                    loss['G/loss_fake'] = g_loss_fake.item()
                    loss['G/loss_rec'] = g_loss_rec.item()
                    loss['G/loss_cls'] = g_loss_cls.item()

                for tag, value in loss.items():
                    writer.add_scalar(tag, value, global_step=epoch * len(train_dataloader) + i)

                print(f"Epoch [{epoch}/{num_epochs}], Step [{i + 1}/{len(train_dataloader)}], Loss: {loss}")

                if (i % 10 == 0) and (i != 0):
                    with torch.no_grad():
                        x_fake_list = [x_fixed]
                        for c_fixed in c_fixed_list:
                            x_fake_list.append(G(x_fixed, c_fixed))
                        x_concat = torch.cat(x_fake_list, dim=3)
                        x_concat = ((x_concat.data.cpu() + 1)/2).clamp_(0, 1)
                        sample_path = os.path.join(out_path, '{}-images.jpg'.format(i))
                        save_image(x_concat, sample_path, nrow=1, padding=0)

                # Save model
                save_checkpoint('last_model.pt', epoch, i + 1, G, D, g_optimizer, d_optimizer, g_loss)
                if g_loss.item() < best_g_loss:
                    best_g_loss = g_loss.item()
                    save_checkpoint('best_model.pt', epoch, i + 1, G, D, g_optimizer, d_optimizer, g_loss)
    except KeyboardInterrupt:
        save_checkpoint('last_model.pt', epoch, i + 1, G, D, g_optimizer, d_optimizer, g_loss)

if __name__ == '__main__':
    train()

