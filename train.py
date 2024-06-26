import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Resize, ToTensor, Compose, Normalize
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import model
import numpy as np
from dataset import ExpW
from model import Generator, Discriminator

def adv_loss(logits, target):
    """
    logits = torch.tensor([0.1])
    target = 1
    adv_loss(logits, target)
    """
    targets = torch.full_like(logits, fill_value=target)
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    return loss

def domain_loss(d_out, c):
    """
    rand = torch.randn(4)
    d_out = torch.exp(rand)/torch.exp(rand).sum()
    c = torch.tensor([1, 2])
    domain_loss(d_out, c)
    """
    v_out = c.long()
    # print(d_out)
    # print(v_out)
    loss = -v_out * torch.log(d_out + 1e-12)
    loss=loss.sum()
    return loss


def rec_l1(x, g_out):
    """
    x = torch.randn((4, 28, 28, 3))
    g_out = torch.randn((4, 28, 28, 3))
    _rec_l1(x, g_out)
    """
    assert x.size() == g_out.size()
    return torch.mean(torch.abs(x-g_out))

def save_checkpoint(filepath, epoch, generator, discriminator, g_optimizer, d_optimizer):
    checkpoint = {
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'g_optimizer_state_dict': g_optimizer.state_dict(),
        'd_optimizer_state_dict': d_optimizer.state_dict()
    }
    print("****************************************************************************************************************************")
    torch.save(checkpoint, filepath)

def load_checkpoint(filepath, generator, discriminator, g_optimizer, d_optimizer, device):
    if os.path.isfile(filepath):
        print(f"Loading checkpoint '{filepath}'")
        checkpoint = torch.load(filepath, map_location=device)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Loaded checkpoint '{filepath}' (epoch {checkpoint['epoch']})")
    else:
        print(f"No checkpoint found at '{filepath}'")
        start_epoch = 0
    return start_epoch

def train():
    batch_size = 4
    lr = 1e-3
    num_epochs = 100
    image_channels = 3
    c_dim = 7
    image_size = 224
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    best_d_loss = float('inf')
    best_g_loss = float('inf')

    transform = Compose([
        Resize((image_size, image_size)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    log_dir = "runs/exp"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    # Data loader
    train_dataset = ExpW(root="data/ExpW/data", transform=transform)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=8,
        shuffle=True,
        drop_last=True
    )

    # Models
    G = Generator(image_channels=image_channels, c_dim=c_dim).to(device)
    D = Discriminator(image_size=image_size, image_channels=image_channels, c_dim=c_dim).to(device)

    # Training loop
    g_optimizer = torch.optim.SGD(params=G.parameters(), lr=lr, momentum=0.9)
    d_optimizer = torch.optim.SGD(params=D.parameters(), lr=lr, momentum=0.9)
    # g_optimizer = torch.optim.Adam(params=G.parameters(), lr=lr)
    # d_optimizer = torch.optim.Adam(params=D.parameters(), lr=lr)

    # Load the best model if it exists
    best_model_path = 'best_model.pt'
    start_epoch = load_checkpoint(best_model_path, G, D, g_optimizer, d_optimizer, device)
    print(start_epoch)

    for epoch in range(num_epochs):
        for i, (x_real, label_org) in enumerate(train_dataloader):
            x_real = x_real.to(device)
            label_org = label_org.to(device)

            rand_idx = torch.randperm(label_org.size(0))
            label_trg = label_org[rand_idx]
            label_trg = label_trg.to(device)  # Move label_trg to device
            label_trg = torch.eye(7, device=device)[label_trg - 1]  # Create one-hot encoding
            label_org = torch.eye(7, device=device)[label_org - 1]  # Create one-hot encoding

            # Train Generator
            x_fake = G(x_real, label_trg)
            out_src, out_cls = D(x_fake)
            g_loss_fake = adv_loss(out_src, 1)
            g_loss_cls = domain_loss(out_cls, label_trg)
            x_reconst = G(x_fake, label_org)
            g_loss_rec = rec_l1(x_real, x_reconst)
            # Compute total loss
            g_loss = g_loss_fake + g_loss_cls + g_loss_rec
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            # Train Discriminator
            out_src, out_cls = D(x_real)
            d_loss_real = adv_loss(out_src, 1)
            d_loss_cls = domain_loss(out_cls, label_org)# lỗi ở đây

            # Compute loss with fake images
            x_fake = G(x_real, label_trg)
            out_src, _ = D(x_fake.detach())
            d_loss_fake = adv_loss(out_src, 0)
            # Compute total loss
            d_loss = d_loss_real + d_loss_fake + d_loss_cls
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            if i % 100 == 0:
                writer.add_scalar('Loss/Discriminator', d_loss.item(), epoch * len(train_dataloader) + i)
                writer.add_scalar('Loss/Generator', g_loss.item(), epoch * len(train_dataloader) + i)

            print(f"Epoch [{epoch}/{num_epochs}], Step [{i+1}/{len(train_dataloader)}], d_loss: {d_loss.item()}, g_loss: {g_loss.item()}")

            #save model
            save_checkpoint('last_model.pt', epoch, G, D, g_optimizer, d_optimizer)
            if d_loss.item() + g_loss.item() < best_d_loss + best_g_loss:
                best_d_loss = d_loss.item()
                best_g_loss = g_loss.item()
                save_checkpoint('best_model.pt', epoch, G, D, g_optimizer, d_optimizer)


if __name__ == '__main__':
    train()
    # print(torch.cuda.is_available())
