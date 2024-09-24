import torch
import numpy as np
from torchvision.utils import save_image
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from model import Generator
from PIL import Image
from torchvision.transforms import Resize, ToTensor, Compose, Normalize

# Tải checkpoint cho mô hình ảnh
def load_checkpoint(filepath, generator, device):
    print(f"Loading checkpoint '{filepath}'")
    checkpoint = torch.load(filepath, map_location=device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    print(f"Loaded checkpoint '{filepath}' (epoch {checkpoint['epoch']})")

# Text classificationt
def classify_text(text, tokenizer, model_nlp, device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    with torch.no_grad():
        outputs = model_nlp(**inputs)

    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=-1).item()
    return predicted_class

def label2onehot(labels, dim):
    batch_size = labels.size(0)
    out = torch.zeros(batch_size, dim)
    out[np.arange(batch_size), labels.long()] = 1
    return out

def create_labels(c_org, c_dim=11):
    c_trg_list = []
    for i in range(c_dim):
        c_trg = label2onehot(torch.ones(c_org.size(0)) * i, c_dim)
        c_trg_list.append(c_trg)
    return c_trg_list

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_dir = "./saved_model"
    model_nlp = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model_nlp.to(device)
    model_nlp.eval()

    text = "hãy làm khuông mặt này vui vẻ hơn"
    # text = "Thêm chút sự kinh hãi vào khuôn mặt này."
    # text = "Làm khuôn mặt này trông hằn học hơn."
    prompt = classify_text(text, tokenizer, model_nlp, device)

    c_dim = 11
    image_size = 224
    g_conv_dim = 64
    repeat_num = 6
    image_path = "63.jpg"
    out_path = "out.jpg"
    checkpoint = "best_model.pt"

    transform = Compose([
        Resize((image_size, image_size)),
        ToTensor(),
        Normalize(mean=[0.5402, 0.4410, 0.3938], std=[0.2914, 0.2657, 0.2609]),
    ])

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device) # Thêm chiều batch
    x_fake_list = [image]

    g = Generator(g_conv_dim, c_dim, repeat_num).to(device)
    load_checkpoint(checkpoint, g, device)

    target = torch.zeros([1, c_dim]).to(device)
    target[0, prompt] = 1

    with torch.no_grad():
        x_fake_list.append(g(image, target))
        x_concat = torch.cat(x_fake_list, dim=3)
        x_concat = ((x_concat.data.cpu() + 1) / 2).clamp_(0, 1)
        save_image(x_concat, out_path, nrow=1, padding=0)

