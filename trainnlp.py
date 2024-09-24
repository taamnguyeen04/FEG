import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AdamW
from sklearn.model_selection import train_test_split
import pandas as pd
# Tải tập dữ liệu từ đường dẫn
file_path = '/home/tam/Desktop/pythonProject1/FEG/data.csv'
data = pd.read_csv(file_path)

# Đặt lại chỉ số để tránh lỗi tiềm ẩn
data = data.reset_index(drop=True)

# Chia và mã hóa dữ liệu
train_texts, val_texts, train_labels, val_labels = train_test_split(
    data['text'], data['label'], test_size=0.2, random_state=42
)
print("alo")

# Đặt lại chỉ số sau khi chia
train_texts = train_texts.reset_index(drop=True)
train_labels = train_labels.reset_index(drop=True)
val_texts = val_texts.reset_index(drop=True)
val_labels = val_labels.reset_index(drop=True)

# Tải mô hình PhoBERT và tokenizer
phobert = AutoModelForSequenceClassification.from_pretrained("vinai/phobert-base-v2", num_labels=8)
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")

train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(list(val_texts), truncation=True, padding=True, max_length=128)

# Chuyển đổi sang PyTorch dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = Dataset(train_encodings, train_labels)
val_dataset = Dataset(val_encodings, val_labels)

# Tạo DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Chuẩn bị optimizer
optimizer = AdamW(phobert.parameters(), lr=5e-5)

# Đưa mô hình lên GPU nếu có
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
phobert.to(device)

# Đặt mô hình về chế độ huấn luyện
phobert.train()

# Vòng lặp huấn luyện
for epoch in range(3):  # Huấn luyện trong 3 epochs
    for batch_idx, batch in enumerate(train_dataloader):
        # Đưa batch lên đúng thiết bị (CPU/GPU)
        batch = {key: val.to(device) for key, val in batch.items()}

        # Forward pass
        outputs = phobert(**batch)
        loss = outputs.loss

        # Backward pass và bước tối ưu hóa
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # In ra loss của batch hiện tại
        print(f"Epoch: {epoch+1}, Batch: {batch_idx+1}, Loss: {loss.item()}")

# Lưu mô hình và tokenizer sau khi huấn luyện
output_dir = "./saved_model"
phobert.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print("Quá trình huấn luyện hoàn tất và mô hình đã được lưu.")
