import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

# 1. Гиперпараметры
BATCH_SIZE = 8          # Можно чуть увеличить, если ПК справляется
MAX_LEN = 256           # Модель будет видеть более длинные контексты диалога
LEARNING_RATE = 0.0005  # Чуть уменьшаем шаг, чтобы при долгом обучении модель училась аккуратнее
EPOCHS = 50             # Поставь для начала 20 или даже 50 эпох
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Используем устройство: {DEVICE}")

# 2. Загрузка датасета Saiga (первая 1000 примеров для быстрой демонстрации)
# ... (здесь идут настройки BATCH_SIZE, EPOCHS и т.д.) ...

# 2. Загрузка датасета Saiga (берем 5000 примеров для более глубокого обучения)
print("Загрузка датасета Saiga с Hugging Face...")
raw_dataset = load_dataset("IlyaGusev/saiga_scored", split="train[:5000]") 

# 3. Сбор словаря символов ... (и дальше код идет без изменений)

# 3. Сбор словаря символов
text_corpus = ""
for item in raw_dataset:
    for msg in item["messages"]:
        text_corpus += msg["content"] + " "

chars = sorted(list(set(text_corpus)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return [stoi[c] for c in s if c in stoi]

print(f"Размер словаря символов: {vocab_size}")

# 4. Класс датасета для PyTorch
class SaigaDataset(Dataset):
    def __init__(self, dataset, max_len):
        self.data = []
        for item in dataset:
            dialogue = ""
            for msg in item["messages"]:
                role = "Пользователь: " if msg["role"] == "user" else "Бот: "
                dialogue += f"{role}{msg['content']}\n"
            
            encoded = encode(dialogue)
            if len(encoded) > max_len:
                self.data.append(encoded[:max_len])
            elif len(encoded) < max_len:
                padded = encoded + [0] * (max_len - len(encoded))
                self.data.append(padded)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokenized = torch.tensor(self.data[idx], dtype=torch.long)
        x = tokenized[:-1]
        y = tokenized[1:]
        return x, y

train_dataset = SaigaDataset(raw_dataset, MAX_LEN + 1)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# 5. Архитектура модели MiniGPT (должна быть одинаковой в обоих файлах!)
class MiniGPT(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embedded = self.token_embedding(x)
        lstm_out, _ = self.lstm(embedded)
        logits = self.fc(lstm_out)
        return logits

# Создание и обучение
model = MiniGPT(vocab_size).to(DEVICE)
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

print("Начало обучения модели...")
model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
        
        optimizer.zero_grad()
        logits = model(x_batch)
        
        loss = criterion(logits.view(-1, vocab_size), y_batch.view(-1))
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Эпоха {epoch+1}/{EPOCHS} | Loss: {total_loss / len(train_loader):.4f}")

# Сохраняем модель вместе со словарем!
torch.save({
    'model_state_dict': model.state_dict(),
    'stoi': stoi,
    'itos': itos,
    'vocab_size': vocab_size
}, "transformer_model.pth")
print("Обучение завершено! Модель сохранена.")