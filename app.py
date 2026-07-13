import streamlit as st
import torch
import torch.nn as nn
import os

# 1. Архитектура модели MiniGPT
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

# Настройка интерфейса Streamlit
st.set_page_config(page_title="Мой Робот ИИ", page_icon="🤖")
st.title("🤖 Мой Робот ИИ на базе датасета Saiga")

# Боковая панель с настройками
with st.sidebar:
    st.header("⚙️ Настройки ИИ")
    
    # НАШ СЛАЙДЕР! Значение по умолчанию 0.3, минимум 0.1, максимум 10.0
    temperature = st.slider(
        "Креативность (Temperature):", 
        min_value=0.1, 
        max_value=10.0, 
        value=0.3, 
        step=0.1,
        help="0.3 — четкий русский язык. 1.0 — фантазии. 10.0 — режим безумия (китайский язык)!"
    )
    
    st.markdown("---")
    st.header("💡 Шпаргалка")
    st.write("Попробуй выкрутить ползунок на 10.0 и написать 'Привет'!")

# 2. Безопасная загрузка модели и словаря
MODEL_PATH = "transformer_model.pth"

if not os.path.exists(MODEL_PATH):
    st.error(f"Файл {MODEL_PATH} не найден!")
else:
    checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    stoi = checkpoint['stoi']
    itos = checkpoint['itos']
    vocab_size = checkpoint['vocab_size']
    
    model = MiniGPT(vocab_size)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Функция генерации текста (теперь принимает temperature со слайдера)
    def generate_text(prompt, max_new_tokens=130, temp=0.3):
        full_prompt = f"Пользователь: {prompt}\nБот: "
        input_ids = [stoi[c] for c in full_prompt if c in stoi]
        if not input_ids:
            input_ids = [0]
            
        context_tensor = torch.tensor([input_ids], dtype=torch.long)
        generated = input_ids.copy()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = model(context_tensor)
                # Делим на выбранную пользователем температуру
                next_token_logits = logits[:, -1, :] / temp
                probs = torch.softmax(next_token_logits, dim=-1)
                
                next_token = torch.multinomial(probs, num_samples=1).item()
                
                generated.append(next_token)
                context_tensor = torch.tensor([generated], dtype=torch.long)
                
                if itos.get(next_token, "") == "\n" and temp < 2.0:
                    break
                    
        result_text = "".join([itos.get(i, "") for i in generated])
        bot_reply = result_text.replace(full_prompt, "").strip()
        return bot_reply if bot_reply else "...(робот задумался)..."

    # Интерфейс чата
    user_input = st.text_input("Напишите что-нибудь вашему ИИ:", placeholder="Пример: Привет!")

    if user_input:
        with st.spinner("Робот думает..."):
            # Передаем температуру из слайдера прямо в функцию
            reply = generate_text(user_input, temp=temperature)
            st.markdown(f"**Ответ робота (при Temp = {temperature}):**")
            st.info(reply)