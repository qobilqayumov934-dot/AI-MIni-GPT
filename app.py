import torch
import torch.nn as nn
import streamlit as st
import os

# === СИНХРОНИЗАЦИЯ НАСТРОЕК ===
block_size = 64 
device = 'cpu'         
n_embd = 128            
n_head = 2              
n_layer = 2             
MODEL_FILE = "transformer_model.pth"

# === НАСТРОЙКА СТРАНИЦЫ ===
st.set_page_config(page_title="Мой Робот ИИ", page_icon="🤖", layout="wide")

# === БОКОВАЯ ПАНЕЛЬ ДЛЯ МАМЫ (В ЛЕВОМ УГЛУ) ===
with st.sidebar:
    st.title("💡 Шпаргалка для Мамы")
    st.write("Привет, мам! Этот ИИ полностью написал твой сын.")
    st.write("Робот учился на специальной базе знаний, поэтому лучше всего он поймет вопросы на эти темы:")
    
    st.markdown("""
    * **Приветствие:** `Привет!`, `Хай, как звать?`
    * **О роботе:** `Кто тебя создал?`, `Чья ты разработка?`, `Что ты умеешь делать?`
    * **Железо:** `Какой графический чип самый мощный?`, `Зачем нужна кастомная вода?`
    * **Прочее:** `Сколько будет два плюс два?`, `Расскажи анекдот про ИИ?`
    """)
    st.info("💡 Пиши вопросы точно так же, как в примерах, чтобы робот ответил правильно!")

# === ЗАГРУЗКА МОДЕЛИ И СЛОВАРЯ ===
if not os.path.exists(MODEL_FILE):
    st.error(f"Файл '{MODEL_FILE}' не найден в папке проекта!")
    st.stop()

checkpoint = torch.load(MODEL_FILE, map_location=device)

if isinstance(checkpoint, dict) and 'chars' in checkpoint:
    chars = checkpoint['chars']
    model_weights = checkpoint['model_state_dict']
else:
    st.error("Обнаружен старый чекпоинт. Переобучи модель с новым train.py")
    st.stop()

vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s if c in stoi] 
decode = lambda l: ''.join([itos[i] for i in l])

# === АРХИТЕКТУРА ===
class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size))
                                     .view(1, 1, block_size, block_size))
    def forward(self, x):
        B, T, C = x.size()
        q, k, v  = self.c_attn(x).split(n_embd, dim=2)
        k = k.view(B, T, n_head, C // n_head).transpose(1, 2)
        q = q.view(B, T, n_head, C // n_head).transpose(1, 2)
        v = v.view(B, T, n_head, C // n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / (k.size(-1) ** 0.5))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = nn.functional.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_fc    = nn.Linear(n_embd, 4 * n_embd)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * n_embd, n_embd)
    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention()
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP()
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class LightTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(vocab_size, n_embd),
            wpe = nn.Embedding(block_size, n_embd),
            h = nn.ModuleList([Block() for _ in range(n_layer)]),
            ln_f = nn.LayerNorm(n_embd),
        ))
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx):
        t = idx.size(1)
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        return self.lm_head(x), None

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            idx = torch.cat((idx, idx_next), dim=1)
            if idx_next.item() == stoi.get('\n', -1):
                break
        return idx

@st.cache_resource
def load_my_model():
    model = LightTransformer().to(device)
    model.load_state_dict(model_weights)
    model.eval()
    return model

model = load_my_model()

# === ГЛАВНЫЙ ИНТЕРФЕЙС ===
st.title("🤖 Мой Карманный Трансформер")
st.subheader("Локальная нейросеть, обученная с нуля")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Напиши боту что-нибудь..."):
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Трюк жесткого маппинга для коротких фраз (борьба со сбоем контекста)
    low_input = user_input.lower().strip()
    if "привет" in low_input:
        prompt = "Привет! Как тебя зовут? Ответ:"
    elif "хай" in low_input or "звать" in low_input:
        prompt = "Хай, как звать? Ответ:"
    elif "кто ты" in low_input:
        prompt = "Здравствуй! Кто ты? Ответ:"
    else:
        prompt = f"\n{user_input.strip()} Ответ:"
    
    context = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    
    if context.size(1) == 0:
        bot_response = "Я не знаю таких символов. Попробуй спросить по-другому!"
    else:
        with torch.no_grad():
            generated_tokens = model.generate(context, max_new_tokens=64)[0].tolist()
        full_text = decode(generated_tokens)
        
        try:
            bot_response = full_text.split("Ответ:")[1].strip()
        except:
            bot_response = full_text[len(prompt):].strip()

    with st.chat_message("assistant"):
        st.markdown(bot_response)
    st.session_state.messages.append({"role": "assistant", "content": bot_response})