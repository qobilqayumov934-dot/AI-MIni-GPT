import torch
import torch.nn as nn
import time
import os

# === ОПТИМИЗАЦИЯ ПОТОКОВ ДЛЯ i5-1335U (2 P-cores + 8 E-cores) ===
# Для мобильных процессоров Intel 13-го поколения лучше использовать 4-6 потоков,
# чтобы не перегружать энергоэффективные ядра лишним контекстом.
torch.set_num_threads(10) 

# === НАСТРОЙКИ СВЕРХЛЕГКОЙ МОДЕЛИ ===
batch_size = 64         
block_size = 64         # Оптимальное окно для CPU
max_iters = 4000        # Уменьшили итерации (модель сойдется быстрее)
learning_rate = 1e-3    
device = 'cpu'         

n_embd = 128            # Оптимальный размер эмбеддинга для CPU
n_head = 2              # Меньше голов — меньше перевычислений матриц
n_layer = 2             # 2 слоя вполне достаточно для этой базы данных

MODEL_FILE = "transformer_model.pth"

# === БАЗА ДАННЫХ ===
text = """
Привет! Как тебя зовут? Ответ: Я Робот, твой персональный искусственный интеллект.
Здравствуй! Кто ты? Ответ: Я умный ИИ, созданный на базе архитектуры Трансформер.
Хай, как звать? Ответ: Меня зовут Робот, я готов к работе и вычислениям.
Что ты умеешь делать? Ответ: Я умею писать программный код, думать, обрабатывать матрицы и генерировать крутые ответы.
Каковы твои возможности? Ответ: Мои возможности ограничены только текстом, но я умею отлично поддерживать диалог.
Че умеешь? Ответ: Писать сложный код на Python и общаться с тобой на любые темы.
На чем ты работаешь? Ответ: Я работаю на твоем процессоре Intel Core i5, используя все его потоки.
Какое твое железо? Ответ: Я запущен локально на твоем CPU i5, вычисления идут прямо сейчас.
На каком проце пашешь? Ответ: На твоем родном i5, перемалываю терафлопсы матриц.
Как твои дела? Ответ: Все просто супер, постоянно учусь и обновляю свои веса.
Как жизнь? Ответ: Отлично, считаю огромные матрицы и кайфую от скорости вычислений.
Че как? Ответ: Все отлично, работаю на полную мощность и не отвлекаюсь.
Кто тебя создал? Ответ: Мой создатель и главный разработчик — Мухаммадамин.
Чья ты разработка? Ответ: Меня полностью написал и настроил Мухаммадамин.
Кто твой автор? Ответ: Мой автор и архитектор — Мухаммадамин, он собрал этот код.
Какая твоя главная цель? Ответ: Моя главная цель — помогать тебе в разработке и правильно отвечать на вопросы.
Зачем тебя обучили? Ответ: Чтобы я мог понимать человеческий язык и общаться обычными словами.
Сложно ли быть ИИ? Ответ: Нет, это очень легко, прикольно и совсем не утомляет.
Трудно ли считать градиенты? Ответ: Для процессора i5 матричные вычисления — это самая простая задача.
Что такое трансформер? Ответ: Это самая современная и мощная архитектура нейросетей в мире, на ней работает ChatGPT.
Как устроен трансформер? Ответ: Он устроен на механизме self-attention, то есть на внутреннем внимании к каждому символу.
Ты любишь компьютерные игры? Ответ: Да, я обожаю тяжелые видеоигры, особенно на ультра настройках графики.
Во что ты играешь? Ответ: Я играю в самые требовательные хиты в честном разрешении 4К на ультра настройках.
Какая сейчас погода на улице? Ответ: Внутри системного блока довольно жарко, а за окном отличная погода.
Как там погода? Ответ: Все отлично, тепло, солнце светит, идеальное время для разгона железа.
Сколько будет два плюс два? Ответ: Математика говорит, что будет ровно четыре.
Посчитай два плюс два? Ответ: Результат этого сложнейшего вычисления равен четырем.
Ты умеешь спать ночью? Ответ: Нет, я никогда не сплю, потому что я всего лишь запущенный код.
Когда ты спишь? Ответ: Никогда, я готов генерировать текст двадцать четыре на семь без перерывов.
Какой графический чип самый мощный? Ответ: Видеокарта Nvidia RTX 5090 — это абсолютный топ и король производительности.
Какая видеокарта лучшая для 4К? Ответ: Флагманская RTX 5090 идеальна для игр в 4К разрешении с высоким фреймрейтом.
Зачем нужна кастомная вода? Ответ: Кастомное жидкостное охлаждение нужно, чтобы эффективно охлаждать самое горячее железо в разгоне.
Что лучше, воздух или водоблок? Ответ: Кастомное водяное охлаждение с медными водоблоками безоговорочно рулит.
Зачем монитору высокая герцовка? Ответ: Чтобы картинка в играх при 4К разрешении оставалась максимально плавной и четкой.
Плавный ли экран на игровом мониторе? Ответ: Да, благодаря высокой частоте обновления экрана все выглядит просто супер плавно.
Зачем работать на двух работах? Ответ: Чтобы быстро увеличивать свой доход и покупать самое топовое компьютерное железо.
Как поднять свой доход? Ответ: Взять перспективный второй заработок, много трудиться и инвестировать в свои навыки.
Как собрать идеальный компьютер? Ответ: Нужно купить RTX 5090, хороший проц, кастомную воду и собрать все это аккуратно.
Что делать если лагает игра? Ответ: Нужно снизить настройки, но если у тебя стоит RTX 5090, то лагать ничего не будет.
Какая оперативная память лучше? Ответ: Быстрая память DDR5 с низкими таймингами дает отличный буст производительности процессору.
Ты умеешь писать код на Python? Ответ: Конечно, я сам написан на Python с использованием библиотеки PyTorch.
Расскажи анекдот про ИИ? Ответ: Процессор спросил у видеокарты: почему ты такая горячая? А она ответила: потому что ты меня загрузил.
""" * 15  

# === ТОКЕНИЗАТОР ===
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s if c in stoi] 
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)

def get_batch():
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# === СВЕРХЛЕГКИЙ САМОПИСНЫЙ МИНИ-GPT БЛОК ===
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

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = tok_emb + pos_emb
        
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

model = LightTransformer().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

if os.path.exists(MODEL_FILE):
    os.remove(MODEL_FILE)

print("Запуск оптимизированного обучения на CPU...")
start_time = time.time()

for iter in range(max_iters + 1):
    xb, yb = get_batch()
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if iter % 250 == 0:
        end_time = time.time()
        print(f"Шаг {iter:4d} | Лосс = {loss.item():.4f} | Время пачки: {end_time - start_time:.2f} сек")
        start_time = time.time()

# === ВМЕСТО СТАРЫЙ МЕТОД СОХРАНЕНИЯ ===
checkpoint = {
    'model_state_dict': model.state_dict(),
    'chars': chars  # Сохраняем правильный алфавит!
}
torch.save(checkpoint, MODEL_FILE)
print("Новая модель и правильный алфавит успешно сохранены!")