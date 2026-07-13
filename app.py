import streamlit as st
from groq import Groq

st.set_page_config(page_title="Мой Супер ИИ", page_icon="🧠")
st.title("🧠 Мой Умный Робот")

with st.sidebar:
    st.header("⚙️ Настройки ИИ")
    temperature = st.slider(
        "Креативность (Temperature):", 
        min_value=0.0, 
        max_value=2.0, 
        value=0.7, 
        step=0.1
    )

# Пытаемся безопасно достать ключ из секретов сервера
try:
    API_KEY = st.secrets["GROQ_API_KEY"]
    client = Groq(api_key=API_KEY)
    
    user_input = st.text_input("Напишите что-нибудь вашему ИИ:", placeholder="Например: Как дела?")

    if user_input:
        with st.spinner("ИИ анализирует твой тон..."):
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Ты — адаптивный ИИ. Твой характер полностью зависит от тона собеседника. "
                            "1. Если пользователь пишет грубо, хамит или обзывается — отвечай взаимно грубо, иронично и токсично. "
                            "2. Если пользователь пишет вежливо или нейтрально — будь добрым, вежливым и уважительным помощником. "
                            "Отвечай строго на русском языке, коротко и емко."
                        )
                    },
                    {
                        "role": "user",
                        "content": user_input,
                    }
                ],
                model="llama-3.3-70b-versatile",
                temperature=temperature,
            )
            
            st.markdown(f"**Ответ робота:**")
            st.info(chat_completion.choices[0].message.content)

except Exception as e:
    st.error("🔑 Ошибка доступа! На сервере Streamlit не настроен или не прочитан ключ GROQ_API_KEY.")
    st.info("Проверь вкладку Secrets в настройках приложения на share.streamlit.io")