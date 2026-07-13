import streamlit as st
from google import genai
from google.genai import types

st.set_page_config(page_title="Мой Супер ИИ", page_icon="🧠")
st.title("🧠 Мой Умный Робот на базе Gemini")

# Пытаемся автоматически достать скрытый ключ из настроек Streamlit
try:
    # КЛЮЧ НЕ В КОДЕ! Он берется из защищенного хранилища сайта
    api_key = st.secrets["GEMINI_API_KEY"]
    client = genai.Client(api_key=api_key)
    
    user_input = st.text_input("Напишите что-нибудь вашему ИИ:", placeholder="Например: Как дела?")

    if user_input:
        with st.spinner("ИИ думает..."):
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=user_input,
            )
            st.info(response.text)

except Exception as e:
    st.error("🔑 Ошибка доступа! На сервере Streamlit не настроен секретный API-ключ.")
    st.info("Администратор: Зайди в настройки (Secrets) на share.streamlit.io и добавь GEMINI_API_KEY")