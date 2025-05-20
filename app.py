import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences # Імпортуємо з правильного місця
import numpy as np
import pickle
import re
import pandas as pd # Потрібен для LabelEncoder.classes_, якщо енкодери зберігалися як pandas об'єкти, хоча зазвичай це numpy array

# --- 0. Налаштування параметрів та назв колонок (з Jupyter Notebook) ---
# Константи, які потрібні для обробки вхідних даних та завантаження артефактів
MAX_LEN = 250 #
OOV_TOKEN = "<OOV>" # # Не використовується безпосередньо в app.py, але був частиною налаштувань токенізатора
MODEL_PATH = 'multi_output_fake_news_model.keras' #
TEXT_TOKENIZER_PATH = 'text_tokenizer.pkl' #
AUTHOR_ENCODER_PATH = 'author_encoder.pkl' #
SOURCE_ENCODER_PATH = 'source_encoder.pkl' #

# --- 1.4. Очищення основного тексту новини (з Jupyter Notebook, cell 7) ---
def clean_text(text): #
    if not isinstance(text, str): #
        text = str(text) #
    text = text.lower() #
    text = re.sub(r'[^a-z0-9\\s\\u0400-\\u04FF\\u0456\\u0457\\u0454\\u0491\\u0490]', '', text) # Додано підтримку кирилиці
    text = re.sub(r'\\s+', ' ', text).strip() #
    return text #

# Функція для завантаження артефактів з кешуванням
@st.cache_resource
def load_artifacts():
    """Завантажує модель Keras, текстовий токенізатор та енкодери."""
    try:
        model = tf.keras.models.load_model(MODEL_PATH) #
    except Exception as e:
        st.error(f"Помилка завантаження моделі ({MODEL_PATH}): {e}")
        st.error("Переконайтеся, що файл моделі знаходиться у тому ж каталозі, що й app.py, або вкажіть правильний шлях.")
        return None, None, None, None

    try:
        with open(TEXT_TOKENIZER_PATH, 'rb') as handle: #
            text_tokenizer = pickle.load(handle) #
    except FileNotFoundError:
        st.error(f"Файл токенізатора '{TEXT_TOKENIZER_PATH}' не знайдено.")
        return None, None, None, None
    except Exception as e:
        st.error(f"Помилка завантаження токенізатора ({TEXT_TOKENIZER_PATH}): {e}")
        return None, None, None, None

    try:
        with open(AUTHOR_ENCODER_PATH, 'rb') as handle: #
            author_encoder = pickle.load(handle) #
    except FileNotFoundError:
        st.error(f"Файл енкодера авторів '{AUTHOR_ENCODER_PATH}' не знайдено.")
        return None, None, None, None
    except Exception as e:
        st.error(f"Помилка завантаження енкодера авторів ({AUTHOR_ENCODER_PATH}): {e}")
        return None, None, None, None

    try:
        with open(SOURCE_ENCODER_PATH, 'rb') as handle: #
            source_encoder = pickle.load(handle) #
    except FileNotFoundError:
        st.error(f"Файл енкодера джерел '{SOURCE_ENCODER_PATH}' не знайдено.")
        return None, None, None, None
    except Exception as e:
        st.error(f"Помилка завантаження енкодера джерел ({SOURCE_ENCODER_PATH}): {e}")
        return None, None, None, None

    return model, text_tokenizer, author_encoder, source_encoder

# Основна частина Streamlit-додатка
if __name__ == '__main__':
    st.title("Аналізатор новин: Fake News Detection")
    st.write("""
    Введіть текст новини для аналізу. Модель спробує визначити, чи є новина фейковою або реальною,
    а також передбачить її можливого автора та джерело на основі навчених даних.
    """)

    # Завантаження артефактів
    model, text_tokenizer, author_encoder, source_encoder = load_artifacts()

    if model and text_tokenizer and author_encoder and source_encoder:
        # Текстове поле для введення новини
        news_text_input = st.text_area("Введіть текст новини тут:", height=200, key="news_text")

        # Кнопка "Аналізувати"
        if st.button("Аналізувати", key="analyze_button"):
            if not news_text_input.strip():
                st.warning("Будь ласка, введіть текст новини для аналізу.")
            else:
                with st.spinner("Аналіз новини..."):
                    # 1. Очищення введеного тексту
                    cleaned_text = clean_text(news_text_input) #

                    # 2. Токенізація та паддінг тексту
                    sequences = text_tokenizer.texts_to_sequences([cleaned_text]) #
                    padded_sequence = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post') #

                    # 3. Отримання передбачень від моделі
                    try:
                        predictions = model.predict(padded_sequence) #
                        label_pred_proba = predictions[0][0][0] # Перший вихід, перший семпл, перше значення
                        author_pred_probas = predictions[1][0]   # Другий вихід, перший семпл
                        source_pred_probas = predictions[2][0]   # Третій вихід, перший семпл
                    except Exception as e:
                        st.error(f"Помилка під час отримання передбачення від моделі: {e}")
                        st.stop()


                    # 4. Обробка передбачень
                    # Мітка 'fake'/'real'
                    predicted_label = "Real" if label_pred_proba > 0.5 else "Fake"
                    label_confidence = label_pred_proba if predicted_label == "Real" else 1 - label_pred_proba

                    # Автор
                    predicted_author_index = np.argmax(author_pred_probas)
                    predicted_author = author_encoder.classes_[predicted_author_index] #
                    author_confidence = author_pred_probas[predicted_author_index]

                    # Джерело
                    predicted_source_index = np.argmax(source_pred_probas)
                    predicted_source = source_encoder.classes_[predicted_source_index] #
                    source_confidence = source_pred_probas[predicted_source_index]

                    # 5. Відображення результатів
                    st.subheader("Результати аналізу:")

                    if predicted_label == "Fake":
                        st.error(f"**Статус новини:** {predicted_label} (Впевненість: {label_confidence:.2%})")
                    else:
                        st.success(f"**Статус новини:** {predicted_label} (Впевненість: {label_confidence:.2%})")

                    st.info(f"**Передбачений автор:** {predicted_author} (Впевненість: {author_confidence:.2%})")
                    st.info(f"**Передбачене джерело:** {predicted_source} (Впевненість: {source_confidence:.2%})")

                    with st.expander("Деталізовані ймовірності (Автор)"):
                        author_probs_df = pd.DataFrame({
                            'Автор': author_encoder.classes_,
                            'Ймовірність': author_pred_probas
                        }).sort_values(by='Ймовірність', ascending=False).head(10) # Показуємо топ-10
                        st.dataframe(author_probs_df.style.format({'Ймовірність': "{:.2%}"}))

                    with st.expander("Деталізовані ймовірності (Джерело)"):
                        source_probs_df = pd.DataFrame({
                            'Джерело': source_encoder.classes_,
                            'Ймовірність': source_pred_probas
                        }).sort_values(by='Ймовірність', ascending=False)
                        st.dataframe(source_probs_df.style.format({'Ймовірність': "{:.2%}"}))
    else:
        st.error("Не вдалося завантажити необхідні артефакти. Додаток не може продовжити роботу.")