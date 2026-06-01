import streamlit as st
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.predict import load_model, predict_image, CLASSES

st.set_page_config(page_title='Fashion Classifier', layout='centered')

st.title('Fashion-MNIST Classifier')
st.markdown('Upload an image of a clothing item and the model predicts which category it belongs to.')
st.markdown(f'**Classes:** {", ".join(CLASSES)}')

MODEL_PATH = Path(__file__).parent.parent / 'models' / 'fashion_best.pt'

@st.cache_resource
def load():
    if not MODEL_PATH.exists():
        st.error('Model not found. Run `python src/train.py` first.')
        st.stop()
    return load_model(str(MODEL_PATH))

model = load()
st.success('Model loaded successfully')

uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png', 'webp'])

if uploaded_file is not None:
    from PIL import Image
    image = Image.open(uploaded_file).convert('RGB')

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image, caption='Uploaded Image', use_container_width=True)

    with col2:
        with st.spinner('Predicting...'):
            results = predict_image(model, image)

        st.subheader('Predictions')
        for i, (label, confidence) in enumerate(results):
            st.markdown(f'**{i + 1}. {label}** — {confidence}%')
            st.progress(confidence / 100)

    st.divider()
    st.caption('Model: FashionCNN (3 Conv2D layers, trained on 1000 Fashion-MNIST samples)')
