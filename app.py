import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import tensorflow as tf

# Load CNN model
model = load_model('model/improved_resnet_model_24_class_82.h5')

# Label kelas
class_labels = ['freshbayam','freshbittergroud', 'freshbrokoli', 'freshbuncis','freshcucumber', 'freshkangkung','freshokra','freshpotato', 'freshsawi', 'freshterong','freshtomato', 'freshwortel', 'rottenbayam','rottenbittergroud', 'rottenbrokoli', 'rottenbuncis','rottencucumber', 'rottenkangkung','rottenokra', 'rottenpotato', 'rottensawi', 'rottenterong','rottentomato', 'rottenwortel']  # Sesuaikan jika lebih dari 2 kelas

nutrition_info = {
    'bayam': "Bayam Mengandung zat besi, vitamin A, C, dan K, serta asam folat; baik untuk darah, mata, dan daya tahan tubuh.",
    'brokoli': "Brokoli mengandung kaya vitamin C dan K, serat, serta antioksidan sulforaphane; bermanfaat untuk tulang, imun, dan pencegahan kanker.",
    'buncis': "Buncis mengandung serat, vitamin A, C, dan K; mendukung pencernaan, jantung, dan kontrol gula darah.",
    'kangkung': "Kangkung mengandung vitamin A, C, zat besi, dan kalsium; baik untuk mata, darah, dan pencernaan.",
    'potato': "kentang mengandung sumber karbohidrat, vitamin C dan B6, serta kalium; memberi energi dan menjaga fungsi otot dan tekanan darah.",
    'sawi': "Sawi mengandung vitamin A, C, dan K, serta kalsium dan folat; mendukung kesehatan tulang, mata, dan imun.",
    'terong': "Terong mengandung serat, vitamin B, antioksidan nasunin, dan mangan; baik untuk jantung dan melindungi sel tubuh.",
    'wortel': "Wortel mengandung Kaya beta-karoten, vitamin K1, serat, dan kalium; bermanfaat untuk mata, kulit, dan sistem kekebalan tubuh.",
    'bittergroud': "Pare mengandung vitamin C, A, folat, dan antioksidan; membantu mengontrol gula darah, meningkatkan imun, dan melancarkan pencernaan.",
    'cucumber': "Mentimun kaya air, vitamin K, dan antioksidan; bermanfaat untuk hidrasi, kesehatan kulit, dan menjaga berat badan.",
    'okra': "Okra mengandung serat, vitamin C, K, folat, dan magnesium; mendukung kesehatan pencernaan, gula darah, dan jantung.",
    'tomato': "Tomat mengandung likopen, vitamin C, K, dan kalium; baik untuk jantung, kulit, dan mencegah penyakit degeneratif.",
}

def preprocess_image(image):
    # Resize ke ukuran input model
    img = image.resize((224, 224))
    
    # Konversi ke array numpy
    img_array = np.array(img)
    
    # PENTING: Gunakan preprocessing yang sama dengan training

    img_array = img_array / 255.0
    
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# UI Streamlit
st.title("Klasifikasi Sayur: Fresh atau Rotten")
st.write("Unggah gambar sayur untuk memprediksi apakah masih segar atau sudah busuk.")

uploaded_file = st.file_uploader("Upload Gambar Sayur", type=["jpg", "jpeg", "png"])



if uploaded_file is not None:
    # Tampilkan gambar
    max_size = 2 * 1024 * 1024  # 2 MB dalam byte

    if uploaded_file.size > max_size:
        st.error("Ukuran gambar terlalu besar. Maksimal 2 MB.")
    else:
        image = Image.open(uploaded_file)
    
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image, width=300)

    
    img = image.resize((224, 224)) 
    img_array = np.array(img) #
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)  

    # Prediksi
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction, axis=1)[0]
    pred_label = class_labels[predicted_index]
    confidence = float(np.max(prediction))

    if "fresh" in pred_label:
        condition = "Segar"
        vegetable = pred_label.replace("fresh", "")
        message = f"**{vegetable.capitalize()} ini masih segar dan layak dikonsumsi.** ✅\n\nKualitas bagus dan aman dimakan."
    else:
        condition = "Rusak"
        vegetable = pred_label.replace("rotten", "")
        message = f"**{vegetable.capitalize()} ini sudah rusak dan sebaiknya tidak dikonsumsi.** ❌"

    # Output hasil
    st.markdown(f"### Prediksi: **{class_labels[predicted_index]}**")
    st.write(f"Tingkat Keyakinan: {confidence:.2%}")
    st.write(f"Tingkat Keyakinan: {predicted_index}")
    st.markdown(message)

    # Informasi gizi
    if vegetable in nutrition_info and condition == "Segar":
        st.markdown("### Kandungan Gizi:")
        st.write(nutrition_info[vegetable])

st.write("Model ini baru bisa mengklasifikasikan 12 jenis sayuran yaitu : Bayam, Brokoli, Buncis, kangkung, kentang, sawi, Terong, Okra, Pare, Timun, Tomat dan wortel")
