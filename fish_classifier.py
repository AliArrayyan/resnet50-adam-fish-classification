import streamlit as st
from PIL import Image
import os
import gdown
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.models import Model
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from io import BytesIO
from datetime import datetime

# ==============================================================================
# 1. STREAMLIT PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="Klasifikasi Ikan",
    page_icon="🐟",
    layout="wide"
)

# ==============================================================================
# 2. MODEL LOADING
# ==============================================================================
# Google Drive file ID for the model
GDRIVE_FILE_ID = '1KJ5oCskJx28tdsW2IZcnnicSm9GtCVEv'

@st.cache_resource
def load_model_from_gdrive():
    """
    Downloads and loads a Keras model in .h5 format from Google Drive.
    Caches the model so it doesn't reload on every interaction.
    """
    url = f'https://drive.google.com/uc?id={GDRIVE_FILE_ID}'
    output = 'model.h5'
    if not os.path.exists(output):
        with st.spinner("Downloading model from Google Drive... (this may take a moment)"):
            gdown.download(url, output, quiet=False)
    
    # Load the model using TensorFlow/Keras
    model = tf.keras.models.load_model(output, compile=False)
    return model

# ==============================================================================
# 3. IMAGE PROCESSING AND PREDICTION
# ==============================================================================
def transform_image_for_prediction(pil_img):
    """
    Preprocesses a PIL image for prediction with the ResNet50 model.
    """
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    
    # Resize to the target size for the model
    pil_img = pil_img.resize((224, 224))
    img_array = keras_image.img_to_array(pil_img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Use the specific preprocessing function for ResNet50
    processed_img = preprocess_input(img_array)
    
    return processed_img

def predict(pil_img, model):
    """
    Makes a prediction and returns the top 3 results with their confidence scores.
    """
    class_names = [
        'Black Sea Sprat', 'Gilt-Head Bream', 'Horse Mackerel', 
        'Red Mullet', 'Red Sea Bream', 'Sea Bass', 
        'Shrimp', 'Striped Red Mullet', 'Trout'
    ]
    
    processed_img = transform_image_for_prediction(pil_img)
    predictions = model.predict(processed_img)[0] # Get the prediction array
    
    # Get the indices of the top 3 predictions
    top_3_indices = np.argsort(predictions)[-3:][::-1]
    
    # Create a list of tuples (class_name, confidence)
    top_3_results = [(class_names[i], float(predictions[i])) for i in top_3_indices]  # Convert to float
    
    return top_3_results

# ==============================================================================
# 4. UPDATED GRAD-CAM VISUALIZATION FUNCTIONS
# ==============================================================================
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    FIXED Grad-CAM function that handles different model output structures
    """
    # Create gradient model
    grad_model = Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        
        # Handle different model output structures
        if isinstance(preds, list):
            preds = preds[0]  # Take first output if multiple outputs exist
            
        # Get the class channel
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # Compute gradients
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Generate heatmap
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-9)
    
    return heatmap.numpy()

def display_gradcam(pil_img, heatmap, alpha=0.5):
    """
    Superimoses the heatmap on the original image.
    """
    img = keras_image.img_to_array(pil_img)
    
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image from the colored heatmap
    jet_heatmap = keras_image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras_image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras_image.array_to_img(superimposed_img)

    return superimposed_img

# ==============================================================================
# 5. SALIENCY MAP FALLBACK (ALTERNATIVE VISUALIZATION)
# ==============================================================================
def generate_saliency_map(model, img_array, target_size=(224, 224)):
    """
    Creates a saliency map as a fallback visualization
    """
    img_array = tf.cast(img_array, tf.float32)
    
    with tf.GradientTape() as tape:
        tape.watch(img_array)
        predictions = model(img_array)
        
        # Handle model output structure
        if isinstance(predictions, list):
            predictions = predictions[0]
            
        top_pred_index = tf.argmax(predictions[0])
        top_class_channel = predictions[0][top_pred_index]

    gradients = tape.gradient(top_class_channel, img_array)
    saliency = tf.reduce_max(tf.abs(gradients), axis=-1)[0]
    
    # Normalize for visualization
    saliency = (saliency - tf.reduce_min(saliency)) 
    saliency /= (tf.reduce_max(saliency) + 1e-9)
    
    return saliency.numpy()

def display_saliency_map(saliency, original_img):
    """
    Displays the saliency map overlaid on the original image
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    
    # Show original image
    ax.imshow(original_img.resize((224, 224)))
    
    # Overlay saliency heatmap
    ax.imshow(
        np.kron(saliency, np.ones((10, 10))),  # Upsample for better visibility
        cmap='jet',
        alpha=0.5
    )
    ax.axis('off')
    
    return fig

# ==============================================================================
# 6. STREAMLIT UI PAGES - WITH BANNER ADDED
# ==============================================================================
def display_banner():
    """Menampilkan banner di bagian atas halaman"""
    # Ganti URL ini dengan URL gambar banner Anda
    BANNER_URL = "banner.png"
    st.image(
        BANNER_URL,
        use_container_width=True,
    )

def homepage():
    """Halaman utama dengan banner"""
    display_banner()
    st.markdown("## Selamat Datang di Aplikasi Klasifikasi Ikan")
    st.markdown("Aplikasi ini menggunakan model *Deep Learning* untuk mengklasifikasikan 9 jenis ikan dan memberikan visualisasi menggunakan Grad-CAM untuk interpretasi.")
    
    # Tambahkan penjelasan aplikasi
    with st.expander("📚 **Tentang Aplikasi Ini** "):
        st.markdown("""
        **Fitur Utama:**
        - Klasifikasi 9 jenis ikan laut
        - Visualisasi daerah penting dengan Grad-CAM
        - Tampilkan top 3 prediksi
        - Riwayat prediksi
        - Antarmuka pengguna yang mudah digunakan
        
        **Cara Penggunaan:**
        1. Pindah ke halaman **Prediction** di sidebar
        2. Unggah gambar ikan
        3. Lihat hasil klasifikasi dan visualisasi
        4. Tinjau riwayat di halaman **History**
        """)
    
    # Tambahkan contoh gambar
    st.markdown("### Contoh Jenis Ikan yang Dikenali")
    st.write("Berikut adalah 9 jenis ikan yang dapat dikenali oleh sistem:")
    
    # Baris 1
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("blackseasprat.png", caption="Black Sea Sprat")
    with col2:
        st.image("giltheadbream.jpeg", caption="Gilt-Head Bream")
    with col3:
        st.image("horsemackerel.png", caption="Horse Mackerel")
    
    # Baris 2
    col4, col5, col6 = st.columns(3)
    with col4:
        st.image("redmullet.png", caption="Red Mullet")
    with col5:
        st.image("redseabream.jpeg", caption="Red Sea Bream")
    with col6:
        st.image("seabass.jpeg", caption="Sea Bass")
    
    # Baris 3
    col7, col8, col9 = st.columns(3)
    with col7:
        st.image("shrimp.png", caption="Shrimp")
    with col8:
        st.image("stripedredmullet.png", caption="Striped Red Mullet")
    with col9:
        st.image("trout.png", caption="Trout")
    
     # TOMBOL CALL-TO-ACTION
    st.markdown("---")
    st.info("Klik **Prediction** di sidebar untuk mengunggah gambar dan melihat hasil prediksi.")

# ==============================================================================
# 7. HISTORY PAGE IMPLEMENTATION
# ==============================================================================
def history_page():
    """Professional history page with prediction records"""
    display_banner()
    st.markdown("## 📜 Prediction History")
    st.write("Review your previous classification results and visualizations")
    
    # Initialize history if not exists
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    if not st.session_state.history:
        st.info("No prediction history yet. Make predictions on the Prediction page to see them appear here.")
        return
    
    # Filter options
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("Prediction Records")
    with col2:
        if st.button("🧹 Clear History", use_container_width=True, type="primary"):
            st.session_state.history = []
            st.experimental_rerun()
    
    # Display history in reverse chronological order
    for i, record in enumerate(reversed(st.session_state.history)):
        # Create card-like expander
        with st.expander(f"**{record['timestamp']}** - {record['image_name']}", expanded=i==0):
            col_img, col_data = st.columns([0.3, 0.7])
            
            with col_img:
                st.image(record['thumbnail'], caption="Uploaded Image", use_container_width=True)
            
            with col_data:
                # Prediction details
                st.subheader(f"Prediction: **{record['prediction']}**")
                
                # FIX: Convert float32 to standard Python float
                confidence = float(record['confidence'])
                st.progress(confidence)
                st.caption(f"Confidence: {confidence*100:.2f}%")
                
                # Top predictions table
                st.markdown("**Top Predictions:**")
                df = pd.DataFrame(record['top_predictions'], columns=['Class', 'Confidence'])
                df['Confidence'] = df['Confidence'].apply(lambda x: f"{x*100:.2f}%")
                st.dataframe(
                    df,
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "Class": st.column_config.TextColumn(width="medium"),
                        "Confidence": st.column_config.ProgressColumn(
                            "Confidence",
                            format="%f",
                            min_value=0,
                            max_value=1
                        )
                    }
                )
                
                # Visualization
                if 'visualization' in record and record['visualization']:
                    st.markdown("**Model Visualization:**")
                    st.image(record['visualization'], use_container_width=True)

# ==============================================================================
# 8. PREDICTION PAGE WITH HISTORY SAVING
# ==============================================================================
def prediction_page():
    """Halaman prediksi dengan banner"""
    display_banner()
    st.markdown("### Unggah Gambar Ikan untuk Klasifikasi")
    st.write("Unggah gambar ikan untuk diklasifikasikan oleh model.")
    
    model = load_model_from_gdrive()
    
    uploaded_file = st.file_uploader("Pilih sebuah gambar...", type=["jpg", "jpeg", "png"])
    
    visualization_created = False
    superimposed_image = None
    
    if uploaded_file is not None:
        pil_image = Image.open(uploaded_file).convert("RGB")
        
        col1, col2 = st.columns([0.9, 1.1]) # Adjust column widths

        with col1:
            st.image(pil_image, caption="Gambar yang Diunggah", use_container_width=True)

        with col2:
            with st.spinner('Mengklasifikasi...'):
                top_3_results = predict(pil_image, model)
            
            predicted_class, confidence = top_3_results[0]

            st.write(f"### Hasil Prediksi: **{predicted_class}**")
            st.write(f"Tingkat Confidence: **{confidence*100:.2f}%**")
            
            st.write("---")
            st.write("#### Top 3 Prediksi")
            df_results = pd.DataFrame(top_3_results, columns=['Kelas Ikan', 'Confidence'])
            df_results['Confidence'] *= 100
            st.dataframe(df_results.style.format({'Confidence': '{:.2f}%'}), use_container_width=True)
            st.bar_chart(df_results.set_index('Kelas Ikan'))

        # --- Visualization Section ---
        st.write("---")
        st.markdown("### Visualisasi Model")
        st.write("Heatmap menunjukkan area pada gambar yang paling memengaruhi keputusan model.")
        
        # Prepare image for visualization
        img_array_for_vis = np.expand_dims(
            keras_image.img_to_array(pil_image.resize((224, 224))), 
            axis=0
        )
        
        # Try Grad-CAM with different layers
        layer_candidates = [
            "conv5_block3_out",        # Standard ResNet50
            "conv5_block3_3_conv",     # Alternative ResNet name
            "conv2d_20",               # Common in sequential models
            "conv2d_3",                # Common in sequential models
            "top_conv",                # Custom models
            "block5_conv3",            # VGG-style networks
            "activation_49"            # DenseNet-style
        ]
        
        with st.spinner("Mencoba membuat visualisasi..."):
            # First try Grad-CAM with different layers
            for layer_name in layer_candidates:
                try:
                    heatmap = make_gradcam_heatmap(
                        img_array_for_vis, 
                        model, 
                        layer_name
                    )
                    superimposed_image = display_gradcam(pil_image, heatmap)
                    st.image(
                        superimposed_image, 
                        caption=f'Grad-CAM Heatmap (Layer: {layer_name})', 
                        use_container_width=True
                    )
                    st.success(f"Berhasil membuat Grad-CAM dengan layer: {layer_name}")
                    visualization_created = True
                    break
                except:
                    continue
            
            # If Grad-CAM failed, try saliency map
            if not visualization_created:
                try:
                    st.warning("Grad-CAM tidak berhasil. Mencoba alternatif saliency map...")
                    saliency = generate_saliency_map(model, img_array_for_vis)
                    fig = display_saliency_map(saliency, pil_image)
                    st.pyplot(fig)
                    st.info("Saliency Map: Area berwarna menunjukkan pengaruh terbesar pada keputusan model")
                    visualization_created = True
                    # Convert fig to PIL for history
                    buf = BytesIO()
                    fig.savefig(buf, format='png')
                    buf.seek(0)
                    superimposed_image = Image.open(buf)
                except Exception as e:
                    st.error(f"Gagal membuat visualisasi: {e}")
                    st.warning("""
                    **Pemecahan masalah:**
                    1. Pastikan model memiliki layer konvolusi
                    2. Coba model berbeda jika masalah berlanjut
                    3. Laporkan masalah ke pengembang
                    """)
            
            if visualization_created:
                st.info("""
                **Interpretasi:**
                - Area **merah/panas**: Sangat memengaruhi prediksi
                - Area **biru/dingin**: Kurang memengaruhi prediksi
                """)
                
        # Create thumbnail for history
        thumbnail = pil_image.copy()
        thumbnail.thumbnail((200, 200))
        
        # Save visualization to bytes
        vis_bytes = None
        if visualization_created and superimposed_image:
            buf = BytesIO()
            if isinstance(superimposed_image, Image.Image):
                superimposed_image.save(buf, format='PNG')
            else:
                plt.savefig(buf, format='png')
            vis_bytes = buf.getvalue()
        
        # Save to history (with float conversion)
        history_record = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'image_name': uploaded_file.name,
            'thumbnail': thumbnail,
            'prediction': top_3_results[0][0],
            'confidence': float(top_3_results[0][1]),  # Convert to float
            'top_predictions': top_3_results,
            'visualization': vis_bytes
        }
        
        if 'history' not in st.session_state:
            st.session_state.history = []
        
        st.session_state.history.append(history_record)

# ==============================================================================
# 9. MAIN APP LOGIC
# ==============================================================================
def main():
    st.sidebar.image("classifish-removebg-preview.png", use_container_width=True)
    st.sidebar.title("Navigasi")
    
    # Use a dictionary for page management
    pages = {
        "Home": homepage,
        "Prediction": prediction_page,
        "History": history_page,
    }
    
    selection = st.sidebar.radio("Pindah ke Halaman", list(pages.keys()))
    
    # Run the selected page function
    pages[selection]()
    
    # Tambahkan footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Tentang Aplikasi")
    st.sidebar.info("""
    Aplikasi Klasifikasi Ikan
    - Dibuat dengan Streamlit dan TensorFlow
    - Model ResNet50 yang telah dilatih ulang
    - [GitHub Repository](https://github.com/AliArrayyan/resnet50-adam-fish-classification.git)
    - [Google Colaboratory](https://colab.research.google.com/drive/1cJTVAzmfHNJz9YVcx968i3WJ0btykRMq?usp=sharing)
    """)

if __name__ == "__main__":
    main()