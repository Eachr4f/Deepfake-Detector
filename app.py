import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import plotly.graph_objects as go

# Configuration de la page
st.set_page_config(
    page_title="Deepfake Detector",
    page_icon="🎭",
    layout="wide"
)

# Styles CSS personnalisés
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #FF4B4B;
        text-align: center;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .real-badge {
        background-color: #28a745;
        color: white;
        padding: 10px 20px;
        border-radius: 20px;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .fake-badge {
        background-color: #dc3545;
        color: white;
        padding: 10px 20px;
        border-radius: 20px;
        font-size: 1.5rem;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Constantes
IMG_HEIGHT = 128
IMG_WIDTH = 128
MODEL_PATH = "deepfake_simple_model.h5"

@st.cache_resource
def load_model():
    """Charge le modèle entraîné"""
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du modèle: {e}")
        st.info("Assurez-vous que le fichier 'deepfake_simple_model.h5' est dans le même dossier.")
        return None

def preprocess_image(image):
    """Prétraite l'image pour la prédiction"""
    # Convertir PIL Image en array
    img_array = np.array(image)
    
    # Redimensionner
    img_resized = cv2.resize(img_array, (IMG_HEIGHT, IMG_WIDTH))
    
    # Normaliser
    img_normalized = img_resized / 255.0
    
    # Ajouter dimension batch
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    return img_batch, img_resized

def create_gauge_chart(confidence, prediction):
    """Crée un graphique jauge pour la confiance"""
    color = "#28a745" if prediction == "REAL" else "#dc3545"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Niveau de Confiance", 'font': {'size': 24}},
        number={'suffix': "%", 'font': {'size': 40}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': '#ffebee'},
                {'range': [50, 75], 'color': '#fff9c4'},
                {'range': [75, 100], 'color': '#e8f5e9'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig

def main():
    # En-tête
    st.markdown('<p class="main-header">🎭 Deepfake Detector</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Charger le modèle
    model = load_model()
    
    if model is None:
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png", width=100)
        st.title("À propos")
        st.info("""
        **Deepfake Detector** utilise un réseau de neurones convolutif (CNN) 
        pour détecter les images deepfakes.
        
        **Comment l'utiliser:**
        1. Téléchargez une image
        2. Attendez l'analyse
        3. Consultez le résultat
        """)
        
        st.markdown("---")
        st.subheader("📊 Statistiques du modèle")
        st.metric("Architecture", "CNN 3 couches")
        st.metric("Paramètres", "~1M")
        st.metric("Input Size", "128x128")
    
    # Zone principale
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Télécharger une image")
        uploaded_file = st.file_uploader(
            "Choisissez une image...",
            type=['jpg', 'jpeg', 'png'],
            help="Formats acceptés: JPG, JPEG, PNG"
        )
        
        if uploaded_file is not None:
            # Afficher l'image
            image = Image.open(uploaded_file)
            st.image(image, caption="Image téléchargée", use_container_width=True)
            
            # Bouton de prédiction
            if st.button("🔍 Analyser l'image", type="primary", use_container_width=True):
                with st.spinner("Analyse en cours..."):
                    # Prétraiter
                    img_batch, img_display = preprocess_image(image)
                    
                    # Prédire
                    prediction_prob = model.predict(img_batch, verbose=0)[0][0]
                    
                    # Déterminer la classe
                    if prediction_prob > 0.5:
                        prediction = "REAL"
                        confidence = prediction_prob
                        badge_class = "real-badge"
                    else:
                        prediction = "FAKE"
                        confidence = 1 - prediction_prob
                        badge_class = "fake-badge"
                    
                    # Afficher les résultats dans col2
                    with col2:
                        st.subheader("📊 Résultat de l'analyse")
                        
                        # Badge de prédiction
                        st.markdown(f'<div class="{badge_class}">{prediction}</div>', 
                                  unsafe_allow_html=True)
                        st.markdown("<br>", unsafe_allow_html=True)
                        
                        # Graphique jauge
                        fig = create_gauge_chart(confidence, prediction)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Probabilités détaillées
                        st.markdown("### 📈 Probabilités")
                        prob_real = prediction_prob * 100
                        prob_fake = (1 - prediction_prob) * 100
                        
                        col_prob1, col_prob2 = st.columns(2)
                        with col_prob1:
                            st.metric("✅ Real", f"{prob_real:.2f}%")
                        with col_prob2:
                            st.metric("❌ Fake", f"{prob_fake:.2f}%")
                        
                        # Barre de progression
                        st.markdown("### 📊 Distribution")
                        st.progress(prediction_prob)
                        
                        # Interprétation
                        st.markdown("### 💡 Interprétation")
                        if confidence > 0.9:
                            st.success(f"✅ **Haute confiance**: L'image est très probablement **{prediction}**")
                        elif confidence > 0.7:
                            st.info(f"ℹ️ **Confiance modérée**: L'image semble être **{prediction}**")
                        else:
                            st.warning(f"⚠️ **Faible confiance**: Résultat incertain, l'image pourrait être **{prediction}**")
    
    with col2:
        if uploaded_file is None:
            st.info("👈 Téléchargez une image pour commencer l'analyse")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>Développé avec ❤️ en utilisant TensorFlow & Streamlit</p>
        <p>⚠️ Cet outil est à titre éducatif. Les résultats peuvent ne pas être parfaits.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()