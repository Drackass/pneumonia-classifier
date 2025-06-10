import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, UnidentifiedImageError
import numpy as np
import matplotlib.pyplot as plt
import requests
from io import BytesIO
import os
import time

# Configuration de l'application
st.set_page_config(
    page_title="Détection de Pneumonie - Version Médicale",
    page_icon="🩺",
    layout="wide"
)

# Chargement du modèle avec vérification
@st.cache_resource
def load_model():
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        st.info(f"Chargement du modèle sur {device}...")
        
        # Création du modèle
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)
        
        # Vérification du fichier modèle
        model_path = 'pneumonia_model.pth'
        if not os.path.exists(model_path):
            st.error(f"Fichier modèle introuvable: {model_path}")
            st.stop()
            
        # Chargement des poids
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        st.success("Modèle chargé avec succès!")
        return model, device
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle: {str(e)}")
        st.stop()

model, device = load_model()

# Transformation des images (identique à l'entraînement)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class_names = ['NORMAL', 'PNEUMONIA']

# Fonction de prédiction améliorée
def predict_image(image):
    try:
        # Vérification de l'image
        if image is None:
            raise ValueError("Aucune image fournie")
            
        # Conversion en RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Transformation et ajout dimension batch
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Prédiction avec mesure du temps
        start_time = time.time()
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            _, predicted = torch.max(outputs, 1)
        inference_time = time.time() - start_time
        
        return {
            'class': class_names[predicted.item()],
            'probability': probabilities[predicted.item()].item(),
            'probabilities': probabilities.tolist(),
            'predicted': predicted.item(),
            'inference_time': inference_time
        }
    except Exception as e:
        st.error(f"Erreur lors de la prédiction: {str(e)}")
        return None

# Interface Streamlit améliorée
st.title("🩺 Détection de Pneumonie par Rayons X")
st.markdown("""
Cette application utilise un modèle d'apprentissage profond pour détecter les signes de pneumonie sur des radiographies pulmonaires.
""")

# Section de chargement d'image
st.sidebar.header("Chargement d'image")
uploaded_file = st.sidebar.file_uploader(
    "Téléchargez une radiographie pulmonaire",
    type=["jpg", "jpeg", "png"],
    help="Formats supportés: JPG, JPEG, PNG"
)

# Exemples d'images de test (chemins locaux ou URLs fiables)
example_images = {
    "Normal (Exemple 1)": "data/test/NORMAL/IM-0027-0001.jpeg",
    "Normal (Exemple 2)": "data/test/NORMAL/NORMAL2-IM-0131-0001.jpeg",
    "Pneumonie (Exemple 1)": "data/test/PNEUMONIA/person101_bacteria_485.jpeg",
    "Pneumonie (Exemple 2)": "data/test/PNEUMONIA/person22_virus_55.jpeg"
}

# Sélection des exemples
st.sidebar.markdown("### Exemples de radiographies")
example_choice = st.sidebar.selectbox("Choisir un exemple:", ["Sélectionner..."] + list(example_images.keys()))

# Affichage principal
col1, col2 = st.columns([1, 1])

if uploaded_file is not None or (example_choice and example_choice != "Sélectionner..."):
    try:
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            image_source = "Fichier uploadé"
        else:
            example_path = example_images[example_choice]
            if example_path.startswith(('http://', 'https://')):
                response = requests.get(example_path)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content))
                image_source = f"Exemple: {example_choice} (URL)"
            else:
                if not os.path.exists(example_path):
                    st.error(f"Fichier exemple introuvable: {example_path}")
                    st.stop()
                image = Image.open(example_path)
                image_source = f"Exemple: {example_choice} (local)"
        
        # Affichage de l'image originale
        with col1:
            st.subheader("Radiographie analysée")
            st.image(image, use_container_width=True, caption=image_source)
            
            # Informations sur l'image
            st.markdown(f"""
            **Détails techniques:**
            - Format: {image.format if image.format else 'Inconnu'}
            - Dimensions: {image.size[0]}x{image.size[1]} pixels
            - Mode couleur: {image.mode}
            """)
        
        # Prédiction
        with st.spinner("Analyse en cours..."):
            result = predict_image(image)
        
        if result:
            with col2:
                st.subheader("Résultats de l'analyse")
                
                # Affichage du résultat avec indicateurs visuels
                if result['class'] == 'PNEUMONIA':
                    st.error(f"## ⚠️ Détection de pneumonie (confiance: {result['probability']*100:.1f}%)")
                    st.warning("""
                    **Recommandations médicales:**  
                    - Consultez un pneumologue ou un médecin généraliste rapidement  
                    - Présentez ces résultats à votre professionnel de santé  
                    - Surveillez les symptômes (fièvre, toux, difficultés respiratoires)
                    """)
                else:
                    st.success(f"## ✅ Radiographie normale (confiance: {result['probability']*100:.1f}%)")
                    st.info("""
                    **Informations:**  
                    - Aucun signe évident de pneumonie détecté  
                    - Consultez tout de même un médecin si vous présentez des symptômes  
                    - Un résultat normal n'exclut pas complètement une pathologie
                    """)
                
                # Graphique des probabilités
                st.subheader("Probabilités de diagnostic")
                fig, ax = plt.subplots(figsize=(8, 3))
                bars = ax.barh(class_names, result['probabilities'], color=['#4CAF50', '#F44336'])
                ax.set_xlim(0, 1)
                ax.set_xlabel('Probabilité')
                ax.bar_label(bars, fmt='%.3f', padding=3)
                st.pyplot(fig)
                
                # Métriques de performance
                st.markdown(f"""
                **Détails techniques:**
                - Temps d'inférence: {result['inference_time']:.3f} secondes
                - Matériel utilisé: {'GPU' if 'cuda' in str(device) else 'CPU'}
                - Classe prédite: {result['predicted']} ({result['class']})
                """)
    
    except UnidentifiedImageError:
        st.error("Erreur: Format d'image non reconnu. Veuillez utiliser une image valide (JPEG, PNG).")
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur de chargement de l'exemple: {str(e)}")
    except Exception as e:
        st.error(f"Erreur inattendue: {str(e)}")

# Section d'information
st.markdown("---")
st.subheader("Informations cliniques")

with st.expander("À propos de ce modèle diagnostique"):
    st.markdown("""
    **Caractéristiques techniques:**
    - Architecture: ResNet18 (transfer learning)
    - Base d'entraînement: ImageNet + jeu de données radiologiques
    - Classes: NORMAL (sans pneumonie) vs PNEUMONIA
    - Précision validée: ~92% sur données de test
    
    **Limitations importantes:**
    1. Ne remplace pas un diagnostic médical professionnel
    2. Performances variables selon la qualité de l'image
    3. Détection principalement des pneumonies bactériennes et virales typiques
    4. Peut avoir des difficultés avec les cas atypiques ou subtils
    """)

with st.expander("Contexte clinique de la pneumonie"):
    st.markdown("""
    **La pneumonie est une infection pulmonaire grave:**
    - Cause principale de mortalité infectieuse chez les enfants
    - Symptômes: fièvre, toux, difficultés respiratoires
    - Diagnostic radiologique essentiel mais complexe
    
    **Importance du diagnostic précoce:**
    - Réduction de la mortalité grâce à un traitement rapide
    - Prévention des complications (abcès, septicémie)
    - Différenciation entre causes bactériennes et virales
    """)

# Footer
st.markdown("---")
st.markdown("""
**Avertissement médical:**  
Cette application fournit une analyse préliminaire automatisée à titre informatif uniquement.  
Elle ne constitue pas un diagnostic médical et ne doit pas remplacer l'évaluation d'un professionnel de santé qualifié.  
En cas de symptômes respiratoires ou de doute, consultez immédiatement un médecin.
""")

# Journal de débogage (caché par défaut)
with st.expander("Journal technique (débogage)"):
    st.code(f"""
    Device: {device}
    Modèle: ResNet18
    Poids chargés: pneumonia_model.pth
    Transformations appliquées: 
        Resize(224x224)
        ToTensor()
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    """)