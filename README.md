# 🎭 Deepfake Detector

Détection d'images deepfake à l'aide d'un réseau de neurones convolutif (CNN) entraîné avec TensorFlow/Keras, accompagnée d'une application web interactive développée avec Streamlit.

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![Accuracy](https://img.shields.io/badge/Test%20Accuracy-87.23%25-success)

---

## 📌 Présentation

Ce projet permet de classer une image de visage comme **réelle (Real)** ou **générée/manipulée (Fake)**. Il couvre toute la chaîne :

1. Chargement et augmentation des données
2. Construction et entraînement d'un CNN
3. Évaluation (accuracy, matrice de confusion, rapport de classification)
4. Déploiement d'une interface web pour tester le modèle sur ses propres images

## ✨ Fonctionnalités

- Modèle CNN léger entraîné sur **128×128 px**
- Augmentation de données (rotation, décalages, flip horizontal)
- Callbacks `EarlyStopping` et `ModelCheckpoint`
- Application Streamlit avec :
  - upload d'image (JPG, JPEG, PNG)
  - verdict **REAL / FAKE** avec badge de couleur
  - jauge de confiance (Plotly)
  - probabilités détaillées pour chaque classe
  - interprétation automatique du niveau de confiance

## 🗂️ Structure du projet

```
.
├── app.py                      # Application Streamlit
├── DeepFake_tfsimple.ipynb     # Notebook d'entraînement et d'évaluation
├── deepfake_simple_model.h5    # Modèle entraîné (généré par le notebook)
├── confusion_matrix.png        # Matrice de confusion
├── training_history.png        # Courbes d'entraînement
└── README.md
```

## 🧠 Architecture du modèle

| Couche | Détails |
|---|---|
| Conv2D + MaxPooling | 32 filtres, 3×3, ReLU |
| Conv2D + MaxPooling | 64 filtres, 3×3, ReLU |
| Conv2D + MaxPooling | 128 filtres, 3×3, ReLU |
| Flatten | 32 768 neurones |
| Dense | 128 neurones, ReLU |
| Dropout | 0.5 |
| Dense (sortie) | 1 neurone, Sigmoid |

- **Paramètres** : ~4,29 M
- **Optimiseur** : Adam (learning rate = 0.001)
- **Loss** : Binary Crossentropy
- **Batch size** : 32
- **Epochs** : 30 maximum (arrêt anticipé à l'epoch 21)

## 📦 Données

Le dataset est organisé en trois dossiers, chacun contenant deux sous-dossiers `Fake` et `Real` :

```
Dataset/
├── Train/        # 50 000 images
├── Validation/   # 13 992 images
└── Test/         #  2 740 images
```

> Le dataset n'est pas inclus dans ce dépôt en raison de sa taille.

## 📊 Résultats

Évaluation sur le jeu de test (2 740 images, classes équilibrées) :

| Métrique | Valeur |
|---|---|
| **Accuracy** | **87,23 %** |
| Loss | 0,3336 |

| Classe | Precision | Recall | F1-score |
|---|---|---|---|
| Fake | 0,83 | 0,93 | 0,88 |
| Real | 0,92 | 0,82 | 0,86 |

Meilleure accuracy de validation atteinte pendant l'entraînement : **93,17 %**.

### Matrice de confusion

![Matrice de confusion](confusion_matrix.png)

### Courbes d'entraînement

![Courbes d'entraînement](training_history.png)

## 🚀 Installation

```bash
# 1. Cloner le dépôt
git clone https://github.com/Eachr4f/deepfake-detector.git
cd deepfake-detector

# 2. Créer un environnement virtuel (recommandé)
python -m venv venv
source venv/bin/activate      # Windows : venv\Scripts\activate

# 3. Installer les dépendances
pip install tensorflow streamlit numpy opencv-python pillow plotly matplotlib seaborn scikit-learn
```

## ▶️ Utilisation

### Lancer l'application

Assurez-vous que `deepfake_simple_model.h5` se trouve dans le même dossier que `app.py`, puis :

```bash
streamlit run app.py
```

L'application s'ouvre sur `http://localhost:8501`. Téléchargez une image, cliquez sur **Analyser l'image** et consultez le résultat.

### Réentraîner le modèle

1. Placer le dataset dans le dossier `Dataset/` (structure ci-dessus)
2. Ouvrir `DeepFake_tfsimple.ipynb` avec Jupyter
3. Exécuter toutes les cellules : le modèle est sauvegardé sous `deepfake_simple_model.h5`

## ⚠️ Limites

- Le modèle est volontairement simple (CNN à 3 blocs) et à but **éducatif** : il ne remplace pas un outil de vérification professionnel.
- L'écart entre la validation (~93 %) et le test (~87 %) indique une sensibilité au type d'images : les performances peuvent baisser sur des deepfakes issus d'autres générateurs.
- Le modèle détecte plus facilement les images *Fake* (recall 93 %) que *Real* (recall 82 %) : certaines images réelles peuvent être classées à tort comme fausses.

## 🔭 Pistes d'amélioration

- Transfer learning (EfficientNet, ResNet, Xception)
- Détection et recadrage du visage avant classification
- Augmentation de données plus variée et test sur d'autres datasets
- Explicabilité avec Grad-CAM
- Déploiement en ligne (Streamlit Cloud, Hugging Face Spaces)

## 🛠️ Technologies

Python · TensorFlow / Keras · Streamlit · OpenCV · Plotly · scikit-learn · Matplotlib · Seaborn

## 👤 Auteur

**Eachr4f** — [github.com/Eachr4f](https://github.com/Eachr4f)

---

⭐ Si ce projet vous est utile, n'hésitez pas à lui donner une étoile.
