import os
import joblib
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from skimage import io, color, transform
import numpy as np
import matplotlib.pyplot as plt

# Fonction pour charger et prétraiter mes images
def load_and_preprocess_images(folder_path, label):
    images = []
    labels = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg"):
            img = io.imread(os.path.join(folder_path, filename))
            img = color.rgb2gray(img) 
            img = transform.resize(img, (64, 64)) 
            images.append(img.flatten())
            labels.append(label)
    return np.array(images), np.array(labels)

# Charger et prétraiter les images de chaussures
shoe_folder = "/Users/nawfoelardjoune/M1/rob/projet/reco/datasets/data/chaussures"
shoe_images, shoe_labels = load_and_preprocess_images(shoe_folder, 0)

# Charger et prétraiter les images de mains
hand_folder = "/Users/nawfoelardjoune/M1/rob/projet/reco/datasets/data/Hands"
hand_images, hand_labels = load_and_preprocess_images(hand_folder, 1)

# Concaténer les données de chaussures et de mains
X = np.concatenate((shoe_images, hand_images), axis=0)
y = np.concatenate((shoe_labels, hand_labels), axis=0)

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
print(X.shape)

# créer le modèle SVM
model = make_pipeline(StandardScaler(), svm.SVC(kernel='linear'))
model.fit(X_train, y_train)

# Enregistrer mon modèle
joblib.dump(model, "modele_reconnaissance.pkl")

# test
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Précision du modèle : {accuracy * 100:.2f}%")

# matrice de confusion
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()