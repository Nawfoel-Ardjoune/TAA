# Robot de Sauvetage avec Apprentissage Artificiel

## Auteur
**Nom:** Nawfoel Ardjoune  
**Niveau:** Master 1 Informatique  
**Projet réalisé pour:** Technique d’Apprentissage Artificiel  

## Contexte
Ce projet s'inscrit dans le cadre du cours de Technique d’Apprentissage Artificiel. Il vise à développer un robot de détection autonome pour sauver les personnes en détresse, appelé SaveIGO. Ce rapport se focalisera principalement sur la partie liée à l'apprentissage artificiel.

## Objectifs
Les principaux objectifs du projet étaient les suivants :
- Effectuer des recherches sur les technologies à utiliser.
- Collecter et traiter des données d'apprentissage.
- Créer un modèle de détection capable de reconnaître des personnes en détresse.
- Tester et évaluer le modèle.
- Intégrer la détection en temps réel dans un environnement vidéo et/ou audio.
- Permettre une utilisation à distance ou un déplacement autonome du robot.

## Données d'Apprentissage
Les données d'apprentissage ont été collectées sur Kaggle, comprenant plus de 11 000 images de chaussures et de mains. Les données ont été prétraitées pour assurer un équilibre adéquat et éviter tout biais.

## Fabrication du Modèle
La fabrication du modèle a été réalisée en utilisant des bibliothèques telles que Sklearn, numpy, et joblib. Un modèle SVM a été choisi en raison de sa pertinence pour les données d'images. Les données ont été normalisées avant d'être utilisées pour l'entraînement du modèle.

## Détection d'Objet en Temps Réel
La détection en temps réel a été effectuée en utilisant la webcam de l'ordinateur, OpenCV, et le modèle YOLO. Les zones d'intérêt détectées par YOLO ont été utilisées pour appliquer le modèle d'apprentissage, permettant de classifier les objets ou les personnes.

## Difficultés Rencontrées
- La difficulté à trouver des données appropriées.
- Le traitement d'images pour les adapter au format requis par les fonctions et les modèles.
- La durée de traitement des images pendant la fabrication du modèle.
- Des problèmes de précision dans le flux vidéo en temps réel.

## Utilisation
Pour utiliser le programme il suffit de lancer le fichier rt_reco.py

## Liste des fichiers
Reco:
    reco.py
    rt_reco.py
    coco.names
    confusion.png
    test_accuracy.png
    yolov3.cfg
    yolov3.weights
    datasets:
        -chaussures.csv
        -HandInfo.csv
        -main_datasets.zip
        -chaussures_datasets.zip
        -data:
            -chaussures => dossier 875 images de chaussures
            -Hands => dossier de 875 images de mains 
            
## Sources
- [Dataset des mains](https://www.kaggle.com/datasets/shyambhu/hands-and-palm-images-dataset)
- [Dataset des chaussures](https://www.kaggle.com/datasets/die9origephit/nike-adidas-and-converse-imaged)
- [OpenCV](https://opencv.org/)
- [YOLO](https://github.com/ultralytics/yolov3)
- Le lien de mon GITHUB : https://github.com/Nawfoel-Ardjoune
**Note:** Le projet n'est pas encore complètement terminé, des bugs subsistent, la detection d'appel à l'aide n'a pas encore été faite pour des raison de matériel, et des améliorations sont envisagées.
