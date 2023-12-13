import cv2
import numpy as np
import joblib
from skimage import color, transform
from sklearn import svm
from skimage.color import rgb2gray

# mon modèle SVM
model_svm = joblib.load("modele_reconnaissance.pkl")

# modèle YOLO
net = cv2.dnn.readNet("yolov3.cfg", "yolov3.weights")
classes_yolo = []
with open("coco.names", "r") as f:
    classes_yolo = [line.strip() for line in f.readlines()]

# traitement de l'image pour SVM
def preprocess_image_svm(img):
    img = rgb2gray(img)
    img = transform.resize(img, (64, 64))
    return img.flatten()

# Capturer le flux vidéo de la webcam
cap = cv2.VideoCapture(0)

# boucle de reconnaissance d'image 
while True:
    ret, frame = cap.read()
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(net.getUnconnectedOutLayersNames())
    
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id_yolo = np.argmax(scores)
            confidence_yolo = scores[class_id_yolo]
            
            if confidence_yolo > 0.50:
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # région d'intérêt
                roi_yolo = frame[y:y+h, x:x+w]
                
                # Afficher le résultat YOLO
                label_yolo = classes_yolo[class_id_yolo]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0,255,0), 2)
                cv2.putText(frame, f"{label_yolo} ({confidence_yolo:.2f})", (x, y + h + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (000, 000, 255), 2)

                # test du roi pour mon modèle
                # if roi_yolo.size != 0:
                #     # traitement de l'image
                #     processed_roi_svm = preprocess_image_svm(roi_yolo)
                #     flattened_roi_svm = processed_roi_svm.flatten().reshape(1, -1)

                #     # prédiction avec mon modèle
                #     prediction_svm = model_svm.predict(flattened_roi_svm)
                #     confidence_svm = model_svm.decision_function(flattened_roi_svm)
                #     # tes j'ai des bug
                #     # print(f"YOLO Prediction: {label_yolo} ({confidence_yolo:.2f})")
                #     print(f"SVM Prediction: {'Chaussure' if prediction_svm > 0.5 else 'Main'} ({confidence_svm[0]:.2f})")
                    
                    # if prediction_svm > 0.5:
                    #     label_svm = "Chaussure"
                    #     color_svm = (0, 255, 0)
                    #     cv2.rectangle(frame, (x, y), (x + w, y + h), color_svm, 2)
                    #     cv2.putText(frame, label_svm, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_svm, 2)
                    # else :
                    #     label_svm = "Main"
                    #     color_svm = (0, 0, 255)
                    #     cv2.rectangle(frame, (x, y), (x + w, y + h), color_svm, 2)
                    #     cv2.putText(frame, label_svm, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_svm, 2)
    
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
