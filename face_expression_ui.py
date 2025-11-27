import tensorflow as tf
import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk

# -------------------------------
# Load the trained model
# -------------------------------
model = tf.keras.models.load_model("face_cnn_model.h5")

# Define emotion labels
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# -------------------------------
# GUI Setup
# -------------------------------
root = tk.Tk()
root.title("Facial Expression Predictor")
root.geometry("500x500")
root.configure(bg="#2c3e50")

result_label = Label(root, text="Upload an image to predict emotion", font=("Arial", 14), bg="#2c3e50", fg="white")
result_label.pack(pady=20)

img_label = Label(root)
img_label.pack()

# -------------------------------
# Function to predict emotion
# -------------------------------
def predict_emotion(image_path):
    global img_label

    # Read image
    img = cv2.imread(image_path)
    if img is None:
        result_label.config(text="❌ Could not load image!", fg="red")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) > 0:
        x, y, w, h = faces[0]  # Take first detected face
        face_img = gray[y:y+h, x:x+w]
    else:
        # Fallback: Assume image is already a cropped face (like FER2013)
        face_img = gray
        x, y, w, h = 0, 0, gray.shape[1], gray.shape[0]


    # Preprocess for model
    face_resized = cv2.resize(face_img, (48, 48))
    face_array = face_resized / 255.0
    face_array = np.expand_dims(face_array, axis=-1)
    face_array = np.expand_dims(face_array, axis=0)

    predictions = model.predict(face_array)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions)

    result_label.config(
        text=f"Prediction: {emotion_labels[predicted_class].upper()} ({confidence*100:.2f}%)",
        fg="lightgreen"
    )

    # Show image on UI
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.rectangle(img_rgb, (x, y), (x+w, y+h), (0, 255, 0), 2)
    img_pil = Image.fromarray(img_rgb)
    img_pil = img_pil.resize((300, 300))
    img_tk = ImageTk.PhotoImage(img_pil)
    img_label.configure(image=img_tk)
    img_label.image = img_tk


# -------------------------------
# Browse button function
# -------------------------------
def browse_file():
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )
    if file_path:
        predict_emotion(file_path)


browse_btn = Button(root, text="Browse Image", command=browse_file, font=("Arial", 12), bg="#27ae60", fg="white")
browse_btn.pack(pady=10)

root.mainloop()
