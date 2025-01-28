import tkinter as tk
from tkinter import filedialog, messagebox
import xml.etree.ElementTree as ET
from joblib import load
from tkinter import *
from nltk.corpus import stopwords
import string
import nltk
ruta_archivo = ""  # Variable global para almacenar la ruta del archivo seleccionado
texto = ""

svm_gender = load('svm_gender_1.joblib')
# Es el modelo entrenado de navie bayes 
NB_country = load('svm_country_1.joblib') 
vectorizer = load('vectorizer_1.joblib')

def preprocess_text(text,stop_words):
    # Convierte a minúsculas
    text = text.lower()
    # Elimina signos de puntuación
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Elimina stopwords
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text


def procesar_archivo():
    Documento = ""
    nltk.download('stopwords')
    stop_words = set(stopwords.words('spanish'))
    Documento = texto.get(1.0, "end-1c")
    x_text = preprocess_text(Documento,stop_words)
    X = vectorizer.transform([x_text])
    genero = svm_gender.predict(X)[0]
    # pais = svm_country.predict(X)[0]
    pais = NB_country.predict(X)[0]
    lbl.config(text="Predicción de género: " + genero)
    lbl2.config(text="Predicción de país: " + pais)

# Crear la ventana principal
ventana = tk.Tk()
ventana.title("Perfilado de Autor")
ventana.geometry("800x600")  # Tamaño inicial de la ventana

# Etiqueta de título
titulo_label = tk.Label(ventana, text="Perfilado de Autor", font=("Arial", 24, "bold"), pady=20)
titulo_label.pack()

# Frame para contener el Text widget y la Scrollbar
frame_texto = tk.Frame(ventana)
frame_texto.pack(pady=10)

# Scrollbar
scrollbar = tk.Scrollbar(frame_texto)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

# Widget Text
texto = tk.Text(frame_texto, yscrollcommand=scrollbar.set, font=("Arial", 12), width=80, height=20)
texto.pack(side=tk.LEFT)

# Configurar la scrollbar
scrollbar.config(command=texto.yview)

# Botón para procesar el archivo
btn_procesar = tk.Button(ventana, text="Procesar", font=("Arial", 14), command=procesar_archivo, bg="#6C3483", fg="white")
btn_procesar.pack(pady=10)

# Labels para mostrar resultados
lbl = tk.Label(ventana, text = "") 
lbl.pack() 
lbl2= tk.Label(ventana, text = "") 
lbl2.pack() 

# Ejecutar el bucle principal de la ventana
ventana.mainloop()
