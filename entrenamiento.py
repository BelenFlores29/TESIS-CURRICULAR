import xml.etree.ElementTree as ET
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.metrics import accuracy_score
from joblib import dump
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB 
from nltk.corpus import stopwords
import string
import nltk




vectorizer = CountVectorizer()
genero_train = []
pais_train = []
lista_train = []
SVM_gender = svm.SVC(kernel='linear')  # Modelo SVM para género
# SVM_country = svm.SVC(kernel='linear')  # Modelo SVM para país
NB_pais= MultinomialNB() # Modelo NB para país

def preprocess_text(text,stop_words):
    # Convierte a minúsculas
    text = text.lower()
    # Elimina signos de puntuación
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Elimina stopwords
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text


def Data_train():
        ruta_train = "C:\\Users\\Belen\\Desktop\\Training\\pan17-author-profiling-training-dataset-2017-03-10\\pan17-author-profiling-training-dataset-2017-03-10\\es\\"    
        archivo_train = open(ruta_train + "truth.txt",'r')
        
        for item in archivo_train.readlines():
            datos_train = item.split(":::")
            genero_train.append(datos_train[1]) 
            pais_train.append(datos_train[2])
            aux_train = ruta_train + datos_train[0] + ".xml"
            root_train = ET.parse(aux_train)
            cdata_element_train = root_train.find('documents')
            ln=""
            for item in cdata_element_train:
                ln += " " + item.text
            lista_train.append(ln)
        return (genero_train, pais_train, lista_train)

# Extracción de características
Y_train_gender,Y_train_country,X_train =Data_train()
nltk.download('stopwords')
stop_words = set(stopwords.words('spanish'))

# Aplica el preprocesamiento a los datos de entrenamiento y prueba
X_train = [preprocess_text(doc,stop_words) for doc in X_train]
X_train_vec = vectorizer.fit_transform(X_train)
# Entrenamiento del modelo SVM para género
SVM_gender.fit(X_train_vec, genero_train)

    # Entrenamiento del modelo SVM para país
    # self.SVM_country.fit(X_train, self.pais_train)

NB_pais.fit(X_train_vec,Y_train_country)
        
dump(SVM_gender, 'svm_gender_1.joblib')
    # dump(self.SVM_country, 'svm_country.joblib')
dump(vectorizer, 'vectorizer_1.joblib')
dump(NB_pais, 'svm_country_1.joblib')


