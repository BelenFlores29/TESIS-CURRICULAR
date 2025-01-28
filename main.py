import xml.etree.ElementTree as ET
import glob 
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB 
from sklearn import tree 
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from nltk.corpus import stopwords
import string
import nltk

genero_test=[]
pais_test=[]
lista_test=[]
genero_train=[]
pais_train=[]
lista_train=[]

def Data_train():
    ruta_train="C:\\Users\\Belen\\Desktop\\Training\\pan17-author-profiling-training-dataset-2017-03-10\\pan17-author-profiling-training-dataset-2017-03-10\\es\\"    
    archivo_train=open("C:\\Users\\Belen\\Desktop\\Training\\pan17-author-profiling-training-dataset-2017-03-10\\pan17-author-profiling-training-dataset-2017-03-10\\es\\truth.txt",'r')
    
    for item in archivo_train.readlines(): #itera sobre todas las lineas del archivo truth
        datos_train=item.split(":::") #divide la cadena item en partes utilizando ":::" como el separador
        genero_train.append(datos_train[1]) 
        pais_train.append(datos_train[2])
        aux_train=ruta_train+datos_train[0]+".xml"
        root_train = ET.parse(aux_train)
        #Busca la etiqueta <Documents>
        cdata_element_train = root_train.find('documents')
        #Texto contenido en la etiqueta <Documents>
        cdata_content_train = cdata_element_train.text
        ln=""
        for item in cdata_element_train: #itera sobre cada elemeto del contenido
            ln+=" "+item.text
        lista_train.append(ln)
    return (genero_train,pais_train,lista_train)


def Data_test():
    ruta_test="C:\\Users\\Belen\\Desktop\\Test\\pan17-author-profiling-test-dataset-2017-03-16\\es\\"
    archivo_test=open("C:\\Users\\Belen\\Desktop\\Test\\pan17-author-profiling-test-dataset-2017-03-16\\es\\truth.txt",'r')
    for item in archivo_test.readlines():
        datos_test=item.split(":::")
        genero_test.append(datos_test[1])
        pais_test.append(datos_test[2])
        aux=ruta_test+datos_test[0]+".xml"
        root = ET.parse(aux)
        cdata_element_test = root.find('documents')
        cdata_content_test = cdata_element_test.text
        ln=""
        for item in cdata_element_test:
            ln+=" "+item.text
        lista_test.append(ln)
    return (genero_test,pais_test,lista_test)


Y_train_gender,Y_train_country,X_train =Data_train()
Y_test_gender,Y_test_country,X_test =Data_test()

nltk.download('stopwords')

stop_words = set(stopwords.words('spanish'))

def preprocess_text(text):
    # Convierte a minúsculas
    text = text.lower()
    # Elimina signos de puntuación
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Elimina stopwords
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Aplica el preprocesamiento a los datos de entrenamiento y prueba
X_train = [preprocess_text(doc) for doc in X_train]
X_test = [preprocess_text(doc) for doc in X_test]

#Extraccion de caracteristicas 
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

#-------------------------------------- SUPORT VECTOR MACHINE --------------------------------------------

SVM_gender=svm.SVC(kernel='linear')
SVM_gender.fit(X_train_vec,Y_train_gender)
Y_predict_gender= SVM_gender.predict(X_test_vec)

SVM_country=svm.SVC(kernel='linear')
SVM_country.fit(X_train_vec,Y_train_country)
Y_predict_country= SVM_country.predict(X_test_vec)

# None, devuelve la precisión para cada clase por separado sin promediar 
accuracy_genero_svm = accuracy_score(Y_test_gender, Y_predict_gender)
accuracy_country_svm = accuracy_score(Y_test_country, Y_predict_country)

f1_genero_svm = f1_score(Y_test_gender,Y_predict_gender, average='macro')
f1_country_svm = f1_score(Y_test_country, Y_predict_country, average='macro')

print('SUPOR VECTOR MACHINE')
print('accuracy score genero: ',accuracy_genero_svm)
print('accuracy score pais: ',accuracy_country_svm)

print('F1 score genero: ', f1_genero_svm)
print('F1 score pais: ', f1_country_svm )


#-------------------------------------- Multinomial NB --------------------------------------------
NB_gender= MultinomialNB()
NB_gender.fit(X_train_vec,Y_train_gender)
Y_predict_gender_NB= NB_gender.predict(X_test_vec)

NB_country= MultinomialNB()
NB_country.fit(X_train_vec,Y_train_country)
Y_predict_country_NB= NB_country.predict(X_test_vec)
 
accuracy_genero_NB = accuracy_score(Y_test_gender, Y_predict_gender_NB)
accuracy_country_NB = accuracy_score(Y_test_country, Y_predict_country_NB)

f1_genero_NB = f1_score(Y_test_gender,Y_predict_gender_NB, average='macro')
f1_country_NB = f1_score(Y_test_country, Y_predict_country_NB, average='macro')

print('MULTINOMIAL NAVIE BAYES')
print('accuracy score genero: ',accuracy_genero_NB)
print('accuracy score pais: ',accuracy_country_NB)

print('F1 score genero: ', f1_genero_NB)
print('F1 score pais: ', f1_country_NB)


#-------------------------------------- Decision Trees --------------------------------------------
Tree_gender = tree.DecisionTreeClassifier()
Tree_gender.fit(X_train_vec,Y_train_gender)
Y_predict_gender_T= Tree_gender.predict(X_test_vec)

Tree_country= tree.DecisionTreeClassifier()
Tree_country.fit(X_train_vec,Y_train_country)
Y_predict_country_T= Tree_country.predict(X_test_vec)

accuracy_genero_T = accuracy_score(Y_test_gender, Y_predict_gender_T)
accuracy_country_T = accuracy_score(Y_test_country, Y_predict_country_T)

f1_genero_T = f1_score(Y_test_gender,Y_predict_gender_T, average='macro')
f1_country_T = f1_score(Y_test_country, Y_predict_country_T, average='macro')

print('Decision Tree')
print('accuracy score genero: ',accuracy_genero_T)
print('accuracy score pais: ',accuracy_country_T)

print('F1 score genero: ', f1_genero_T)
print('F1 score pais: ', f1_country_T)

