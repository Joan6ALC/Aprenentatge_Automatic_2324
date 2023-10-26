from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.svm import SVC
import numpy as np
from scipy.spatial import distance_matrix
from sklearn.preprocessing import MinMaxScaler


def kernel_lineal(x1, x2):
    return x1.dot(x2.T)


def kernel_rbf(x1, x2, gamma=10):
    return np.exp(-gamma * distance_matrix(x1, x2) ** 2)


X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, n_repeated=0,
                           n_classes=2, n_clusters_per_class=1, class_sep=0.5,
                           random_state=8)
# En realitat ja no necessitem canviar les etiquetes Scikit ho fa per nosaltres

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

## FEINA 1
# Els dos algorismes es beneficien d'estandaritzar les dades

scaler = MinMaxScaler()
X_transformed = scaler.fit_transform(X_train)
X_test_transformed = scaler.transform(X_test)

# Entrenam un SVM linear (classe SVC)
svm_linear = SVC(C=1.0, kernel='linear', random_state=33)
svm_linear.fit(X_transformed, y_train)
y_linear = svm_linear.predict(X_test_transformed)

# Entrenam el nostre SVM linear
svm_meu = SVC(C=1.0, kernel=kernel_lineal, random_state=33)
svm_meu.fit(X_transformed, y_train)
y_meu = svm_meu.predict(X_test_transformed)

accuracy_linear = np.mean(y_linear == y_test)
accuracy_meu = np.mean(y_meu == y_test)

print("Accuracy SVM linear: ", accuracy_linear)
print("Accuracy SVM meu: ", accuracy_meu)
print("")

## FEINA 2

# Els dos algorismes es beneficien d'estandaritzar les dades

scaler = StandardScaler()
X_transformed = scaler.fit_transform(X_train)
X_test_transformed = scaler.transform(X_test)

# Entrenam un SVM rbf (classe SVC)
svm_rbf = SVC(C=1.0, kernel='rbf', random_state=33)
svm_rbf.fit(X_transformed, y_train)
y_rbf = svm_rbf.predict(X_test_transformed)

# Entrenam el nostre SVM rbf
svm_rbf2 = SVC(C=1.0, kernel=kernel_rbf, random_state=33)
svm_rbf2.fit(X_transformed, y_train)
y_rbf2 = svm_rbf2.predict(X_test_transformed)

accuracy_rbf = np.mean(y_rbf == y_test)
accuracy_meu_rbf = np.mean(y_rbf2 == y_test)

print("Accuracy SVM rbf: ", accuracy_rbf)
print("Accuracy SVM meu rbf: ", accuracy_meu_rbf)
print("")
