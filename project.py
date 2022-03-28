import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

#Outcome = 1 Diabet
#Outcome = 0 Sağlıklı/Healthy

data = pd.read_csv("diabetes.csv")
result = data.head()
# print(result)

seker_hastalari = data[data.Outcome == 1]
saglikli_insanlar = data[data.Outcome == 0]
# print(seker_hastalari)
# print(saglikli_insanlar)

plt.scatter(saglikli_insanlar.Age,saglikli_insanlar.Glucose,color="green",label="Sağlıklı", alpha=0.4)
plt.scatter(seker_hastalari.Age,seker_hastalari.Glucose, color="red", label ="Hasta", alpha = 0.4)
plt.xlabel("Age")
plt.ylabel("Glucose")
plt.legend()
plt.show()

#x and y axis
y = data.Outcome.values
x_ham_veri = data.drop(["Outcome"],axis=1) #independent variable
# print(x_ham_veri)

#Normalization
x = (x_ham_veri - np.min(x_ham_veri))/(np.max(x_ham_veri)-np.min(x_ham_veri))
# print(x)

#train test
x_train , x_test , y_train , y_test = train_test_split(x, y, test_size=0.1, random_state=1) #yapay zekayı eğitmek için

#KNN model
sayac = 1
for k in range(1,11):
    knn = KNeighborsClassifier(n_neighbors= k )
    knn.fit(x_train,y_train) # eğit
    print(sayac,"  ", "Doğruluk oranı: %", knn.score(x_test,y_test)*100)
    sayac += 1

#k değerini 3 bulduk. en iyi başarı oranı

#Yeni bir hasta tahmini için :

scaler = MinMaxScaler()
scaler.fit_transform(x_ham_veri)

new_prediction = knn.predict(scaler.transform(np.array([[6,148,72,35,0,33.6,0.627,50]])))
new_prediction[0]
print(new_prediction)
