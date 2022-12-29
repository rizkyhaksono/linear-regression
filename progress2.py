import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn

dataset = pd.read_csv('test.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

dataku = pd.DataFrame(dataset)
plt.scatter(dataku.Umur, dataku.Angkatan)
plt.xlabel("Umur")
plt.ylabel("Jenis Kelamin")
plt.title("Grafik Masa Umur Mahasiswa")
plt.show()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

plt.figure(figsize = (10, 8))
plt.scatter(X_train, y_train, color = 'blue')
plt.plot(X_train, regressor.predict(X_train), color = 'red')
plt.title('Masa Umur Terhadap Angkatan')
plt.xlabel('Masa Umur')
plt.ylabel('Masa Angkatan')
plt.show()