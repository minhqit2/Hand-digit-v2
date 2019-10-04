#train model file
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
import joblib

from sklearn.datasets import fetch_mldata

mnist = fetch_mldata('mnist-original', data_home='./')
N,d = mnist.data.shape
print(N)
print(d)

x_all = mnist.data 
y_all = mnist.target
# in thử 1 số trong dataset

plt.imshow(x_all.T[:,3000].reshape(28,28))
plt.axis("off")
plt.show()

#lọc lại chỉ còn 2 chữ số 0 và 1
x0 = x_all[np.where(y_all == 0)[0]]
x1 = x_all[np.where(y_all == 1)[0]]
y0 = np.zeros(x0.shape[0])
y1 = np.ones(x1.shape[0])
#gộp 0 và 1 lại thành một dataset
x = np.concatenate((x0,x1), axis = 0)
y = np.concatenate((y0,y1))
print("chay di ne")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
#chia dữ liệu thành 2 tập train và set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1000)
#train model
model = LogisticRegression(C=1e5)
model.fit(x_train, y_train)

#check aceuracy
y_prediction = model.predict(x_test)
print("Accuracy"+str(100*accuracy_score(y_test, y_prediction)))

#save model
from sklearn.externals import joblib
joblib.dump(model, "digits.pkl", compress = 3)

# mở file hard_digit  v1 ra và 
#model = joblib.load("digits.pkl")
#trong đoạn code for rect in rects bổ sung thêm đoạn code  để lấy vùng ảnh[1] đưa về mảng 28*28