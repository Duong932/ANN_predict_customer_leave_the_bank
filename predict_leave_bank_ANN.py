# import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
tf.__version__


###################################--------------PART 1: DATA PREPROCESSING----------------#############################
# import the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values
#print(X)
#print(y)


# ENCODING CATEGORICAL DATA
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])
#print(X)


#ONE HOT ENCODING THE "GEOGRAPHY" COLUMN
from sklearn.compose import ColumnTransformer
from sklearn. preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
#print(X)


# SPLITTING THE DATASET INTO THE TRAINING SET AND TEST SET
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


#FEATURE SCALING
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



###################################--------------PART 2: BUILDING THE ANN----------------###############################
# initialzing the ANN
ann = tf.keras.models.Sequential()


# adding the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))  # 'relu' - Rectified Linear Unit: ĐƠN VỊ CHỈNH LƯU TUYẾN TÍNH


# adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))


# adding the output layer
# units = 1 là vì cần 1 neuron ở output để mã hóa dữ liệu là 1 hoặc 0
# activation = 'sigmoid' là vì output cần 1 sự dựa đoán, và dự đoán chính là xác suất,
# ở output cần 1 dự đoán binary mà hàm sigmoid là hàm có biên từ 0 đến 1, nếu không phải dự đoán binary thì thay thành 'softmax'
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))



###################################--------------PART 3: TRAINING THE ANN----------------###############################
# compiling the ANN
# optimizer = 'adam' chọn trình tối ưu hóa adam
# loss = 'binary_crossentropy' vì hàm mất mát luôn tồn tại ở output, mà output là dự đoán binary nên chọn binary
# nếu dự đoán output không phải binary thì loss = 'cross_
# metrics = ['accuracy'] là chọn chỉ số
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# training the ANN on the training set
# batch_size = 32 là kích thước lô mặc định, 32 là phổ biến
ann.fit(X_train, y_train, batch_size=32, epochs = 100)



"""
Homework 1:
Use our ANN model to predict if the customer with the following informations will leave the bank: 
Geography: France
Credit Score: 600
Gender: Male
Age: 40 years old
Tenure: 3 years
Balance: $ 60000
Number of Products: 2
Does this customer have a credit card? Yes
Is this customer an Active Member: Yes
Estimated Salary: $ 50000
So, should we say goodbye to that customer?
"""

# SOLUTION
# 2 cặp ngoặc vuông là tạo ra mảng 2 chiều
# sc.transform là dự đoán từ tập test set
# > 0.5 là để dự đoán ra 0 hoặc 1, còn nếu muốn dự đoán ra xác suất thì bỏ >0.5 đi
#print((ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]]))) > 0.5)


""""
Homework 2:
Use our ANN model to predict if the customer with the following informations will leave the bank: 
Geography: Spain
Credit Score: 608
Gender: Female
Age: 41 years old
Tenure: 2 years
Balance: $ 90000
Number of Products: 2
Does this customer have a credit card? No
Is this customer an Active Member: No
Estimated Salary: $ 20000
So, should we say goodbye to that customer?
"""

#print((ann.predict(sc.transform([[0, 0, 0, 608, 0, 41, 2, 90000, 2, 0, 0, 20000]]))) > 0.5)


y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

