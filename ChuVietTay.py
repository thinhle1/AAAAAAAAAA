import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt
import seaborn as sns
import os
#Ép chạy trên CPU
#os.environ['CUDA_VISIBLE_DEVICE']='-1'
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import warnings


warnings.filterwarnings('ignore')

sns.set()
dataset = read_csv('A_Z Handwritten Data/A_Z Handwritten Data.csv').astype('float32')

dataset.rename(columns={'0':'label'}, inplace=True)
#Chia x làm datatrain , theo cột và y nhãn chữ
X = dataset.drop('label', axis = 1 )
y = dataset['label']
#print("Shape:",X.shape)
#print("Culoms count:",len(X.iloc[1]))
# dataset đang có 372037 dòng và 784 cột .
from sklearn.utils import shuffle
#Gán nhãn
alphabets_mapper = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',
                    11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',
                    21:'V',22:'W',23:'X',24:'Y',25:'Z'}

dataset_alphabets = dataset.copy()
label_size = dataset.groupby('label').size()
#label_size.plot.barh(figsize=(10,10))
#plt.show()
#dataset['label'] : In tất cả cột từ cột của label từ 0 -> 25 (26 chữ cái)
#Gán nhãn vào
dataset['label'] = dataset['label'].map(alphabets_mapper)
#NHẬN 1 CHỮ
import cv2
img = cv2.imread('Anh Test/ChuA.png',0)
img = np.array(img).astype('float32')
img = img.reshape(-1,784)
#93010
#standard_sca.fit(img)
X_train , X_test, Y_train, Y_test = train_test_split(X,y)
#Chuẩn hóa dữ liệu vào khoảng [0,1]
standard_scaler = MinMaxScaler()
standard_scaler.fit(X_train)
# điều chỉnh dữ liệu
X_train = standard_scaler.transform(X_train)
X_test = standard_scaler.transform(X_test)
#Đổi tất cả dữ liệu đã chuyển hóa thành float
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test = X_test.reshape( X_test.shape[0],28, 28, 1).astype('float32')
#72919840
img = standard_scaler.transform(img)
img = img.reshape(1,28,28,1).astype('float32')
#rint('X_train[0] :\n',X_train[0])
Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)
#print('X_train.shape: ',X_train.shape)
# Có 279027 dữ liệu để train theo dạng 28*28*1
#print('Y_train.shape: ',Y_train.shape)
#Tương tự có 279027 nhãn từ 0->26 thuộc 26 chữ cái
print("X_test.shape :", X_test.shape)
#result (93010, 28, 28, 1)
#print("Y_test.shape :", Y_test.shape)
# Vẽ các chữ đem train
X_shuffle = shuffle(X_train)
plt.figure(figsize = (12,10))
row, colums = 4, 4
for i in range(16):
    plt.subplot(colums, row, i+1)
    plt.imshow(X_shuffle[i].reshape(28,28),interpolation='nearest', cmap='Greys')
plt.show()
#print(img.shape)
#print(type(X_test[0].shape))
#print(type(img.shape))
#XÂY DỰNG MODEL
model = Sequential()
##Cắt từng phần nhỏ của ma trận vào hidden layer || cắt shape 28*28
model.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation='relu'))
# lấy các đặc tính đặc biệt trong ma trận
model.add(MaxPooling2D(pool_size=(2, 2)))
# Tránh hiện tượng overfitting
model.add(Dropout(0.3))
#chuyển thành ma trận 1 chiều
model.add(Flatten(input_shape=(28, 28, 1)))
#tạo đầu ra lần 1 là 128
model.add(Dense(128, activation='relu'))
#Layer thứ hai (và cuối cùng) là lớp softmax có 26 nút, với mỗi nút tương đương với điểm xác suất,
# và tổng các giá trị của 10 nút này là 1 (tương đương 100%).
model.add(Dense(len(y.unique()), activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
model.fit(X_train, Y_train, epochs=1, batch_size=100, verbose=2)
#chúng đánh giá các chất lượng của mô hình bằng tập kiểm thử
test_lost,  test_acc = model.evaluate(X_test,Y_test, verbose=0)
print("Tỷ lệ:{}%".format(test_acc))
###
# Nhận diện 1 chữ trong data được cắt
predictions = model.predict(X_test)
print(predictions[0])
print(np.argmax(predictions[0]))
print(alphabets_mapper[np.argmax(predictions[0])])

#NHẬN LOẠT CHỮ
plt.figure(figsize = (12,10))
row, colums = 4, 4
for i in range(16):
    plt.subplot(row,colums,i+1)
    plt.imshow(X_test[i+5].reshape(28,28),interpolation='nearest', cmap='Greys')
    plt.xticks([])
    plt.yticks([])
    prob = model.predict(X_test[i+5].reshape(1,28,28,1))
    pred = int(np.argmax(prob, axis=1))
    response = alphabets_mapper[pred] +" (" + str(round(prob[0][pred] * 100,2)) + " %)"
    plt.title(response)
plt.show()
########################################
def predictions_test():
    predictions = model.predict(X_test)
    print(predictions[0])
    print(alphabets_mapper[np.argmax(predictions[0])])
    plt.figure(figsize=(12, 10))
    row, colums = 3, 3
    plt.subplot(colums, row, 5)
    plt.xticks([])
    #, validation_data = (X_test, Y_test)
    plt.yticks([])
    plt.imshow(X_test[0].reshape(28, 28), interpolation='nearest', cmap='Greys')
    plt.xlabel(alphabets_mapper[np.argmax(predictions[0])])
    plt.show()
######################
#####GIAO DIỆN
from tkinter import *
from PIL import Image, ImageTk
from tkinter import filedialog
from tkinter import messagebox
root = Tk()
root.geometry("550x300+300+150")
root.resizable(width=True, height=True)
def openfn():
    filename = filedialog.askopenfilename(title='open')
    return filename
#def data_config
def HienThi(x):
    Prediction = model.predict(x)
    return alphabets_mapper[np.argmax(Prediction)]
def open_img():
    x = openfn()
    imgage = Image.open(x)
    imgage = imgage.resize((100, 100), Image.ANTIALIAS)
    imgage = ImageTk.PhotoImage(imgage)
    panel = Label(root, image=imgage)
    panel.image = imgage
    panel.pack()
    im = cv2.imread(x,0)
    im = np.array(im).astype('float32')
    im = im.reshape(-1, 784)
    im = standard_scaler.transform(im)
    im = im.reshape(im.shape[0], 28, 28, 1).astype('float32')
    messagebox.showinfo("NHẬN DẠNG CHỮ CÁI VIẾT TAY","Đây là chữ {}".format(HienThi(im)))
    plt.figure(figsize=(12, 10))
    row, colums = 4, 4
    plt.subplot(colums, row, 7)
    plt.imshow(im.reshape(28, 28), interpolation='nearest', cmap='Greys')
    plt.show()
btn = Button(root, text='Chọn hình để nhận dạng', command=open_img).pack()

root.mainloop()
##############################

# Nhận diện 1 chữ mà không có giao diện
prediction = model.predict(img)
plt.figure(figsize=(12, 10))
#row, colums = 4, 4
plt.subplot(colums, row, 7)
plt.imshow(img.reshape(28,28), interpolation='nearest', cmap='Greys')
plt.xticks([])
plt.yticks([])
plt.xlabel('Chữ đang test là chữ :{} '.format(alphabets_mapper[np.argmax(prediction)]))
plt.show()


