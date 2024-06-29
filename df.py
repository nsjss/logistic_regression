import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
train_data = pd.read_csv("train.csv")

x_train = train_data.drop('Survived', axis = 1).drop('Name', axis = 1)
x_train = x_train.replace("male",1).replace("female",0).values.T
y_train = np.zeros(train_data.shape[0])
print("x_train",x_train.shape)

y_train = train_data.Survived.values.reshape(1,y_train.shape[0])
print("y_train",y_train)
print("y_train",y_train.shape)

n = x_train.shape[0]
print(np.zeros((n,1)))
print("wshape",np.zeros((n,1)).T.shape)

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def model(x, y, lr, iter):    
    m = x_train.shape[1]
    n = x_train.shape[0]    
    w = np.zeros((n,1))
    b = 0   
    cost_list = []    
    for i in range(iter):   
        if(i%500==0):
            lr=lr/1.05
        z = np.dot(w.T, x) + b
        if(i==1):
            print("zshape",z.shape)
        a = sigmoid(z)  
        cost = -(1/m)*np.sum( y*np.log(a) + (1-y)*np.log(1-a))
        dW = (1/m)*np.dot(a-y, x.T)
        dB = (1/m)*np.sum(a - y)        
        w = w - lr*dW.T
        b = b - lr*dB
        cost_list.append(cost)        
        if(i%100 == 0):
            print("cost is : ", cost) 

        #print("cost",cost_list)     
    return w, b, cost_list
iter = 10000
lr = 0.01
w, b, cost_list = model(x_train, y_train, lr,iter)

plt.plot(list(range(iter)), cost_list)
test_data = pd.read_csv("test.csv")

x_test = test_data.drop('Survived', axis = 1).drop('Name', axis = 1)
x_test = x_test.replace("male",1).replace("female",0).values.T
y_test = np.zeros(test_data.shape[0])
y_test = test_data.Survived.values.reshape(1,y_test.shape[0])

def accuracy(X, Y, W, B):    
    z = np.dot(W.T, X) + B
    a = sigmoid(z)    
    a = a > 0.5  
    a = np.array(a)   
    acc = (1 - np.sum(np.absolute(a - Y))/Y.shape[1])*100    
    print("Accuracy of the model is : ", round(acc, 2), "%")

accuracy(x_test, y_test, w, b)

plt.show()