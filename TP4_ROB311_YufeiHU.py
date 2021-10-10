import os
import numpy as np
import skimage.feature
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import (KNeighborsClassifier,
                               NeighborhoodComponentsAnalysis)
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn.metrics import classification_report


#=======================================================
'''data loading'''
#=======================================================
# path="database/train/happy/Training_430147.jpg"
# img = plt.imread(path)
path="database/train/"
folders=os.listdir(path)
lable_list=[] # save the emotion lable  [0 'angry', 1'disgust', 2'fear', 3'happy', 4'neutral', 5'sad', 6'surprise']
imgs=[] #save the imamges
for i in range(len(folders)):
    path_emotion=path+folders[i]
    filenames=sorted(os.listdir(path_emotion))
    #print("..................")
    #print(filenames)
    for j in range(len(filenames)):
        lable_list.append(i)
        img=plt.imread(path_emotion+"/"+filenames[j])
        imgs.append(img)

imgs_np=np.array(imgs)
print(imgs_np.shape)    #(28709, 48, 48)
print("train/angry number: ",lable_list.count(0))
print("train/disgust number: ",lable_list.count(1))
print("train/fear number: ",lable_list.count(2))
print("train/happy number: ",lable_list.count(3))
print("train/neutral number: ",lable_list.count(4))
print("train/sad number: ",lable_list.count(5))
print("train/surprise number: ",lable_list.count(6))
lable_np=np.array(lable_list)
print(lable_np.shape)
print(lable_np)

plt.figure(0, figsize=(12,20))
fig_no = 0
for expression in os.listdir("database/train/"):
    print(expression)
    for i in range(1,6):
        fig_no = fig_no + 1
        plt.subplot(7,5,fig_no)
        print("train/" + expression + "/" +os.listdir("database/train/" + expression)[i])
        img = plt.imread("database/train/" + expression + "/" +os.listdir("database/train/" + expression)[i])
        plt.imshow(img, cmap="gray")
plt.tight_layout()
plt.show()
#=======================================================
'''Local Binary Patterns (LBP)(feature extraction)'''
#=======================================================
img = imgs_np[0]
# print(img)
# exit()
img_lbp = skimage.feature.local_binary_pattern(img, 8,1.0,method='var')
# numpy historgram operation
plt.subplot(121)
plt.imshow(img,cmap ='gray')
plt.subplot(122)
plt.imshow(img_lbp)
plt.show()
#=======================================================
'''KNN classifier (Expression Recognition)'''
#=======================================================
X = [[0], [1], [2], [3]]
y = [0, 0, 1, 1]
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X, y)
print(neigh.predict([[1.1]]))
print(neigh.predict_proba([[0.9]]))

X = imgs_np
Y = lable_np
X_feature = []
Y_feature = []

for i in range(X.shape[0]):
    print('working on {}th img...'.format(i+1))
    x_lbp = skimage.feature.local_binary_pattern(X[i], 8,1.0,method='var')
    if np.isnan(x_lbp).any():
        continue
    X_feature.append(x_lbp)
    Y_feature.append(Y[i])
X_feature = np.array(X_feature)
X_feature = np.reshape(X_feature,(X_feature.shape[0],-1))
X_feature = preprocessing.normalize(X_feature[:],norm='l2')
Y_feature = np.array(Y_feature)
print(Y_feature.shape)
print(X_feature.shape)
# exit()
X_train, X_test, Y_train, _test = train_test_split(X_feature, Y_feature, stratify=Y_feature, test_size=0.1, random_state=42)
print(X_train.shape)
''' here we can change the n_neighbors to check the performance'''
clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(X_train, Y_train)
print('the process of train is finished!')
Y_pred = clf.predict(X_test)
Y_true = _test
target_names = ['class0','class1','class2','class3','class4','class5','class6']
print(classification_report(Y_true,Y_pred,target_names=target_names))


