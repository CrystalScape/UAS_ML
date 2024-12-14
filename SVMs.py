# Model 
from sklearn.svm import SVC
# Preprocessing Data 
import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler , LabelEncoder
from sklearn import metrics
import random
# Visualizer 
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import PCA
# save
import pickle as pkl

dt1 = pd.read_excel(r'fruit.xlsx')
dt1

print(dt1['name'].unique())

x = dt1.drop(columns=['name']).values
MM = MinMaxScaler().fit(x)
x_norms = MM.transform(x)
x_norms

y = dt1['name'].values
le = LabelEncoder().fit(y)
y_trns = le.transform(y)
y_trns

x_train , x_test , y_train , y_test = train_test_split(x_norms , y_trns , test_size=0.2)
x_train.shape , y_train.shape , x_test.shape , y_test.shape

def fitnes(y_true ,x , p): 
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    models = SVC(kernel=kernels[int(p[0])] , gamma=p[1] , C=p[2]).fit(x, y_true)
    y_pred= models.predict(x)
    loss = metrics.mean_squared_error(y_true , y_pred)
    acc = metrics.accuracy_score(y_true , y_pred)
    return loss , acc
ite = 10
def spwan_pop(): 
    return np.array([np.random.randint(0,3) , np.abs(np.random.randn()) , np.abs(np.random.random())])
params = []
loss = []
accs = []
gammas = []
cs = []
for i in tqdm(range(ite) , 'Fine Tuning...') : 
    pop = spwan_pop()
    fitnesh , acc = fitnes(y_train , x_train, pop)
    params.append(pop)
    loss.append(fitnesh)
    accs.append(acc)
    gammas.append(pop[1])
    cs.append(pop[2])
best = params[np.argmin(loss)]
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
modelsh = SVC(kernel=kernels[int(best[0])] , gamma=best[1] , C=best[2]).fit(x_train, y_train)
fileSVM = open(r'svm.pkl' , 'wb')
pkl.dump(modelsh , fileSVM)        
print(f'Model Berhasil di save dengan parameter : Kernel : {kernels[int(best[0])]} | gamma : {best[1]} | C : {best[2]} | loss : {np.min(loss)} | acc : {np.max(accs)}')

plt.figure(figsize=(20 , 5))
plt.title('FineTune')
plt.scatter(x=np.argmin(loss) , y=np.max(accs))
plt.text(x=np.argmin(loss) , y=np.max(accs), s = f'loss : {np.min(loss)} | acc : {np.max(accs)}')
plt.plot(loss , label = 'Loss')
plt.plot(accs , label = 'Acc')
plt.plot(gammas , label = 'gamma')
plt.plot(cs , label = 'C')
plt.grid()
plt.legend()
plt.show()

PCAcomponent = PCA(2).fit(x_norms , y_trns)
componen = PCAcomponent.transform(x_norms)
print(componen)
xmin , xmax = componen[:,0].min() - 1, componen[:,0].max() + 1
ymin , ymax = componen[:,1].min() - 1, componen[:,1].max() + 1
xx,yy = np.meshgrid(
    np.arange(xmin , xmax , 0.2) , np.arange(ymin , ymax , 0.2) 
)
loadM = SVC(kernel=kernels[int(best[0])] , gamma=best[1] , C=best[2]).fit(componen, y_trns)
z = loadM.predict(np.c_[xx.ravel(),yy.ravel()])
z = z.reshape(xx.shape)
plt.title(f"SVM {kernels[int(best[0])]}")
plt.contourf(xx, yy ,z , alpha = 0.7)
plt.scatter(x=componen[:,0] , y=componen[:,1] , c=y_trns)
plt.show()
