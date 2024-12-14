#model
# https://www.geeksforgeeks.org/elbow-method-for-optimal-value-of-k-in-kmeans/
from sklearn.cluster import KMeans # Cluster
from sklearn.linear_model import LogisticRegression # clasfic
import pickle
from sklearn import metrics
#data
import pandas as pd 
import numpy as np 
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
#Visual
import matplotlib.pyplot as plt 
from tqdm import tqdm

dt = pd.read_csv('Dataset UnSupervised\Wine Clustering\wine-clustering.csv')
real = pd.read_csv('wine_data.csv')['class']
print(real.unique())
print(dt.head())

def distanceCalculation_distortion(x , kmeanCenter): 
    jarak = x[: , np.newaxis , :] - kmeanCenter[np.newaxis , : , :]
    sums = np.sum(np.square(jarak) , axis=2)
    ecludean  = np.sqrt(sums)
    distor = sum(np.square(np.min(ecludean , axis=1))) / x.shape[0]
    return distor

def fin_k(x): 
    diff1 = np.diff(x)
    diff2 = np.diff(diff1)
    k_best = np.argmax(diff2) + 2 # di tambah 2 karena diff mengurangi dimensi
    return k_best

mm = MinMaxScaler().fit(dt)
x = mm.transform(dt)
print(x)

dist = []
iner = []
ks = range(10)
for k in tqdm(range(10)): 
    print(k + 1)
    model_cluter_test = KMeans(n_clusters=k + 1 , random_state=42).fit(x)
    iner.append(model_cluter_test.inertia_)
    dist.append(distanceCalculation_distortion(x , model_cluter_test.cluster_centers_))
best_iner = fin_k(iner) + 1 # menyesuai kan jumlah k dengan + 1
best_dist = fin_k(dist) + 1
print(best_iner , best_dist)
plt.title('Distortion and inertia')
plt.plot(dist , label = 'Distorsion')
plt.scatter(ks,dist)
plt.plot(iner , label = 'Inertia')
plt.scatter(ks,iner)
plt.axvline(x=best_dist , color='red', linestyle='--', label='Elbow Point Dist')
plt.axvline(x=best_iner , color='blue', linestyle='--', label='Elbow Point iner')
plt.text(x=best_dist , y=dist[best_dist] , s = f'best K : {best_dist}')
plt.text(x=best_iner , y=iner[best_iner] , s = f'best K : {best_iner}')
plt.grid()
plt.legend()
plt.show()

model_cluter = KMeans(best_iner , random_state=42).fit(x)
kelompokan = model_cluter.predict(x)

Pcas = PCA(n_components=2).fit(x)
pcaclas = PCA(n_components=2).fit(model_cluter.cluster_centers_)
componen_center = pcaclas.transform(model_cluter.cluster_centers_)
componen = Pcas.transform(x)
plt.title('Cluter result')
plt.scatter(x=componen[:,0] , y=componen[:,1] , c=kelompokan)
plt.scatter(x=componen_center[:,0], y=componen_center[:,1] , c='red' , s = 250)
plt.text(x=componen_center[0][0], y=componen_center[0][1] , s='Centroit')
plt.text(x=componen_center[1][0], y=componen_center[1][1] , s='Centroit')
plt.text(x=componen_center[2][0], y=componen_center[2][1] , s='Centroit')
plt.grid()
plt.show()

dt['class'] = kelompokan
pd.DataFrame(data=dt).to_csv('New_wine.csv' , index=False) # Save result

