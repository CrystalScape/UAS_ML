# Model
import torch
from torch import nn 
from torch.nn import Sequential , Module
import numpy as np 
#data 
import pandas as pd 
from sklearn.preprocessing import LabelEncoder , MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix , ConfusionMatrixDisplay
#visual
import matplotlib.pyplot as plt 
from tqdm import tqdm

df = pd.read_excel('Pumpkin_Seeds_Dataset.xlsx')
print(df['Class'].unique())
labels = df['Class'].unique()

x = df.drop(columns=['Class'])
mm = MinMaxScaler().fit(x)
xmm = mm.transform(x)
xmm = torch.from_numpy(xmm).float()
print(xmm)

y = df['Class']
le = LabelEncoder().fit(y)
yle = le.transform(y)
yle = torch.from_numpy(yle).long()
print(yle)

x_train , x_test ,y_train , y_test = train_test_split(xmm, yle , train_size=0.7)
print(x_train.shape , y_train.shape , x_test.shape , y_test.shape)

class MultiLayerPerceptron(Module): 
    def __init__(self, n_lab):
        super(MultiLayerPerceptron , self).__init__()
        self.Seq = Sequential(
            nn.Linear(12 , 1200), 
            nn.ReLU(),
            nn.Linear(1200 , 1000),
            nn.ReLU(),
            nn.Linear(1000 , 500),
            nn.ReLU()
        )
        self.final_n = nn.Linear(500 , n_lab)
        self.activ = nn.Sigmoid()
    def forward(self, x):
        x = self.Seq(x)
        x = self.final_n(x)
        x = self.activ(x)
        return x 
    
model = MultiLayerPerceptron(len(labels))
print(model(x_train[0]))
optims = torch.optim.Adam(model.parameters() , 0.001)
loss = nn.CrossEntropyLoss()

#for i in tqdm(range(1000)): 
#    optims.zero_grad()
#    prediks_prop = model(x_train)
#    prediks = prediks_prop.argmax(-1)
#    jmlh = (prediks == y_train).sum(0)
#    lossh = loss(prediks_prop , y_train)
#    lossh.backward()
#    optims.step()
#    print(f'Loss : {lossh} | Acc : {jmlh} | real : {y_train.shape} | selisih : {y_train.shape[0]-jmlh}')
#torch.save(model , 'MLP.pth')

modelLoad = torch.load('MLP.pt')
prediksi = modelLoad(x_test).argmax(-1)
acc = metrics.accuracy_score(y_test , prediksi.detach().numpy())
loss = metrics.mean_squared_error(y_test , prediksi.detach().numpy())
confus = confusion_matrix(y_test , prediksi.detach().numpy())
plotconfus = ConfusionMatrixDisplay(confus,display_labels=labels)
plotconfus.plot()
plt.show()
