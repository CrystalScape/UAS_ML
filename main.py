# PCA : https://towardsdatascience.com/the-mathematics-behind-principal-component-analysis-fff2d7f4b643
import pickle
import pandas as pd 
import numpy as np 
import streamlit as st 
from sklearn.preprocessing import MinMaxScaler
import torch 
from torch import nn 
from torch.nn import Sequential, Module
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
st.set_page_config('MACHINE LEARNING UAS 🤖' , '📏' , layout='wide')

pils = st.selectbox('Pilih :' , ['SVM' , 'MLP'])
if pils == 'SVM' : 
    st.title('SVM 🤖') 
    st.write('''Melakukan klasifikasi buah berdasarkan bobot warna
             dan diameterdengan menggunakan algoritma SVM''')
    x = pd.read_excel('fruit.xlsx')
    st.header('Perview Data')
    st.dataframe(x)
    xl = x['name'].unique()
    x = x.drop(columns=['name'])
    modelSVM = pickle.load(open('svm.pkl' , 'rb'))
    minmax = MinMaxScaler().fit(x)
    diamet = float(st.select_slider('Diameter 📏' , np.round(np.arange(1,21 , 0.10) , 2)))
    bobot = float(st.select_slider('Bobot ⚖️' , np.round(np.arange(1,501 , 0.11) , 2)))
    r = float(st.select_slider('Red 🔴' , np.round(np.arange(1,256) , 2)))
    g = float(st.select_slider('Green 🟢' , np.round(np.arange(1,256) , 2)))
    b = float(st.select_slider('Blue 🔵' , np.round(np.arange(1,256) , 2)))
    x_in = minmax.transform([[diamet ,bobot , r , g , b]])
    predict = modelSVM.predict(x_in)
    st.write('buah yang di klasifikasi kan berdasar kan kriteria di atas : ',xl[predict][0])
    
if pils == 'MLP' : 
    st.title('Multi Layer Perceptron 🤖')
    st.write('Melakukan klasifikasi jenis labu berdasarkan kriteria nya')
    df = pd.read_excel('Pumpkin_Seeds_Dataset.xlsx')
    st.dataframe(df)
    area = float(st.select_slider('Area 📏' , np.round(np.arange(1,151 , 0.10) , 2)))
    perme = float(st.select_slider('Perimeter 📏' , np.round(np.arange(1,1500 , 0.10) , 2)))
    mal = float(st.select_slider('MAL 📏' , np.round(np.arange(1,1000 , 0.10) , 2)))
    mial = float(st.select_slider('MIAL 📏' , np.round(np.arange(1,1000 , 0.10) , 2)))
    ca = float(st.select_slider('CA 📏' , np.round(np.arange(1,1000 , 0.10) , 2)))
    ed = float(st.select_slider('ED 📏' , np.round(np.arange(1,1000 , 0.10) , 2)))
    E = float(st.select_slider('Ecc 📏' , np.round(np.arange(0,1 , 0.1) , 2)))
    sol = float(st.select_slider('Solidiy 📏' , np.round(np.arange(0,1 , 0.1) , 2)))
    ex = float(st.select_slider('Ex 📏' , np.round(np.arange(0,1 , 0.1) , 2)))
    Ro = float(st.select_slider('Ro 📏' , np.round(np.arange(0,1 , 0.1) , 2)))
    Ar = ex = float(st.select_slider('Ex 📏' , np.round(np.arange(1,3 , 0.10) , 2)))
    C = float(st.select_slider('Comp 📏' , np.round(np.arange(0,1 , 0.1) , 2)))
    labels = df['Class'].unique()
    model = MultiLayerPerceptron(len(labels))
    x = df.drop(columns=['Class'])
    mm = MinMaxScaler().fit(x)
    x_i = mm.transform([[area , perme , mal , mial , ca , ed , E , sol , ex , Ro , Ar , C]])
    molod = torch.load('MLP.pth')
    prediks = molod(torch.from_numpy(x_i).float())
    st.write(labels[torch.argmax(prediks).detach().numpy()])
