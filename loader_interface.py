import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn import linear_model

import matplotlib.pyplot as plt

def regression(df,col1,col2):
    """"
    A function that takes a dataframe and the name of two columns 
    returns a linear regression using the two columns 
    and plots the result directly. 
    """
    x = df[col1].values
    y = df[col2].values

    x = x.reshape(918 , 1)
    y = y.reshape(918 , 1)    

    regr = linear_model.LinearRegression()
    regr.fit(x, y)


    plt.scatter(x, y,  color='red')

    fig, ax = plt.subplots()
    ax.scatter(x, y)
    ax.scatter(x, regr.predict(x),color='blue')

    st.pyplot(fig)

def kmeans(df,col1,col2,k):   
    """"
    A function that takes a dataframe and the name of two columns, as well as K value
    computes a kmeans using the two columns with k = K that is passed in input
    and plots the result directly.
    """
    kmeans = KMeans(n_clusters=k).fit(df[[col1,col2]])
    centroids = kmeans.cluster_centers_
    st.write(centroids)

    fig, ax = plt.subplots()
    ax.scatter(df[col1], df[col2], c= kmeans.labels_.astype(float), s=50, alpha=0.5)
    ax.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)

    st.pyplot(fig)


def display_gen(df):
    """"
    A function that takes a dataframe and computes and prints in streamlit interface
    different metrics.
    """

    with st.expander("Data"):
        st.write(df)

    st.markdown("<h2 style='text-align: center;'>General Figures</h2>", unsafe_allow_html=True)

    
    col1,col2,col3 = st.columns([1,1,1])
    selectcol = col2.selectbox("Aggregation column",df.columns[:],index=0)
 
    grp = df.groupby(selectcol,as_index=False).size().values.tolist()
    
    if len(grp) < 6:
        cols = st.columns(len(grp))
        for i in range(len(grp)):
            g = grp[i]
            cols[i].metric(selectcol + "(" + str(g[0]) + ")",g[1])
    else:
        grp = df.groupby(selectcol,as_index=False).size()     
        fig = px.histogram(grp, x=selectcol, y="size")
        st.plotly_chart(fig, use_container_width=True)
    

    st.markdown("<h2 style='text-align: center;'>Aggregation by column</h2>", unsafe_allow_html=True)

    possible_rows = df.columns

    col1,col2,col3 = st.columns([1,1,1])
    x_axis_select = col1.selectbox("X-axis",possible_rows[:],index=0)
    color_select = col2.selectbox("Color",possible_rows[:],index=1)
    barmode = col3.selectbox("Bar Mode",['stack','group'])

    
    df_grp = df.groupby(by=[x_axis_select,color_select]).size().to_frame('size')
    df_grp = df_grp.reset_index()
    fig = px.bar(df_grp, x=x_axis_select, y='size',color=color_select,barmode=barmode)
    st.plotly_chart(fig, use_container_width=True)
    




def render():
    # the main render function that is called when the heart interface is choosen

    uploaded_file = st.sidebar.file_uploader("Choose a file")
    options = st.sidebar.selectbox('Mode',("Display","Kmeans","Regression"))
    df = None
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state['df'] = df


    if 'df' in st.session_state:
        if options == 'Display':    
            display_gen(df)
    
        elif options == "Kmeans":
            possible_rows = df.columns

            col1,col2,col3,col4 = st.columns(4)
            x_axis_select = col1.selectbox("X-axis",possible_rows[:],index=0)
            y_axis_select = col2.selectbox("Y-axis",possible_rows[:],index=1)
            
            k_value = col3.number_input("K-Value",value=3)
            col4.write("")
            btn_load = col4.button("Load")
            if btn_load and is_numeric_dtype(df[x_axis_select]) and is_numeric_dtype(df[y_axis_select]):           
                kmeans(df,x_axis_select,y_axis_select,k_value)

        elif options == "Regression":
            possible_rows = df.columns
            col1,col2,col3 = st.columns(3)
            x_axis_select = col1.selectbox("X-axis",possible_rows[:])
            y_axis_select = col2.selectbox("Y-axis",possible_rows[:])
            btn_load = col3.button("Load")
            if btn_load and is_numeric_dtype(df[x_axis_select]) and is_numeric_dtype(df[y_axis_select]):           
                regression(df,x_axis_select,y_axis_select)
