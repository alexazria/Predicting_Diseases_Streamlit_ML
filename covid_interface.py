import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn import linear_model

import matplotlib.pyplot as plt
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype

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

    # plot it as in the example at http://scikit-learn.org/
    plt.scatter(x, y,  color='red')

    fig, ax = plt.subplots()
    ax.scatter(x, y)
    ax.scatter(x, regr.predict(x),color='blue')

    # plt.plot(x, regr.predict(x), color='blue', linewidth=3)
    # plt.xticks(())
    # plt.yticks(())
    # plt.show()
    st.pyplot(fig)

def kmeans(df,col1,col2):  
    """"
    A function that takes a dataframe and the name of two columns, as well as K value
    computes a kmeans using the two columns with k = K that is passed in input
    and plots the result directly.
    
    """

    kmeans = KMeans(n_clusters=3).fit(df[[col1,col2]])
    centroids = kmeans.cluster_centers_
    st.write(centroids)

    fig, ax = plt.subplots()
    ax.scatter(df[col1], df[col2], c= kmeans.labels_.astype(float), s=50, alpha=0.5)
    ax.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)

    st.pyplot(fig)
 

def label_gender(row):
    # Converts gender into one column with a label.
    if row['Gender_Female'] == 1 :
        return 'F'
    elif row['Gender_Male'] == 1 :
        return 'M'
    else:
        return 'Transgender'

def label_age(row):
    # Converts age categories into one column.
    if row['Age_0-9'] == 1 :
        return '0-9'
    elif row['Age_10-19'] == 1 :
        return '10-19'
    elif row['Age_20-24'] == 1 :
        return '20-24'
    elif row['Age_25-59'] == 1 :
        return '25-59'
    else:
        return '60+'

def label_severity(row):
    # Converts severity columns into one column.
    if row['Severity_None'] == 1 :
        return 'None'
    elif row['Severity_Moderate'] == 1 :
        return 'Moderate'
    elif row['Severity_Mild'] == 1 :
        return 'Mild'
    else:
        return 'Severe'

def label_nbsym(row):
    # Counts the number of symptomes.
    nb_sym =  row['Fever'] + row['Tiredness']+ row['Dry-Cough'] + row['Sore-Throat'] + row['Pains']+ row['Runny-Nose'] + row['Diarrhea']
    return nb_sym

   

def display_gen(df):
    #A function that takes a dataframe and computes and prints in streamlit interface
    #different metrics.
    with st.expander("Data"):
        st.write(df)

    st.markdown("<h2 style='text-align: center;'>General Figures</h2>", unsafe_allow_html=True)

    col1,col2,col3 = st.columns(3)

    col1.metric('Number of Contact',int(len(df[df['Contact_Yes']==1])))
    col2.metric('Number of Non Contact',int(len(df[df['Contact_No']==1])))
    col3.metric('Number of Not sure',int(len(df[df['Contact_Dont-Know']==1])))


    ##age distribution per sex

    st.markdown("<h2 style='text-align: center;'>Distributions</h2>", unsafe_allow_html=True)

    

    fig = px.histogram(df, x="Country", y="count", color="gender", marginal="violin",
                   hover_data=df.columns)

    col1,col2 = st.columns([1,1])
    col1.plotly_chart(fig, use_container_width=True)

    
    df_agg = df.groupby(['Country','severity'],as_index=False)['NbSym'].mean()
    col2.write("Country")
    col2.table(df_agg)
    df_agg = df.groupby(['age','severity'],as_index=False)['NbSym'].mean()
    col2.write("Age")
    col2.table(df_agg)
    # col2.write("Stats by HeartDisease")
    # col2.table(df.groupby('HeartDisease',as_index=False).apply(agg))
    # col2.write("Stats by RestingECG")
    # col2.table(df.groupby('RestingECG',as_index=False).apply(agg))
    

    st.markdown("<h2 style='text-align: center;'>Aggergation by column</h2>", unsafe_allow_html=True)

    possible_rows = df.columns

    col1,col2,col3 = st.columns([1,1,1])
    x_axis_select = col1.selectbox("X-axis",possible_rows[:],index=2)
    color_select = col2.selectbox("Color",possible_rows[:],index=1)
    barmode = col3.selectbox("Bar Mode",['stack','group'])

    
    df_grp = df.groupby(by=[x_axis_select,color_select]).size().to_frame('size')
    df_grp = df_grp.reset_index()
    fig = px.bar(df_grp, x=x_axis_select, y='size',color=color_select,barmode=barmode)
    st.plotly_chart(fig, use_container_width=True)


def render():
    # the main render function that is called when the heart interface is choosen.
    if 'df_covid' not in st.session_state:
        df = pd.read_csv('attachments/Cleaned-Data.csv')  
        df['count'] = 1
        df['gender'] = df.apply(lambda row: label_gender(row), axis=1)
        df['age'] = df.apply(lambda row: label_age(row), axis=1)
        df['severity'] = df.apply(lambda row: label_severity(row), axis=1)
        df['NbSym'] = df.apply(lambda row: label_nbsym(row), axis=1)
        
        st.session_state['df_covid'] = df
    else:
        df = st.session_state['df_covid']


    options = st.sidebar.selectbox('Mode',("Display","Kmeans","Regression"))
    

    if options == 'Display':    
        display_gen(df)
    elif options == "Kmeans":
        possible_rows = df.columns

        fig = px.scatter_matrix(df,
            dimensions=['age','severity','NbSym'],
            color="gender", symbol="gender",
            title="Scatter matrix",
            labels={col:col.replace('_', ' ') for col in df.columns}) # remove underscore
        config = {
                    'toImageButtonOptions': {
                    'format': 'png', # one of png, svg, jpeg, webp
                    'filename': 'weekprofile',
                    'height': 500,
                    'width': 2000,
                    'scale': 5 # Multiply title/legend/axis/canvas sizes by this factor
                    }
                } 
        fig.update_traces(diagonal_visible=False)
        fig.update_layout(height=1000)
        st.plotly_chart(fig, use_container_width=True,config=config,height=1000)




        col1,col2,col3 = st.columns(3)
        x_axis_select = col1.selectbox("X-axis",possible_rows[:])
        y_axis_select = col2.selectbox("Y-axis",possible_rows[:])

        btn_load = col3.button("Load")
        if btn_load and is_numeric_dtype(df[x_axis_select]) and is_numeric_dtype(df[y_axis_select]):          
            kmeans(df,x_axis_select,y_axis_select)

    elif options == "Regression":
        possible_rows = df.columns

        fig = px.scatter_matrix(df,
            dimensions=['age','severity','NbSym'],
            color="gender", symbol="gender",
            title="Scatter matrix",
            labels={col:col.replace('_', ' ') for col in df.columns}) # remove underscore
        config = {
                    'toImageButtonOptions': {
                    'format': 'png', # one of png, svg, jpeg, webp
                    'filename': 'weekprofile',
                    'height': 500,
                    'width': 2000,
                    'scale': 5 # Multiply title/legend/axis/canvas sizes by this factor
                    }
                } 
        fig.update_traces(diagonal_visible=False)
        fig.update_layout(height=1000)
        st.plotly_chart(fig, use_container_width=True,config=config,height=1000)



        col1,col2,col3 = st.columns(3)
        x_axis_select = col1.selectbox("X-axis",possible_rows[:],index=0)
        y_axis_select = col2.selectbox("Y-axis",possible_rows[:],index=1)
        btn_load = col3.button("Load")
        if btn_load and is_numeric_dtype(df[x_axis_select]) and is_numeric_dtype(df[y_axis_select]):           
            regression(df,x_axis_select,y_axis_select)


       


    #     subrender(dataset_select,options)



    # # param_form = st.form("parameters")    


    # # col1,col2 = param_form.columns(2)

    # # ligne = col1.number_input('Ligne',value=132)
    # # date = col2.text_input('Date',value='2021-03-31')



    # df_trips_bis = df_trips[df_trips['service_id'].isin(df_cal[df_cal['date']==date].service_id.tolist())]
    # df_trips_bis['short_name'] = df_trips_bis.route_id.str.split(":").str[1]

    # lines = list(df_trips_bis.short_name.unique())

    # for l in lines:
    #     try:
    #         load_timetable(date,str(l),'Classic week day')
    #     except Exception as inst:
    #         st.write(str(l),"failed")
    #         st.write(str(inst))
 
# st.set_page_config(page_title="Project", page_icon=None, layout='wide')

# render()