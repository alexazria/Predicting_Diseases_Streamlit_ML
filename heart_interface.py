import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn import linear_model
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
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
    
    #an expander that allows to have a drop window of all the raw data in df
    with st.expander("Data"):
        st.write(df)

    st.markdown("<h2 style='text-align: center;'>General Figures</h2>", unsafe_allow_html=True)

    col1,col2,col3 = st.columns(3)

    col1.metric('Number of males',int(len(df[df['Sex']=='M'])))
    col2.metric('Number of Females',int(len(df[df['Sex']=='F'])))
    col3.metric('Detected Heart Disease',int(len(df[df['HeartDisease']==1])))


    ##age distribution per sex

    st.markdown("<h2 style='text-align: center;'>Distributions</h2>", unsafe_allow_html=True)

    df['count'] = 1
    fig = px.histogram(df, x="Age", y="count", color="Sex", marginal="violin",
                   hover_data=df.columns)

    col1,col2 = st.columns([1,1])
    col1.plotly_chart(fig, use_container_width=True)

    # An aggregation function that computes the average for RestingBP/Cholesterol/HeartDisease/avg_max_HR as well
    # as the number of patients with a heart disease.
    def agg(row):
        d = {}
        d['avg_restingBP'] = row['RestingBP'].mean()
        d['avg_Cholestrol'] = row['Cholesterol'].mean()
        d['total_HeartDisease'] = row['HeartDisease'].sum()
        d['avg_max_HR'] = row['MaxHR'].sum()
        return pd.Series(d)

    #different groupbys for different columns to get the stats.
    df_agg = df.groupby('Sex',as_index=False).apply(agg)
    col2.write("Stats by Sex")
    col2.table(df_agg)
    col2.write("Stats by HeartDisease")
    col2.table(df.groupby('HeartDisease',as_index=False).apply(agg))
    col2.write("Stats by RestingECG")
    col2.table(df.groupby('RestingECG',as_index=False).apply(agg))
    

    st.markdown("<h2 style='text-align: center;'>Aggergation by column</h2>", unsafe_allow_html=True)

   #Create a drop list for columns in the dataframe and allows the aggregation by axis as well as a color 
   # and the choice between stacked or grouped bar charts.
    possible_rows = df.columns

    col1,col2,col3 = st.columns([1,1,1])
    x_axis_select = col1.selectbox("X-axis",possible_rows[:],index=0)
    color_select = col2.selectbox("Color",possible_rows[:],index=1)
    barmode = col3.selectbox("Bar Mode",['stack','group'])

    
    df_grp = df.groupby(by=[x_axis_select,color_select]).size().to_frame('size')
    df_grp = df_grp.reset_index()
    fig = px.bar(df_grp, x=x_axis_select, y='size',color=color_select,barmode=barmode)
    st.plotly_chart(fig, use_container_width=True)
    

    st.markdown("<h2 style='text-align: center;'>Average over column</h2>", unsafe_allow_html=True)

    # If above column is numeric an average is computed over the x-axis.
    if is_numeric_dtype(df[color_select]):
        df_grp = df.groupby(by=[x_axis_select],as_index=False)[color_select].mean()
        fig = px.line(df_grp, x=x_axis_select, y=color_select)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning('A non numeric column, cannot compute the average')


def render():
    # The main render function that is called when the heart interface is choosen.    
    if 'df_heart' not in st.session_state:
        df = pd.read_csv('attachments/heart.csv')  
        st.session_state['df_heart'] = df
    else:
        df = st.session_state['df_heart']


    options = st.sidebar.selectbox('Mode',("Display","Kmeans","Regression"))
    

    if options == 'Display':    
        display_gen(df)
    elif options == "Kmeans":
        possible_rows = df.columns

        fig = px.scatter_matrix(df,
            dimensions=['Age','ChestPainType','RestingBP','Cholesterol','FastingBS','RestingECG','MaxHR'],
            color="Sex", symbol="Sex",
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

        fig = px.scatter_matrix(df,
            dimensions=['Age','ChestPainType','RestingBP','Cholesterol','FastingBS','RestingECG','MaxHR'],
            color="Sex", symbol="Sex",
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