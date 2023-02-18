import streamlit as st
import pandas as pd
import time
import plotly.express as px
import datetime
from importlib import import_module
from glob import glob
import json
from os import path
import sys
import logging

logging.basicConfig(level=logging.DEBUG)

st.set_page_config(page_title="Data Hub", page_icon=None, layout='wide', initial_sidebar_state='auto')


##
# Streamlit HUB - Encapsulate multiple Streamlit Apps.
##


# Load Configuration
if not path.exists("config.json"):

    st.error("No configuration file 'config.json' found")
    sys.exit(1)

with open("config.json", "r") as config_file:
    config = json.load(config_file)
    VIEWS = config["views"]

    #### View selectors
    dic = {name: import_module(mod).render for mod, name in VIEWS.items()}

# Run app

st.sidebar.title("Data Science Hub")
view_list_ordered = list(dic.keys())
app_view = st.sidebar.selectbox("Select App", view_list_ordered)
dic[app_view]()
