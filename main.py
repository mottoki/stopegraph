import streamlit as st
import numpy as np
import pandas as pd
from math import degrees, radians, tan, sin, cos, atan, asin, acos, sqrt, pi, pow
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# ------------- CONFIG ----------------------------------

st.set_page_config(page_title='Corephoto', page_icon=None, layout="wide")

hide_table_row_index = """
    <style>
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """

st.markdown(hide_table_row_index, unsafe_allow_html=True)

st.sidebar.title("Select Input Files")

# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

# Key columns
key_level = ['level']
key_stope = ['stope']
key_unsupp = ['unsupported (ft)']
key_length = ['length (ft)']
key_q = ["q'"]
key_a = ['a']
key_b = ['b']
key_c = ['c']

# -------------------- Input file -------------------------------------
uploaded_file = st.sidebar.file_uploader("Stope Performance Data")

if uploaded_file is not None:
    # Can be used wherever a "file-like" object is accepted:
    if uploaded_file.name.endswith('.csv'):
        xl = pd.read_csv(uploaded_file)
        sheets = ['Sheet1']

    ds = pd.DataFrame()
    for sheet in sheets:
        if uploaded_file.name.endswith('.csv'):
            df1 = xl

        dfn = pd.DataFrame()
        for col in df1.columns:
            if any(subs == col.lower() for subs in key_level):
                dfn['Level'] = df1[col]
            if any(subs == col.lower() for subs in key_stope):
                dfn['Stope'] = df1[col]
            if any(subs == col.lower() for subs in key_unsupp):
                dfn['Unsupported'] = df1[col]
            if any(subs == col.lower() for subs in key_length):
                dfn['Length'] = df1[col]
            if any(subs == col.lower() for subs in key_q):
                dfn["Q'"] = df1[col]
            if any(subs == col.lower() for subs in key_a):
                dfn["A"] = df1[col]
            if any(subs == col.lower() for subs in key_b):
                dfn["B"] = df1[col]
            if any(subs == col.lower() for subs in key_c):
                dfn["C"] = df1[col]
        ds = pd.concat([ds, dfn], ignore_index=True)

    # Filter
    # level
    container = st.sidebar.container()
    all_selected = st.sidebar.checkbox("Select all", key="level")
    levels = sorted(set(ds['Level']))
    if all_selected:
        level_selection = container.multiselect("Level", (levels), (levels))
        if level_selection: ds = ds[ds["Level"].isin(level_selection)]
    else:
        level_selection = container.multiselect("Level", (levels))
        if level_selection: ds = ds[ds["Level"].isin(level_selection)]

    # Stope
    container = st.sidebar.container()
    all_selected = st.sidebar.checkbox("Select all", key="stope")
    stopes = sorted(set(ds['Stope']))
    if all_selected:
        stope_selection = container.multiselect("Stope", (stopes), (stopes))
        if stope_selection: ds = ds[ds["Stope"].isin(stope_selection)]
    else:
        stope_selection = container.multiselect("Stope", (stopes))
        if stope_selection: ds = ds[ds["Stope"].isin(stope_selection)]

else:
    ds = None

# ----------------- FUNCTIONS -------------------------
# Evaluate a polynomial in reverse order using Horner's Rule,
# for example: a3*x^3+a2*x^2+a1*x+a0 = ((a3*x+a2)x+a1)x+a0
def horners_poly(lst, x):
    total = 0
    for a in reversed(lst):
        total = total*x+a
    return total

def simple_poly(lst, x):
  n, tmp = 0, 0
  for a in lst:
    tmp = tmp + (a * (x**n))
    n += 1

  return tmp

# ------------------- MAIN ---------------------------------
if ds is not None:
    # Calculation
    ds["S"] = (ds['Unsupported']*ds['Length'])/(ds['Unsupported']*2+ds['Length']*2)
    ds['S'] = ds['S']*0.3048 # Convert from feet to metre
    ds["N"] = ds["Q'"]*ds["A"]*ds["B"]*ds["C"]

    # GRAPH
    y = [0.1,1,10,100,643]
    y_lustz = [0.1, 1, 10, 100, 368.75]
    y_ustz = [0.1, 1, 10, 50]
    y_lstz = [0.1, 1, 10, 115.45]
    uustz = [1.4, 2.8, 5.9, 14.1, 25]
    lustz = [3.2424, 4.4583, 8.2411, 17.4279, 25]
    ustz = [5.1338, 7.5656, 11.2133, 14.0504]
    lstz = [6.8901, 9.1868, 12.6994, 18.2385]
    columns = ['y', 'y_lustz', 'y_ustz', 'y_lstz', 'uustz', 'lustz', 'ustz', 'lstz']

    df = pd.DataFrame([y, y_lustz, y_ustz, y_lstz, uustz, lustz, ustz, lstz]).T
    df.columns = columns

    # Plotly
    fig = go.Figure()
    myDict = {}
    for col in columns[4:]:
        if col == 'ustz':
            degree = 6
            y_lst = df['y_ustz'].dropna()
            x_lst = df[col].dropna()
        elif col == 'lstz':
            degree = 5
            y_lst = df['y_lstz'].dropna()
            x_lst = df[col].dropna()
        elif col == 'lustz':
            degree = 5
            y_lst = df['y_lustz']
            x_lst = df[col]
        else:
            degree = 5
            y_lst = df['y']
            x_lst = df[col]
        X = x_lst.values.reshape(-1, 1)
        x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        poly = PolynomialFeatures(degree)
        poly.fit(X)
        X_poly = poly.transform(X)
        x_range_poly = poly.transform(x_range)
        model = LinearRegression(fit_intercept=False)
        model.fit(X_poly, y_lst)
        y_poly = model.predict(x_range_poly)

        # fig.add_trace(go.Scatter(x=df[col], y=y_lst, mode='markers'))
        fig.add_traces(go.Scatter(x=x_range.squeeze(), y=y_poly, name=col,
            line_color='#000000', hoverinfo='skip'))

        myDict[col] = list(model.coef_)

    # Categorise data
    ds['lustz_check'] = ds.apply(lambda row: simple_poly(myDict['lustz'], row['S']), axis=1)
    ds['uustz_check'] = ds.apply(lambda row: simple_poly(myDict['uustz'], row['S']), axis=1)
    ds['ustz_check'] = ds.apply(lambda row: simple_poly(myDict['ustz'], row['S']), axis=1)
    ds['lstz_check'] = ds.apply(lambda row: simple_poly(myDict['lstz'], row['S']), axis=1)
    ds['cat'] = 'Stable Zone'
    mask = ds['uustz_check'] > ds["N"]
    ds.loc[mask, 'cat'] = 'Unsupported Transition Zone'
    mask = ds['lustz_check'] > ds["N"]
    ds.loc[mask, 'cat'] = 'Stable with support'
    mask = ds['ustz_check'] > ds["N"]
    ds.loc[mask, 'cat'] = 'Supported Transition Zone'
    mask = ds['lstz_check'] > ds["N"]
    ds.loc[mask, 'cat'] = 'Caved Zone'

    # fig.add_annotation(x=5, y=1,
    #     text="Stable Zone",
    #     showarrow=False,
    #     textangle=-30,
    #     font=dict(family="Courier New, monospace", size=16,),
    #     bordercolor="#c7c7c7",
    #     borderwidth=2,
    #     borderpad=4,
    #     bgcolor="#ff7f0e",
    #     opacity=0.8)

    f = px.scatter(ds, x='S', y='N', hover_data=['Level', 'Stope'], color='cat')

    for i in range(len(f.data)):
        fig.add_trace(f.data[i-1])

    fig.update_traces(marker=dict(size=10,
        line=dict(width=2, color='DarkSlateGrey')),)
    # fig.update_yaxes(type="log")

    fig.update_layout(
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#FFFFFF',
        height=700,)

    fig.add_shape(
        type="rect", xref="paper", yref="paper",
        x0=0, y0=0, x1=1.0, y1=1.0,
        line=dict(color="black", width=2))

    fig.update_xaxes(gridcolor='lightgrey', tickfont=dict(size=16),
        zeroline=True, zerolinewidth=3, zerolinecolor='lightgrey',
        tickformat=",.0f")

    fig.update_yaxes(gridcolor='lightgrey', tickfont=dict(size=16),
        zeroline=True, zerolinewidth=3, zerolinecolor='lightgrey',
        type="log", range=[-1, 3])

    fig.update_layout(
        # title_text="Double Y Axis Example",
        xaxis=dict(title_text='<b>N</b>'),
        yaxis=dict(title_text="<b>S (m)</b>"))

    st.plotly_chart(fig, use_container_width=True)
