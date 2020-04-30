import pandas as pd
import requests
import numpy as np
from datetime import datetime
import streamlit as st
import matplotlib
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.express as px
from sodapy import Socrata
from collections import Counter
import seaborn as sns
from scipy import stats
from scipy.optimize import curve_fit


st.title('APP COVID19 Colombia 2020')

# Example authenticated client (needed for non-public datasets):
client = Socrata("www.datos.gov.co", 
                "ymgH2QpK9Z5cSKBNlKgtuzWZP", 
                username="esteban.silvav@udea.edu.co", 
                password="Nomeacuerdodatosabiertos_1")

results = client.get("gt2j-8ykr", limit=10000)



# Convert to pandas DataFrame
df = pd.DataFrame.from_records(results)

# DataFrame with the information
#.apply(lambda x: datetime.strptime(x, '%Y%m%d%H'))
df['fecha_de_notificaci_n'] = df['fecha_de_notificaci_n'].apply(lambda row: datetime.strptime(row, '%Y-%m-%dT%H:%M:%S.%f'))
df['fecha_de_notificaci_n'] = pd.to_datetime(df['fecha_de_notificaci_n'])

df['fis'] = df['fis'].replace('Asintomático', '1999-01-01T00:00:00.000')
df['fis'] = df['fis'].apply(lambda x: datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%f'))
df['fis'] = pd.to_datetime(df['fis'])

df['fecha_diagnostico'] = df['fecha_diagnostico'].apply(lambda row: datetime.strptime(row, '%Y-%m-%dT%H:%M:%S.%f'))
df['fecha_diagnostico'] = pd.to_datetime(df['fecha_diagnostico'])

to_remove_in_df = df['fecha_de_muerte'][0]


df['fecha_recuperado'] = df['fecha_recuperado'].replace(to_remove_in_df, datetime.now().strftime('%Y-%m-%dT00:00:00.000'))
df['fecha_recuperado'] = df['fecha_recuperado'].replace(np.nan, datetime.now().strftime('%Y-%m-%dT00:00:00.000'))
df['fecha_recuperado'] = df['fecha_recuperado'].apply(lambda row: datetime.strptime(row, '%Y-%m-%dT%H:%M:%S.%f'))
df['fecha_recuperado'] = pd.to_datetime(df['fecha_recuperado'])

tiemp_diag = []
for ii in range(len(df['id_de_caso'])):
    dif_days = (df['fecha_diagnostico'][ii] - df['fis'][ii]).days
    if dif_days < 200:
        tiemp_diag.append(dif_days)
    else:
        tiemp_diag.append(np.nan)
df['Tiempo_diagnostico'] = tiemp_diag

tiemp_recup = []
for ii in range(len(df['id_de_caso'])):
    dif_days = (df['fecha_recuperado'][ii] - df['fis'][ii]).days
    if (dif_days < 200) and (df['fecha_de_muerte'][ii] == to_remove_in_df):
        tiemp_recup.append(dif_days)
    else:
        tiemp_recup.append(np.nan)
df['Tiempo_recuperado'] = tiemp_recup

dfcum_sum = df.groupby(df['fecha_diagnostico'])['id_de_caso'].count()
cum_sum = np.cumsum(dfcum_sum)

# DATOS
st.title('Datos')
df

# Última actualización
st.title('Última actualización')
st.write(datetime.today().strftime('%d-%m-%Y'))

# Número de casos
st.title('Casos')
'## Número total de casos:',len(df)
'## Número de nuevos casos:',cum_sum[-1]-cum_sum[-2]
'## Número de muertos:',df['id_de_caso'][df['fecha_de_muerte'] != to_remove_in_df].count()
'## Número de recuperados:',df['id_de_caso'][df['atenci_n'] == 'Recuperado'].count()


# Tiempo promedio de diagnóstico
st.title('Tiempo medio y promedio de Diagnóstico')


#plt.hist(df['Tiempo_diagnostico'], bins=50)
#plt.xlabel('Tiempo de Diagnóstico')
#st.pyplot()

hist_values_diagn = np.histogram(df['Tiempo_diagnostico'], bins=50, range=(df['Tiempo_diagnostico'].min(),df['Tiempo_diagnostico'].max()))[0]
'## #días = fecha_diagnóstico - fecha_inicio_síntomas'
st.write('El número medio de días es: ',np.argmax(hist_values_diagn))
st.write('El número promedio de días es: ',round(np.mean(df['Tiempo_diagnostico']),2))
st.bar_chart(hist_values_diagn)

# Tiempo promedio de recuperación
st.title('Tiempo medio y promedio de Recuperación')
hist_values_recup = np.histogram(df['Tiempo_recuperado'], bins=50, range=(df['Tiempo_recuperado'].min(),df['Tiempo_recuperado'].max()))[0]
'## #días = fecha_recuperado - fecha_inicio_síntomas'
st.write('El número medio de días es: ',np.argmax(hist_values_recup))
st.write('El número promedio de días es: ',round(np.mean(df['Tiempo_recuperado']),2))
st.bar_chart(hist_values_recup)


#plt.hist(df['Tiempo_recuperado'], bins=50)
#plt.xlabel('Tiempo de Recuperación')
#st.pyplot()

# Número de casos por día
st.title('Acumulado y Casos por Día')
#df_groupedbyday = df.groupby('Tiempo_diagnostico')['id_de_caso'].count()

st.line_chart(cum_sum)
st.bar_chart(dfcum_sum)


st.title('Fit a curva experimental')
r'''
$$n_e(t) = n_e(0) \times R_0^{t}$$
'''

x_to_fit = np.array(range(len(cum_sum)))

def exponenial_func(x, a, b):
#    return a*np.exp(b*x)+c
    return a*b**(x)

popt, pcov = curve_fit(exponenial_func, x_to_fit, cum_sum, p0=(1, 2))

st.write('n_e(0)=', round(popt[0],3))
st.write('R_0=', round(popt[1],3))


xx = np.linspace(1, 50, 1000)
yy = exponenial_func(xx, *popt)

trace1 = go.Scatter(
                  x=x_to_fit,
                  y=cum_sum,
                  mode='markers',
                  marker=go.Marker(color='rgb(255, 127, 14)'),
                  name='Data'
                  )

trace2 = go.Scatter(
                  x=xx,
                  y=yy,
                  mode='lines',
                  marker=go.Marker(color='rgb(31, 119, 180)'),
                  name='Fit'
                  )

annotation = go.Annotation(
                  x=20,
                  y=10,
                  text='',
                  showarrow=False
                  )
layout = go.Layout(
                title='Fit datos COVID19, Colombia',
                plot_bgcolor='rgb(229, 229, 229)',
                  xaxis=go.XAxis(zerolinecolor='rgb(255,255,255)', gridcolor='rgb(255,255,255)'),
                  yaxis=go.YAxis(zerolinecolor='rgb(255,255,255)', gridcolor='rgb(255,255,255)'),
                  annotations=[annotation]
                )

data = [trace1, trace2]
fig = go.Figure(data=data, layout=layout)

st.write(fig)


# Gráficas por ciudad
st.title('Acumulado por Ciudad y R0 Estimado')
vars_ops = [ii for ii in df['ciudad_de_ubicaci_n'].unique()]
ciudad = st.selectbox('Ciudad',vars_ops ,index=2)

df_ciudad = df.loc[df['ciudad_de_ubicaci_n'] == ciudad]
df_city = df_ciudad.groupby(df['fecha_diagnostico'])['id_de_caso'].count()

cum_sum_ciudad = np.cumsum(df_city)
st.line_chart(cum_sum_ciudad)
st.bar_chart(df_city)

r'''
$$n_e(t) = n_e(0) \times R_0^t$$
'''

x_to_fit = np.array(range(len(cum_sum_ciudad)))

def exponenial_func(x, a, b):
#    return a*np.exp(b*x)+c
    return a*b**x

popt_citu, pcov_city = curve_fit(exponenial_func, x_to_fit, cum_sum_ciudad, p0=(1, 2))

st.write('n_e(0)=', round(popt_citu[0],3))
st.write('R_0=', round(popt_citu[1],3))

xx = np.linspace(1, 50, 1000)
yy = exponenial_func(xx, *popt_citu)

trace1 = go.Scatter(
                  x=x_to_fit,
                  y=cum_sum_ciudad,
                  mode='markers',
                  marker=go.Marker(color='rgb(255, 127, 14)'),
                  name='Data'
                  )

trace2 = go.Scatter(
                  x=xx,
                  y=yy,
                  mode='lines',
                  marker=go.Marker(color='rgb(31, 119, 180)'),
                  name='Fit'
                  )

annotation = go.Annotation(
                  x=20,
                  y=10,
                  text='',
                  showarrow=False
                  )
layout = go.Layout(
                title='Fit datos COVID19, '+ ciudad,
                plot_bgcolor='rgb(229, 229, 229)',
                  xaxis=go.XAxis(zerolinecolor='rgb(255,255,255)', gridcolor='rgb(255,255,255)'),
                  yaxis=go.YAxis(zerolinecolor='rgb(255,255,255)', gridcolor='rgb(255,255,255)'),
                  annotations=[annotation]
                )

data = [trace1, trace2]
fig = go.Figure(data=data, layout=layout)

st.write(fig)



