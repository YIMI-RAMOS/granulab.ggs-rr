"""GranuloLab · Granulometría comparativa GGS vs RR (fines didácticos)."""
from io import BytesIO
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import least_squares

st.set_page_config(page_title="GranuloLab | GGS vs RR", page_icon="⚗️", layout="wide", initial_sidebar_state="expanded")
st.markdown('''<style>
.stApp {background:linear-gradient(145deg,#f1f7fb 0%,#f6fbf9 60%,#fff 100%);color:#162b43}
[data-testid="stSidebar"]{background:#10263b}
[data-testid="stSidebar"] *{color:#eef8ff!important}
h1,h2,h3{color:#15354d!important;letter-spacing:-.02em}
[data-testid="stMetric"]{background:white;padding:16px;border:1px solid #d9e8ea;border-radius:15px;box-shadow:0 4px 16px #1b59700d}
.hero{background:linear-gradient(110deg,#103450,#087f86);padding:26px;border-radius:20px;color:white;margin-bottom:20px}
.hero h1{color:white!important;margin:0;font-size:2.3rem}
.hero p{color:#c5f0e8;margin:8px 0 0}
[data-testid="stDataFrame"]{border-radius:14px;overflow:hidden}
</style>''', unsafe_allow_html=True)

# Aberturas referenciales ASTM E11; el usuario puede añadir otras series/aberturas manualmente.
CATALOGO = {
    '3/8 in': 9500, '#4': 4750, '#5': 4000, '#6': 3350, '#7': 2800,
    '#8': 2360, '#10': 2000, '#12': 1700, '#14': 1400, '#16': 1180,
    '#18': 1000, '#20': 850, '#25': 710, '#30': 600, '#35': 500,
    '#40': 425, '#45': 355, '#50': 300, '#60': 250, '#70': 212,
    '#80': 180, '#100': 150, '#120': 125, '#140': 106, '#170': 90,
    '#200': 75, '#230': 63, '#270': 53, '#325': 45, '#400': 38,
    '#450': 32, '#500': 25, '#635': 20,
}
DEFAULT = ['#10', '#16', '#20', '#30', '#40', '#50', '#70', '#100', '#140', '#200']


def ggs(x, k, m):
    x=np.asarray(x,dtype=float)
    return 100*np.minimum((x/k)**m, 1.0)


def rr(x, d0, n):
    x=np.asarray(x,dtype=float)
    return 100*(-np.expm1(-(x/d0)**n))


def fit_models(x,y):
    """Mínimos cuadrados no lineales EN escala % pasante; ambos modelos 2 parámetros."""
    x=np.asarray(x,dtype=float); y=np.asarray(y,dtype=float)
    xmax=max(x); xmin=min(x)
    if len(x)<3 or len(np.unique(y))<3 or np.allclose(y,y[0]):
        raise ValueError('Se requieren al menos 3 mallas y 3 porcentajes pasantes distintos para un ajuste interpretable.')
    fg=least_squares(lambda v:ggs(x,*v)-y,
                     x0=[xmax*1.2, 0.8],bounds=([xmax,0.05],[xmax*1e5,15]),max_nfev=2500)
    fr=least_squares(lambda v:rr(x,*v)-y,
                     x0=[max(np.median(x),xmin),1.2],bounds=([xmin/1e4,0.05],[xmax*1e5,15]),max_nfev=2500)
    if not fg.success or not fr.success:
        raise ValueError('No se logró completar el ajuste. Revisa los datos y las mallas.')
    return fg.x,fr.x


def metrics(y, pred):
    sse=float(np.sum((y-pred)**2)); tss=float(np.sum((y-np.mean(y))**2))
    return {'RMSE (p.p.)':np.sqrt(sse/len(y)), 'R²':(1-sse/tss if tss>0 else np.nan),
            'MAE (p.p.)':float(np.mean(np.abs(y-pred)))}


def interpolate_p(x,y,q):
    """Interpolación logarítmica solo si el objetivo está dentro de la curva observada."""
    if not (min(y)<=q<=max(y)):
        return np.nan
    x=np.asarray(x);y=np.asarray(y)
    for i in range(len(y)-1):
        if (y[i]-q)*(y[i+1]-q)<=0 and y[i]!=y[i+1]:
            return float(np.exp(np.interp(q,[y[i+1],y[i]],[np.log(x[i+1]),np.log(x[i])])))
    return np.nan


def example_masses(labels):
    """Ejemplo ficticio reproducible, NO datos de laboratorio."""
    sizes=np.array([CATALOGO[k] for k in labels],dtype=float)
    # Retenidos por diferencia de una curva RR ficticia y fondo.
    p=rr(sizes, 320, 1.12)
    retained=np.r_[100-p[0],p[:-1]-p[1:]]
    return np.round(np.maximum(retained,0)*5,2),round(float(p[-1]*5),2)

st.markdown('<div class="hero"><h1>⚗️ GranuloLab</h1><p>Análisis granulométrico interactivo · comparación Gates–Gaudin–Schuhmann (GGS) y Rosin–Rammler (RR)</p></div>', unsafe_allow_html=True)
st.info('Selecciona solo las mallas que usaste, ingresa la masa retenida en cada una y la del fondo. El programa calcula el % pasante y ajusta ambos modelos. Los valores precargados son un **ejemplo simulado**, no mediciones reales.')

with st.sidebar:
    st.header('⚙️ Configuración')
    st.caption('Serie de aberturas de referencia en micrómetros (µm). Comprueba la norma y abertura de tus tamices reales.')
    seleccion=st.multiselect('Mallas que deseo analizar', list(CATALOGO), default=DEFAULT,
                             help='Puedes marcar más de 20 mallas y omitir cualquiera de ellas.')
    st.divider()
    st.subheader('Añadir mallas propias')
    custom_label=st.text_input('Nombre de la malla (opcional)',placeholder='Ej.: Malla especial A')
    custom_size=st.number_input('Abertura propia (µm)',min_value=0.1,value=100.0,step=1.0)
    if 'custom' not in st.session_state:st.session_state.custom={}
    if st.button('➕ Añadir malla'):
        if not custom_label.strip():st.warning('Escribe un nombre para la malla.')
        elif custom_label in CATALOGO or custom_label in st.session_state.custom:st.warning('Ese nombre ya existe.')
        else:
            st.session_state.custom[custom_label.strip()]=custom_size
            st.rerun()
    if st.session_state.custom:
        st.caption('Mallas propias activas: '+', '.join(st.session_state.custom))
        if st.button('Quitar mallas propias'):
            st.session_state.custom={};st.rerun()
    st.divider()
    st.caption('Modelos ajustados por mínimos cuadrados no lineales respecto al porcentaje pasante. No se extrapola el D80 observado fuera de las mallas medidas.')

selected={k:CATALOGO[k] for k in seleccion}
selected.update(st.session_state.custom)
items=sorted(selected.items(),key=lambda kv:-kv[1])
if len(items)<3:
    st.warning('Selecciona por lo menos **3 mallas de diferentes aberturas** para comparar GGS y RR.')
    st.stop()
if len(set(v for k,v in items))!=len(items):
    st.error('Hay dos mallas con igual abertura. Elige mallas de aberturas diferentes.');st.stop()

labels=[a for a,b in items]; sizes=np.array([b for a,b in items],float)
if 'mass_map' not in st.session_state:
    st.session_state.mass_map={}
if 'pan_mass' not in st.session_state:
    st.session_state.pan_mass=0.0
# Memoria de las pesadas previas para mantener valores al cambiar de selección.
example,pan_example=example_masses([k for k in labels if k in CATALOGO]) if all(k in CATALOGO for k in labels) else (np.zeros(len(labels)),0.0)
if not st.session_state.mass_map:
    st.session_state.mass_map={k:float(m) for k,m in zip(labels,example)}
    st.session_state.pan_mass=pan_example

left,right=st.columns([1.7,1],gap='large')
with left:
    st.subheader('1. Mallas y masas retenidas')
    df0=pd.DataFrame({'Malla':labels,'Abertura (µm)':sizes,'Masa retenida (g)':[float(st.session_state.mass_map.get(k,0)) for k in labels]})
    edited=st.data_editor(df0,hide_index=True,use_container_width=True,num_rows='fixed',key='mesh_editor',
                          disabled=['Malla','Abertura (µm)'],column_config={
                              'Masa retenida (g)':st.column_config.NumberColumn(min_value=0.0,step=0.01,format='%.2f',required=True),
                              'Abertura (µm)':st.column_config.NumberColumn(format='%.2f'),
                          })
    for _,r in edited.iterrows():st.session_state.mass_map[str(r['Malla'])]=float(r['Masa retenida (g)'])
with right:
    st.subheader('2. Fondo y controles')
    pan=st.number_input('Masa en el fondo / bandeja (g)',min_value=0.0,step=0.01,value=float(st.session_state.pan_mass),format='%.2f',help='Fracción menor que la abertura de la malla más fina.')
    st.session_state.pan_mass=float(pan)
    if st.button('🧪 Cargar ejemplo simulado',use_container_width=True):
        vals,p=example_masses([k for k in labels if k in CATALOGO]) if all(k in CATALOGO for k in labels) else (np.zeros(len(labels)),0.0)
        st.session_state.mass_map={k:float(m) for k,m in zip(labels,vals)}
        st.session_state.pan_mass=float(p)
        st.session_state.pop('mesh_editor',None)
        st.rerun()
    if st.button('🧹 Poner masas en cero',use_container_width=True):
        st.session_state.mass_map={k:0.0 for k in labels};st.session_state.pan_mass=0.0
        st.session_state.pop('mesh_editor',None);st.rerun()
    st.caption(f'Mallas seleccionadas: **{len(items)}**. No existe un límite de 20 mallas.')

m=np.asarray(edited['Masa retenida (g)'].to_numpy(),float)
if np.any(~np.isfinite(m)) or np.any(m<0):st.error('Todas las masas deben ser números finitos ≥ 0.');st.stop()
total=float(np.sum(m)+pan)
if total<=0:st.warning('Agrega masas retenidas o utiliza el ejemplo para obtener resultados.');st.stop()
ret=100*m/total
cum=np.cumsum(ret)
passing=np.clip(100-cum,0,100)
result=pd.DataFrame({'Malla':labels,'Abertura (µm)':sizes,'Masa retenida (g)':m,
                     'Retenido (%)':ret,'Retenido acumulado (%)':cum,'Pasante acumulado (%)':passing})

st.divider()
st.subheader('3. Resultados metalúrgicos')
fit_error=None
try:
    pg,pr=fit_models(sizes,passing)
    predg=ggs(sizes,*pg);predr=rr(sizes,*pr)
    mg=metrics(passing,predg);mr=metrics(passing,predr)
except (ValueError,TypeError,OverflowError) as e:
    fit_error=str(e)

cols=st.columns(4)
cols[0].metric('Mallas analizadas',str(len(items)))
cols[1].metric('Masa total',f'{total:,.2f} g')
cols[2].metric('P80 observado',f'{interpolate_p(sizes,passing,80):,.1f} µm' if np.isfinite(interpolate_p(sizes,passing,80)) else 'Fuera de rango')
cols[3].metric('P50 observado',f'{interpolate_p(sizes,passing,50):,.1f} µm' if np.isfinite(interpolate_p(sizes,passing,50)) else 'Fuera de rango')

if pan==0:st.warning('La masa del fondo es cero: verifica si realmente no existe fracción más fina que la última malla. Si no mediste el fondo, el pasante en la malla menor no es confiable.')
if fit_error:
    st.warning('No hay suficientes datos útiles para ajustar ambos modelos: '+fit_error)
else:
    comparison=pd.DataFrame({'Modelo':['GGS','RR'], 'RMSE (p.p.)':[mg['RMSE (p.p.)'],mr['RMSE (p.p.)']],
                             'R²':[mg['R²'],mr['R²']], 'MAE (p.p.)':[mg['MAE (p.p.)'],mr['MAE (p.p.)']]})
    st.markdown(f'**Parámetros GGS:** k = `{pg[0]:.2f} µm`, m = `{pg[1]:.4f}`  ·  **Parámetros RR:** d₀ = `{pr[0]:.2f} µm`, n = `{pr[1]:.4f}`')
    winner='GGS' if mg['RMSE (p.p.)']<mr['RMSE (p.p.)'] else 'RR'
    st.success(f'Para las mallas seleccionadas, **{winner}** presenta el RMSE menor en % pasante. Esto no garantiza mejor predicción fuera del rango medido.')
    st.dataframe(comparison.style.format({'RMSE (p.p.)':'{:.3f}','R²':'{:.4f}','MAE (p.p.)':'{:.3f}'}),use_container_width=True,hide_index=True)

st.subheader('4. Gráficos dinámicos')
tab1,tab2,tab3=st.tabs(['📈 Curvas comparativas','📊 Retenido por malla','🎯 Residuos de ajuste'])
with tab1:
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=sizes,y=passing,mode='markers+lines',name='Datos experimentales',line=dict(color='#102f50',width=2),marker=dict(size=9,color='#173c62')))
    if not fit_error:
        xs=np.geomspace(max(min(sizes)*0.8,0.01),max(sizes)*1.15,300)
        fig.add_trace(go.Scatter(x=xs,y=ggs(xs,*pg),mode='lines',name='GGS',line=dict(color='#00a78e',width=3)))
        fig.add_trace(go.Scatter(x=xs,y=rr(xs,*pr),mode='lines',name='RR',line=dict(color='#f08044',width=3,dash='dash')))
    fig.update_layout(template='plotly_white',height=440,hovermode='x unified',xaxis_title='Abertura (µm) · escala log',yaxis_title='Pasante acumulado (%)',yaxis_range=[0,102],legend=dict(orientation='h',y=1.15),margin=dict(t=48))
    fig.update_xaxes(type='log')
    st.plotly_chart(fig,use_container_width=True)
with tab2:
    fig2=go.Figure(go.Bar(x=labels,y=ret,marker_color=['#128b9d' if i%2==0 else '#44b6a1' for i in range(len(labels))],text=[f'{v:.1f}%' for v in ret],textposition='outside'))
    fig2.update_layout(template='plotly_white',height=430,xaxis_title='Malla (de gruesa a fina)',yaxis_title='Retenido individual (%)',margin=dict(t=45))
    st.plotly_chart(fig2,use_container_width=True)
with tab3:
    if fit_error:st.info('Completa datos útiles para generar residuos.')
    else:
        res=go.Figure()
        res.add_trace(go.Bar(x=labels,y=passing-predg,name='GGS',marker_color='#00a78e'))
        res.add_trace(go.Bar(x=labels,y=passing-predr,name='RR',marker_color='#f08044'))
        res.add_hline(y=0,line_dash='dot',line_color='#334155')
        res.update_layout(barmode='group',template='plotly_white',height=430,yaxis_title='Observado − calculado (puntos porcentuales)',xaxis_title='Malla')
        st.plotly_chart(res,use_container_width=True)

st.subheader('5. Tabla de análisis y descargas')
st.dataframe(result.style.format({c:'{:.3f}' for c in result.select_dtypes(include=np.number).columns}),use_container_width=True,hide_index=True)
if not fit_error:
    result['Pasante GGS (%)']=predg
    result['Pasante RR (%)']=predr
    result['Residuo GGS (p.p.)']=passing-predg
    result['Residuo RR (p.p.)']=passing-predr

csv=result.to_csv(index=False).encode('utf-8-sig')
def excel_bytes():
    out=BytesIO()
    with pd.ExcelWriter(out,engine='openpyxl') as writer:
        result.to_excel(writer,index=False,sheet_name='Granulometria')
        if not fit_error:
            comparison.to_excel(writer,index=False,sheet_name='Comparacion')
            pd.DataFrame({'Modelo':['GGS','RR'],'Parametro de tamaño (µm)':[pg[0],pr[0]],'Exponente':[pg[1],pr[1]]}).to_excel(writer,index=False,sheet_name='Parametros')
    return out.getvalue()
c1,c2=st.columns(2)
c1.download_button('⬇️ Descargar CSV',csv,'granulometria_ggs_rr.csv','text/csv',use_container_width=True)
c2.download_button('⬇️ Descargar Excel',excel_bytes(),'granulometria_ggs_rr.xlsx','application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',use_container_width=True)

with st.expander('📚 Fundamento, supuestos y límites del análisis'):
    st.markdown(r'''
**GGS:** $P(x)=100\min[(x/k)^m,1]$, con $k>0$ en µm y $m>0$. Se impone $k\geq$ mayor abertura medida y se limita a 100 %.

**RR:** $P(x)=100\{1-\exp[-(x/d_0)^n]\}$, con $d_0>0$ en µm y $n>0$.

**Base:** $\%R_i=100m_i/M$; $\%P_i=100-\sum_{j\le i}\%R_j$, de mayor a menor abertura. Se incluye el material del fondo en $M$.

**Ajuste:** mínimos cuadrados *no lineales* sobre % pasante. El RMSE se informa en puntos porcentuales (p.p.); R² se calcula en el espacio original de % pasante. Los extremos 0 % y 100 % se conservan porque no se usan logaritmos de los datos experimentales.

**P50 y P80 observados:** interpolación entre aberturas contiguas en escala logarítmica; no se calcula si el porcentaje no está comprendido entre las mediciones. Son estimaciones aproximadas dependientes de la malla elegida.

**Calidad de los datos:** no se calcula cierre respecto a la masa inicial porque no se ingresa esa masa; si hubo pérdidas, regístralas externamente. Si no se conoce el fondo, no asumas que es 0. No se estima incertidumbre del tamiz ni de la balanza. Si hay pocos puntos distintos, el ajuste no es robusto.

**Catálogo:** aberturas nominales de referencia; verifica tu serie real (ASTM, Tyler, ISO) antes de usarlas en un informe.

**Referencias de ecuaciones:** [SysCAD](https://help.syscad.net/Size_Distribution_Definition) y [Itasca](https://docs.itascacg.com/itasca900/pfc/docproject/source/manual/examples/tutorials/granulometry/granulometry.html).
    ''')
