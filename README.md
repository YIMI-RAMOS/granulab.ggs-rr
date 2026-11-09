# GranuloLab — Comparador GGS y RR

Aplicación web educativa de análisis granulométrico para Ingeniería Metalúrgica, construida con Streamlit.

## Funciones
- Más de 30 mallas nominales de referencia, multiselección y mallas personalizadas.
- Pesos retenidos editables, fondo de tamices, porcentajes retenidos y pasantes.
- Comparación Gates–Gaudin–Schuhmann y Rosin–Rammler por mínimos cuadrados no lineales.
- Métricas RMSE, R², MAE; P50/P80 observado si el rango de medición lo permite.
- Gráficos Plotly interactivos, tabla y exportación CSV/Excel.

## Ejecutar
```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Publicar en Streamlit Community Cloud
1. Crear un repositorio nuevo de GitHub, por ejemplo `granulolab-ggs-rr` (recomendado para no reemplazar FiolaLab).
2. Subir **app.py**, **requirements.txt** y **README.md** (descomprimir el ZIP antes).
3. En https://share.streamlit.io seleccionar repositorio, rama y ruta `app.py`.
4. Pulsar Deploy y compartir el enlace resultante.

Los datos iniciales son simulados. Comprueba la apertura real de los tamices y no extrapoles modelos fuera del rango experimental sin justificación.
