
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go

df = pd.read_csv(r"C:\Users\Asus\Desktop\Clases\Analítica\Proy\icfes_cols_filtered_no_dupes_BOM.csv")

#unica estimación del modelo
modelo = smf.ols("""punt_global ~ C(cole_naturaleza, Treatment(reference="OFICIAL")) +C(cole_calendario, Treatment(reference="A")) +
C(cole_area_ubicacion, Treatment(reference="URBANO")) + C(cole_jornada, Treatment(reference="MAÑANA")) + 
C(fami_educacionmadre, Treatment(reference="Secundaria (Bachillerato) completa")) +
C(fami_educacionpadre, Treatment(reference="Secundaria (Bachillerato) completa"))
""", data=df).fit(cov_type="HC3")

coef_table = pd.DataFrame({
    "Coeficiente": modelo.params,
    "Error Std": modelo.bse,
    "z": modelo.tvalues,
    "p-value": modelo.pvalues
}).reset_index()

anova_table = sm.stats.anova_lm(modelo, typ=2).reset_index()

app = dash.Dash(__name__)
server = app.server

#---
# Graficos de continuas
#---

punt_cols = [
    "punt_ingles",
    "punt_matematicas",
    "punt_sociales_ciudadanas",
    "punt_c_naturales",
    "punt_lectura_critica",
    "punt_global"
]

# =========================
# 
fig_hist = make_subplots(
    rows=1, cols=len(punt_cols),
    subplot_titles=punt_cols
)

for i, col in enumerate(punt_cols, start=1):
    fig_hist.add_trace(
        go.Histogram(
            x=df[col].dropna(),
            nbinsx=30,
            showlegend=False
        ),
        row=1, col=i
    )

fig_hist.update_layout(
    title="Distribuciones de las variables de puntaje",
    height=350,
    template="simple_white",
    margin=dict(l=20, r=20, t=50, b=20)
)

# =========================
# 
corr = df[punt_cols].corr(method="pearson")

fig_corr = px.imshow(
    corr,
    text_auto=".3f",
    color_continuous_scale="RdBu_r",
    zmin=-1, zmax=1,
    title="Matriz de correlaciones (Pearson)"
)

fig_corr.update_layout(
    height=600,
    template="simple_white",
    margin=dict(l=40, r=40, t=60, b=40)
)

# =========================
# 
dff = df[punt_cols].dropna().sample(
    n=min(4000, df[punt_cols].dropna().shape[0]),
    random_state=42
)

fig_pair = px.scatter_matrix(
    dff,
    dimensions=[
        "punt_global",
        "punt_matematicas",
        "punt_lectura_critica",
        "punt_c_naturales",
        "punt_ingles"
    ],
    opacity=0.15,
    title="Pairplot (Scatter Matrix) – Puntajes ICFES"
)

fig_pair.update_layout(
    height=700,
    template="simple_white",
    margin=dict(l=40, r=40, t=60, b=40)
)


app.layout = html.Div([

    html.H1("Dashboard ICFES – Departamento del Cesar"),

    dcc.Tab(label="Variables Continuas", children=[
    dcc.Graph(figure=fig_hist),
    dcc.Graph(figure=fig_corr),
    dcc.Graph(figure=fig_pair),
]),
])

if __name__ == "__main__":
    app.run(debug=True)

