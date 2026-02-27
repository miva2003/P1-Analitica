import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import statsmodels.formula.api as smf

# Carga de datos
df = pd.read_csv(r"C:\Users\manes\Downloads\P1-Analitica\analisis_de_datos\pregunta2\icfes_cols_filtered_no_dupes_BOM(in).csv")

# Reordenar categorías de educación
orden = [
    'Sin respuesta',
    'Ninguno',
    'Primaria incompleta',
    'Primaria completa',
    'Secundaria (Bachillerato) incompleta',
    'Secundaria (Bachillerato) completa',
    'Técnica o tecnológica incompleta',
    'Técnica o tecnológica completa',
    'Educación profesional incompleta',
    'Educación profesional completa',
    'Postgrado',
    'No sabe',
    'No Aplica'
]

df["fami_educacionmadre"] = pd.Categorical(
    df["fami_educacionmadre"],
    categories=orden,
    ordered=True
)

df["fami_educacionpadre"] = pd.Categorical(
    df["fami_educacionpadre"],
    categories=orden,
    ordered=True
)

# Estimación de modelo de regresión

modelo = smf.ols("""
punt_global ~ C(cole_naturaleza, Treatment(reference="OFICIAL")) +
C(cole_calendario, Treatment(reference="A")) +
C(cole_area_ubicacion, Treatment(reference="URBANO")) +
C(cole_jornada, Treatment(reference="MAÑANA")) + 
C(fami_educacionmadre, Treatment(reference="Secundaria (Bachillerato) completa")) +
C(fami_educacionpadre, Treatment(reference="Secundaria (Bachillerato) completa"))
""", data=df).fit(cov_type="HC3")

conf_int = modelo.conf_int()
conf_int.columns = ["IC_inf", "IC_sup"]

coef_table = pd.concat([
    modelo.params,
    modelo.pvalues,
    conf_int
], axis=1)

coef_table.columns = [
    "Coeficiente",
    "p-value",
    "IC_inf",
    "IC_sup"
]

coef_table = coef_table.reset_index()

# Grafico para intervalos de confianza para el modelo de regresión
def grafico_intervalos(df_plot, titulo):

    df_plot = df_plot.copy()
    df_plot["significativo"] = df_plot["p-value"] < 0.05

    fig = go.Figure()

    for _, row in df_plot.iterrows():

        color = "#1f77b4" if row["significativo"] else "#B0B0B0"

        fig.add_trace(go.Scatter(
            x=[row["IC_inf"], row["IC_sup"]],
            y=[row["index"], row["index"]],
            mode="lines",
            line=dict(color=color, width=8),
            showlegend=False
        ))

        fig.add_trace(go.Scatter(
            x=[row["Coeficiente"]],
            y=[row["index"]],
            mode="markers",
            marker=dict(color=color, size=12),
            showlegend=False
        ))

    fig.add_vline(x=0, line_dash="dash", line_color="black")

    fig.update_layout(
        title=titulo,
        template="simple_white",
        height=500
    )

    return fig


# Dash
app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([

    html.Div([
        html.H1("Dashboard ICFES – Análisis de Brechas",
                className="header-title")
    ], className="header"),

    html.Div([

        dcc.Dropdown(
            id="pregunta-selector",
            options=[
                {"label": "Brecha Oficial vs No Oficial", "value": "naturaleza"},
                {"label": "Influencia Educación Familiar", "value": "educacion"},
                {"label": "Diferencias por Calendario", "value": "calendario"},
            ],
            value="naturaleza",
            clearable=False,
        ),

        html.Br(),

        dcc.Dropdown(
            id="educacion-selector",
            options=[
                {"label": "Educación Madre", "value": "madre"},
                {"label": "Educación Padre", "value": "padre"},
                {"label": "Ambos", "value": "ambos"},
            ],
            value="madre",
            clearable=False,
        ),

        html.Div([

            html.Div(
                dcc.Graph(id="grafico_principal"),
                className="card column"
            ),

            html.Div(
                dcc.Graph(id="intervalos"),
                className="card column"
            ),

        ], className="row"),

        html.Div(
            id="interpretacion",
            className="interpretacion"
        )

    ], className="main-container")

])


# Mostrar filtro educación solo si aplica
@app.callback(
    Output("educacion-selector", "style"),
    Input("pregunta-selector", "value")
)
def mostrar_filtro(pregunta):
    if pregunta == "educacion":
        return {"marginTop": "10px"}
    return {"display": "none"}


# Callback principal para actualizar gráficos e interpretación
@app.callback(
    Output("grafico_principal", "figure"),
    Output("intervalos", "figure"),
    Output("interpretacion", "children"),
    Input("pregunta-selector", "value"),
    Input("educacion-selector", "value")
)
def actualizar_dashboard(pregunta, tipo_edu):

    if pregunta == "naturaleza":

        fig = px.violin(
            df,
            x="cole_naturaleza",
            y="punt_global",
            box=True,
            points="outliers",
            title="Distribución del Puntaje según Naturaleza del Colegio"
        )

        df_plot = coef_table[coef_table["index"].str.contains("cole_naturaleza")]

        fig_ic = grafico_intervalos(df_plot,
                                    "Efecto Ajustado – Naturaleza del Colegio")

        texto = "Comparación respecto a colegios oficiales (categoría base)."


    elif pregunta == "calendario":

        fig = px.violin(
            df,
            x="cole_calendario",
            y="punt_global",
            box=True,
            points="outliers",
            title="Distribución del Puntaje según Calendario"
        )

        df_plot = coef_table[coef_table["index"].str.contains("cole_calendario")]

        fig_ic = grafico_intervalos(df_plot,
                                    "Efecto Ajustado – Calendario Académico")

        texto = "Comparación respecto al calendario A (categoría base)."


    else:

        if tipo_edu == "madre":

            fig = px.box(
                df,
                y="fami_educacionmadre",
                x="punt_global",
                orientation="h",
                title="Puntaje según Educación de la Madre",
                category_orders={"fami_educacionmadre": orden}
            )

        elif tipo_edu == "padre":

            fig = px.box(
                df,
                y="fami_educacionpadre",
                x="punt_global",
                orientation="h",
                title="Puntaje según Educación del Padre",
                category_orders={"fami_educacionpadre": orden}
            )

        else:

            df_melt = df.melt(
                id_vars="punt_global",
                value_vars=["fami_educacionmadre", "fami_educacionpadre"],
                var_name="Tipo",
                value_name="Nivel"
            )

            fig = px.box(
                df_melt,
                y="Nivel",
                x="punt_global",
                orientation="h",
                facet_col="Tipo",
                title="Puntaje según Educación de los Padres",
                category_orders={"Nivel": orden}
            )

        df_plot = coef_table[coef_table["index"].str.contains("fami_educacion")]

        fig_ic = grafico_intervalos(df_plot,
                                    "Efecto Ajustado – Educación Familiar")

        texto = "Comparación respecto a secundaria completa (categoría base)."

    return fig, fig_ic, texto


if __name__ == "__main__":
    app.run(debug=True)