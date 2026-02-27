import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import statsmodels.formula.api as smf
import re

# Cargar datos
df = pd.read_csv(r"C:\Users\manes\Downloads\P1-Analitica\clean_data\filtered_icfes_data.csv")

# Ordenar niveles educativos para gráficas y tablas
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
    df["fami_educacionmadre"], categories=orden, ordered=True
)
df["fami_educacionpadre"] = pd.Categorical(
    df["fami_educacionpadre"], categories=orden, ordered=True
)

# Modelo de regresión
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

coef_table = pd.concat([modelo.params, modelo.pvalues, conf_int], axis=1)
coef_table.columns = ["Coeficiente", "p-value", "IC_inf", "IC_sup"]
coef_table = coef_table.reset_index()


# Etiquetas legibles
def etiqueta_legible(nombre):
    reemplazos = {
        "Intercept": "Intercepto",
        'C(cole_naturaleza, Treatment(reference="OFICIAL"))[T.NO OFICIAL]': "Colegio No Oficial",
        'C(cole_calendario, Treatment(reference="A"))[T.B]': "Calendario B",
        'C(cole_calendario, Treatment(reference="A"))[T.OTRO]': "Calendario Otro",
        'C(cole_area_ubicacion, Treatment(reference="URBANO"))[T.RURAL]': "Área Rural",
    }
    if nombre in reemplazos:
        return reemplazos[nombre]
    m = re.match(r'C\(fami_educacion(madre|padre).*\)\[T\.(.*)\]', nombre)
    if m:
        quien = "Madre" if m.group(1) == "madre" else "Padre"
        return f"{quien}: {m.group(2)}"
    m2 = re.match(r'C\(cole_jornada.*\)\[T\.(.*)\]', nombre)
    if m2:
        return f"Jornada {m2.group(1).capitalize()}"
    return nombre


coef_table["etiqueta"] = coef_table["index"].apply(etiqueta_legible)


# Forest plot horizontal
def grafico_forest(df_plot, titulo, variable=None):

    df_plot = df_plot.copy()
    df_plot["significativo"] = df_plot["p-value"] < 0.05

    # Ordenar educación por nivel, resto por coeficiente
    if variable in ["fami_educacionmadre", "fami_educacionpadre"]:
        df_plot["nivel"] = df_plot["index"].str.extract(r'\[T\.(.*)\]')
        df_plot["nivel"] = pd.Categorical(
            df_plot["nivel"], categories=orden, ordered=True
        )
        df_plot = df_plot.sort_values("nivel")
    else:
        df_plot = df_plot.sort_values("Coeficiente")

    fig = go.Figure()

    # Referencia alrededor del 0
    fig.add_vrect(
        x0=-5, x1=5,
        fillcolor="rgba(200,200,200,0.08)",
        layer="below", line_width=0
    )

    # Cada coeficiente con su IC
    for _, row in df_plot.iterrows():
        color      = "#1f3c88" if row["significativo"] else "#AAAAAA"
        fill_color = "#1f3c88" if row["significativo"] else "#CCCCCC"

        # Línea del IC
        fig.add_trace(go.Scatter(
            x=[row["IC_inf"], row["IC_sup"]],
            y=[row["etiqueta"], row["etiqueta"]],
            mode="lines",
            line=dict(color=color, width=2.5),
            showlegend=False,
            hoverinfo="skip"
        ))

        # Marcadores de extremos del IC
        fig.add_trace(go.Scatter(
            x=[row["IC_inf"], row["IC_sup"]],
            y=[row["etiqueta"], row["etiqueta"]],
            mode="markers",
            marker=dict(symbol="line-ns", color=color, size=10,
                        line=dict(width=2, color=color)),
            showlegend=False,
            hoverinfo="skip"
        ))

        # Diamante central = estimación puntual
        fig.add_trace(go.Scatter(
            x=[row["Coeficiente"]],
            y=[row["etiqueta"]],
            mode="markers",
            marker=dict(
                symbol="diamond",
                color=fill_color,
                size=10,
                line=dict(width=1.5, color="white")
            ),
            customdata=[[row["IC_inf"], row["IC_sup"], row["p-value"]]],
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Coef: %{x:.3f}<br>"
                "IC 95%%: [%{customdata[0][0]:.3f}, %{customdata[0][1]:.3f}]<br>"
                "p-value: %{customdata[0][2]:.4f}<extra></extra>"
            ),
            showlegend=False
        ))

    # Línea vertical de nulidad
    fig.add_vline(x=0, line_dash="dash", line_color="#333333", line_width=1.2)

    # Leyenda manual
    for col, label in [("#1f3c88", "Significativo (p < 0.05)"),
                        ("#AAAAAA", "No significativo")]:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(symbol="diamond", color=col, size=10),
            name=label, showlegend=True
        ))

    fig.update_layout(
        title=dict(text=titulo, font=dict(size=14, color="#1f3c88")),
        template="simple_white",
        height=430,
        margin=dict(l=10, r=20, t=55, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, font=dict(size=11)),
        xaxis=dict(title="Efecto sobre puntaje global", zeroline=False),
        yaxis=dict(title="", tickfont=dict(size=11))
    )

    return fig


# Tabla de coeficientes
def grafico_tabla(df_plot):
    df_plot = df_plot.copy()
    df_plot["Significativo"] = df_plot["p-value"].apply(
        lambda x: "Sí" if x < 0.05 else "No"
    )

    fig = go.Figure(data=[go.Table(
        columnwidth=[3, 1.2, 1.2, 1.2, 1.2, 1],
        header=dict(
            values=["Variable", "Coef.", "p-value", "IC inf", "IC sup", "Sig."],
            fill_color="#d6eaff",
            align="center",
            font=dict(size=14, color="#1f3c88"),
            height=38
        ),
        cells=dict(
            values=[
                df_plot["etiqueta"],
                round(df_plot["Coeficiente"], 2),
                df_plot["p-value"].apply(lambda x: f"{x:.4f}" if x >= 0.0001 else f"{x:.2e}"),
                round(df_plot["IC_inf"], 2),
                round(df_plot["IC_sup"], 2),
                df_plot["Significativo"]
            ],
            align="center",
            font=dict(size=13),
            height=34,
            fill_color=[["#f0f6ff" if i % 2 == 0 else "white"
                         for i in range(len(df_plot))]]
        )
    )])

    fig.update_layout(
        height=430,
        margin=dict(l=20, r=20, t=20, b=20)
    )
    return fig


# Interpretación: una frase corta por coeficiente significativo
def generar_interpretacion(df_plot, contexto):

    df_sig = df_plot[df_plot["p-value"] < 0.05].copy()
    df_sig = df_sig.sort_values("Coeficiente", key=abs, ascending=False)

    if len(df_sig) == 0:
        return [
            html.P(
                "Ningún coeficiente resultó estadísticamente significativo (p < 0.05).",
                style={"color": "#888", "fontSize": "14px", "fontStyle": "italic", "textAlign": "center"}
            ),
            html.P(
                "* Efectos estimados manteniendo las demás variables del modelo constantes.",
                style={
                    "fontSize": "11px", "color": "#999999", "fontStyle": "italic",
                    "textAlign": "center", "marginTop": "14px",
                    "borderTop": "1px solid #eeeeee", "paddingTop": "10px"
                }
            )
        ]

    items = []

    for _, row in df_sig.iterrows():
        coef     = row["Coeficiente"]
        etq      = row["etiqueta"]
        magnitud = abs(round(coef, 2))
        signo    = f"+{magnitud}" if coef > 0 else f"−{magnitud}"

        # Frase corta según contexto
        if contexto == "naturaleza":
            frase = f"Colegio No Oficial: {signo} pts vs. Oficial"
        elif contexto == "calendario":
            frase = f"{etq}: {signo} pts vs. Calendario A"
        else:
            nivel = etq.split(': ')[-1] if ': ' in etq else etq
            quien = etq.split(':')[0]   if ': ' in etq else ""
            frase = f"{quien} con '{nivel}': {signo} pts vs. bachillerato completo"

        color_bg   = "#eef4ff" if coef > 0 else "#fff0f0"
        color_bord = "#1f3c88" if coef > 0 else "#c0392b"
        color_text = "#1f3c88" if coef > 0 else "#c0392b"

        items.append(
            html.Div([
                html.Span(signo, style={
                    "fontWeight": "700",
                    "fontSize": "14px",
                    "color": color_text,
                    "display": "block",
                    "marginBottom": "4px"
                }),
                html.Span(frase, style={
                    "fontSize": "13px",
                    "color": "#2c3e50",
                    "lineHeight": "1.5"
                })
            ], style={
                "textAlign": "center",
                "padding": "12px 16px",
                "borderRadius": "8px",
                "backgroundColor": color_bg,
                "borderTop": f"3px solid {color_bord}",
                "marginBottom": "8px"
            })
        )

    n = len(df_sig)
    encabezado = html.P(
        f"{n} efecto{'s' if n > 1 else ''} significativo{'s' if n > 1 else ''} (p < 0.05):",
        style={"fontWeight": "600", "color": "#1f3c88",
               "fontSize": "13px", "marginBottom": "10px", "textAlign": "center"}
    )

    nota = html.P(
        "* Efectos estimados manteniendo las demás variables del modelo constantes.",
        style={
            "fontSize": "11px",
            "color": "#999999",
            "fontStyle": "italic",
            "textAlign": "center",
            "marginTop": "14px",
            "marginBottom": "0",
            "borderTop": "1px solid #eeeeee",
            "paddingTop": "10px"
        }
    )

    return [encabezado] + items + [nota]


# Dash
app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([

    # Header
    html.Div([
        html.H1("Pruebas saber 11 - Departamento de Cesar, Colombia",
                className="header-title"),
        html.H2("Nicolás Salazar, Juan Camilo Medina y María Inés Velásquez",
                className="header-subtitle"),
        html.P("Analítica del puntaje global en la prueba saber 11, con énfasis en la influencia de la naturaleza del colegio, el calendario académico y la educación familiar.",
               className="header-desc")
    ], className="header"),

    html.Div([

        # Filtros superiores
        html.Div([
            dcc.Dropdown(
                id="pregunta-selector",
                options=[
                    {"label": "Brecha Oficial vs No Oficial",  "value": "naturaleza"},
                    {"label": "Influencia Educación Familiar", "value": "educacion"},
                    {"label": "Diferencias por Calendario",    "value": "calendario"},
                ],
                value="naturaleza",
                clearable=False,
                style={"minWidth": "260px"}
            ),
            dcc.Dropdown(
                id="educacion-selector",
                options=[
                    {"label": "Educación Madre", "value": "madre"},
                    {"label": "Educación Padre", "value": "padre"},
                    {"label": "Ambos",           "value": "ambos"},
                ],
                value="madre",
                clearable=False,
                style={"minWidth": "220px"}
            ),
        ], style={
            "display": "flex", "gap": "16px",
            "alignItems": "center", "flexWrap": "wrap",
            "marginBottom": "20px"
        }),

        # Box/Violin y tabla de coeficientes
        html.Div([
            html.Div(
                dcc.Graph(id="grafico_principal"),
                className="card",
                style={"flex": "1", "minWidth": "0"}
            ),
            html.Div(
                dcc.Graph(id="tabla_coef"),
                className="card",
                style={"flex": "1", "minWidth": "0"}
            ),
        ], style={"display": "flex", "gap": "25px", "marginBottom": "25px"}),

        # Forest plot e interpretación
        html.Div([
            html.Div(
                dcc.Graph(id="forest_plot"),
                className="card",
                style={"flex": "1", "minWidth": "0"}   # ← antes era 1.3
            ),
            html.Div(
                id="interpretacion",
                className="card",
                style={
                    "flex": "1",                         # ← mismo flex que forest
                    "minWidth": "0",
                    "overflowY": "auto",
                    "maxHeight": "460px",
                    "padding": "20px 22px"
                }
            ),
        ], style={"display": "flex", "gap": "25px", "marginBottom": "30px"}),

    ], className="main-container")

])


# Callbacks
@app.callback(
    Output("educacion-selector", "style"),
    Input("pregunta-selector", "value")
)
def mostrar_filtro(pregunta):
    if pregunta == "educacion":
        return {"minWidth": "220px"}
    return {"display": "none", "minWidth": "220px"}


@app.callback(
    Output("grafico_principal", "figure"),
    Output("forest_plot",       "figure"),
    Output("tabla_coef",        "figure"),
    Output("interpretacion",    "children"),
    Input("pregunta-selector",  "value"),
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
            title="Distribución del puntaje global según la naturaleza del colegio",
            color="cole_naturaleza",
            color_discrete_sequence=["#1f3c88", "#4a90d9"],
            labels={
                "cole_naturaleza": "Naturaleza del colegio",
                "punt_global": "Puntaje global en la prueba"
            }
        )
        fig.update_layout(template="simple_white", showlegend=False, height=430)

        df_plot    = coef_table[coef_table["index"].str.contains("cole_naturaleza")]
        fig_forest = grafico_forest(df_plot, "Forest Plot – Naturaleza del Colegio")
        tabla      = grafico_tabla(df_plot)
        interp     = generar_interpretacion(df_plot, "naturaleza")

    elif pregunta == "calendario":

        fig = px.violin(
            df,
            x="cole_calendario",
            y="punt_global",
            box=True,
            points="outliers",
            title="Distribución del puntaje global en la prueba según el calendario académico",
            color="cole_calendario",
            color_discrete_sequence=["#1f3c88", "#4a90d9", "#a0c4ff"],
            labels={
                "cole_calendario": "Calendario académico del establecimiento",
                "punt_global": "Puntaje global en la prueba"
            }
        )
        fig.update_layout(template="simple_white", showlegend=False, height=430)

        df_plot    = coef_table[coef_table["index"].str.contains("cole_calendario")]
        fig_forest = grafico_forest(df_plot, "Forest Plot – Calendario Académico")
        tabla      = grafico_tabla(df_plot)
        interp     = generar_interpretacion(df_plot, "calendario")

    else:

        if tipo_edu == "madre":
            fig = px.box(
                df,
                y="fami_educacionmadre",
                x="punt_global",
                orientation="h",
                category_orders={"fami_educacionmadre": orden},
                title="Distribución del puntaje global en la prueba según el nivel educativo de la madre",
                color_discrete_sequence=["#1f3c88"],
                labels={
                    "fami_educacionmadre": "Nivel educativo de la madre",
                    "punt_global": "Puntaje global en la prueba"
                }
            )
            df_plot    = coef_table[coef_table["index"].str.contains("fami_educacionmadre")]
            fig_forest = grafico_forest(df_plot, "Forest Plot – Educación Madre",
                                        variable="fami_educacionmadre")

        elif tipo_edu == "padre":
            fig = px.box(
                df,
                y="fami_educacionpadre",
                x="punt_global",
                orientation="h",
                category_orders={"fami_educacionpadre": orden},
                title="Distribución del puntaje global en la prueba según el nivel educativo del padre",
                color_discrete_sequence=["#4a90d9"],
                labels={
                    "fami_educacionpadre": "Nivel educativo del padre",
                    "punt_global": "Puntaje global en la prueba"
                }
            )
            df_plot    = coef_table[coef_table["index"].str.contains("fami_educacionpadre")]
            fig_forest = grafico_forest(df_plot, "Forest Plot – Educación Padre",
                                        variable="fami_educacionpadre")

        else:
            df_melt = df.melt(
                id_vars="punt_global",
                value_vars=["fami_educacionmadre", "fami_educacionpadre"],
                var_name="Tipo", value_name="Nivel"
            )
            df_melt["Tipo"] = df_melt["Tipo"].map({
                "fami_educacionmadre": "Madre",
                "fami_educacionpadre": "Padre"
            })
            fig = px.box(
                df_melt,
                y="Nivel",
                x="punt_global",
                orientation="h",
                facet_col="Tipo",
                facet_col_spacing=0.1,
                category_orders={"Nivel": orden, "Tipo": ["Madre", "Padre"]},
                title="Distribución del puntaje global en la prueba según el nivel educativo de los padres",
                color="Tipo",
                color_discrete_map={"Madre": "#1f3c88", "Padre": "#4a90d9"},
                labels={
                    "Nivel": "Nivel educativo",
                    "punt_global": "Puntaje global en la prueba",
                    "Tipo": "Progenitor"
                }
            )
            fig.for_each_annotation(lambda a: a.update(
                text=f"<b>{a.text.split(chr(61))[-1]}</b>",
                font=dict(size=13, color="#1f3c88")
            ))
            fig.update_layout(showlegend=False)
            df_plot    = coef_table[coef_table["index"].str.contains("fami_educacion")]
            fig_forest = grafico_forest(df_plot, "Forest Plot – Educación Familiar",
                                        variable="fami_educacionmadre")

        fig.update_layout(template="simple_white", height=430)
        tabla  = grafico_tabla(df_plot)
        interp = generar_interpretacion(df_plot, "educacion")

    return fig, fig_forest, tabla, interp


if __name__ == "__main__":
    app.run(debug=True)