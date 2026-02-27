import dash
from dash import dcc, html, callback_context
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import statsmodels.formula.api as smf
import re

# ==================== LOAD DATA ====================

df = pd.read_csv("../clean_data/filtered_icfes_data.csv")

# ORder education level
orden = [
    'Sin respuesta', 'Ninguno', 'Primaria incompleta', 'Primaria completa',
    'Secundaria (Bachillerato) incompleta', 'Secundaria (Bachillerato) completa',
    'Técnica o tecnológica incompleta', 'Técnica o tecnológica completa',
    'Educación profesional incompleta', 'Educación profesional completa',
    'Postgrado', 'No sabe', 'No Aplica'
]

df["fami_educacionmadre"] = pd.Categorical(df["fami_educacionmadre"], categories=orden, ordered=True)
df["fami_educacionpadre"] = pd.Categorical(df["fami_educacionpadre"], categories=orden, ordered=True)

# ==================== OLS REGRESSION ====================

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

# ==================== TAGS FUNCTIONS ====================

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

# ==================== GRAPH FUNCTIONS ====================

def grafico_forest(df_plot, titulo, variable=None):
    df_plot = df_plot.copy()
    df_plot["significativo"] = df_plot["p-value"] < 0.05

    # Order according to variable type
    if variable in ["fami_educacionmadre", "fami_educacionpadre"]:
        df_plot["nivel"] = df_plot["index"].str.extract(r'\[T\.(.*)\]')
        df_plot["nivel"] = pd.Categorical(df_plot["nivel"], categories=orden, ordered=True)
        df_plot = df_plot.sort_values("nivel")
    else:
        df_plot = df_plot.sort_values("Coeficiente")

    fig = go.Figure()

    # Reference
    fig.add_vrect(x0=-5, x1=5, fillcolor="rgba(200,200,200,0.08)", layer="below", line_width=0)

    # Coefficients of interest
    for _, row in df_plot.iterrows():
        color = "#1f3c88" if row["significativo"] else "#AAAAAA"
        
        # Intervalo de confianza
        fig.add_trace(go.Scatter(
            x=[row["IC_inf"], row["IC_sup"]], y=[row["etiqueta"], row["etiqueta"]],
            mode="lines", line=dict(color=color, width=2.5),
            showlegend=False, hoverinfo="skip"
        ))
        
        # Marcadores extremos
        fig.add_trace(go.Scatter(
            x=[row["IC_inf"], row["IC_sup"]], y=[row["etiqueta"], row["etiqueta"]],
            mode="markers", marker=dict(symbol="line-ns", color=color, size=10),
            showlegend=False, hoverinfo="skip"
        ))
        
        # Puntual 
        fig.add_trace(go.Scatter(
            x=[row["Coeficiente"]], y=[row["etiqueta"]],
            mode="markers", marker=dict(symbol="diamond", color=color, size=10),
            customdata=[[row["IC_inf"], row["IC_sup"], row["p-value"]]],
            hovertemplate=(
                "<b>%{y}</b><br>Coef: %{x:.3f}<br>"
                "IC 95%%: [%{customdata[0][0]:.3f}, %{customdata[0][1]:.3f}]<br>"
                "p-value: %{customdata[0][2]:.4f}<extra></extra>"
            ),
            showlegend=False
        ))

    # Null line
    fig.add_vline(x=0, line_dash="dash", line_color="#333333", line_width=1.2)

    # Legend
    for col, label in [("#1f3c88", "Significativo (p < 0.05)"), ("#AAAAAA", "No significativo")]:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(symbol="diamond", color=col, size=10),
            name=label, showlegend=True
        ))

    fig.update_layout(
        title=dict(text=titulo, font=dict(size=14, color="#1f3c88", family="Poppins")),
        template="simple_white", height=400, margin=dict(l=10, r=20, t=50, b=30),
        legend=dict(orientation="v", yanchor="top", y=0.98, xanchor="left", x=0.02, bgcolor="rgba(255,255,255,0.8)"),
        xaxis=dict(title="Efecto sobre puntaje global", zeroline=False),
        yaxis=dict(title="", tickfont=dict(size=11))
    )
    return fig

def grafico_tabla(df_plot):
    df_plot = df_plot.copy()
    df_plot["Significativo"] = df_plot["p-value"].apply(lambda x: "Sí" if x < 0.05 else "No")

    fig = go.Figure(data=[go.Table(
        columnwidth=[3, 1.2, 1.2, 1.2, 1.2, 1],
        header=dict(
            values=["Variable", "Coef.", "p-value", "IC inf", "IC sup", "Sig."],
            fill_color="#1f3c88", align="center",
            font=dict(size=13, color="white", family="Poppins"), height=36
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
            align="center", font=dict(size=12, family="Poppins"), height=32,
            fill_color=[["#f0f6ff" if i % 2 == 0 else "white" for i in range(len(df_plot))]]
        )
    )])

    fig.update_layout(height=400, margin=dict(l=10, r=10, t=10, b=10))
    return fig

def generar_interpretacion(df_plot, contexto):
    df_sig = df_plot[df_plot["p-value"] < 0.05].copy()
    df_sig = df_sig.sort_values("Coeficiente", key=abs, ascending=False)

    if len(df_sig) == 0:
        return html.Div([
            html.P("No se encontraron efectos estadísticamente significativos (p < 0.05).",
                   style={"color": "#666", "fontSize": "13px", "textAlign": "center", "marginTop": "20px"})
        ])

    items = []
    for _, row in df_sig.iterrows():
        coef = row["Coeficiente"]
        etq = row["etiqueta"]
        magnitud = abs(round(coef, 2))
        signo = f"+{magnitud}" if coef > 0 else f"−{magnitud}"
        
        color_bg = "#e8f4f8" if coef > 0 else "#fef0f0"
        color_text = "#1f3c88" if coef > 0 else "#c0392b"

        items.append(html.Div([
            html.Span(signo, style={
                "fontWeight": "700", "fontSize": "16px", "color": color_text, 
                "display": "block", "marginBottom": "5px"
            }),
            html.Span(etq, style={"fontSize": "12px", "color": "#333"})
        ], style={
            "textAlign": "center", "padding": "14px", "borderRadius": "8px",
            "backgroundColor": color_bg, "marginBottom": "8px",
            "borderLeft": f"4px solid {color_text}"
        }))

    return html.Div([
        html.H4(f"Efectos Significativos (coef.) ({len(df_sig)})", 
                style={"color": "#1f3c88", "fontSize": "14px", "marginBottom": "12px", "textAlign": "center"}),
        *items,
        html.P("* Efectos estimados con las demás variables constantes.",
               style={"fontSize": "10px", "color": "#999", "marginTop": "12px", 
                      "textAlign": "center", "fontStyle": "italic"})
    ])

def grafico_desglose_materias(variable, titulo):
    """Crea gráfico de barras comparando puntajes por materia según la variable especificada"""
    materias = ['punt_ingles', 'punt_matematicas', 'punt_sociales_ciudadanas', 
                'punt_c_naturales', 'punt_lectura_critica']
    nombres_materias = ['Inglés', 'Matemáticas', 'Sociales y Ciudadanas', 
                        'Ciencias Naturales', 'Lectura Crítica']
    
    # AVG scores of category and subject
    datos = []
    for materia, nombre in zip(materias, nombres_materias):
        for categoria in df[variable].unique():
            if pd.notna(categoria):
                promedio = df[df[variable] == categoria][materia].mean()
                datos.append({
                    'Materia': nombre,
                    'Categoría': str(categoria),
                    'Promedio': promedio
                })
    
    df_materias = pd.DataFrame(datos)
    
    fig = px.bar(
        df_materias,
        x='Materia',
        y='Promedio',
        color='Categoría',
        barmode='group',
        title=titulo,
        color_discrete_sequence=['#1f3c88', '#4a90d9', '#a0c4ff'],
        labels={'Promedio': 'Puntaje Promedio', 'Materia': 'Área de Conocimiento'}
    )
    
    fig.update_layout(
        template="simple_white",
        height=400,
        font=dict(family="Poppins", size=12),
        legend=dict(title="", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(tickangle=-15)
    )
    
    return fig

# Pie plot student distr
def crear_distribucion_estudiantes(variable):
    conteo = df[variable].value_counts()
    
    fig = go.Figure(data=[go.Pie(
        labels=conteo.index,
        values=conteo.values,
        hole=0.4,
        marker=dict(colors=['#1f3c88', '#4a90d9', '#a0c4ff', '#7cb9e8', '#c8e0f4']),
        textinfo='percent+label',
        textposition='auto',
        hovertemplate='%{label}<br>%{value:,} estudiantes<br>%{percent}<extra></extra>'
    )])
    
    fig.update_layout(
        template="simple_white",
        height=300,
        margin=dict(l=10, r=10, t=30, b=10),
        font=dict(family="Poppins", size=11),
        showlegend=False
    )
    
    return fig

# insight card
def crear_hallazgo_clave(texto, icono="💡"):
    return html.Div([
        html.Div([
            html.Span(icono, className="key-finding-icon"),
            html.Span("¡Hallazgos!", className="key-finding-title")
        ]),
        html.P(texto, className="key-finding-text")
    ], className="key-finding")

# ==================== DASH APP ====================

app = dash.Dash(__name__)
app.title = "MinEducación"
server = app.server

app.layout = html.Div([
    
    # Header
    html.Div([
        html.Div([
            html.Button(
                "☰", id="menu-toggle-btn",
                style={
                    "display": "none",
                    "backgroundColor": "transparent",
                    "border": "none",
                    "fontSize": "28px",
                    "color": "#1f3c88",
                    "cursor": "pointer",
                    "marginRight": "15px"
                },
                className="menu-toggle-button"
            ),
            html.Img(src="/assets/images/min_educacion_logo.png", 
                     style={"height": "80px", "marginRight": "30px", "borderRadius": "50%"}),
            html.Div([
                html.H1("Análisis Pruebas Saber 11", 
                        style={"margin": "0", "color": "#1f3c88", "fontSize": "28px", 
                               "fontWeight": "600", "fontFamily": "Poppins"}),
                html.P("Departamento del Cesar, Colombia", 
                       style={"margin": "5px 0 0 0", "color": "#4a6fa5", "fontSize": "16px", 
                              "fontFamily": "Poppins"}),
                html.P("Análisis estadístico del rendimiento académico en las pruebas saber 11", 
                       style={"margin": "8px 0 0 0", "color": "#666", "fontSize": "13px", 
                              "fontFamily": "Poppins", "maxWidth": "600px"})
            ], style={"flex": "1"})
        ], style={"display": "flex", "alignItems": "center", "maxWidth": "1400px", 
                  "margin": "0 auto", "padding": "25px 30px", "width": "100%"})
    ], className="dashboard-header"),

    # Main container
    html.Div([
        
        # Side nav panel
        html.Div([
            html.Div([
                html.Button(
                    "✕", id="menu-close-btn",
                    style={
                        "display": "none",
                        "backgroundColor": "transparent",
                        "border": "none",
                        "fontSize": "24px",
                        "color": "#1f3c88",
                        "cursor": "pointer",
                        "marginBottom": "15px"
                    },
                    className="menu-close-button"
                )
            ], style={"width": "100%"}),
            html.H3("Secciones", style={"color": "#1f3c88", "fontSize": "16px", "marginBottom": "20px", 
                                         "fontFamily": "Poppins", "fontWeight": "600"}), 
            
            dcc.RadioItems(
                id="seccion-selector",
                options=[
                    {"label": "Resumen General", "value": "resumen"},
                    {"label": "Colegios Oficiales vs No Oficiales", "value": "naturaleza"},
                    {"label": "Calendario Académico", "value": "calendario"},
                    {"label": "Influencia Educación Familiar", "value": "educacion"}
                ],
                value="resumen",
                labelStyle={"display": "block", "marginBottom": "14px", "cursor": "pointer", 
                           "fontSize": "13px", "fontFamily": "Poppins"},
                inputStyle={"marginRight": "10px"}
            ),
            
            # Education filter
            html.Div(id="educacion-container", children=[
                html.Hr(style={"margin": "20px 0", "border": "none", "borderTop": "1px solid #ddd"}),
                html.P("Desglose:", style={"fontSize": "12px", "color": "#666", "marginBottom": "10px", 
                                            "fontFamily": "Poppins", "fontWeight": "500"}),
                dcc.RadioItems(
                    id="educacion-selector",
                    options=[
                        {"label": "Educación Madre", "value": "madre"},
                        {"label": "Educación Padre", "value": "padre"},
                        {"label": "Ambos", "value": "ambos"}
                    ],
                    value="madre",
                    labelStyle={"display": "block", "marginBottom": "10px", "fontSize": "12px", 
                               "fontFamily": "Poppins"},
                    inputStyle={"marginRight": "8px"}
                )
            ], style={"display": "none"}),
            
            # Stats view filter 
            html.Div(id="vista-container", children=[
                html.Hr(style={"margin": "20px 0", "border": "none", "borderTop": "1px solid #ddd"}),
                html.P("Vista Estadística:", style={"fontSize": "12px", "color": "#666", "marginBottom": "10px", 
                                                     "fontFamily": "Poppins", "fontWeight": "500"}),
                dcc.RadioItems(
                    id="vista-selector",
                    options=[
                        {"label": "Forest Plot", "value": "forest"},
                        {"label": "Tabla", "value": "tabla"},
                        {"label": "Resumen", "value": "resumen"}
                    ],
                    value="forest",
                    labelStyle={"display": "block", "marginBottom": "10px", "fontSize": "12px", 
                               "fontFamily": "Poppins"},
                    inputStyle={"marginRight": "8px"}
                )
            ], style={"display": "none"})
            
        ], id="sidebar-nav", style={
            "width": "240px", "backgroundColor": "white", "padding": "25px 20px",
            "boxShadow": "2px 0 10px rgba(0,0,0,0.05)", "minHeight": "calc(100vh - 150px)",
            "position": "sticky", "top": "0", "overflowY": "auto"
        }),

        # Main container
        html.Div(id="contenido-principal", style={
            "flex": "1", "padding": "30px", "backgroundColor": "#f4f7fb", "minWidth": "0"
        })

    ], id="main-container", style={"display": "flex", "width": "100%"})

])

# ==================== CALLBACKS ====================

@app.callback(
    Output("sidebar-nav", "style"),
    Input("menu-toggle-btn", "n_clicks"),
    Input("menu-close-btn", "n_clicks"),
    State("sidebar-nav", "style"),
    prevent_initial_call=True
)
def toggle_sidebar(toggle_clicks, close_clicks, current_style):
    ctx = callback_context
    
    # Default styles for desktop (sidebar visible)
    sidebar_visible = {
        "width": "240px", "backgroundColor": "white", "padding": "25px 20px",
        "boxShadow": "2px 0 10px rgba(0,0,0,0.05)", "minHeight": "calc(100vh - 150px)",
        "position": "sticky", "top": "0", "overflowY": "auto"
    }
    
    # Hidden sidebar (for mobile)
    sidebar_hidden = {
        "width": "240px", "backgroundColor": "white", "padding": "25px 20px",
        "boxShadow": "2px 0 10px rgba(0,0,0,0.05)", "minHeight": "calc(100vh - 150px)",
        "position": "fixed", "top": "0", "left": "-280px", "overflowY": "auto",
        "zIndex": "1000", "transition": "left 0.3s ease"
    }
    
    # Sidebar visible on mobile (slid in from left)
    sidebar_mobile_visible = {
        "width": "240px", "backgroundColor": "white", "padding": "25px 20px",
        "boxShadow": "2px 0 10px rgba(0,0,0,0.05)", "minHeight": "100vh",
        "position": "fixed", "top": "0", "left": "0", "overflowY": "auto",
        "zIndex": "1000", "transition": "left 0.3s ease"
    }
    
    if not ctx.triggered:
        return sidebar_visible
    
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    # If toggle button clicked, show sidebar on mobile
    if button_id == "menu-toggle-btn":
        return sidebar_mobile_visible
    # If close button clicked, hide sidebar on mobile
    elif button_id == "menu-close-btn":
        return sidebar_hidden
    
    return sidebar_visible

@app.callback(
    Output("educacion-container", "style"),
    Output("vista-container", "style"),
    Input("seccion-selector", "value")
)
def mostrar_selectores(seccion):
    if seccion == "educacion":
        return {"display": "block"}, {"display": "none"}
    elif seccion in ["naturaleza", "calendario"]:
        return {"display": "none"}, {"display": "block"}
    return {"display": "none"}, {"display": "none"}

@app.callback(
    Output("contenido-principal", "children"),
    Input("seccion-selector", "value"),
    Input("educacion-selector", "value"),
    Input("vista-selector", "value")
)
def actualizar_contenido(seccion, tipo_edu, vista_estadistica):
    
    # ========== Summary ==========
    if seccion == "resumen":
        n_estudiantes = len(df)
        promedio_global = df['punt_global'].mean()
        
        # Coefficients from regression model (adjusted effects)
        coef_naturaleza = coef_table[coef_table["index"].str.contains("cole_naturaleza")]["Coeficiente"].values
        brecha_naturaleza = coef_naturaleza[0] if len(coef_naturaleza) > 0 else 0
        
        coef_calendario_b = coef_table[coef_table["index"].str.contains(r"cole_calendario.*\[T\.B\]")]["Coeficiente"].values
        brecha_calendario = abs(coef_calendario_b[0]) if len(coef_calendario_b) > 0 else 0
        
        # Family education coefficients (selecting most vs least educated)
        coef_madre = coef_table[coef_table["index"].str.contains("fami_educacionmadre")]["Coeficiente"]
        brecha_educacion = coef_madre.max() - coef_madre.min() if len(coef_madre) > 0 else 0
        
        coef_padre = coef_table[coef_table["index"].str.contains("fami_educacionpadre")]["Coeficiente"]
        brecha_educacion_dad = coef_padre.max() - coef_padre.min() if len(coef_padre) > 0 else 0
        
        return html.Div([
            html.H2("Resumen General del Análisis", 
                    style={"color": "#1f3c88", "fontSize": "22px", "marginBottom": "25px", 
                           "fontFamily": "Poppins"}),
            
            # Main metrics
            html.Div([
                html.Div([
                    html.H3(f"{n_estudiantes:,}", style={"fontSize": "32px", "color": "#1f3c88", 
                                                           "margin": "0", "fontFamily": "Poppins"}),
                    html.P("Estudiantes Analizados", style={"fontSize": "13px", "color": "#666", 
                                                              "margin": "5px 0 0 0"})
                ], className="metric-card", style={"padding": "25px", "borderRadius": "10px", 
                         "boxShadow": "0 2px 8px rgba(0,0,0,0.08)", "textAlign": "center", "flex": "1", "minWidth": "0"}),
                
                html.Div([
                    html.H3(f"{promedio_global:.1f}", style={"fontSize": "32px", "color": "#1f3c88", 
                                                              "margin": "0", "fontFamily": "Poppins"}),
                    html.P("Puntaje Promedio Global", style={"fontSize": "13px", "color": "#666", 
                                                               "margin": "5px 0 0 0"})
                ], className="metric-card", style={"padding": "25px", "borderRadius": "10px", 
                         "boxShadow": "0 2px 8px rgba(0,0,0,0.08)", "textAlign": "center", "flex": "1", "minWidth": "0"}),
                
                html.Div([
                    html.H3(f"{brecha_naturaleza:.1f}", style={"fontSize": "32px", "color": "#c0392b" if brecha_naturaleza > 0 else "#27ae60", 
                                                     "margin": "0", "fontFamily": "Poppins"}),
                    html.P("Brecha Oficial vs Privado", style={"fontSize": "13px", "color": "#666", 
                                                                 "margin": "5px 0 0 0"})
                ], className="metric-card", style={"padding": "25px", "borderRadius": "10px", 
                         "boxShadow": "0 2px 8px rgba(0,0,0,0.08)", "textAlign": "center", "flex": "1", "minWidth": "0"}),
                html.Div([
                    html.H3(f"{round(brecha_calendario,1):,}", style={"fontSize": "32px", "color": "#c0392b" if brecha_calendario > 0 else "#27ae60", 
                                                           "margin": "0", "fontFamily": "Poppins"}),
                    html.P("Brecha Calendario (A vs B)", style={"fontSize": "13px", "color": "#666", 
                                                              "margin": "5px 0 0 0"})
                ], className="metric-card", style={"padding": "25px", "borderRadius": "10px", 
                         "boxShadow": "0 2px 8px rgba(0,0,0,0.08)", "textAlign": "center", "flex": "1", "minWidth": "0"})
                
                
                # html.Div([
                #     html.H3(f"{brecha_educacion:.1f}", style={"fontSize": "32px", "color": "#c0392b" if brecha_educacion > 0 else "#27ae60", 
                #                                               "margin": "0", "fontFamily": "Poppins"}),
                #     html.P("Brecha Educación Madre", style={"fontSize": "13px", "color": "#666", 
                #                                                "margin": "5px 0 0 0"})
                # ], className="metric-card", style={"padding": "25px", "borderRadius": "10px", 
                #          "boxShadow": "0 2px 8px rgba(0,0,0,0.08)", "textAlign": "center", "flex": "1", "minWidth": "0"}),
                
                # html.Div([
                #     html.H3(f"{brecha_educacion_dad:.1f}", style={"fontSize": "32px", "color": "#c0392b" if brecha_educacion_dad > 0 else "#27ae60", 
                #                                      "margin": "0", "fontFamily": "Poppins"}),
                #     html.P("Brecha Educación Padre", style={"fontSize": "13px", "color": "#666", 
                #                                                  "margin": "5px 0 0 0"})
                # ], className="metric-card", style={"padding": "25px", "borderRadius": "10px", 
                #          "boxShadow": "0 2px 8px rgba(0,0,0,0.08)", "textAlign": "center", "flex": "1", "minWidth": "0"})
                
            ], style={"display": "flex", "gap": "20px", "marginBottom": "30px", "flexWrap": "wrap"}),
            
            # Analysis description
            html.Div([
                html.H3("Objetivo del Análisis", style={"color": "#1f3c88", "fontSize": "16px", 
                                                         "marginBottom": "12px", "fontFamily": "Poppins"}),
                html.P([
                    "Este tablero presenta un análisis estadístico del rendimiento de estudiantes del ",
                    html.B("Departamento del Cesar"), " en las pruebas Saber 11. ",
                    "Se examinan tres factores clave:"
                ], style={"fontSize": "13px", "lineHeight": "1.7", "color": "#333", "marginBottom": "15px"}),
                
                html.Ul([
                    html.Li([html.B("Naturaleza del colegio:"), " Diferencias entre colegios oficiales y no oficiales"], 
                            style={"marginBottom": "8px"}),
                    html.Li([html.B("Calendario académico:"), " Impacto del calendario A, B"], 
                            style={"marginBottom": "8px"}),
                    html.Li([html.B("Educación familiar:"), " Influencia del nivel educativo de los padres"], 
                            style={"marginBottom": "8px"})
                ], style={"fontSize": "13px", "lineHeight": "1.7", "color": "#333", "paddingLeft": "20px"}),
                
                html.P([
                    "Los análisis utilizan ", html.B("regresión OLS"), 
                    " para estimar el efecto de cada factor."
                ], style={"fontSize": "13px", "lineHeight": "1.7", "color": "#333", "marginTop": "15px"})
                
            ], style={"backgroundColor": "white", "padding": "25px", "borderRadius": "10px", 
                     "boxShadow": "0 2px 8px rgba(0,0,0,0.08)"})
        ])
    
    # ========== School type ==========
    elif seccion == "naturaleza":
        fig = px.violin(
            df, x="cole_naturaleza", y="punt_global", box=True, points="outliers",
            title="Distribución del Puntaje Global por Naturaleza del Colegio",
            color="cole_naturaleza", color_discrete_sequence=["#1f3c88", "#4a90d9"],
            labels={"cole_naturaleza": "Naturaleza del Colegio", "punt_global": "Puntaje Global"}
        )
        fig.update_layout(template="simple_white", showlegend=False, height=400, 
                         font=dict(family="Poppins"))

        df_plot = coef_table[coef_table["index"].str.contains("cole_naturaleza")]
        
        # Get coefficient (adjusted effect)
        coef_naturaleza = df_plot["Coeficiente"].values[0] if len(df_plot) > 0 else 0
        pct_oficial = (df['cole_naturaleza'] == 'OFICIAL').sum() / len(df) * 100
        
        hallazgo_text = (f"El efecto estimado de asistir a un colegio no oficial (privado) es de {coef_naturaleza:+.1f} puntos "
                        f"en el puntaje global. "
                        f"Los colegios oficiales representan el {pct_oficial:.1f}% de la muestra.")
        
        # Select view
        if vista_estadistica == "forest":
            fig_estadistica = grafico_forest(df_plot, "Efectos Estimados - Naturaleza del Colegio")
            vista_grafico = dcc.Graph(figure=fig_estadistica)
        elif vista_estadistica == "tabla":
            fig_estadistica = grafico_tabla(df_plot)
            vista_grafico = dcc.Graph(figure=fig_estadistica)
        else:  # summary
            vista_grafico = generar_interpretacion(df_plot, "naturaleza")
        
        # Subjects
        fig_materias = grafico_desglose_materias("cole_naturaleza", "Comparación por Área de Conocimiento")
        
        # Dist
        fig_dist = crear_distribucion_estudiantes("cole_naturaleza")

        return html.Div([
            html.H2("Colegios Oficiales vs No Oficiales", 
                    style={"color": "#1f3c88", "fontSize": "22px", "marginBottom": "20px", 
                           "fontFamily": "Poppins"}),
            
            # Insight
            crear_hallazgo_clave(hallazgo_text, "📊"),
            
            # Top row
            html.Div([
                html.Div(dcc.Graph(figure=fig), style={"flex": "2", "backgroundColor": "white", 
                        "padding": "15px", "borderRadius": "10px", "boxShadow": "0 2px 8px rgba(0,0,0,0.08)"}),
                html.Div(vista_grafico, style={"flex": "1.5", "backgroundColor": "white", "padding": "20px", 
                        "borderRadius": "10px", "boxShadow": "0 2px 8px rgba(0,0,0,0.08)",
                        "maxHeight": "440px", "overflowY": "auto"}),
                html.Div([
                    html.P("Distribución de Estudiantes", className="distribution-title"),
                    dcc.Graph(figure=fig_dist, config={"displayModeBar": False})
                ], style={"flex": "1", "backgroundColor": "white", "padding": "15px", 
                         "borderRadius": "10px", "boxShadow": "0 2px 8px rgba(0,0,0,0.08)"})
            ], style={"display": "flex", "gap": "15px", "marginBottom": "20px"}),
            
            # bottom row
            html.Div([
                html.Div(dcc.Graph(figure=fig_materias), style={"backgroundColor": "white", 
                        "padding": "15px", "borderRadius": "10px", "boxShadow": "0 2px 8px rgba(0,0,0,0.08)"})
            ])
        ])
    
    # ========== Calendar ==========
    elif seccion == "calendario":
        fig = px.violin(
            df, x="cole_calendario", y="punt_global", box=True, points="outliers",
            title="Distribución del Puntaje Global por Calendario Académico",
            color="cole_calendario", color_discrete_sequence=["#1f3c88", "#4a90d9", "#a0c4ff"],
            labels={"cole_calendario": "Calendario Académico", "punt_global": "Puntaje Global"}
        )
        fig.update_layout(template="simple_white", showlegend=False, height=400, 
                         font=dict(family="Poppins"))

        df_plot = coef_table[coef_table["index"].str.contains("cole_calendario")]
        
        # Get coefficient for calendar B (adjusted effect vs calendar A)
        coef_cal_b = df_plot[df_plot["index"].str.contains(r"\[T\.B\]")]["Coeficiente"].values
        efecto_cal_b = coef_cal_b[0] if len(coef_cal_b) > 0 else 0
        pct_cal_a = (df['cole_calendario'] == 'A').sum() / len(df) * 100
        
        hallazgo_text = (f"El calendario A representa el {pct_cal_a:.1f}% de los estudiantes. "
                        f"El efecto estimado del calendario B es de {efecto_cal_b:+.1f} puntos "
                        f"respecto al calendario A.")
        
        if vista_estadistica == "forest":
            fig_estadistica = grafico_forest(df_plot, "Efectos Estimados - Calendario Académico")
            vista_grafico = dcc.Graph(figure=fig_estadistica)
        elif vista_estadistica == "tabla":
            fig_estadistica = grafico_tabla(df_plot)
            vista_grafico = dcc.Graph(figure=fig_estadistica)
        else:  
            vista_grafico = generar_interpretacion(df_plot, "calendario")
        
        fig_materias = grafico_desglose_materias("cole_calendario", "Comparación por Área de Conocimiento")
        
        fig_dist = crear_distribucion_estudiantes("cole_calendario")

        return html.Div([
            html.H2("Diferencias por Calendario Académico", 
                    style={"color": "#1f3c88", "fontSize": "22px", "marginBottom": "20px", 
                           "fontFamily": "Poppins"}),
            
            crear_hallazgo_clave(hallazgo_text, "📅"),
            
            html.Div([
                html.Div(dcc.Graph(figure=fig), style={"flex": "2", "backgroundColor": "white", 
                        "padding": "15px", "borderRadius": "10px", "boxShadow": "0 2px 8px rgba(0,0,0,0.08)"}),
                html.Div(vista_grafico, style={"flex": "1.5", "backgroundColor": "white", "padding": "20px", 
                        "borderRadius": "10px", "boxShadow": "0 2px 8px rgba(0,0,0,0.08)",
                        "maxHeight": "440px", "overflowY": "auto"}),
                html.Div([
                    html.P("Distribución de Estudiantes", className="distribution-title"),
                    dcc.Graph(figure=fig_dist, config={"displayModeBar": False})
                ], style={"flex": "1", "backgroundColor": "white", "padding": "15px", 
                         "borderRadius": "10px", "boxShadow": "0 2px 8px rgba(0,0,0,0.08)"})
            ], style={"display": "flex", "gap": "15px", "marginBottom": "20px"}),
            
            html.Div([
                html.Div(dcc.Graph(figure=fig_materias), style={"backgroundColor": "white", 
                        "padding": "15px", "borderRadius": "10px", "boxShadow": "0 2px 8px rgba(0,0,0,0.08)"})
            ])
        ])
    
    # ========== Family education ==========
    else:  # education
        if tipo_edu == "madre":
            fig = px.box(
                df, y="fami_educacionmadre", x="punt_global", orientation="h",
                category_orders={"fami_educacionmadre": orden},
                title="Puntaje Global según Nivel Educativo de la Madre",
                color_discrete_sequence=["#1f3c88"],
                labels={"fami_educacionmadre": "Nivel Educativo Madre", "punt_global": "Puntaje Global"}
            )
            df_plot = coef_table[coef_table["index"].str.contains("fami_educacionmadre")]
            fig_forest = grafico_forest(df_plot, "Efectos Estimados - Educación Madre", 
                                       variable="fami_educacionmadre")
            
            # Get coefficient range (max - min effect)
            if len(df_plot) > 0:
                coef_max = df_plot["Coeficiente"].max()
                coef_min = df_plot["Coeficiente"].min()
                rango_efecto = coef_max - coef_min
            else:
                rango_efecto = 0
            
            hallazgo_text = (f"La educación materna muestra efectos significativos en el rendimiento. "
                           f"El rango de efectos estimados entre el nivel más alto y más bajo de educación materna "
                           f"es de {rango_efecto:.1f} puntos.")
            
        elif tipo_edu == "padre":
            fig = px.box(
                df, y="fami_educacionpadre", x="punt_global", orientation="h",
                category_orders={"fami_educacionpadre": orden},
                title="Puntaje Global según Nivel Educativo del Padre",
                color_discrete_sequence=["#4a90d9"],
                labels={"fami_educacionpadre": "Nivel Educativo Padre", "punt_global": "Puntaje Global"}
            )
            df_plot = coef_table[coef_table["index"].str.contains("fami_educacionpadre")]
            fig_forest = grafico_forest(df_plot, "Efectos Estimados - Educación Padre", 
                                       variable="fami_educacionpadre")
            
            # Get coefficient range (max - min effect)
            if len(df_plot) > 0:
                coef_max = df_plot["Coeficiente"].max()
                coef_min = df_plot["Coeficiente"].min()
                rango_efecto = coef_max - coef_min
            else:
                rango_efecto = 0
            
            hallazgo_text = (f"La educación paterna también muestra efectos determinantes. "
                           f"El rango de efectos estimados entre el nivel más alto y más bajo de educación paterna "
                           f"es de {rango_efecto:.1f} puntos.")
            
        else:  # both
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
                df_melt, y="Nivel", x="punt_global", orientation="h", facet_col="Tipo",
                category_orders={"Nivel": orden, "Tipo": ["Madre", "Padre"]},
                title="Puntaje Global según Nivel Educativo de los Padres",
                color="Tipo", color_discrete_map={"Madre": "#1f3c88", "Padre": "#4a90d9"},
                labels={"Nivel": "Nivel Educativo", "punt_global": "Puntaje Global"}
            )
            fig.update_layout(showlegend=False, autosize=True)
            df_plot = coef_table[coef_table["index"].str.contains("fami_educacion")]
            fig_forest = grafico_forest(df_plot, "Efectos Estimados - Educación Familiar", 
                                       variable="fami_educacionmadre")
            
            # Get coefficient range for both parents
            if len(df_plot) > 0:
                coef_max = df_plot["Coeficiente"].max()
                coef_min = df_plot["Coeficiente"].min()
                rango_efecto = coef_max - coef_min
            else:
                rango_efecto = 0
            
            hallazgo_text = (f"El nivel educativo de ambos padres influye significativamente en el desempeño. "
                           f"El rango de efectos estimados para educación familiar es de {rango_efecto:.1f} puntos, "
                           f"evidenciando la importancia de programas de apoyo para familias con menor nivel educativo.")

        fig.update_layout(template="simple_white", height=400, font=dict(family="Poppins"))
        tabla = grafico_tabla(df_plot)
        interp = generar_interpretacion(df_plot, "educacion")

        return html.Div([
            html.H2("Influencia de la Educación Familiar", 
                    style={"color": "#1f3c88", "fontSize": "22px", "marginBottom": "20px", 
                           "fontFamily": "Poppins"}),
            
            crear_hallazgo_clave(hallazgo_text, "🎓"),
            
            html.Div([
                html.Div(dcc.Graph(figure=fig, config={"responsive": True}), style={"flex": "1", "backgroundColor": "white", 
                        "padding": "15px", "borderRadius": "10px", "boxShadow": "0 2px 8px rgba(0,0,0,0.08)", 
                        "minWidth": "0", "overflow": "hidden"}),
                html.Div(interp, style={"flex": "1", "backgroundColor": "white", "padding": "20px", 
                        "borderRadius": "10px", "boxShadow": "0 2px 8px rgba(0,0,0,0.08)", 
                        "maxHeight": "440px", "overflowY": "auto", "minWidth": "0"})
            ], style={"display": "flex", "gap": "20px", "marginBottom": "20px"}),
            
            html.Div([
                html.Div(dcc.Graph(figure=fig_forest, config={"responsive": True}), style={"flex": "1", "backgroundColor": "white", 
                        "padding": "15px", "borderRadius": "10px", "boxShadow": "0 2px 8px rgba(0,0,0,0.08)", 
                        "minWidth": "0", "overflow": "hidden"}),
                html.Div(dcc.Graph(figure=tabla, config={"responsive": True}), style={"flex": "1", "backgroundColor": "white", 
                        "padding": "15px", "borderRadius": "10px", "boxShadow": "0 2px 8px rgba(0,0,0,0.08)", 
                        "minWidth": "0", "overflow": "hidden"})
            ], style={"display": "flex", "gap": "20px"})
        ])

if __name__ == "__main__":
    app.run(debug=True)
