
import dash
from dash import dcc, html
import pandas as pd

app = dash.Dash(__name__)
server = app.server

df = pd.read_csv(r"C:\Users\Asus\Desktop\Clases\Analítica\Proy\icfes_cols_filtered_no_dupes_BOM.csv")

app.layout = html.Div([
    html.H2("Dash ICFES Cesar"),
    html.Div(f"Filas: {df.shape[0]} | Columnas: {df.shape[1]}")
])

if __name__ == "__main__":
    app.run(debug=True)

