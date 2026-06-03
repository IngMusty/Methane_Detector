import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import base64
import io
import json
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from statsmodels.tsa.arima.model import ARIMA

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import cm
import openpyxl
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from openpyxl.chart import LineChart, Reference
from datetime import datetime

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="PFEMS · Methane Monitor",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}]
)
server = app.server

COLOURS = {
    "bg": "#0A0E17", "surface": "#111827", "card": "#161D2E",
    "border": "#1E2D45", "accent": "#00D4AA", "accent2": "#FF6B35",
    "accent3": "#4B9FFF", "warn": "#FFB800", "danger": "#FF3B5C",
    "text": "#E8EDF5", "muted": "#6B7A99", "grid": "#1A2540",
}

CARD_STYLE = {
    "backgroundColor": COLOURS["card"],
    "border": "1px solid " + COLOURS["border"],
    "borderRadius": "12px", "padding": "24px", "marginBottom": "20px",
}

HEADING_STYLE = {
    "fontFamily": "monospace", "color": COLOURS["accent"],
    "fontSize": "11px", "letterSpacing": "3px",
    "textTransform": "uppercase", "marginBottom": "16px",
}

def blank_fig(msg="Upload data to begin"):
    fig = go.Figure()
    fig.add_annotation(text=msg, x=0.5, y=0.5, xref="paper", yref="paper",
                       showarrow=False, font=dict(color=COLOURS["muted"], size=14))
    fig.update_layout(
        paper_bgcolor=COLOURS["card"], plot_bgcolor=COLOURS["card"],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=20, r=20, t=20, b=20),
    )
    return fig

def chart_layout(title="", height=350):
    return dict(
        title=dict(text=title, font=dict(color=COLOURS["muted"], size=12), x=0.01),
        paper_bgcolor=COLOURS["card"], plot_bgcolor=COLOURS["card"],
        font=dict(color=COLOURS["text"]), height=height,
        margin=dict(l=50, r=20, t=45, b=50),
        xaxis=dict(gridcolor=COLOURS["grid"], zeroline=False,
                   tickfont=dict(color=COLOURS["muted"], size=11)),
        yaxis=dict(gridcolor=COLOURS["grid"], zeroline=False,
                   tickfont=dict(color=COLOURS["muted"], size=11)),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=COLOURS["muted"], size=11),
                    orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified",
    )

# ── Core analytics ──────────────────────────────────────────────────────────────
def compute_residual(df, produced_col, flare_col, export_col, fuel_col, injection_col):
    utilisation = df[flare_col] + df[export_col] + df[fuel_col] + df[injection_col]
    return df[produced_col] - utilisation

def apply_methane_fraction(residual, ch4_fraction):
    return residual * ch4_fraction

def run_arima_anomaly(residual, sigma_threshold=3.0):
    try:
        vals = residual.values
        # Fixed order for speed on free tier — no AIC grid search
        model = ARIMA(vals, order=(1, 0, 1))
        result = model.fit(method_kwargs={"warn_convergence": False})
        fitted = result.fittedvalues
        errors = vals - fitted
        std = np.std(errors) or 1.0
        z_scores = errors / std
        anomaly_flags = z_scores > sigma_threshold
        return anomaly_flags, np.abs(errors), z_scores, fitted, (1, 0, 1)
    except Exception:
        n = len(residual)
        return np.zeros(n, bool), np.zeros(n), np.zeros(n), residual.values, (1, 0, 1)

def run_isolation_forest(residual, contamination=0.02):
    try:
        lags = pd.DataFrame({
            "val":  residual.values,
            "lag1": np.roll(residual.values, 1),
            "lag2": np.roll(residual.values, 2),
            "rolling_mean": pd.Series(residual.values).rolling(7, min_periods=1).mean().values,
            "rolling_std":  pd.Series(residual.values).rolling(7, min_periods=1).std().fillna(0).values,
        })
        X = StandardScaler().fit_transform(lags.values)
        iso = IsolationForest(contamination=contamination, random_state=42, n_estimators=100)
        preds = iso.fit_predict(X)
        scores = -iso.score_samples(X)
        return preds == -1, scores
    except Exception:
        return np.zeros(len(residual), bool), np.zeros(len(residual))

def run_autoencoder_anomaly(residual, contamination=0.02, window=14):
    try:
        vals = residual.values
        scaled = StandardScaler().fit_transform(vals.reshape(-1, 1)).flatten()
        n = len(scaled)
        if n <= window:
            return np.zeros(n, bool), np.zeros(n)
        windows = np.array([scaled[i:i+window] for i in range(n - window)])
        pca = PCA(n_components=min(2, window - 1))
        errors = np.mean((windows - pca.inverse_transform(pca.fit_transform(windows))) ** 2, axis=1)
        full_errors = np.concatenate([np.zeros(window), errors])
        threshold = np.percentile(full_errors[window:], (1 - contamination) * 100)
        return full_errors > threshold, full_errors
    except Exception:
        return np.zeros(len(residual), bool), np.zeros(len(residual))

def compute_ensemble(arima_flags, iso_flags, ae_flags, z_scores, min_votes=2):
    vote_count = arima_flags.astype(int) + iso_flags.astype(int) + ae_flags.astype(int)
    ensemble_alert = (vote_count >= min_votes) & (z_scores > 0)
    return ensemble_alert, vote_count

def compute_snr(residual, window=30):
    s = pd.Series(residual)
    signal = s.rolling(window, min_periods=1).mean().abs()
    noise  = s.rolling(window, min_periods=1).std().fillna(1)
    return signal / noise.replace(0, np.nan).fillna(1)

def compute_mdlf(residual, produced, window=90):
    res_s  = pd.Series(residual.values)
    prod_s = pd.Series(produced.values)
    sigma  = res_s.rolling(window, min_periods=window // 3).std()
    mean_p = prod_s.rolling(window, min_periods=window // 3).mean()
    mdlf   = (3 * sigma / mean_p.replace(0, np.nan)) * 100
    return mdlf.ffill().bfill()

def classify_alert(score, snr_val):
    if score > 0.8 and snr_val > 3:   return "CRITICAL", COLOURS["danger"]
    elif score > 0.5 or snr_val > 2:  return "WARNING",  COLOURS["warn"]
    elif score > 0.2:                  return "WATCH",    COLOURS["accent3"]
    else:                              return "NORMAL",   COLOURS["accent"]

def _status_pill(text, colour):
    return html.Span(text, style={"fontFamily": "monospace", "color": colour,
                                   "fontSize": "11px", "letterSpacing": "2px"})

def _kpi_card(label, value, colour):
    return html.Div(style={**CARD_STYLE, "borderTop": "2px solid " + colour,
                           "marginBottom": 0, "textAlign": "center", "padding": "16px"},
                    children=[
        html.Div(str(value), style={"fontFamily": "monospace", "fontSize": "28px",
                                    "fontWeight": "700", "color": colour}),
        html.Div(label, style={"color": COLOURS["muted"], "fontSize": "10px",
                               "letterSpacing": "2px", "marginTop": "6px"}),
    ])

def _anomaly_fig(dates, residual, flags, scores, title):
    from plotly.subplots import make_subplots
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=dates, y=residual, mode="lines", name="Residual",
                              line=dict(color=COLOURS["accent3"], width=1.2), opacity=0.6),
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=dates, y=scores, mode="lines", name="Score",
                              line=dict(color=COLOURS["warn"], width=1.5)),
                  secondary_y=True)
    if flags.any():
        fig.add_trace(go.Scatter(x=dates[flags], y=residual[flags], mode="markers",
                                  name="Flagged",
                                  marker=dict(color=COLOURS["danger"], size=7, symbol="x")),
                      secondary_y=False)
    fig.update_layout(**chart_layout(title, 300))
    fig.update_yaxes(gridcolor=COLOURS["grid"], zeroline=False,
                     tickfont=dict(color=COLOURS["muted"], size=10), secondary_y=False)
    fig.update_yaxes(tickfont=dict(color=COLOURS["warn"], size=10),
                     secondary_y=True, showgrid=False)
    return fig

def _build_alert_table(alerts):
    if not alerts:
        return html.Div(
            "No ensemble-confirmed alerts. Requires at least 2 detectors to agree "
            "on a positive-direction residual deviation.",
            style={"color": COLOURS["muted"], "fontSize": "13px"})
    hs = {"fontFamily": "monospace", "fontSize": "10px", "letterSpacing": "2px",
          "color": COLOURS["muted"], "padding": "10px 12px",
          "borderBottom": "1px solid " + COLOURS["border"],
          "textAlign": "left", "whiteSpace": "nowrap"}
    cs = {"padding": "9px 12px", "fontSize": "12px", "color": COLOURS["text"],
          "borderBottom": "1px solid " + COLOURS["grid"], "whiteSpace": "nowrap"}
    cols = ["Date", "Residual", "CH4 Frac", "Score", "Votes",
            "z-score", "SNR", "ARIMA", "ISO", "AE", "Level"]
    header = html.Tr([html.Th(c, style=hs) for c in cols])
    rows = []
    for a in alerts[-200:]:
        rows.append(html.Tr([
            html.Td(a["Date"],    style=cs),
            html.Td(a["Res"],     style=cs),
            html.Td(a["CH4"],     style=cs),
            html.Td(a["Score"],   style=cs),
            html.Td(a["Votes"],   style=cs),
            html.Td(a["Z"],       style=cs),
            html.Td(a["SNR"],     style=cs),
            html.Td(a["ARIMA"],   style={**cs, "color": COLOURS["danger"] if a["ARIMA"] == "●" else COLOURS["muted"]}),
            html.Td(a["ISO"],     style={**cs, "color": COLOURS["danger"] if a["ISO"]   == "●" else COLOURS["muted"]}),
            html.Td(a["AE"],      style={**cs, "color": COLOURS["danger"] if a["AE"]    == "●" else COLOURS["muted"]}),
            html.Td(a["Level"],   style={**cs, "color": a["colour"],
                                         "fontFamily": "monospace", "fontWeight": "700"}),
        ]))
    return html.Div(style={"overflowX": "auto"}, children=[
        html.Div(f"{len(alerts)} ensemble-confirmed alerts (majority voting, positive z-score)",
                 style={"color": COLOURS["muted"], "fontSize": "12px", "marginBottom": "12px"}),
        html.Table([html.Thead(header), html.Tbody(rows)],
                   style={"width": "100%", "borderCollapse": "collapse"})
    ])

# ── Layout ──────────────────────────────────────────────────────────────────────
app.layout = html.Div(
    style={"backgroundColor": COLOURS["bg"], "minHeight": "100vh"},
    children=[
        # Top bar
        html.Div(style={
            "backgroundColor": COLOURS["surface"],
            "borderBottom": "1px solid " + COLOURS["border"],
            "padding": "0 40px", "display": "flex", "alignItems": "center",
            "justifyContent": "space-between", "height": "64px",
            "position": "sticky", "top": 0, "zIndex": 1000,
        }, children=[
            html.Div([
                html.Span("PFEMS", style={"fontFamily": "monospace", "color": COLOURS["accent"],
                                          "fontSize": "18px", "fontWeight": "700",
                                          "letterSpacing": "2px", "marginRight": "12px"}),
                html.Span("Physics-Guided Fugitive Emissions Monitoring System",
                          style={"color": COLOURS["muted"], "fontSize": "12px"}),
            ], style={"display": "flex", "alignItems": "center"}),
            html.Div(id="status-pill", children=[
                _status_pill("● NO DATA", COLOURS["muted"])
            ]),
        ]),

        html.Div(style={"padding": "32px 40px", "maxWidth": "1600px", "margin": "0 auto"},
                 children=[

            # 01 Field Config
            html.Div(style=CARD_STYLE, children=[
                html.P("01 / FIELD CONFIGURATION", style=HEADING_STYLE),
                dbc.Row([
                    dbc.Col([
                        html.Label("Field / Asset Name",
                                   style={"color": COLOURS["muted"], "fontSize": "12px"}),
                        dcc.Input(id="field-name", type="text",
                                  placeholder="e.g. Jubilee FPSO",
                                  style={"width": "100%", "backgroundColor": COLOURS["surface"],
                                         "border": "1px solid " + COLOURS["border"],
                                         "borderRadius": "6px", "padding": "8px 12px",
                                         "color": COLOURS["text"], "fontSize": "13px"}),
                    ], md=4),
                    dbc.Col([
                        html.Label("Operator / Company",
                                   style={"color": COLOURS["muted"], "fontSize": "12px"}),
                        dcc.Input(id="operator-name", type="text",
                                  placeholder="e.g. Tullow Oil, GNPC",
                                  style={"width": "100%", "backgroundColor": COLOURS["surface"],
                                         "border": "1px solid " + COLOURS["border"],
                                         "borderRadius": "6px", "padding": "8px 12px",
                                         "color": COLOURS["text"], "fontSize": "13px"}),
                    ], md=4),
                    dbc.Col([
                        html.Label("Methane Fraction (CH4) — field-specific",
                                   style={"color": COLOURS["muted"], "fontSize": "12px"}),
                        dcc.Input(id="ch4-fraction", type="number",
                                  placeholder="e.g. 0.77",
                                  min=0.5, max=1.0, step=0.01, value=0.77,
                                  style={"width": "100%", "backgroundColor": COLOURS["surface"],
                                         "border": "1px solid " + COLOURS["border"],
                                         "borderRadius": "6px", "padding": "8px 12px",
                                         "color": COLOURS["text"], "fontSize": "13px"}),
                        html.Div("Range 0.50 to 1.00 · Jubilee default 0.77",
                                 style={"color": COLOURS["muted"], "fontSize": "11px",
                                        "marginTop": "4px"}),
                    ], md=4),
                ], className="g-3"),
            ]),

            # 02 Upload
            html.Div(style=CARD_STYLE, children=[
                html.P("02 / DATA INGESTION", style=HEADING_STYLE),
                dcc.Upload(id="upload-data",
                    children=html.Div([
                        html.Div("↑", style={"fontSize": "32px", "color": COLOURS["accent"],
                                             "marginBottom": "8px"}),
                        html.Div("Drop CSV file here or click to browse",
                                 style={"color": COLOURS["text"], "fontSize": "14px"}),
                        html.Div("Columns: date, produced gas, flaring, export, fuel gas, injection",
                                 style={"color": COLOURS["muted"], "fontSize": "12px"}),
                    ], style={"textAlign": "center", "padding": "20px"}),
                    style={"border": "1px dashed " + COLOURS["border"],
                           "borderRadius": "8px", "cursor": "pointer",
                           "backgroundColor": COLOURS["surface"]},
                    multiple=False),
                html.Div(id="upload-status",
                         style={"marginTop": "12px", "fontSize": "13px",
                                "color": COLOURS["muted"]}),
            ]),

            # 03 Column mapping
            html.Div(id="column-mapping-section", style={"display": "none"}, children=[
                html.Div(style=CARD_STYLE, children=[
                    html.P("03 / COLUMN MAPPING", style=HEADING_STYLE),
                    html.P("Map your CSV columns to PFEMS variables.",
                           style={"color": COLOURS["muted"], "fontSize": "13px",
                                  "marginBottom": "20px"}),
                    dbc.Row([
                        dbc.Col([html.Label("Date", style={"color": COLOURS["muted"], "fontSize": "12px"}),
                                 dcc.Dropdown(id="col-date", placeholder="Select...")], md=2),
                        dbc.Col([html.Label("Produced Gas", style={"color": COLOURS["muted"], "fontSize": "12px"}),
                                 dcc.Dropdown(id="col-produced", placeholder="Select...")], md=2),
                        dbc.Col([html.Label("Gas Flaring", style={"color": COLOURS["muted"], "fontSize": "12px"}),
                                 dcc.Dropdown(id="col-flare", placeholder="Select...")], md=2),
                        dbc.Col([html.Label("Gas Export", style={"color": COLOURS["muted"], "fontSize": "12px"}),
                                 dcc.Dropdown(id="col-export", placeholder="Select...")], md=2),
                        dbc.Col([html.Label("Fuel Gas", style={"color": COLOURS["muted"], "fontSize": "12px"}),
                                 dcc.Dropdown(id="col-fuel", placeholder="Select...")], md=2),
                        dbc.Col([html.Label("Gas Injection", style={"color": COLOURS["muted"], "fontSize": "12px"}),
                                 dcc.Dropdown(id="col-injection", placeholder="Select...")], md=2),
                    ], className="g-3"),
                    html.Div(style={"marginTop": "20px", "display": "flex",
                                    "gap": "24px", "alignItems": "center",
                                    "flexWrap": "wrap"}, children=[
                        html.Button("RUN ANALYSIS", id="run-btn", n_clicks=0, style={
                            "backgroundColor": COLOURS["accent"], "color": COLOURS["bg"],
                            "border": "none", "borderRadius": "6px", "padding": "10px 24px",
                            "fontFamily": "monospace", "fontSize": "11px",
                            "letterSpacing": "2px", "fontWeight": "700", "cursor": "pointer",
                        }),
                        html.Div([
                            html.Label("Contamination %",
                                       style={"color": COLOURS["muted"], "fontSize": "12px"}),
                            dcc.Slider(id="contamination-slider", min=1, max=5, step=1, value=2,
                                       marks={i: {"label": str(i) + "%",
                                                  "style": {"color": COLOURS["muted"]}}
                                              for i in [1, 2, 3, 5]}),
                        ], style={"flex": "1", "maxWidth": "260px"}),
                        html.Div([
                            html.Label("Sigma Threshold",
                                       style={"color": COLOURS["muted"], "fontSize": "12px"}),
                            dcc.Slider(id="sigma-slider", min=2.0, max=4.0, step=0.5, value=3.0,
                                       marks={v: {"label": str(v),
                                                  "style": {"color": COLOURS["muted"]}}
                                              for v in [2.0, 2.5, 3.0, 3.5, 4.0]}),
                        ], style={"flex": "1", "maxWidth": "260px"}),
                    ]),
                    html.Div(id="arima-order-display",
                             style={"marginTop": "10px", "fontSize": "12px",
                                    "color": COLOURS["muted"]}),
                ]),
            ]),

            # KPI
            html.Div(id="kpi-section", style={"display": "none"}, children=[
                html.P("04 / SYSTEM STATUS", style={**HEADING_STYLE, "marginBottom": "12px"}),
                dbc.Row(id="kpi-cards", className="g-3 mb-4"),
            ]),

            # Charts
            html.Div(id="charts-section", style={"display": "none"}, children=[

                html.P("05 / METHANE RESIDUAL SIGNAL", style=HEADING_STYLE),
                html.Div(style=CARD_STYLE, children=[
                    dcc.Graph(id="residual-chart", figure=blank_fig(),
                              config={"displayModeBar": False}),
                ]),

                dbc.Row([
                    dbc.Col([
                        html.P("06 / ARIMA ANOMALY DETECTION", style=HEADING_STYLE),
                        html.Div(style=CARD_STYLE, children=[
                            dcc.Graph(id="arima-chart", figure=blank_fig(),
                                      config={"displayModeBar": False}),
                        ]),
                    ], md=6),
                    dbc.Col([
                        html.P("07 / ISOLATION FOREST DETECTION", style=HEADING_STYLE),
                        html.Div(style=CARD_STYLE, children=[
                            dcc.Graph(id="xgb-chart", figure=blank_fig(),
                                      config={"displayModeBar": False}),
                        ]),
                    ], md=6),
                ]),

                dbc.Row([
                    dbc.Col([
                        html.P("08 / AUTOENCODER RECONSTRUCTION ERROR", style=HEADING_STYLE),
                        html.Div(style=CARD_STYLE, children=[
                            dcc.Graph(id="lstm-chart", figure=blank_fig(),
                                      config={"displayModeBar": False}),
                        ]),
                    ], md=6),
                    dbc.Col([
                        html.P("09 / SIGNAL-TO-NOISE RATIO", style=HEADING_STYLE),
                        html.Div(style=CARD_STYLE, children=[
                            dcc.Graph(id="snr-chart", figure=blank_fig(),
                                      config={"displayModeBar": False}),
                        ]),
                    ], md=6),
                ]),

                html.P("10 / MINIMUM DETECTABLE LEAK FRACTION", style=HEADING_STYLE),
                html.Div(style=CARD_STYLE, children=[
                    dcc.Graph(id="mdlf-chart", figure=blank_fig(),
                              config={"displayModeBar": False}),
                    html.Div(id="mdlf-summary",
                             style={"marginTop": "12px", "fontSize": "12px",
                                    "color": COLOURS["muted"]}),
                ]),

                html.P("11 / ALERT LOG", style=HEADING_STYLE),
                html.Div(style=CARD_STYLE, children=[html.Div(id="alert-table")]),

                html.P("12 / MODEL ENSEMBLE AGREEMENT", style=HEADING_STYLE),
                html.Div(style=CARD_STYLE, children=[
                    dcc.Graph(id="ensemble-chart", figure=blank_fig(),
                              config={"displayModeBar": False}),
                ]),

                html.P("13 / EXPORT REPORTS", style=HEADING_STYLE),
                html.Div(style=CARD_STYLE, children=[
                    html.P("Download full analysis results.",
                           style={"color": COLOURS["muted"], "fontSize": "13px",
                                  "marginBottom": "20px"}),
                    html.Div(style={"display": "flex", "gap": "12px", "flexWrap": "wrap"},
                             children=[
                        html.Button("↓ PDF REPORT", id="btn-pdf", n_clicks=0,
                                    style={"backgroundColor": COLOURS["danger"], "color": "#fff",
                                           "border": "none", "borderRadius": "6px",
                                           "padding": "10px 20px", "fontFamily": "monospace",
                                           "fontSize": "11px", "cursor": "pointer"}),
                        html.Button("↓ EXCEL ALERT LOG", id="btn-excel", n_clicks=0,
                                    style={"backgroundColor": "#1D6F42", "color": "#fff",
                                           "border": "none", "borderRadius": "6px",
                                           "padding": "10px 20px", "fontFamily": "monospace",
                                           "fontSize": "11px", "cursor": "pointer"}),
                        html.Button("↓ CSV RESIDUAL", id="btn-csv", n_clicks=0,
                                    style={"backgroundColor": COLOURS["accent3"], "color": "#fff",
                                           "border": "none", "borderRadius": "6px",
                                           "padding": "10px 20px", "fontFamily": "monospace",
                                           "fontSize": "11px", "cursor": "pointer"}),
                    ]),
                    html.Div(id="export-status",
                             style={"marginTop": "12px", "fontSize": "13px",
                                    "color": COLOURS["muted"]}),
                ]),

                dcc.Download(id="download-pdf"),
                dcc.Download(id="download-excel"),
                dcc.Download(id="download-csv"),
            ]),

            dcc.Store(id="stored-data"),
            dcc.Store(id="stored-results"),
        ]),

        html.Div(style={
            "borderTop": "1px solid " + COLOURS["border"],
            "padding": "20px 40px", "display": "flex",
            "justifyContent": "space-between", "marginTop": "40px",
        }, children=[
            html.Span("PFEMS · Savanna Dynamics Limited",
                      style={"color": COLOURS["muted"], "fontSize": "11px"}),
            html.Span("Data processed in-session only · Not stored server-side",
                      style={"color": COLOURS["muted"], "fontSize": "11px"}),
        ]),
    ]
)

# ── Callback: parse upload ──────────────────────────────────────────────────────
@app.callback(
    Output("stored-data", "data"),
    Output("upload-status", "children"),
    Output("column-mapping-section", "style"),
    Output("col-date", "options"),
    Output("col-produced", "options"),
    Output("col-flare", "options"),
    Output("col-export", "options"),
    Output("col-fuel", "options"),
    Output("col-injection", "options"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
)
def parse_upload(contents, filename):
    hidden  = {"display": "none"}
    visible = {"display": "block"}
    empty   = []
    if contents is None:
        return None, "", hidden, empty, empty, empty, empty, empty, empty
    try:
        _, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)
        df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
        cols = [{"label": c, "value": c} for c in df.columns]
        status = html.Span([
            html.Span("✓ ", style={"color": COLOURS["accent"]}),
            filename + " loaded · " + str(len(df)) + " rows"
        ], style={"fontSize": "13px", "color": COLOURS["text"]})
        return (df.to_json(date_format="iso", orient="split"),
                status, visible, cols, cols, cols, cols, cols, cols)
    except Exception as e:
        return (None, html.Span("Error: " + str(e),
                                style={"color": COLOURS["danger"]}),
                hidden, empty, empty, empty, empty, empty, empty)

# ── Callback: run analysis ──────────────────────────────────────────────────────
@app.callback(
    Output("stored-results", "data"),
    Output("kpi-section", "style"),
    Output("charts-section", "style"),
    Output("kpi-cards", "children"),
    Output("residual-chart", "figure"),
    Output("arima-chart", "figure"),
    Output("xgb-chart", "figure"),
    Output("lstm-chart", "figure"),
    Output("snr-chart", "figure"),
    Output("mdlf-chart", "figure"),
    Output("mdlf-summary", "children"),
    Output("ensemble-chart", "figure"),
    Output("alert-table", "children"),
    Output("status-pill", "children"),
    Output("arima-order-display", "children"),
    Input("run-btn", "n_clicks"),
    State("stored-data", "data"),
    State("col-date", "value"),
    State("col-produced", "value"),
    State("col-flare", "value"),
    State("col-export", "value"),
    State("col-fuel", "value"),
    State("col-injection", "value"),
    State("contamination-slider", "value"),
    State("sigma-slider", "value"),
    State("ch4-fraction", "value"),
    State("field-name", "value"),
    State("operator-name", "value"),
    prevent_initial_call=True,
)
def run_analysis(n_clicks, stored_data, col_date, col_produced, col_flare,
                 col_export, col_fuel, col_injection,
                 contamination_pct, sigma_thresh, ch4_fraction,
                 field_name, operator_name):

    show = {"display": "block"}
    hide = {"display": "none"}

    empties = (None, hide, hide, [],
               blank_fig("Select all columns first"),
               blank_fig(), blank_fig(), blank_fig(), blank_fig(),
               blank_fig(), "", blank_fig(),
               html.Div("No results."),
               _status_pill("NO DATA", COLOURS["muted"]), "")

    if not stored_data or not all([col_date, col_produced, col_flare,
                                   col_export, col_fuel, col_injection]):
        return empties

    try:
        contamination  = (contamination_pct or 2) / 100
        ch4_frac       = float(ch4_fraction or 0.77)
        sigma          = float(sigma_thresh or 3.0)
        field_label    = field_name or "Unnamed Field"
        operator_label = operator_name or "Unnamed Operator"

        df = pd.read_json(io.StringIO(stored_data), orient="split")
        df[col_date] = pd.to_datetime(df[col_date], errors="coerce")
        df = df.sort_values(col_date).reset_index(drop=True)
        for col in [col_produced, col_flare, col_export, col_fuel, col_injection]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        raw_residual = compute_residual(df, col_produced, col_flare,
                                        col_export, col_fuel, col_injection)
        residual     = apply_methane_fraction(raw_residual, ch4_frac)
        dates        = df[col_date]

        arima_flags, arima_scores, z_scores, arima_fitted, arima_order = \
            run_arima_anomaly(residual, sigma_threshold=sigma)
        iso_flags, iso_scores   = run_isolation_forest(residual, contamination)
        ae_flags,  ae_scores    = run_autoencoder_anomaly(residual, contamination)
        ensemble_alert, vote_count = compute_ensemble(
            arima_flags, iso_flags, ae_flags, z_scores)

        def norm(arr):
            mn, mx = arr.min(), arr.max()
            return (arr - mn) / (mx - mn + 1e-9)

        arima_norm = norm(arima_scores)
        iso_norm   = norm(iso_scores)
        ae_norm    = norm(ae_scores)
        ens_score  = (arima_norm + iso_norm + ae_norm) / 3

        snr  = compute_snr(residual)
        mdlf = compute_mdlf(residual, df[col_produced])

        alerts = []
        for idx in np.where(ensemble_alert)[0]:
            level, colour = classify_alert(
                float(ens_score[idx]),
                float(snr.iloc[idx]) if idx < len(snr) else 1.0)
            alerts.append({
                "Date":   dates.iloc[idx].strftime("%Y-%m-%d"),
                "Res":    f"{residual.iloc[idx]:.4f}",
                "CH4":    f"{ch4_frac:.2f}",
                "Score":  f"{ens_score[idx]:.3f}",
                "Votes":  str(int(vote_count[idx])),
                "Z":      f"{z_scores[idx]:.2f}",
                "SNR":    f"{snr.iloc[idx]:.2f}" if idx < len(snr) else "—",
                "ARIMA":  "●" if arima_flags[idx] else "○",
                "ISO":    "●" if iso_flags[idx]   else "○",
                "AE":     "●" if ae_flags[idx]    else "○",
                "Level":  level, "colour": colour,
            })

        n_crit = sum(1 for a in alerts if a["Level"] == "CRITICAL")
        n_warn = sum(1 for a in alerts if a["Level"] == "WARNING")
        n_watch= sum(1 for a in alerts if a["Level"] == "WATCH")
        mdlf_global = float((3 * residual.std() / df[col_produced].mean()) * 100)

        kpi_data = [
            ("ENSEMBLE ALERTS", len(alerts),         COLOURS["accent2"]),
            ("CRITICAL",        n_crit,               COLOURS["danger"]),
            ("WARNING",         n_warn,               COLOURS["warn"]),
            ("WATCH",           n_watch,              COLOURS["accent3"]),
            ("DATA POINTS",     str(len(df)),         COLOURS["accent"]),
            ("MEAN MDLF",       f"{mdlf.mean():.2f}%", COLOURS["muted"]),
        ]
        kpi_cards = [dbc.Col(_kpi_card(l, v, c), md=2) for l, v, c in kpi_data]

        # Residual chart
        fig_res = go.Figure()
        fig_res.add_trace(go.Scatter(
            x=dates, y=residual, mode="lines",
            name="CH4 Residual",
            line=dict(color=COLOURS["accent3"], width=1.5),
            fill="tozeroy", fillcolor="rgba(75,159,255,0.08)"))
        if ensemble_alert.any():
            fig_res.add_trace(go.Scatter(
                x=dates[ensemble_alert], y=residual[ensemble_alert],
                mode="markers", name="Ensemble Alert",
                marker=dict(color=COLOURS["danger"], size=7,
                            symbol="circle-open", line=dict(width=2))))
        fig_res.add_hline(y=0, line_dash="dot",
                          line_color=COLOURS["muted"], line_width=1)
        fig_res.add_hline(y=float(residual.std() * sigma), line_dash="dash",
                          line_color=COLOURS["danger"], line_width=1,
                          annotation_text=f"+{sigma}σ threshold",
                          annotation_font_color=COLOURS["danger"],
                          annotation_font_size=10)
        fig_res.update_layout(**chart_layout(
            "METHANE BALANCE RESIDUAL · " + field_label, 340))

        fig_arima = _anomaly_fig(dates, residual, arima_flags, arima_norm,
                                  "ARIMA(1,0,1) ANOMALY SCORES")
        fig_iso   = _anomaly_fig(dates, residual, iso_flags, iso_norm,
                                  "ISOLATION FOREST ANOMALY SCORES")
        fig_ae    = _anomaly_fig(dates, residual, ae_flags, ae_norm,
                                  "AUTOENCODER RECONSTRUCTION ERROR")

        fig_snr = go.Figure()
        fig_snr.add_trace(go.Scatter(x=dates, y=snr, mode="lines", name="SNR",
                                      line=dict(color=COLOURS["warn"], width=1.5)))
        fig_snr.add_hline(y=2, line_dash="dash", line_color=COLOURS["accent2"],
                          annotation_text="Warning (SNR=2)",
                          annotation_font_color=COLOURS["accent2"],
                          annotation_font_size=10)
        fig_snr.add_hline(y=3, line_dash="dash", line_color=COLOURS["danger"],
                          annotation_text="Critical (SNR=3)",
                          annotation_font_color=COLOURS["danger"],
                          annotation_font_size=10)
        fig_snr.update_layout(**chart_layout("SIGNAL-TO-NOISE RATIO (30-DAY ROLLING)", 320))

        fig_mdlf = go.Figure()
        fig_mdlf.add_trace(go.Scatter(x=dates, y=mdlf, mode="lines", name="MDLF %",
                                       line=dict(color=COLOURS["accent"], width=1.5),
                                       fill="tozeroy",
                                       fillcolor="rgba(0,212,170,0.08)"))
        fig_mdlf.add_hline(y=mdlf_global, line_dash="dash",
                           line_color=COLOURS["muted"], line_width=1,
                           annotation_text=f"Global {mdlf_global:.2f}%",
                           annotation_font_color=COLOURS["muted"],
                           annotation_font_size=10)
        fig_mdlf.update_layout(**chart_layout(
            "MINIMUM DETECTABLE LEAK FRACTION (90-DAY ROLLING, % OF PRODUCTION)", 320))

        mdlf_text = (f"Global MDLF: {mdlf_global:.2f}% · "
                     f"Rolling mean: {mdlf.mean():.2f}% · "
                     f"Best: {mdlf.min():.2f}% · Worst: {mdlf.max():.2f}%")

        fig_ens = go.Figure()
        fig_ens.add_trace(go.Scatter(x=dates, y=arima_norm, mode="lines",
                                      name="ARIMA", opacity=0.7,
                                      line=dict(color=COLOURS["accent3"], width=1)))
        fig_ens.add_trace(go.Scatter(x=dates, y=iso_norm, mode="lines",
                                      name="Isolation Forest", opacity=0.7,
                                      line=dict(color=COLOURS["warn"], width=1)))
        fig_ens.add_trace(go.Scatter(x=dates, y=ae_norm, mode="lines",
                                      name="Autoencoder", opacity=0.7,
                                      line=dict(color=COLOURS["accent"], width=1)))
        fig_ens.add_trace(go.Scatter(x=dates, y=ens_score, mode="lines",
                                      name="Ensemble Score",
                                      line=dict(color=COLOURS["accent2"], width=2.5)))
        fig_ens.update_layout(**chart_layout("MODEL ENSEMBLE AGREEMENT", 320))

        if n_crit > 0:   pill = _status_pill(f"● {n_crit} CRITICAL",  COLOURS["danger"])
        elif n_warn > 0: pill = _status_pill(f"● {n_warn} WARNING",   COLOURS["warn"])
        elif n_watch > 0:pill = _status_pill(f"● {n_watch} WATCH",    COLOURS["accent3"])
        else:            pill = _status_pill("● NORMAL",               COLOURS["accent"])

        stored = json.dumps({
            "n_alerts": len(alerts), "field": field_label,
            "operator": operator_label, "ch4_frac": ch4_frac,
            "mdlf_global": mdlf_global,
            "contamination_pct": contamination_pct,
        })

        return (stored, show, show, kpi_cards,
                fig_res, fig_arima, fig_iso, fig_ae, fig_snr,
                fig_mdlf, mdlf_text, fig_ens,
                _build_alert_table(alerts), pill,
                "ARIMA order: (1, 0, 1) fixed for deployment speed")

    except Exception as e:
        err_pill = _status_pill("● ERROR", COLOURS["danger"])
        err_div  = html.Div("Analysis error: " + str(e),
                            style={"color": COLOURS["danger"], "fontSize": "13px"})
        return (None, hide, hide, [],
                blank_fig("Error — check logs"),
                blank_fig(), blank_fig(), blank_fig(), blank_fig(),
                blank_fig(), "", blank_fig(),
                err_div, err_pill, "Error: " + str(e))

# ── Export callback ─────────────────────────────────────────────────────────────
@app.callback(
    Output("download-pdf",   "data"),
    Output("download-excel", "data"),
    Output("download-csv",   "data"),
    Output("export-status",  "children"),
    Input("btn-pdf",   "n_clicks"),
    Input("btn-excel", "n_clicks"),
    Input("btn-csv",   "n_clicks"),
    State("stored-results",       "data"),
    State("stored-data",          "data"),
    State("col-date",             "value"),
    State("col-produced",         "value"),
    State("col-flare",            "value"),
    State("col-export",           "value"),
    State("col-fuel",             "value"),
    State("col-injection",        "value"),
    State("contamination-slider", "value"),
    State("sigma-slider",         "value"),
    State("ch4-fraction",         "value"),
    State("field-name",           "value"),
    State("operator-name",        "value"),
    prevent_initial_call=True,
)
def handle_exports(btn_pdf, btn_excel, btn_csv,
                   stored_results, stored_data,
                   col_date, col_produced, col_flare, col_export,
                   col_fuel, col_injection,
                   contamination_pct, sigma_thresh,
                   ch4_fraction, field_name, operator_name):

    ctx = callback_context
    if not ctx.triggered or not stored_results or not stored_data:
        return None, None, None, html.Span("Run analysis first.",
                                            style={"color": COLOURS["warn"]})

    triggered      = ctx.triggered[0]["prop_id"].split(".")[0]
    contamination  = (contamination_pct or 2) / 100
    ch4_frac       = float(ch4_fraction or 0.77)
    sigma          = float(sigma_thresh or 3.0)
    field_label    = field_name or "Unnamed Field"
    operator_label = operator_name or "Unnamed Operator"
    stored_meta    = json.loads(stored_results)
    mdlf_global    = stored_meta.get("mdlf_global")
    cont_pct       = contamination_pct or 2

    df = pd.read_json(io.StringIO(stored_data), orient="split")
    df[col_date] = pd.to_datetime(df[col_date], errors="coerce")
    df = df.sort_values(col_date).reset_index(drop=True)
    for col in [col_produced, col_flare, col_export, col_fuel, col_injection]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    raw_residual   = compute_residual(df, col_produced, col_flare,
                                      col_export, col_fuel, col_injection)
    residual       = apply_methane_fraction(raw_residual, ch4_frac)
    dates_list     = df[col_date].dt.strftime("%Y-%m-%d").tolist()

    arima_flags, arima_scores, z_scores, _, arima_order = \
        run_arima_anomaly(residual, sigma_threshold=sigma)
    iso_flags, iso_scores = run_isolation_forest(residual, contamination)
    ae_flags,  ae_scores  = run_autoencoder_anomaly(residual, contamination)
    ensemble_alert, vote_count = compute_ensemble(
        arima_flags, iso_flags, ae_flags, z_scores)

    def norm(arr):
        mn, mx = arr.min(), arr.max()
        return (arr - mn) / (mx - mn + 1e-9)

    ens_score = (norm(arima_scores) + norm(iso_scores) + norm(ae_scores)) / 3
    snr       = compute_snr(residual)
    mdlf      = compute_mdlf(residual, df[col_produced])

    alerts = []
    for idx in np.where(ensemble_alert)[0]:
        level, colour = classify_alert(
            float(ens_score[idx]),
            float(snr.iloc[idx]) if idx < len(snr) else 1.0)
        alerts.append({
            "Date":   dates_list[idx] if idx < len(dates_list) else "—",
            "Res":    f"{residual.iloc[idx]:.4f}",
            "CH4":    f"{ch4_frac:.2f}",
            "Score":  f"{ens_score[idx]:.3f}",
            "Votes":  str(int(vote_count[idx])),
            "Z":      f"{z_scores[idx]:.2f}",
            "SNR":    f"{snr.iloc[idx]:.2f}" if idx < len(snr) else "—",
            "ARIMA":  "●" if arima_flags[idx] else "○",
            "ISO":    "●" if iso_flags[idx]   else "○",
            "AE":     "●" if ae_flags[idx]    else "○",
            "Level":  level, "colour": colour,
        })

    n_crit  = sum(1 for a in alerts if a["Level"] == "CRITICAL")
    n_warn  = sum(1 for a in alerts if a["Level"] == "WARNING")
    n_watch = sum(1 for a in alerts if a["Level"] == "WATCH")
    ts      = datetime.now().strftime("%Y%m%d_%H%M")

    if triggered == "btn-pdf":
        buf  = io.BytesIO()
        doc  = SimpleDocTemplate(buf, pagesize=A4,
                                  leftMargin=2*cm, rightMargin=2*cm,
                                  topMargin=2*cm, bottomMargin=2*cm)
        acc  = colors.HexColor("#00D4AA")
        drk  = colors.HexColor("#0A0E17")
        mut  = colors.HexColor("#6B7A99")
        dng  = colors.HexColor("#FF3B5C")
        wrn  = colors.HexColor("#FFB800")
        ts_  = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
        t_st = ParagraphStyle("t", fontSize=20, textColor=acc,
                               fontName="Helvetica-Bold", spaceAfter=4)
        s_st = ParagraphStyle("s", fontSize=10, textColor=mut,
                               fontName="Helvetica", spaceAfter=6)
        h_st = ParagraphStyle("h", fontSize=11, textColor=acc,
                               fontName="Helvetica-Bold",
                               spaceAfter=8, spaceBefore=16)
        b_st = ParagraphStyle("b", fontSize=9,
                               textColor=colors.HexColor("#333333"),
                               fontName="Helvetica", spaceAfter=6, leading=14)
        story = [
            Spacer(1, 1.5*cm),
            Paragraph("PFEMS", t_st),
            Paragraph("Physics-Guided Fugitive Emissions Monitoring System", s_st),
            Paragraph("Savanna Dynamics Limited", s_st),
            Spacer(1, 0.3*cm),
            Paragraph(f"<b>Field:</b> {field_label}", s_st),
            Paragraph(f"<b>Operator:</b> {operator_label}", s_st),
            Paragraph(f"<b>CH4 Fraction:</b> {ch4_frac:.2f}", s_st),
            Paragraph(f"<b>Generated:</b> {ts_}", s_st),
            Spacer(1, 0.5*cm),
            Paragraph("EXECUTIVE SUMMARY", h_st),
        ]
        mean_r = float(np.mean(residual.values))
        std_r  = float(np.std(residual.values))
        dr     = f"{dates_list[0]} to {dates_list[-1]}" if dates_list else "—"
        mf     = f"{mdlf_global:.2f}%" if mdlf_global else "—"
        status = ("CRITICAL" if n_crit > 0 else
                  "WARNING"  if n_warn > 0 else
                  "WATCH"    if n_watch > 0 else "NORMAL")
        story.append(Paragraph(
            f"Field: <b>{field_label}</b> | Operator: <b>{operator_label}</b> | "
            f"Period: {dr} | Points: {len(residual):,} | CH4 frac: {ch4_frac:.2f}<br/><br/>"
            f"Status: <b>{status}</b> — {n_crit} critical, {n_warn} warning, "
            f"{n_watch} watch alerts.<br/>"
            f"Mean residual: {mean_r:.4f} MMscf/day (std: {std_r:.4f}). "
            f"Global MDLF: {mf}.", b_st))
        story.append(Paragraph("KEY METRICS", h_st))
        kd = [["Metric","Value"],
              ["Field", field_label], ["Operator", operator_label],
              ["CH4 Fraction", f"{ch4_frac:.2f}"],
              ["Data points", str(len(residual))],
              ["Period", dr], ["Mean residual", f"{mean_r:.4f}"],
              ["Std deviation", f"{std_r:.4f}"], ["Global MDLF", mf],
              ["Total alerts", str(len(alerts))],
              ["Critical", str(n_crit)], ["Warning", str(n_warn)],
              ["Watch", str(n_watch)]]
        kt = Table(kd, colWidths=[10*cm, 6*cm])
        kt.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),drk), ("TEXTCOLOR",(0,0),(-1,0),acc),
            ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
            ("FONTSIZE",(0,0),(-1,-1),9),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),
             [colors.HexColor("#F8F9FA"), colors.white]),
            ("GRID",(0,0),(-1,-1),0.5,colors.HexColor("#DDDDDD")),
            ("PADDING",(0,0),(-1,-1),6), ("LEFTPADDING",(0,0),(-1,-1),10),
        ]))
        story.append(kt)
        story.append(PageBreak())
        story.append(Paragraph("ALERT LOG", h_st))
        if alerts:
            ah = ["Date","Residual","CH4","Score","Votes","z","SNR",
                  "ARIMA","ISO","AE","Level"]
            ar = [ah] + [[a["Date"],a["Res"],a["CH4"],a["Score"],
                          a["Votes"],a["Z"],a["SNR"],
                          a["ARIMA"],a["ISO"],a["AE"],a["Level"]]
                         for a in alerts[:150]]
            at = Table(ar, colWidths=[2.2*cm,2*cm,1.5*cm,1.5*cm,1.2*cm,
                                       1.5*cm,1.5*cm,1.2*cm,1.2*cm,1.2*cm,1.8*cm])
            ats = TableStyle([
                ("BACKGROUND",(0,0),(-1,0),drk),("TEXTCOLOR",(0,0),(-1,0),acc),
                ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
                ("FONTSIZE",(0,0),(-1,-1),7.5),
                ("ROWBACKGROUNDS",(0,1),(-1,-1),
                 [colors.HexColor("#F8F9FA"),colors.white]),
                ("GRID",(0,0),(-1,-1),0.3,colors.HexColor("#DDDDDD")),
                ("PADDING",(0,0),(-1,-1),3),("ALIGN",(0,0),(-1,-1),"CENTER"),
            ])
            for i, a in enumerate(alerts[:150], 1):
                if a["Level"] == "CRITICAL":
                    ats.add("TEXTCOLOR",(10,i),(10,i),dng)
                    ats.add("FONTNAME",(10,i),(10,i),"Helvetica-Bold")
                elif a["Level"] == "WARNING":
                    ats.add("TEXTCOLOR",(10,i),(10,i),wrn)
            at.setStyle(ats)
            story.append(at)
        else:
            story.append(Paragraph("No alerts detected.", b_st))
        story.append(PageBreak())
        story.append(Paragraph("METHODOLOGY", h_st))
        story.append(Paragraph(
            f"<b>Residual:</b> (produced - flaring - export - fuel - injection) x {ch4_frac:.2f}.<br/>"
            f"<b>ARIMA(1,0,1):</b> Fixed order for deployment. z-score positive direction only.<br/>"
            f"<b>Isolation Forest:</b> Lag features, 100 estimators, contamination={cont_pct}%.<br/>"
            f"<b>PCA Autoencoder:</b> 14-obs windows, 2 components, reconstruction error.<br/>"
            f"<b>Ensemble:</b> Majority voting (2 of 3) + positive z-score restriction.<br/>"
            f"<b>MDLF:</b> 3 x rolling sigma / rolling mean production, 90-day window.<br/>"
            f"<b>Reference:</b> Abdul Hameed M. A Scalable Data-Driven Framework for Fugitive "
            f"Methane Detection. Discover Sustainability, Springer Nature, 2025 (under review).",
            b_st))
        story.append(Spacer(1, 1*cm))
        story.append(Paragraph(
            f"PFEMS v1.1 · Savanna Dynamics Limited · {ts_} · In-session only",
            ParagraphStyle("ft", fontSize=7, textColor=mut, fontName="Helvetica")))
        doc.build(story)
        buf.seek(0)
        return (dcc.send_bytes(buf.read(), f"PFEMS_{field_label}_{ts}.pdf"),
                None, None,
                html.Span("✓ PDF downloaded", style={"color": COLOURS["accent"]}))

    elif triggered == "btn-excel":
        buf = io.BytesIO()
        wb  = openpyxl.Workbook()
        hf  = PatternFill("solid", fgColor="0A0E17")
        hfn = Font(color="00D4AA", bold=True, name="Consolas", size=10)
        nfn = Font(name="Calibri", size=10)
        cfn = Font(color="FF3B5C", bold=True, name="Calibri", size=10)
        wfn = Font(color="FFB800", bold=True, name="Calibri", size=10)
        bdr = Border(bottom=Side(style="thin", color="1E2D45"),
                     right=Side(style="thin",  color="1E2D45"))
        ws  = wb.active
        ws.title = "Alert Log"
        hdrs = ["Date","Residual CH4","CH4 Frac","Score","Votes",
                "z-score","SNR","ARIMA","ISO","AE","Level"]
        for c, h in enumerate(hdrs, 1):
            cell = ws.cell(row=1, column=c, value=h)
            cell.fill = hf; cell.font = hfn
            cell.alignment = Alignment(horizontal="center")
        for ri, a in enumerate(alerts, 2):
            vals = [a["Date"], float(a["Res"]), float(a["CH4"]),
                    float(a["Score"]), int(a["Votes"]),
                    float(a["Z"]),
                    float(a["SNR"]) if a["SNR"] != "—" else "",
                    1 if a["ARIMA"]=="●" else 0,
                    1 if a["ISO"]  =="●" else 0,
                    1 if a["AE"]   =="●" else 0,
                    a["Level"]]
            for ci, val in enumerate(vals, 1):
                cell = ws.cell(row=ri, column=ci, value=val)
                cell.border = bdr
                cell.font = (cfn if ci==11 and a["Level"]=="CRITICAL" else
                             wfn if ci==11 and a["Level"]=="WARNING"  else nfn)
                cell.alignment = Alignment(horizontal="center")
        for col in ws.columns:
            ws.column_dimensions[col[0].column_letter].width = min(
                max(len(str(c.value or "")) for c in col)+4, 30)
        ws2 = wb.create_sheet("Residual Signal")
        for c, h in enumerate(["Date","Residual CH4","Score","SNR","MDLF%"], 1):
            cell = ws2.cell(row=1, column=c, value=h)
            cell.fill = hf; cell.font = hfn
            cell.alignment = Alignment(horizontal="center")
        mdlf_list = list(mdlf.values)
        for i, (d, r) in enumerate(zip(dates_list, residual.values), 2):
            ws2.cell(row=i,column=1,value=str(d)).font=nfn
            ws2.cell(row=i,column=2,value=float(r)).font=nfn
            ws2.cell(row=i,column=3,
                     value=float(ens_score[i-2]) if i-2<len(ens_score) else "").font=nfn
            ws2.cell(row=i,column=4,
                     value=float(snr.values[i-2]) if i-2<len(snr) else "").font=nfn
            ws2.cell(row=i,column=5,
                     value=float(mdlf_list[i-2]) if i-2<len(mdlf_list) else "").font=nfn
        for col in ws2.columns:
            ws2.column_dimensions[col[0].column_letter].width = 22
        ws3 = wb.create_sheet("Summary")
        rows3 = [("PFEMS · Savanna Dynamics Limited",None),
                 (f"Field: {field_label}",None),
                 (f"Operator: {operator_label}",None),
                 (f"CH4: {ch4_frac:.2f}",None),
                 ("",""),("METRIC","VALUE"),
                 ("Data points",len(residual)),
                 ("Mean residual",round(float(np.mean(residual.values)),4)),
                 ("Std deviation",round(float(np.std(residual.values)),4)),
                 ("Total alerts",len(alerts)),
                 ("Critical",n_crit),("Warning",n_warn),("Watch",n_watch)]
        for ri,(lab,val) in enumerate(rows3,1):
            c1=ws3.cell(row=ri,column=1,value=lab)
            c1.font=(Font(name="Consolas",size=14,bold=True,color="00D4AA")
                     if ri==1 else hfn if lab=="METRIC" else nfn)
            if val is not None:
                c2=ws3.cell(row=ri,column=2,value=val)
                c2.font=hfn if lab=="METRIC" else nfn
        ws3.column_dimensions["A"].width=40
        ws3.column_dimensions["B"].width=20
        wb.save(buf); buf.seek(0)
        return (None,
                dcc.send_bytes(buf.read(), f"PFEMS_{field_label}_{ts}.xlsx"),
                None,
                html.Span("✓ Excel downloaded", style={"color": COLOURS["accent"]}))

    elif triggered == "btn-csv":
        df_out = pd.DataFrame({
            "date":                  dates_list,
            "residual_ch4_MMscf":    list(residual.values),
            "ch4_fraction":          ch4_frac,
            "ensemble_score":        list(ens_score),
            "snr":                   list(snr.values),
            "z_score":               list(z_scores),
            "arima_flag":            arima_flags.astype(int).tolist(),
            "isolation_forest_flag": iso_flags.astype(int).tolist(),
            "autoencoder_flag":      ae_flags.astype(int).tolist(),
            "ensemble_alert":        ensemble_alert.astype(int).tolist(),
            "mdlf_pct":              list(mdlf.values),
        })
        return (None, None,
                dcc.send_string(df_out.to_csv(index=False),
                                f"PFEMS_{field_label}_{ts}.csv"),
                html.Span("✓ CSV downloaded", style={"color": COLOURS["accent"]}))

    return None, None, None, ""


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
