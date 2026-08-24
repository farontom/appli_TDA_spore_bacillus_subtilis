# -*- coding: utf-8 -*-
"""
SpectroMapper — interface Dash pour explorer un graphe Mapper construit à
partir de spectres OPTIR / Raman (mono-bloc ou multibloc).
"""
import time
import numpy as np
from dash import Dash, html, dcc, Input, Output, State, callback_context
import dash_cytoscape as cyto
import plotly.graph_objects as go
import tda_tools_mapper_dash as ttmd
from sklearn.preprocessing import LabelEncoder

app = Dash(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# DONNÉES
# ─────────────────────────────────────────────────────────────────────────────

_df_optir = ttmd.load_data(mode="mono", sheet=0)
X0, meta0, _, _, feature_names0 = ttmd.prepare_data(_df_optir)

_df_raman = ttmd.load_data(mode="mono", sheet=1)
X1, meta1, _, _, feature_names1 = ttmd.prepare_data(_df_raman)

_df_multi = ttmd.load_data(mode="multi")
meta_mb, X_optir_mb, X_raman_mb, X_mb = ttmd.prepare_multiblock_data(_df_multi)

souches    = sorted(meta0["souches"].unique())
treatments = ["UVB wet", "UVC wet", "UVB dry", "UVC dry"]

# Cache mémoire : évite de relancer tout le calcul Mapper (long) quand
# l'utilisateur clique juste sur des nœuds dans le graphe déjà affiché.
# Clé = (souche, traitement, mode, source, lens, bandes) → dernier résultat
# calculé pour cette combinaison exacte de paramètres.
all_graphs = {}

LENS_MONO = [{"label": "PCA", "value": "pca"}, {"label": "PLS-DA", "value": "plsda"}]
LENS_MB   = [{"label": "MFA", "value": "mfa"}, {"label": "MB-PCA", "value": "mbpca"},
             {"label": "MB-PLS", "value": "mbpls"}, {"label": "O2-PLS", "value": "o2pls"},
             {"label": "MCOA", "value": "mcoa"}]


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS SPECTRE / BANDES
# ─────────────────────────────────────────────────────────────────────────────

def _col_to_float(c):
    """
    Extrait le nombre d'onde contenu dans le nom d'une colonne et le
    convertit en flottant. Cette valeur est utilisée pour le tri des
    variables spectrales et la sélection des bandes.
    """
    for p in reversed(c.split("_")):
        try:
            return float(p)
        except ValueError:
            continue
    raise ValueError(c)


def _add_band_traces(fig, x_s, y_s, bands, color, col):
    """Ajoute la courbe avec zones retenues (couleur) et exclues (gris),
    plus un scatter invisible pour capturer la box select.

    Le spectre est découpé en segments continus dans les zones retenues
    (pour pouvoir les remplir en couleur indépendamment les uns des autres),
    et affiché en gris fin partout ailleurs. Le scatter invisible superposé
    sert uniquement de "cible" fiable pour l'outil box-select de Plotly
    (une simple ligne ne capte pas toujours bien la sélection)."""
    kwargs = dict(row=1, col=col) if col is not None else {}

    if not bands:
        fig.add_trace(go.Scatter(x=x_s, y=y_s, mode="lines",
                                  line=dict(color=color), name="spectre",
                                  showlegend=False), **kwargs)
    else:
        mask = np.zeros(len(x_s), dtype=bool)
        for band in bands:
            lo, hi = min(band), max(band)
            mask |= (x_s >= lo) & (x_s <= hi)

        fig.add_trace(go.Scatter(x=x_s, y=y_s, mode="lines",
                                  line=dict(color="lightgray", width=1),
                                  name="exclu", showlegend=False), **kwargs)

        in_band = False
        seg_x, seg_y = [], []
        for xi, yi, mi in zip(x_s, y_s, mask):
            if mi:
                seg_x.append(xi); seg_y.append(yi); in_band = True
            else:
                if in_band and seg_x:
                    fig.add_trace(go.Scatter(
                        x=seg_x, y=seg_y, mode="lines",
                        line=dict(color=color, width=2),
                        fill="tozeroy", name="retenu",
                        showlegend=False), **kwargs)
                    seg_x, seg_y = [], []
                in_band = False
        if in_band and seg_x:
            fig.add_trace(go.Scatter(
                x=seg_x, y=seg_y, mode="lines",
                line=dict(color=color, width=2),
                name="retenu", showlegend=False), **kwargs)

    # Scatter invisible sur toute la plage → capte selectedData fiablement
    fig.add_trace(go.Scatter(
        x=x_s, y=y_s,
        mode="markers",
        marker=dict(opacity=0, size=4),
        name="__sel__",
        showlegend=False,
        hoverinfo="skip",
    ), **kwargs)


def _filter_cols(X, bands, prefix):
    """Retourne la liste de colonnes retenues (ou toutes si bands vide).
    prefix permet de ne filtrer qu'un bloc spectral (ex. "OPTIR_") en mode
    multi ; None en mode mono où il n'y a qu'un seul bloc de colonnes."""
    pool = ([c for c in X.columns if c.startswith(prefix)]
            if prefix else list(X.columns))
    if not bands:
        return pool
    kept = []
    for c in pool:
        try:
            v = _col_to_float(c)
        except ValueError:
            continue
        if any(min(b) <= v <= max(b) for b in bands):
            kept.append(c)
    return kept


def _make_band_preview_mono(sheet, bands):
    """Preview spectre mono-bloc avec bandes surlignées."""
    X_all = X0 if sheet == "0" else X1
    fn    = feature_names0 if sheet == "0" else feature_names1
    name  = "OPTIR" if sheet == "0" else "Raman"
    x_arr = fn.astype(float)
    order = np.argsort(x_arr)
    x_s   = x_arr[order]
    y_s   = X_all.mean(axis=0).values[order]
    fig   = go.Figure()
    _add_band_traces(fig, x_s, y_s, bands, "steelblue", None)
    fig.update_xaxes(autorange="reversed")
    n_kept = len(_filter_cols(X_all, bands, None))
    fig.update_layout(
        title=f"{name} — {n_kept}/{len(X_all.columns)} vars",
        height=300, margin=dict(t=40, b=30, l=40, r=20),
        showlegend=False, dragmode="select",
    )
    return fig


def _make_band_preview_bloc(prefix, color, bands):
    """Preview d'un seul bloc (OPTIR ou Raman) avec bandes surlignées."""
    cols  = [c for c in X_mb.columns if c.startswith(prefix)]
    x_arr = np.array([_col_to_float(c) for c in cols])
    order = np.argsort(x_arr)
    x_s   = x_arr[order]
    y_s   = X_mb[cols].mean(axis=0).values[order]
    fig   = go.Figure()
    _add_band_traces(fig, x_s, y_s, bands, color, None)
    fig.update_xaxes(autorange="reversed")
    n_kept = len(_filter_cols(X_mb[cols], bands, None))
    name   = prefix.rstrip("_")
    fig.update_layout(
        title=f"{name} — {n_kept}/{len(cols)} vars",
        height=280, margin=dict(t=40, b=30, l=40, r=20),
        showlegend=False, dragmode="select",
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# STYLESHEET
# ─────────────────────────────────────────────────────────────────────────────

stylesheet = [
    {"selector": "node", "style": {
        "label": "data(size)",
        "width": "mapData(size, 1, 100, 35, 80)",
        "height": "mapData(size, 1, 100, 35, 80)",
        "background-color": "data(color)",
        "border-width": 3,
        "border-color": "data(border_color)",
        "border-opacity": 0.9,
    }},
    {"selector": "edge",
     "style": {"line-color": "#000000", "width": 1, "opacity": 0.4}},
    {"selector": "node:selected", "style": {
        "border-width": 6, "border-color": "#FF0000", "border-opacity": 1,
        "overlay-color": "#FF0000", "overlay-opacity": 0.20,
        "overlay-padding": 8, "z-index": 999,
    }},
]


# ─────────────────────────────────────────────────────────────────────────────
# LAYOUT
# ─────────────────────────────────────────────────────────────────────────────

def _section(title, children, open_=True):
    return html.Details(
        [html.Summary(f"▶ {title}",
                      style={"cursor": "pointer", "fontWeight": "bold",
                             "fontSize": "16px", "marginBottom": "8px"})]
        + children,
        open=open_,
        style={"marginBottom": "16px", "borderLeft": "3px solid #ddd",
               "paddingLeft": "12px"},
    )


app.layout = html.Div([

    html.H2("SpectroMapper"),

    dcc.Store(id="bands_store",  data=[]),   # mono
    dcc.Store(id="lens_data",    data={}),   # données lens
    dcc.Store(id="bands_optir",  data=[]),   # multi OPTIR
    dcc.Store(id="bands_raman",  data=[]),   # multi Raman

    # ── Contrôles principaux ─────────────────────────────────────────────────
    html.Div([
        html.Div([html.Label("Souche"),
                  dcc.Dropdown(id="souche_dropdown",
                               options=[{"label": s, "value": s} for s in souches],
                               value=souches[0], clearable=False)],
                 style={"flex": "1", "marginRight": "15px"}),
        html.Div([html.Label("Traitement"),
                  dcc.Dropdown(id="treatment_dropdown",
                               options=[{"label": t, "value": t} for t in treatments],
                               value="UVB wet", clearable=False)],
                 style={"flex": "1"}),
    ], style={"display": "flex", "marginBottom": "12px"}),

    html.Div([
        html.Div([html.Label("Mode"),
                  dcc.RadioItems(id="mode_radio",
                                 options=[{"label": " Mono-bloc", "value": "mono"},
                                          {"label": " Multi-bloc", "value": "multi"}],
                                 value="mono", inline=True,
                                 inputStyle={"marginRight": "4px"},
                                 labelStyle={"marginRight": "18px"})],
                 style={"flex": "1", "marginRight": "15px"}),
        html.Div([html.Label("Source spectrale"),
                  dcc.RadioItems(id="sheet_radio",
                                 options=[{"label": " OPTIR ", "value": "0"},
                                          {"label": " Raman ", "value": "1"}],
                                 value="0", inline=True,
                                 inputStyle={"marginRight": "4px"},
                                 labelStyle={"marginRight": "18px"})],
                 id="sheet_div", style={"flex": "1", "marginRight": "15px"}),
        html.Div([html.Label("Lens"),
                  dcc.Dropdown(id="lens_dropdown", options=LENS_MONO,
                               value="plsda", clearable=False)],
                 style={"flex": "1"}),
    ], style={"display": "flex", "alignItems": "flex-end", "marginBottom": "12px"}),

    # ── Sélection de bandes spectrales ───────────────────────────────────────
    _section("Sélection de bandes spectrales", [
        html.P([
            "Faites un ",
            html.B("glisser-sélectionner (box select)"),
            " sur le graphe pour ajouter une bande. "
            "Les zones colorées sont retenues, les grises exclues.",
        ], style={"margin": "0 0 6px 0", "fontSize": "13px", "color": "#555"}),
        html.Div([
            html.Button("🗑 Effacer toutes les bandes", id="clear_bands_btn",
                        n_clicks=0,
                        style={"marginBottom": "8px", "padding": "5px 12px",
                               "cursor": "pointer", "backgroundColor": "#fee",
                               "border": "1px solid #d62728", "borderRadius": "4px",
                               "color": "#d62728", "fontWeight": "bold"}),
            html.Span(id="bands_summary",
                      style={"marginLeft": "15px", "color": "#555",
                             "fontSize": "13px"}),
        ]),
        # Mode mono : un seul graphe
        html.Div(id="band_selector_mono_div", children=[
            dcc.Graph(id="band_selector", style={"height": "300px"},
                      config={"modeBarButtonsToRemove": ["zoom2d", "pan2d", "zoomIn2d",
                                                          "zoomOut2d", "autoScale2d", "resetScale2d"],
                              "modeBarButtonsToAdd": ["select2d"],
                              "displayModeBar": True}),
        ]),
        # Mode multi : deux graphes indépendants
        html.Div(id="band_selector_multi_div", style={"display": "none"}, children=[
            html.Div([
                html.Div([
                    html.P("OPTIR", style={"fontWeight": "bold", "color": "steelblue",
                                            "margin": "0 0 4px 0"}),
                    dcc.Graph(id="band_selector_optir", style={"height": "280px"},
                              config={"modeBarButtonsToRemove": ["zoom2d", "pan2d", "zoomIn2d",
                                                                  "zoomOut2d", "autoScale2d", "resetScale2d"],
                                      "modeBarButtonsToAdd": ["select2d"],
                                      "displayModeBar": True}),
                ], style={"flex": "1", "marginRight": "10px"}),
                html.Div([
                    html.P("Raman", style={"fontWeight": "bold", "color": "crimson",
                                            "margin": "0 0 4px 0"}),
                    dcc.Graph(id="band_selector_raman", style={"height": "280px"},
                              config={"modeBarButtonsToRemove": ["zoom2d", "pan2d", "zoomIn2d",
                                                                  "zoomOut2d", "autoScale2d", "resetScale2d"],
                                      "modeBarButtonsToAdd": ["select2d"],
                                      "displayModeBar": True}),
                ], style={"flex": "1"}),
            ], style={"display": "flex"}),
        ]),
    ], open_=True),

    # ── Type de représentation graphique ─────────────────────────────────────
    html.Div([
        html.Label("Représentation graphique"),
        dcc.Dropdown(
            id="layout_dropdown",
            options=[
                {"label": "Force dirigée", "value": "cose"},
                {"label": "Concentrique", "value": "concentric"},
                {"label": "Circulaire", "value": "circle"},
                {"label": "Hiérarchique", "value": "breadthfirst"},
                {"label": "Grille", "value": "grid"},
            ],
            value="cose",
            clearable=False,
        ),
    ], style={"marginBottom": "20px"}),

    # ── Paramètres Mapper ────────────────────────────────────────────────────
    _section("Paramètres", [

        html.H4("Paramètres DBSCAN", style={"marginBottom": "5px"}),

        html.Div([
            html.Div([html.Label("epsilon"),
                      dcc.Slider(id="eps_slider", min=0.5, max=20, step=0.5, value=5,
                                 tooltip={"placement": "bottom"})],
                     style={"flex": "1", "marginRight": "20px"}),

            html.Div([html.Label("minimum d'un noeud"),
                      dcc.Slider(id="min_samples_slider", min=1, max=5, step=1, value=3,
                                 tooltip={"placement": "bottom"})],
                     style={"flex": "1", "marginRight": "20px"}),
        ], style={"display": "flex", "marginBottom": "20px"}),

        html.H4("Paramètres de Mapper", style={"marginBottom": "5px"}),

        html.Div([
            html.Div([html.Label("nombres de cubes"),
                      dcc.Slider(id="n_cubes_slider", min=2, max=10, step=1, value=5,
                                 tooltip={"placement": "bottom"})],
                     style={"flex": "1", "marginRight": "20px"}),

            html.Div([html.Label("pourcentage de chevauchement"),
                      dcc.Slider(id="overlap_slider", min=0.1, max=0.5, step=0.05,
                                 value=0.2, tooltip={"placement": "bottom"})],
                     style={"flex": "1", "marginRight": "20px"}),
        ], style={"display": "flex", "marginBottom": "20px"}),

        html.H4("Dimension de la projection", style={"marginBottom": "5px"}),

        html.Div([
            html.Div([html.Label("nombres de composantes"),
                      dcc.Slider(id="n_components_slider", min=2, max=10, step=1,
                                 value=7, tooltip={"placement": "bottom"})],
                     style={"flex": "1"}),
        ], style={"display": "flex"}),

    ], open_=True),

    # ── Graphe Mapper ────────────────────────────────────────────────────────
    cyto.Cytoscape(
        id="graph",
        elements=[],
        layout={"name": "cose"},
        style={"width": "100%", "height": "700px"},
        boxSelectionEnabled=True,
        autounselectify=False,
        stylesheet=stylesheet,
    ),

    html.Div([
        html.Div(id="color_legend", style={"flex": "1"}),
        html.Div([
            html.H4("Légende pureté (bordure)"),
            html.Div([
                html.Span("■ ", style={"color": "#2ca02c", "fontSize": "18px"}),
                html.Span(" 100% pur  "),
                html.Span("■ ", style={"color": "#ff7f0e", "fontSize": "18px"}),
                html.Span("70–99% mixte  "),
                html.Span("■ ", style={"color": "#d62728", "fontSize": "18px"}),
                html.Span("< 70% très mélangé"),
            ]),
        ], style={"flex": "1"}),
    ], style={"display": "flex", "marginTop": "10px"}),

    html.Hr(),

    # ── Sélection & Spectres ─────────────────────────────────────────────────
    _section("Sélection & Spectres", [
        html.Div(id="selection_info"),
        html.Div([
            dcc.Graph(id="mean_spectrum_left", style={"height": "500px", "flex": "1"}),
            dcc.Graph(id="mean_spectrum_right", style={"height": "500px", "flex": "1"}),
        ], style={"display": "flex"}),
        html.H4("Spectre différentiel (traitement − control)", style={"marginTop": "8px"}),
        html.P(id="diff_info",
               style={"color": "#666", "fontSize": "13px", "margin": "0 0 4px 0"}),
        html.Div([
            dcc.Graph(id="diff_spectrum_left", style={"height": "400px", "flex": "1"}),
            dcc.Graph(id="diff_spectrum_right", style={"height": "400px", "flex": "1"}),
        ], style={"display": "flex"}),
    ]),

    _section("Espace projection", [
        dcc.Graph(id="lens_plot", style={"height": "900px"}),
    ]),

    _section("Test Mann-Whitney (sélection vs reste)", [
        html.P(id="mw_info",
               style={"color": "#666", "fontSize": "13px", "margin": "0 0 4px 0"}),
        html.Div([
            dcc.Graph(id="mw_plot_left", style={"height": "500px", "flex": "1"}),
            dcc.Graph(id="mw_plot_right", style={"height": "500px", "flex": "1"}),
        ], style={"display": "flex"}),
    ], open_=False),

    _section("Importance des variables — Loadings Dim 1/2/3", [
        html.P(id="loadings_info",
               style={"color": "#666", "fontSize": "13px", "margin": "0 0 6px 0"}),
        html.Div([
            dcc.Graph(id="loadings_dim1", style={"height": "700px", "flex": "1"}),
            dcc.Graph(id="loadings_dim2", style={"height": "700px", "flex": "1"}),
            dcc.Graph(id="loadings_dim3", style={"height": "700px", "flex": "1"}),
        ], style={"display": "flex"}),
    ], open_=False),

], style={"padding": "20px", "fontFamily": "sans-serif"})


def _make_lens_fig(lens_store, sel_idx=None, meta_cur=None):
    """Construit le lens plot depuis lens_store (toujours à jour).
    sel_idx : indices sélectionnés pour la surbrillance."""
    if not lens_store:
        return go.Figure()
    dim_x     = np.array(lens_store["dim_x"])
    dim_y     = np.array(lens_store["dim_y"])
    color     = lens_store["color"]
    methodes  = lens_store["methodes"]
    lens_nm   = lens_store.get("lens", "")
    souche    = lens_store.get("souche", "")
    treatment = lens_store.get("treatment", "")
    source    = lens_store.get("source", "")

    fig = go.Figure()
    fig.add_scatter(
        x=dim_x, y=dim_y, mode="markers",
        marker=dict(color=color, colorscale="Viridis", size=7),
        text=methodes, name="tous",
        hovertemplate="%{text}<extra></extra>",
    )
    if sel_idx is not None and meta_cur is not None:
        idx_arr      = np.array(sel_idx)
        methodes_sel = meta_cur["methodes"].iloc[sel_idx].values
        souches_sel  = meta_cur["souches"].iloc[sel_idx].values
        fig.add_scatter(
            x=dim_x[idx_arr], y=dim_y[idx_arr],
            mode="markers",
            marker=dict(color="red", size=12, symbol="circle-open",
                        line=dict(width=2.5)),
            name="sélection",
            text=[f"{m} — {s}" for m, s in zip(methodes_sel, souches_sel)],
            hovertemplate="%{text}<extra></extra>",
        )
    fig.update_layout(
        title=f"Espace lens ({lens_nm.upper()}) — {souche} / {treatment} [{source}]",
        xaxis_title="Dim 1",
        yaxis_title="Dim 2",
        height=900,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CALLBACKS UTILITAIRES
# ─────────────────────────────────────────────────────────────────────────────

@app.callback(
    Output("lens_dropdown", "options"),
    Output("lens_dropdown", "value"),
    Output("sheet_div",     "style"),
    Input("mode_radio", "value"),
)
def sync_mode_ui(mode):
    visible = {"flex": "1", "marginRight": "15px"}
    hidden  = {"flex": "1", "marginRight": "15px", "display": "none"}
    if mode == "mono":
        return LENS_MONO, "plsda", visible
    return LENS_MB, "mfa", hidden


@app.callback(
    Output("color_legend", "children"),
    Input("treatment_dropdown", "value"),
)
def update_legend(treatment):
    def dot(c):
        return html.Div(style={"width": "20px", "height": "20px",
                                "backgroundColor": c, "borderRadius": "50%",
                                "display": "inline-block", "marginRight": "8px"})
    return html.Div([
        html.H4("Légende couleur nœud"),
        html.Div([dot("#FDE725"), html.Span("Control")]),
        html.Br(),
        html.Div([dot("#4B0082"), html.Span(treatment)]),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# CALLBACK BANDES — sélection + preview
# ─────────────────────────────────────────────────────────────────────────────

@app.callback(
    Output("band_selector_mono_div",  "style"),
    Output("band_selector_multi_div", "style"),
    Input("mode_radio", "value"),
)
def toggle_band_selectors(mode):
    if mode == "multi":
        return {"display": "none"}, {"display": "block"}
    return {"display": "block"}, {"display": "none"}


def _extract_band_from_selection(selected_data, current_bands, clear):
    """Transforme une box-select Plotly en nouvelle bande [xmin, xmax],
    ajoutée à la liste des bandes déjà choisies (les bandes s'accumulent :
    chaque nouveau glisser-sélectionner ajoute une bande supplémentaire,
    seul le bouton "Effacer" repart de zéro)."""
    bands = list(current_bands) if current_bands else []
    if clear:
        return []
    if selected_data:
        pts = selected_data.get("points", [])
        xs  = [p["x"] for p in pts if p.get("curveNumber") is not None]
        if xs:
            xmin, xmax = min(xs), max(xs)
            # On ignore les sélections trop étroites (probablement un clic
            # accidentel plutôt qu'un vrai glisser-sélectionner de bande).
            if abs(xmax - xmin) > 1:
                bands.append([xmin, xmax])
    return bands


# Callback mono
@app.callback(
    Output("bands_store",   "data"),
    Output("band_selector", "figure"),
    Output("bands_summary", "children"),
    Input("band_selector",   "selectedData"),
    Input("clear_bands_btn", "n_clicks"),
    Input("mode_radio",      "value"),
    Input("sheet_radio",     "value"),
    State("bands_store",     "data"),
)
def update_bands_mono(selected_data, clear_clicks, mode, sheet, current_bands):
    # Ce callback a plusieurs Inputs (sélection, clic "effacer", changement
    # de mode/source) : callback_context permet de savoir lequel a
    # déclenché l'appel, pour ne traiter la sélection que si c'est bien elle
    # qui est à l'origine du déclenchement (sinon on la relirait à tort à
    # chaque changement de mode ou de source spectrale).
    ctx       = callback_context
    triggered = ctx.triggered[0]["prop_id"] if ctx.triggered else ""
    do_clear  = "clear_bands_btn" in triggered
    sel       = selected_data if "band_selector.selectedData" in triggered else None
    bands     = _extract_band_from_selection(sel, current_bands, do_clear)
    fig       = _make_band_preview_mono(sheet, bands)
    total     = len(X0.columns) if sheet == "0" else len(X1.columns)
    n_kept    = len(_filter_cols(X0 if sheet == "0" else X1, bands, None))
    summary   = (f"{len(bands)} bande(s) → {n_kept}/{total} vars"
                 if bands else f"Toutes les {total} variables")
    return bands, fig, summary


# Callback OPTIR (multi)
@app.callback(
    Output("bands_optir",         "data"),
    Output("band_selector_optir", "figure"),
    Input("band_selector_optir", "selectedData"),
    Input("clear_bands_btn",     "n_clicks"),
    State("bands_optir",         "data"),
)
def update_bands_optir(selected_data, clear_clicks, current_bands):
    ctx       = callback_context
    triggered = ctx.triggered[0]["prop_id"] if ctx.triggered else ""
    do_clear  = "clear_bands_btn" in triggered
    sel       = selected_data if "band_selector_optir.selectedData" in triggered else None
    bands     = _extract_band_from_selection(sel, current_bands, do_clear)
    fig       = _make_band_preview_bloc("OPTIR_", "steelblue", bands)
    return bands, fig


# Callback Raman (multi)
@app.callback(
    Output("bands_raman",         "data"),
    Output("band_selector_raman", "figure"),
    Input("band_selector_raman", "selectedData"),
    Input("clear_bands_btn",     "n_clicks"),
    State("bands_raman",         "data"),
)
def update_bands_raman(selected_data, clear_clicks, current_bands):
    ctx       = callback_context
    triggered = ctx.triggered[0]["prop_id"] if ctx.triggered else ""
    do_clear  = "clear_bands_btn" in triggered
    sel       = selected_data if "band_selector_raman.selectedData" in triggered else None
    bands     = _extract_band_from_selection(sel, current_bands, do_clear)
    fig       = _make_band_preview_bloc("Raman_", "crimson", bands)
    return bands, fig


# ─────────────────────────────────────────────────────────────────────────────
# CALLBACK représentation graphique
# ─────────────────────────────────────────────────────────────────────────────

@app.callback(
    Output("graph", "layout"),
    Input("layout_dropdown", "value"),
)
def update_layout(layout):
    if layout == "cose":
        return {
            "name": "cose",
            "animate": False,
            "nodeRepulsion": 1500000,
            "idealEdgeLength": 150,
            "edgeElasticity": 100,
            "gravity": 0.2,
            "numIter": 3000,
        }
    return {"name": layout, "animate": False}


# ─────────────────────────────────────────────────────────────────────────────
# CALLBACK PRINCIPAL — Mapper + Loadings
# ─────────────────────────────────────────────────────────────────────────────

@app.callback(
    Output("graph",         "elements"),
    Output("graph",         "selectedNodeData"),
    Output("loadings_dim1", "figure"),
    Output("loadings_dim2", "figure"),
    Output("loadings_dim3", "figure"),
    Output("loadings_info", "children"),
    Output("lens_data",     "data"),
    Input("souche_dropdown",     "value"),
    Input("treatment_dropdown",  "value"),
    Input("mode_radio",          "value"),
    Input("sheet_radio",         "value"),
    Input("lens_dropdown",       "value"),
    Input("eps_slider",          "value"),
    Input("min_samples_slider",  "value"),
    Input("n_cubes_slider",      "value"),
    Input("overlap_slider",      "value"),
    Input("n_components_slider", "value"),
    Input("bands_store",         "data"),
    Input("bands_optir",         "data"),
    Input("bands_raman",         "data"),
)
def update_graph(souche, treatment, mode, sheet, lens,
                  eps, min_samples, n_cubes, perc_overlap, n_components,
                  bands_mono, bands_optir, bands_raman):
    t_total_start = time.perf_counter()

    # ── Données brutes ────────────────────────────────────────────────────────
    if mode == "multi":
        X_all, meta_all = X_mb, meta_mb
    else:
        X_all, meta_all = (X0, meta0) if sheet == "0" else (X1, meta1)

    # Ne garde que les individus de la souche choisie, en ne comparant que
    # control vs. le traitement sélectionné (les 3 autres traitements sont
    # écartés pour cette vue).
    X_cur, meta_cur = ttmd.get_control_vs_method(X_all, meta_all, souche, treatment)

    # ── Filtrage par bandes ───────────────────────────────────────────────────
    # Si l'utilisateur a sélectionné des bandes spectrales via les graphes
    # d'aperçu, on ne garde que les variables (nombres d'onde) qui tombent
    # dans une de ces bandes ; sinon toutes les variables sont utilisées.
    if mode == "multi":
        cols_o = _filter_cols(X_cur, bands_optir or [], "OPTIR_")
        cols_r = _filter_cols(X_cur, bands_raman or [], "Raman_")
        kept   = cols_o + cols_r
        X_filt = X_cur[kept] if kept else X_cur
    else:
        kept   = _filter_cols(X_cur, bands_mono or [], None)
        X_filt = X_cur[kept] if kept else X_cur

    if X_filt.shape[1] == 0:
        # Filet de sécurité : si les bandes choisies ne recoupent aucune
        # variable (ex. bande hors plage du spectre), on retombe sur toutes
        # les variables plutôt que de planter avec une matrice vide.
        X_filt = X_cur

    # ── Lens ──────────────────────────────────────────────────────────────────
    t_projection_start = time.perf_counter()
    model_cur, Z_cur = ttmd.compute_lens(
        X_filt, meta=meta_cur, lens=lens, n_components=n_components)
    temps_projection = time.perf_counter() - t_projection_start

    # ── Variance expliquée ───────────────────────────────────────────────────
    ev = ttmd.explained_information(model_cur, X_filt, lens, Z_cur)

    # ── Mapper ────────────────────────────────────────────────────────────────
    t_mapper_start = time.perf_counter()
    graph = ttmd.build_mapper_graph(
        Z_cur, X_filt, eps=eps, min_samples=min_samples,
        n_cubes=n_cubes, perc_overlap=perc_overlap)
    temps_mapper = time.perf_counter() - t_mapper_start

    t_format_start = time.perf_counter()
    purity_dict = ttmd.node_purity(graph, meta_cur)
    le_local    = LabelEncoder()
    color_cur   = le_local.fit_transform(meta_cur["methodes"])
    elements    = ttmd.graph_to_cytoscape(graph, meta_cur, color_cur, purity_dict)
    temps_format = time.perf_counter() - t_format_start

    # Cache — clé incluant les bandes pour que le cache se mette à jour
    _bands_key = (
        tuple(tuple(b) for b in sorted(bands_mono or [])),
        tuple(tuple(b) for b in sorted(bands_optir or [])),
        tuple(tuple(b) for b in sorted(bands_raman or [])),
    )
    _cache_key = (souche, treatment, mode, sheet, lens, _bands_key)
    all_graphs[_cache_key] = {
        "X":            X_cur,
        "X_filt":       X_filt,
        "meta":         meta_cur,
        "Z":            Z_cur,
        "color":        color_cur,
        "model":        model_cur,
        "graph":        graph,
        "purity":       purity_dict,
        "bands_mono":   bands_mono or [],
        "bands_optir":  bands_optir or [],
        "bands_raman":  bands_raman or [],
        "ev":           ev,
    }

    # ── Loadings ──────────────────────────────────────────────────────────────
    t_loadings_start = time.perf_counter()
    loadings = ttmd.extract_loadings(model_cur, X_filt, lens, top_n=20)
    temps_loadings = time.perf_counter() - t_loadings_start
    temps_total = time.perf_counter() - t_total_start

    # ── Log ───────────────────────────────────────────────────────────────────
    n_nodes = len(graph.get("nodes", {}))
    n_edges = sum(len(t) for t in graph.get("links", {}).values())
    
    # Pureté moyenne des nœuds
    if purity_dict:
        mean_purity = np.mean([v["purity"] for v in purity_dict.values()])
        prop_mixed = np.mean([v["purity"] < 1 for v in purity_dict.values()])
    else:
        mean_purity = np.nan
        prop_mixed = np.nan
    
    print("-" * 70)
    print(f"Mode={mode} | Source={sheet} | Lens={lens} | "
          f"Souche={souche} | Traitement={treatment}")
    print(f"n_components={n_components} | n_cubes={n_cubes} | "
          f"overlap={perc_overlap:.2f} | eps={eps:.2f} | min_samples={min_samples}")
    print(f"Individus={X_filt.shape[0]} | Variables={X_filt.shape[1]} | "
          f"Nœuds={n_nodes} | Arêtes={n_edges}")
    print(f"Pureté moyenne={mean_purity:.3f} | "
          f"Nœuds mixtes={prop_mixed * 100:.1f}%")
    print(f"Projection={temps_projection:.3f}s | Mapper={temps_mapper:.3f}s | "
          f"Mise en forme={temps_format:.3f}s | Loadings={temps_loadings:.3f}s | "
          f"Total={temps_total:.3f}s")

    def _load_fig(df_load, dim_label):
        if df_load is None or df_load.empty:
            f = go.Figure()
            f.update_layout(title=f"Loadings {dim_label} — non disponible")
            return f
        colors = ["#d62728" if s == "positive" else "#1f77b4"
                  for s in df_load["sign"]]
        f = go.Figure(go.Bar(
            x=df_load["loading"], y=df_load["variable"],
            orientation="h", marker_color=colors,
            hovertemplate="%{y}<br>loading = %{x:.4f}<extra></extra>",
        ))
        f.add_vline(x=0, line_width=1, line_color="black")
        f.update_layout(
            title=f"Loadings {dim_label} ({lens.upper()})",
            xaxis_title=f"Loading {dim_label}",
            yaxis=dict(tickfont=dict(size=10), autorange="reversed"),
            margin=dict(l=160, r=20, t=40, b=40), showlegend=False,
        )
        return f

    def _dim_label(i, name):
        pct = ev.get(i)
        return f"{name} ({pct}%)" if pct is not None else name

    lf1 = _load_fig(loadings.get("dim1"), _dim_label(0, "Dim 1"))
    lf2 = _load_fig(loadings.get("dim2"), _dim_label(1, "Dim 2"))
    lf3 = _load_fig(loadings.get("dim3"), _dim_label(2, "Dim 3"))

    _any_bands = any([bands_mono, bands_optir, bands_raman])
    band_note = (f" ({X_filt.shape[1]}/{X_cur.shape[1]} vars après filtrage)"
                 if _any_bands else "")
    load_info = (f"Top 20 positifs 🔴 + top 20 négatifs 🔵 / dimension "
                 f"| {lens.upper()}{band_note}")

    # ── Lens plot (affiché dès le chargement) ─────────────────────────────────
    source_g = {("mono", "0"): "OPTIR", ("mono", "1"): "Raman"}.get(
        (mode, sheet), mode.upper())
    lens_store = {
        "dim_x":     Z_cur[:, 0].tolist(),
        "dim_y":     (Z_cur[:, 1] if Z_cur.shape[1] > 1
                      else np.zeros(len(Z_cur))).tolist(),
        "color":     color_cur.tolist(),
        "methodes":  meta_cur["methodes"].tolist(),
        "souches":   meta_cur["souches"].tolist(),
        "ev":        {str(k): v for k, v in ev.items()},
        "lens":      lens,
        "souche":    souche,
        "treatment": treatment,
        "source":    source_g,
    }
    return elements, [], lf1, lf2, lf3, load_info, lens_store


# ─────────────────────────────────────────────────────────────────────────────
# CALLBACK SÉLECTION
# ─────────────────────────────────────────────────────────────────────────────

@app.callback(
    Output("selection_info",      "children"),
    Output("mean_spectrum_left",  "figure"),
    Output("mean_spectrum_right", "figure"),
    Output("diff_spectrum_left",  "figure"),
    Output("diff_spectrum_right", "figure"),
    Output("diff_info",           "children"),
    Output("lens_plot",           "figure"),
    Output("mw_plot_left",        "figure"),
    Output("mw_plot_right",       "figure"),
    Output("mw_info",             "children"),
    Input("graph",               "selectedNodeData"),
    Input("souche_dropdown",     "value"),
    Input("treatment_dropdown",  "value"),
    Input("mode_radio",          "value"),
    Input("sheet_radio",         "value"),
    Input("lens_dropdown",       "value"),
    Input("bands_store",         "data"),
    Input("bands_optir",         "data"),
    Input("bands_raman",         "data"),
    Input("lens_data",           "data"),
)
def update_selection(nodes, souche, treatment, mode, sheet, lens,
                      bands_mono, bands_optir, bands_raman, lens_store):

    empty = go.Figure()

    # Chercher la clé la plus récente pour ce (souche, treatment, mode, sheet, lens)
    _bands_key = (
        tuple(tuple(b) for b in sorted(bands_mono or [])),
        tuple(tuple(b) for b in sorted(bands_optir or [])),
        tuple(tuple(b) for b in sorted(bands_raman or [])),
    )
    _cache_key = (souche, treatment, mode, sheet, lens, _bands_key)

    current = all_graphs.get(_cache_key)
    if current is None:
        return ("Pas de données", empty, empty, empty, empty, "",
                _make_lens_fig(lens_store), empty, empty, "")

    X_cur    = current["X"]
    X_filt   = current["X_filt"]
    meta_cur = current["meta"]
    
    
    
    
    

    source_label = {("mono", "0"): "OPTIR", ("mono", "1"): "Raman"}.get(
        (mode, sheet), mode.upper())

    if not nodes:
        return ("Aucune sélection", empty, empty, empty, empty, "",
                _make_lens_fig(lens_store), empty, empty, "")

    # Un individu peut appartenir à plusieurs nœuds sélectionnés en même
    # temps (le Mapper autorise le chevauchement) : on déduplique via un set
    # pour ne pas le compter deux fois dans les spectres/tests ci-dessous.
    idx      = sorted({i for n in nodes for i in n["members"]})
    X_sel    = X_cur.iloc[idx]
    meta_sel = meta_cur.iloc[idx]

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _spectrum_fig(X_sub, prefix, color, title, x_axis=None):
        cols = ([c for c in X_sub.columns if c.startswith(prefix)]
                if prefix else list(X_sub.columns))
        if not cols:
            return go.Figure()
        x_vals = (x_axis.astype(float) if x_axis is not None
                  else np.array([_col_to_float(c) for c in cols]))
        order  = np.argsort(x_vals)
        x_s, mean_s, std_s = (x_vals[order],
                               X_sub[cols].mean(axis=0).values[order],
                               X_sub[cols].std(axis=0).values[order])
        f = go.Figure()
        f.add_trace(go.Scatter(
            x=np.concatenate([x_s, x_s[::-1]]),
            y=np.concatenate([mean_s + std_s, (mean_s - std_s)[::-1]]),
            fill="toself", opacity=0.2, fillcolor=color,
            line=dict(color="rgba(0,0,0,0)"), name="±std"))
        f.add_trace(go.Scatter(x=x_s, y=mean_s, mode="lines",
                               name="mean", line=dict(color=color)))
        f.update_xaxes(autorange="reversed")
        f.update_layout(title=title,
                        xaxis_title="Nombre d'onde", yaxis_title="Intensité")
        return f

    def _diff_fig(X_sub, meta_sub, prefix, color, title, treatment_name,
                  x_axis=None):
        cols = ([c for c in X_sub.columns if c.startswith(prefix)]
                if prefix else list(X_sub.columns))
        if not cols:
            return go.Figure(), ""
        mask_c = meta_sub["methodes"] == "control"
        mask_t = meta_sub["methodes"] == treatment_name
        if mask_c.sum() == 0 or mask_t.sum() == 0:
            f = go.Figure()
            f.update_layout(title=f"{title} — groupes incomplets")
            return f, "Sélection incomplète"
        x_vals = (x_axis.astype(float) if x_axis is not None
                  else np.array([_col_to_float(c) for c in cols]))
        order  = np.argsort(x_vals)
        x_s    = x_vals[order]
        diff   = (X_sub.loc[mask_t, cols].mean(axis=0).values[order] -
                  X_sub.loc[mask_c, cols].mean(axis=0).values[order])
        f = go.Figure()
        f.add_hline(y=0, line_width=1, line_color="black", line_dash="dot")
        f.add_trace(go.Scatter(x=x_s, y=np.where(diff > 0, diff, 0),
                               fill="tozeroy",
                               fillcolor="rgba(214,39,40,0.25)",
                               line=dict(color="rgba(0,0,0,0)"),
                               name="traitement > control"))
        f.add_trace(go.Scatter(x=x_s, y=np.where(diff < 0, diff, 0),
                               fill="tozeroy",
                               fillcolor="rgba(31,119,180,0.25)",
                               line=dict(color="rgba(0,0,0,0)"),
                               name="traitement < control"))
        f.add_trace(go.Scatter(x=x_s, y=diff, mode="lines",
                               line=dict(color=color), name="Δ"))
        f.update_xaxes(autorange="reversed")
        f.update_layout(title=title,
                        xaxis_title="Nombre d'onde", yaxis_title="Δ Intensité")
        return f, (f"control : {mask_c.sum()} | {treatment_name} : {mask_t.sum()}")

    # ── Spectres moyens ───────────────────────────────────────────────────────
    if mode == "multi":
        fig_left  = _spectrum_fig(X_sel, "OPTIR_", "steelblue",
                                  f"OPTIR — {len(idx)} individus")
        fig_right = _spectrum_fig(X_sel, "Raman_", "crimson",
                                  f"Raman — {len(idx)} individus")
    else:
        fn   = feature_names0 if sheet == "0" else feature_names1
        name = "OPTIR" if sheet == "0" else "Raman"
        fig_left  = _spectrum_fig(X_sel, None, "steelblue",
                                  f"{name} — {len(idx)} individus", x_axis=fn)
        fig_right = go.Figure()

    # ── Spectre différentiel ──────────────────────────────────────────────────
    if mode == "multi":
        diff_left,  info_l = _diff_fig(X_sel, meta_sel, "OPTIR_", "steelblue",
                                       "Δ OPTIR", treatment)
        diff_right, info_r = _diff_fig(X_sel, meta_sel, "Raman_", "crimson",
                                       "Δ Raman", treatment)
        diff_info = f"OPTIR : {info_l} | Raman : {info_r}"
    else:
        fn   = feature_names0 if sheet == "0" else feature_names1
        name = "OPTIR" if sheet == "0" else "Raman"
        diff_left, diff_info = _diff_fig(X_sel, meta_sel, None, "steelblue",
                                         f"Δ {name}", treatment, x_axis=fn)
        diff_right = go.Figure()

    # ── Mann-Whitney ──────────────────────────────────────────────────────────
    # On compare la sélection de l'utilisateur au "reste du Mapper", c'est-
    # à-dire tous les individus présents dans au moins un nœud du graphe
    # mais pas dans la sélection courante (pas à toute la population brute :
    # certains individus peuvent ne faire partie d'aucun nœud si le Mapper
    # ne les a pas rattachés à un cube).
    graph = current["graph"]
    idx_in_mapper = sorted({
        i for members in graph["nodes"].values() for i in members
    })
    n_reste = len(set(idx_in_mapper) - set(idx))

    def _safe_float(s):
        try:
            return float(s)
        except (ValueError, TypeError):
            return None

    def _mw_fig(df_mw, prefix, bloc_name, bands=None):
        if df_mw is None or df_mw.empty:
            return go.Figure()

        sub = (
            df_mw[df_mw["variable"].str.startswith(prefix)].copy()
            if prefix else df_mw.copy()
        )
        if sub.empty:
            return go.Figure()

        sub["x_val"] = sub["variable"].apply(
            lambda c: next(
                (float(p) for p in reversed(str(c).split("_"))
                 if _safe_float(p) is not None),
                np.nan
            )
        )
        sub = sub.dropna(subset=["x_val"]).sort_values("x_val", ascending=False)
        if sub.empty:
            return go.Figure()

        colors = np.where(sub["significant"], "#d62728", "#aaaaaa")

        f = go.Figure()
        f.add_trace(go.Scatter(
            x=sub["x_val"],
            y=-np.log10(sub["pvalue_corrected"].clip(lower=1e-300)),
            mode="markers",
            marker=dict(color=colors, size=5),
            text=sub["variable"],
            hovertemplate=(
                "%{text}"
                "<br>p corrigée = %{customdata:.3e}"
                "<br>-log10(p corrigée) = %{y:.2f}<extra></extra>"
            ),
            customdata=sub["pvalue_corrected"],
        ))
        f.add_hline(
            y=-np.log10(0.05),
            line_dash="dash",
            line_color="orange",
            annotation_text="p corrigée = 0.05"
        )
        if bands:
            xmin = min(min(b) for b in bands)
            xmax = max(max(b) for b in bands)
            f.update_xaxes(range=[xmax, xmin])
        else:
            f.update_xaxes(autorange="reversed")
        f.update_layout(
            title=f"Variables discriminantes — {bloc_name}",
            xaxis_title="Nombre d'onde",
            yaxis_title="-log10(p corrigée)",
        )
        return f

    if n_reste < 2 or len(idx) < 2:
        mw_left  = go.Figure()
        mw_right = go.Figure()
        mw_info  = f"Test non calculé : sélection = {len(idx)}, reste = {n_reste}"
    else:
        df_mw = ttmd.mann_whitney_selection(X_filt, idx, idx_in_mapper=idx_in_mapper)

        if mode == "multi":
            mw_left  = _mw_fig(df_mw, "OPTIR_", "OPTIR", current.get("bands_optir", []))
            mw_right = _mw_fig(df_mw, "Raman_", "Raman", current.get("bands_raman", []))
        else:
            name = "OPTIR" if sheet == "0" else "Raman"
            mw_left  = _mw_fig(df_mw, None, name, current.get("bands_mono", []))
            mw_right = go.Figure()

        if df_mw is not None and not df_mw.empty:
            n_sig = int(df_mw["significant"].sum())
            mw_info = (
                f"{n_sig} variables significatives "
                f"(Benjamini-Hochberg, FDR < 0.05) "
                f"sur {len(df_mw)} testées | "
                f"sélection : {len(idx)} vs reste du Mapper : {n_reste}"
            )
        else:
            mw_info = f"Test non calculé : sélection = {len(idx)}, reste = {n_reste}"

    # ── Info sélection ────────────────────────────────────────────────────────
    counts      = meta_sel["methodes"].value_counts()
    n_control   = counts.get("control", 0)
    n_treatment = {m: v for m, v in counts.items() if m != "control"}
    count_rows  = [
        html.Tr([html.Td("control", style={"paddingRight": "20px", "color": "#888"}),
                 html.Td(str(n_control),
                         style={"fontWeight": "bold", "textAlign": "right"})]),
    ] + [
        html.Tr([html.Td(m, style={"paddingRight": "20px", "color": "#4B0082"}),
                 html.Td(str(n), style={"fontWeight": "bold", "textAlign": "right"})])
        for m, n in n_treatment.items()
    ]

    purity_dict = current.get("purity", {})
    purity_rows = []
    for n in nodes:
        nid = n["id"]
        if nid in purity_dict:
            p   = purity_dict[nid]
            col = ("#2ca02c" if p["purity"] == 1
                   else "#ff7f0e" if p["purity"] >= 0.7 else "#d62728")
            purity_rows.append(html.Tr([
                html.Td(nid, style={"paddingRight": "15px", "fontSize": "12px"}),
                html.Td(f"{p['purity']:.0%}",
                        style={"color": col, "fontWeight": "bold"}),
                html.Td(p["dominant"],
                        style={"paddingLeft": "10px", "fontSize": "12px"}),
                html.Td(f"entropie={p['entropy']:.2f}",
                        style={"paddingLeft": "10px", "color": "#888",
                               "fontSize": "11px"}),
            ]))

    has_bands = any([current.get("bands_mono"), current.get("bands_optir"),
                      current.get("bands_raman")])
    band_note = (f" | {X_filt.shape[1]}/{X_cur.shape[1]} vars (bandes)"
                 if has_bands else "")

    info = html.Div([
        html.H3(f"{len(idx)} individus sélectionnés"),
        html.Table(count_rows, style={"marginBottom": "6px", "fontSize": "15px"}),
        html.Details([
            html.Summary("Pureté par nœud",
                         style={"cursor": "pointer", "color": "#555"}),
            html.Table(purity_rows,
                       style={"fontSize": "13px", "marginTop": "4px"}),
        ]) if purity_rows else html.Span(""),
        html.P(f"Souches : {list(meta_sel['souches'].unique())}",
               style={"margin": "4px 0"}),
        html.P(f"Mode : {mode.upper()} | Source : {source_label} "
               f"| Lens : {lens.upper()}{band_note}",
               style={"margin": "2px 0", "color": "#666", "fontSize": "13px"}),
    ])

    lens_sel = _make_lens_fig(lens_store, sel_idx=idx, meta_cur=meta_cur)
    return (info, fig_left, fig_right, diff_left, diff_right, diff_info,
            lens_sel, mw_left, mw_right, mw_info)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)