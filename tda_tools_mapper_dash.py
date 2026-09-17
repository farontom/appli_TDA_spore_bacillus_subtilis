# -*- coding: utf-8 -*-
"""
tda_tools_mapper_dash — fonctions de traitement pour l'app SpectroMapper.

Contient : chargement/préparation des données, calcul du lens (PCA, PLS-DA,
méthodes multibloc), construction du graphe Mapper, pureté des nœuds,
export Cytoscape, test de Mann-Whitney et extraction des loadings.
"""
import numpy as np
import pandas as pd
import prince
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import DBSCAN
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
import kmapper as km
from sklearn.manifold import TSNE
import umap

# ─────────────────────────────────────────────────────────────────────────────
# FICHIERS
# ─────────────────────────────────────────────────────────────────────────────

FILE_MONO  = "02-sans correction ACP.xlsx"
FILE_MULTI = "02-sans correction ACP_concat.xlsx"


def load_data(mode="mono", sheet=0):
    if mode == "multi":
        return pd.read_excel(FILE_MULTI)
    return pd.read_excel(FILE_MONO, sheet_name=int(sheet))


# ─────────────────────────────────────────────────────────────────────────────
# PRÉPARATION
# ─────────────────────────────────────────────────────────────────────────────

def prepare_data(df, color_col="souches"):
    """
    Prends les colonnes 17 et 18, correspondant respectivement aux variables
    souches et méthodes. Ignore les colonnes précédentes ainsi que la colonne
    19, puis conserve toutes les colonnes à partir de la colonne 20.
    """
    # Indices figés car le fichier Excel a toujours la même structure de colonnes :
    # 0-16 = métadonnées diverses non utilisées, 17-18 = souches/méthodes,
    # 19 = colonne tampon ignorée, 20+ = variables spectrales.
    meta = df.iloc[:, [17, 18]].copy()
    meta.columns = ["souches", "methodes"]
    meta = meta.reset_index(drop=True)
    X = df.iloc[:, 20:].copy()
    X = X.apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    X.columns = X.columns.astype(str)
    le = LabelEncoder()
    color_values = le.fit_transform(meta[color_col])
    return X, meta, color_values, le, X.columns


def prepare_multiblock_data(df):
    """
    Prend les colonnes 17 et 18, correspondant respectivement aux variables
    souches et méthodes. Ignore les colonnes précédentes ainsi que la colonne
    19, puis conserve toutes les colonnes à partir de la colonne 20.

    Les variables spectrales sont ensuite séparées en deux blocs selon leur
    préfixe (« OPTIR » ou « RAMAN »), converties en données numériques, puis
    concaténées afin de constituer une matrice multibloc.
    """
    metadata = df.iloc[:, [17, 18]].copy()
    metadata.columns = ["souches", "methodes"]
    metadata = metadata.reset_index(drop=True)
    spec = df.iloc[:, 20:].copy()
    spec.columns = spec.columns.astype(str)
    cols_o = [c for c in spec.columns if c.upper().startswith("OPTIR")]
    cols_r = [c for c in spec.columns if c.upper().startswith("RAMAN")]
    X_optir = spec[cols_o].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    X_raman = spec[cols_r].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    X_concat = pd.concat([X_optir, X_raman], axis=1).reset_index(drop=True)
    return metadata, X_optir, X_raman, X_concat


def get_control_vs_method(X, meta, souche, methode, control_name="control"):
    """
    Sélectionne les individus appartenant à une souche donnée et conserve
    uniquement les échantillons contrôle et ceux correspondant au traitement
    étudié.
    """
    mask_souche = meta["souches"] == souche
    X_s    = X[mask_souche]
    meta_s = meta.loc[mask_souche]
    mask   = meta_s["methodes"].isin([control_name, methode])
    return X_s[mask].reset_index(drop=True), meta_s[mask].reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# LENS
# ─────────────────────────────────────────────────────────────────────────────

def compute_lens(X, meta=None, lens="pca", n_components=7, color_name="methodes"):
    """
    Calcule une représentation de faible dimension (lens) des données
    spectrales. (<10 arbitraire par rapport au temps de calcul si on augmente n_components)

    Selon la méthode choisie, la projection est obtenue à l'aide d'une
    PCA, d'une PLS-DA ou d'une méthode multibloc (MFA, MB-PCA, MB-PLS
    ou O2-PLS). Cette représentation est ensuite utilisée comme fonction
    de projection (lens) pour construire le graphe Mapper.
    """
    if lens == "pca":
        # On ne peut pas demander plus de composantes que d'individus - 1
        # ni plus que de variables : sklearn lèverait une erreur sinon.
        n_comp = min(n_components, X.shape[0] - 1, X.shape[1])
        model  = PCA(n_components=n_comp)
        Z      = model.fit_transform(X)
        return model, Z
    elif lens == "plsda":
        le     = LabelEncoder()
        y      = le.fit_transform(meta[color_name])
        n_comp = min(n_components, X.shape[0] - 1, X.shape[1])
        model  = PLSRegression(n_components=n_comp)
        model.fit(X, y)
        return model, model.x_scores_
    elif lens == "tsne":

        perplexity = min(30, X.shape[0] - 1)

        model = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            random_state=42,
            method="exact"
        )
    
        Z = model.fit_transform(X)
    
        return model, Z
    elif lens == "umap":

        n_comp = min(n_components, X.shape[0] - 2)
    
        model = umap.UMAP(
            n_components=n_comp,
            n_neighbors=min(15, X.shape[0] - 1),
            min_dist=0.1,
            metric="euclidean",
            random_state=42,
        )
    
        Z = model.fit_transform(X)
    
        return model, Z
    elif lens in ["mfa", "mbpca", "mbpls", "o2pls", "mcoa"]:
        return _compute_lens_MB(X, lens=lens, n_components=n_components)


def _split_blocks(X):
    """
    Certaines méthodes multibloc nécessitent que les blocs de données soient
    séparés, tandis que d'autres les considèrent comme un unique bloc de
    variables. Les deux représentations sont donc construites.
    """
    cols_o = [c for c in X.columns if c.startswith("OPTIR_")]
    cols_r = [c for c in X.columns if c.startswith("Raman_")]
    return X[cols_o].astype(float), X[cols_r].astype(float)


def _mbpls_symmetric(Xo, Xr, n_components, max_iter=500, tol=1e-10):
    """
    MB-PLS symétrique (SUM-PLS)
    Maximise la covariance entre les deux blocs de façon symétrique.
    Retourne un dict avec T_common (n × k), W_o, W_r, T, U, P_o, P_r.
    """
    Xo = np.array(Xo, dtype=float)
    Xr = np.array(Xr, dtype=float)
    Xo_res = Xo - Xo.mean(axis=0)
    Xr_res = Xr - Xr.mean(axis=0)

    W_o, W_r   = [], []
    T_list, U_list = [], []
    P_o_list, P_r_list = [], []
    T_common_list = []

    for _ in range(n_components):
        # Initialisation : 1ère colonne de Xr_res
        u = Xr_res[:, 0].copy()

        # Boucle NIPALS : on fait rebondir le score entre les deux blocs
        # (OPTIR → Raman → OPTIR → ...) jusqu'à ce qu'il se stabilise.
        for _ in range(max_iter):
            # Loading OPTIR
            w_o = Xo_res.T @ u
            w_o /= np.linalg.norm(w_o) + 1e-12
            # Score OPTIR
            t = Xo_res @ w_o

            # Loading Raman
            w_r = Xr_res.T @ t
            w_r /= np.linalg.norm(w_r) + 1e-12
            # Score Raman
            u_new = Xr_res @ w_r

            if np.linalg.norm(u_new - u) < tol:
                u = u_new
                break
            u = u_new

        # Normalisation des scores avant fusion afin d'éviter qu'un bloc domine
        # le score commun uniquement par son échelle.
        t_norm = t / (np.linalg.norm(t) + 1e-12)
        u_norm = u / (np.linalg.norm(u) + 1e-12)

        # Score commun symétrique OPTIR/Raman utilisé comme coordonnée latente.
        t_common = (t_norm + u_norm) / 2

        # Loadings de régression (pour la déflation)
        tt = float(t @ t)
        uu = float(u @ u)
        p_o = Xo_res.T @ t / (tt + 1e-12)
        p_r = Xr_res.T @ u / (uu + 1e-12)

        # Déflation séquentielle : on retire la composante qu'on vient de
        # trouver, pour que la prochaine itération cherche une direction
        # différente (non redondante) dans les données restantes.
        Xo_res = Xo_res - np.outer(t, p_o)
        Xr_res = Xr_res - np.outer(u, p_r)

        W_o.append(w_o); W_r.append(w_r)
        T_list.append(t); U_list.append(u)
        P_o_list.append(p_o); P_r_list.append(p_r)
        T_common_list.append(t_common)

    return {
        "W_o":      np.column_stack(W_o),
        "W_r":      np.column_stack(W_r),
        "T":        np.column_stack(T_list),
        "U":        np.column_stack(U_list),
        "P_o":      np.column_stack(P_o_list),
        "P_r":      np.column_stack(P_r_list),
        "T_common": np.column_stack(T_common_list),
        "x_weights_": np.column_stack(W_o),   # alias pour extract_loadings
        "x_scores_":  np.column_stack(T_list),
        "x_loadings_":np.column_stack(P_o_list),
    }


def _o2pls(Xo, Xr, n_components, n_orth=1, max_iter=500, tol=1e-10):
    """
    O2-PLS
    1. MB-PLS symétrique pour identifier la direction commune
    2. Retire la variation orthogonale (spécifique) de chaque bloc
       via déflation dans l'espace OBSERVATION (scores)
    3. MB-PLS symétrique sur les résidus = variation commune pure

    n_orth : nombre de composantes orthogonales à retirer par bloc.
    """
    Xo = np.array(Xo, dtype=float)
    Xr = np.array(Xr, dtype=float)
    Xo_c = Xo - Xo.mean(axis=0)
    Xr_c = Xr - Xr.mean(axis=0)

    # ── Étape 1 : direction commune initiale ─────────────────────────────────
    mb_init = _mbpls_symmetric(Xo_c, Xr_c, n_components=1,
                                max_iter=max_iter, tol=tol)
    t_joint = mb_init["T"][:, 0]    # score commun OPTIR (n,)
    u_joint = mb_init["U"][:, 0]    # score commun Raman  (n,)

    def _remove_orth(X_res, t_ref):
        """
        Retire les composantes orthogonales à t_ref de X_res.
        t_ref : vecteur de référence dans l'espace observation (n,)
        Retourne X_deflated, liste des scores orthogonaux.
        """
        t_scores_orth = []
        for _ in range(n_orth):
            # Score candidat : 1ère CP de X_res
            U_svd, s, Vt = np.linalg.svd(X_res, full_matrices=False)
            t_cand = U_svd[:, 0] * s[0]   # ~ 1ère composante principale

            # Projection orthogonale à t_ref dans l'espace obs
            tt_ref = float(t_ref @ t_ref)
            if tt_ref < 1e-12:
                break
            t_orth = t_cand - (float(t_cand @ t_ref) / tt_ref) * t_ref
            norm_orth = float(t_orth @ t_orth)
            if norm_orth < 1e-12:
                break

            # Loading orthogonal et déflation
            p_orth = X_res.T @ t_orth / norm_orth
            X_res  = X_res - np.outer(t_orth, p_orth)
            t_scores_orth.append(t_orth / np.sqrt(norm_orth))

        return X_res, t_scores_orth

    # ── Étape 2 : déflater Xo (référence = score joint OPTIR) ────────────────
    Xo_filt, T_orth_o_list = _remove_orth(Xo_c.copy(), t_joint)

    # ── Étape 3 : déflater Xr (référence = score joint Raman) ────────────────
    Xr_filt, T_orth_r_list = _remove_orth(Xr_c.copy(), u_joint)

    # ── Étape 4 : MB-PLS symétrique sur la variation commune ─────────────────
    mb_common = _mbpls_symmetric(Xo_filt, Xr_filt, n_components,
                                  max_iter=max_iter, tol=tol)

    n = Xo.shape[0]
    mb_common["T_orth_o"] = (np.column_stack(T_orth_o_list)
                              if T_orth_o_list else np.zeros((n, 0)))
    mb_common["T_orth_r"] = (np.column_stack(T_orth_r_list)
                              if T_orth_r_list else np.zeros((n, 0)))
    return mb_common


def _mcoa(blocks, n_components=3, max_iter=500, tol=1e-10):
    """
    MCOA — Multiple Co-Inertia Analysis

    Cherche les axes qui maximisent la somme des co-inerties entre
    chaque bloc et un tableau de référence commun Z.

    Algorithme NIPALS multi-bloc :
      1. ACP sur chaque bloc Xi → scores Ui (n × k)
      2. Init : Z = moyenne des Ui[:,0]
      3. Pour chaque composante :
         a. Calculer les loadings : ai = Ui.T @ Z / ||Ui.T @ Z||
         b. Calculer les scores   : ti = Ui @ ai
         c. Mettre à jour Z = somme pondérée des ti
         d. Normaliser Z
         e. Répéter jusqu'à convergence
      4. Déflater chaque Ui

    Paramètres
    ----------
    blocks : liste de arrays (n × pi) — les blocs centrés/réduits
    n_components : int
    Retourne un dict avec T_common (n × k), loadings, scores par bloc.
    """
    blocks = [np.array(b, dtype=float) for b in blocks]
    K      = len(blocks)

    # ── Étape 1 : ACP sur chaque bloc ────────────────────────────────────────
    pca_scores = []
    pca_models = []
    for Xk in blocks:
        Xk_c = Xk - Xk.mean(axis=0)
        n_comp_k = min(n_components * 2, Xk_c.shape[0] - 1, Xk_c.shape[1])
        model_k  = PCA(n_components=n_comp_k)
        Uk       = model_k.fit_transform(Xk_c)
        pca_scores.append(Uk)
        pca_models.append(model_k)

    T_common_list = []
    A_list        = [[] for _ in range(K)]   # loadings par bloc
    T_list        = [[] for _ in range(K)]   # scores par bloc

    # Copie de travail des scores ACP
    U_res = [U.copy() for U in pca_scores]

    for comp in range(n_components):
        # Init Z : première colonne du premier bloc résiduel
        Z = U_res[0][:, 0].copy()
        Z /= np.linalg.norm(Z) + 1e-12

        for _ in range(max_iter):
            # a. Loading de chaque bloc sur Z
            a_k_list = []
            t_k_list = []
            for k, Uk in enumerate(U_res):
                a_k = Uk.T @ Z
                n_a = np.linalg.norm(a_k)
                if n_a > 1e-12:
                    a_k /= n_a
                t_k = Uk @ a_k
                a_k_list.append(a_k)
                t_k_list.append(t_k)

            # b. Nouveau Z = moyenne des scores
            Z_new = np.mean(np.column_stack(t_k_list), axis=1)
            n_z   = np.linalg.norm(Z_new)
            if n_z > 1e-12:
                Z_new /= n_z

            if np.linalg.norm(Z_new - Z) < tol:
                Z = Z_new
                break
            Z = Z_new

        # Score commun = Z non normalisé (remettre l'échelle)
        t_common = np.mean(np.column_stack(t_k_list), axis=1)
        T_common_list.append(t_common)

        for k in range(K):
            A_list[k].append(a_k_list[k])
            T_list[k].append(t_k_list[k])

        # c. Déflation : retirer la contribution de t_common de chaque Uk
        for k, Uk in enumerate(U_res):
            t_k = t_k_list[k]
            tt  = float(t_k @ t_k)
            if tt > 1e-12:
                p_k    = Uk.T @ t_k / tt
                U_res[k] = Uk - np.outer(t_k, p_k)

    result = {
        "T_common":   np.column_stack(T_common_list),
        "pca_models": pca_models,
        "A":          [np.column_stack(A_list[k]) for k in range(K)],
        "T_blocks":   [np.column_stack(T_list[k]) for k in range(K)],
        # Alias pour explained_variance et extract_loadings
        "x_scores_":  np.column_stack(T_list[0]),   # scores bloc 1 (OPTIR)
        "x_weights_": np.column_stack(A_list[0]),    # loadings bloc 1
        "x_loadings_":np.column_stack(A_list[0]),
    }
    return result


def _compute_lens_MB(X, lens="mfa", n_components=3):
    """
    Fonction utilisée par `compute_lens` pour calculer les méthodes
    de réduction de dimension multibloc (MFA, MB-PCA, MB-PLS et O2-PLS).
    """
    Xo, Xr = _split_blocks(X)
    n_comp  = max(2, min(n_components, Xo.shape[0] - 1, Xo.shape[1], Xr.shape[1]))
    if lens == "mfa":
        groups = {"OPTIR": list(Xo.columns), "Raman": list(Xr.columns)}
        X_full = pd.concat([Xo, Xr], axis=1)
        model  = prince.MFA(n_components=n_comp).fit(X_full, groups=groups)
        return model, model.row_coordinates(X_full).values
    elif lens == "mbpca":
        X_mb  = pd.concat([Xo / np.sqrt(Xo.shape[1]), Xr / np.sqrt(Xr.shape[1])], axis=1)
        model = PCA(n_components=n_comp)
        return model, model.fit_transform(X_mb)
    elif lens == "mbpls":
        model = _mbpls_symmetric(Xo.values, Xr.values, n_comp)
        return model, model["T_common"]
    elif lens == "o2pls":
        model = _o2pls(Xo.values, Xr.values, n_comp)
        return model, model["T_common"]
    elif lens == "mcoa":
        blocks = [Xo.values, Xr.values]
        model  = _mcoa(blocks, n_components=n_comp)
        return model, model["T_common"]


# ─────────────────────────────────────────────────────────────────────────────
# MAPPER
# ─────────────────────────────────────────────────────────────────────────────

def build_mapper_graph(Z, X, eps=2, min_samples=1, n_cubes=10, perc_overlap=0.3):
    """
    Construit le graphe Mapper à partir de la projection (lens) des données.

    Les individus sont recouverts par un ensemble de cubes chevauchants
    (`Cover`), puis regroupés localement à l'aide de l'algorithme DBSCAN.
    Les clusters obtenus sont ensuite reliés pour former le graphe Mapper.
    """
    mapper = km.KeplerMapper()
    return mapper.map(
        Z,
        X.values if isinstance(X, pd.DataFrame) else X,
        clusterer=DBSCAN(eps=eps, min_samples=min_samples),
        cover=km.Cover(n_cubes=n_cubes, perc_overlap=perc_overlap),
    )


# ─────────────────────────────────────────────────────────────────────────────
# PURETÉ DES NŒUDS
# ─────────────────────────────────────────────────────────────────────────────

def node_purity(graph, meta, control_name="control"):
    """
    Pour chaque nœud, calcule la proportion de l'étiquette majoritaire
    (pureté = max(p_control, p_traitement)).
    Retourne un dict {node_id: {"purity": float, "dominant": str, "entropy": float}}
    """
    result = {}
    for node_id, members in graph["nodes"].items():
        members  = list(members)
        methodes = meta["methodes"].iloc[members]
        counts   = methodes.value_counts()
        total    = len(members)
        props    = counts / total
        # Pureté = poids de la classe majoritaire dans le nœud.
        # 1.0 = nœud homogène (control seul ou traitement seul),
        # proche de 0.5 = nœud très mélangé (les deux classes s'y côtoient).
        purity   = props.max()
        dominant = props.idxmax()
        # Entropie de Shannon normalisée (0=pur, 1=max mixte)
        n_classes = len(props)
        if n_classes > 1:
            entropy = -sum(p * np.log2(p) for p in props if p > 0)
            entropy /= np.log2(n_classes)   # normalisation 0–1
        else:
            entropy = 0.0
        result[node_id] = {
            "purity":   round(float(purity), 3),
            "dominant": dominant,
            "entropy":  round(float(entropy), 3),
            "n_control":   int(counts.get(control_name, 0)),
            "n_treatment": int(total - counts.get(control_name, 0)),
            "total":    total,
        }
    return result


# ─────────────────────────────────────────────────────────────────────────────
# CYTOSCAPE
# ─────────────────────────────────────────────────────────────────────────────

def graph_to_cytoscape(graph, meta, color_values, purity_dict=None):
    """
    Convertit le graphe Mapper en une liste d'éléments compatibles avec
    Cytoscape. Les nœuds sont colorés selon la répartition des classes
    qu'ils contiennent, tandis que leur bordure est colorée en fonction
    de leur pureté. Les arêtes du graphe sont ensuite ajoutées afin de
    reconstituer la structure topologique.
    """
    elements  = []
    cmap      = plt.cm.viridis
    n_classes = len(np.unique(color_values))

    for node_id, members in graph["nodes"].items():
        members = list(members)
        encoded = color_values[members]
        unique, counts = np.unique(encoded, return_counts=True)
        proportions    = counts / counts.sum()
        # Mélange pondéré des couleurs de chaque classe présente dans le nœud
        # (proportion au prorata des effectifs) : un nœud 100% control aura
        # la couleur "pure" de control, un nœud mixte une couleur intermédiaire.
        blended = np.sum(
            [np.array(cmap(u / max(n_classes - 1, 1))) * p
             for u, p in zip(unique, proportions)], axis=0,
        )
        color_hex = "#{:02x}{:02x}{:02x}".format(
            int(blended[0] * 255), int(blended[1] * 255), int(blended[2] * 255),
        )

        # Couleur de bordure selon la pureté
        border_color = "#aaaaaa"   # gris par défaut (pas de pureté)
        purity_val   = None
        if purity_dict and node_id in purity_dict:
            p = purity_dict[node_id]["purity"]
            purity_val = p
            if p == 1:
                border_color = "#2ca02c"   # vert — parfaitement pur
            elif p >= 0.7:
                border_color = "#ff7f0e"   # orange — majoritairement pur
            else:
                border_color = "#d62728"   # rouge — très mélangé

        data = {
            "id":           node_id,
            "label":        str(len(members)),
            "size":         len(members),
            "color":        color_hex,
            "members":      members,
            "border_color": border_color,
        }
        if purity_val is not None:
            data["purity"] = purity_val

        elements.append({"data": data})

    for source, targets in graph["links"].items():
        for target in targets:
            elements.append({"data": {"source": source, "target": target}})

    return elements


# ─────────────────────────────────────────────────────────────────────────────
# TEST MANN-WHITNEY  (sélection vs reste)
# ─────────────────────────────────────────────────────────────────────────────

def mann_whitney_selection(X_cur, idx_sel, alpha=0.05, idx_in_mapper=None):
    """
    Test Mann-Whitney variable par variable :
    groupe A = individus sélectionnés (idx_sel)
    groupe B = individus dans le Mapper mais PAS sélectionnés

    idx_in_mapper : indices présents dans au moins un noeud.
    Si None, utilise tous les individus de X_cur.
    """
    if idx_in_mapper is None:
        idx_in_mapper = list(range(len(X_cur)))

    idx_rest = np.setdiff1d(idx_in_mapper, idx_sel)

    if len(idx_sel) < 2:
        return pd.DataFrame(columns=["variable", "stat", "pvalue", "significant"])
    if len(idx_rest) < 2:
        return pd.DataFrame(columns=["variable", "stat", "pvalue", "significant"])
    X_sel  = X_cur.iloc[idx_sel].values
    X_rest = X_cur.iloc[idx_rest].values

    rows = []
    for j, col in enumerate(X_cur.columns):
        try:
            stat, pval = mannwhitneyu(X_sel[:, j], X_rest[:, j],
                                      alternative="two-sided")
        except Exception:
            stat, pval = np.nan, 1.0
        rows.append({"variable": col, "stat": stat, "pvalue": pval})

    df = pd.DataFrame(rows)
    # Correction pour tests multiples : on teste une p-value par variable
    # spectrale (potentiellement des centaines), donc sans correction on
    # aurait énormément de faux positifs. "fdr_by" est plus conservateur que
    # "fdr_bh" mais reste valide même si les variables sont corrélées entre
    # elles (ce qui est le cas ici, les nombres d'onde voisins sont liés).
    reject, pvals_corrected, _, _ = multipletests(
        df["pvalue"].values,
        alpha=alpha,
        method="fdr_by"
    )
    df["pvalue_corrected"] = pvals_corrected
    df["significant"] = reject

    return df.sort_values("pvalue")


# ─────────────────────────────────────────────────────────────────────────────
# LOADINGS
# ─────────────────────────────────────────────────────────────────────────────

def _raw_loadings_for_dim(model, X, lens, dim_idx):
    """
    Récupère les coefficients (loadings) associés à une dimension donnée
    du modèle de projection. Selon la méthode utilisée, les loadings sont
    obtenus directement à partir du modèle ou estimés à partir des
    corrélations entre les variables d'origine et les composantes latentes.
    """
    feature_names = list(X.columns)
    if lens == "pca":
        if dim_idx >= model.components_.shape[0]:
            return np.array([]), feature_names
        raw = model.components_[dim_idx]
    elif lens == "plsda":
        if dim_idx >= model.x_weights_.shape[1]:
            return np.array([]), feature_names
        raw = model.x_weights_[:, dim_idx]
    elif lens in ("mbpls", "o2pls"):
        # model est un dict avec x_weights_ alias
        W = model["W_o"] if isinstance(model, dict) else model.x_weights_
        if dim_idx >= W.shape[1]:
            return np.array([]), feature_names
        raw = W[:, dim_idx]
    elif lens == "mcoa":
        # Loadings MCOA = corrélations entre variables originales et scores communs
        # On utilise les deux blocs et on concatène leurs corrélations
        Z_dim = model["T_common"][:, dim_idx] if dim_idx < model["T_common"].shape[1] else None
        if Z_dim is None:
            return np.array([]), feature_names
        raw = np.array([
            np.corrcoef(X[c].values.astype(float), Z_dim)[0, 1]
            for c in feature_names
        ])
    elif lens == "mfa":
        try:
            corr     = model.column_correlations(
                pd.DataFrame(X.values, columns=feature_names))
            dim_cols = [c for c in corr.columns if c[1] == dim_idx]
            if not dim_cols:
                return np.array([]), feature_names
            raw          = corr[dim_cols].mean(axis=1).values
            feature_names = list(corr.index)
        except Exception:
            try:
                Z   = model.row_coordinates(
                    pd.DataFrame(X.values, columns=feature_names)).values[:, dim_idx]
                raw = np.array([np.corrcoef(X[c].values, Z)[0, 1]
                                for c in feature_names])
            except Exception:
                return np.array([]), feature_names
    elif lens == "mbpca":
        if dim_idx >= model.components_.shape[0]:
            return np.array([]), feature_names
        raw = model.components_[dim_idx]

    else:
        return np.array([]), feature_names

    raw = np.array(raw, dtype=float)
    if len(raw) != len(feature_names):
        n             = min(len(raw), len(feature_names))
        raw           = raw[:n]
        feature_names = feature_names[:n]
    return raw, feature_names


def extract_loadings(model, X, lens, top_n=20):
    """
    Extrait les variables les plus contributives pour les trois premières
    dimensions du modèle. Pour chaque dimension, les `top_n` plus fortes
    contributions positives et négatives sont conservées afin de faciliter
    leur interprétation.
    """
    result = {}
    for dim_idx, dim_name in [(0, "dim1"), (1, "dim2"), (2, "dim3")]:
        raw, fnames = _raw_loadings_for_dim(model, X, lens, dim_idx)
        if len(raw) == 0:
            result[dim_name] = pd.DataFrame(columns=["variable", "loading", "sign"])
            continue
        df_load = pd.DataFrame({"variable": fnames, "loading": raw})
        # On garde les top_n variables qui tirent le plus la dimension dans
        # chaque sens : un loading positif fort et un loading négatif fort
        # ont tous les deux une contribution importante, juste en sens opposé.
        top_pos = df_load[df_load["loading"] > 0].nlargest(top_n, "loading").assign(sign="positive")
        top_neg = df_load[df_load["loading"] < 0].nsmallest(top_n, "loading").assign(sign="negative")
        result[dim_name] = pd.concat([top_pos, top_neg]).sort_values("loading")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# VARIANCE EXPLIQUÉE PAR DIMENSION
# ─────────────────────────────────────────────────────────────────────────────

def explained_information(model, X, lens, Z):
    """
    Quantifie l'information portée par chaque dimension latente du modèle.

    La métrique utilisée dépend de la méthode de réduction de dimension :
    variance expliquée (PCA, MB-PCA), inertie (MFA) ou contribution à la
    structure commune/covariance (PLS-DA, MB-PLS, O2-PLS).

    Retourne un dictionnaire contenant le pourcentage d'information
    expliqué par chaque dimension.
    """
    result = {}
    # PCA / MB-PCA
    # Les composantes maximisent la variance de X.
    # On utilise directement la variance expliquée fournie par le modèle.
    if lens in ("pca", "mbpca"):
        if hasattr(model, "explained_variance_ratio_"):
            for i, v in enumerate(model.explained_variance_ratio_):
                result[i] = round(float(v) * 100, 1)
        return result
    # MFA
    # Les composantes maximisent l'inertie globale des différents blocs.
    # on l'estime à partir de la variance des coordonnées projetées.
    elif lens == "mfa":
        Z_arr = np.asarray(Z, dtype=float)

        inertia = np.var(Z_arr, axis=0, ddof=1)
        total = inertia.sum()

        if total > 0:
            for i, v in enumerate(inertia):
                result[i] = round(v / total * 100, 1)

        return result
    # PLS-DA
    # Les composantes sont évaluées selon leur capacité à expliquer la
    # covariance entre les scores latents T et la réponse Y.
    elif lens == "plsda":
        T = np.asarray(model.x_scores_, dtype=float)
        y = np.asarray(model.y_scores_[:, 0], dtype=float)
        covs = []
        for k in range(T.shape[1]):
            covs.append(np.cov(T[:, k], y)[0, 1] ** 2)
        total = np.sum(covs)
        if total > 0:
            for i, c in enumerate(covs):
                result[i] = round(c / total * 100, 1)
        return result
    # MB-PLS
    # Les composantes représentent la structure commune entre les blocs.
    # Leur importance est estimée à partir de la variance des scores communs.
    elif lens == "mbpls":
        T = np.asarray(model["T"], dtype=float)
        covs = []
        for k in range(T.shape[1]):
            covs.append(np.var(T[:, k], ddof=1))
        total = np.sum(covs)
        if total > 0:
            for i, c in enumerate(covs):
                result[i] = round(c / total * 100, 1)
        return result
    # O2PLS
    # Même principe que MB-PLS, mais après élimination des variations
    # spécifiques à chaque bloc. Seule la structure commune est quantifiée.
    elif lens == "o2pls":
        T = np.asarray(model["T"], dtype=float)
        covs = []
        for k in range(T.shape[1]):
            covs.append(np.var(T[:, k], ddof=1))
        total = np.sum(covs)
        if total > 0:
            for i, c in enumerate(covs):
                result[i] = round(c / total * 100, 1)
        return result
    # MCOA
    # La variance expliquée est calculée sur les scores communs T_common,
    # qui représentent le consensus entre tous les blocs.
    elif lens == "mcoa":
        T = np.asarray(model["T_common"], dtype=float)
        covs = [np.var(T[:, k], ddof=1) for k in range(T.shape[1])]
        total = np.sum(covs)
        if total > 0:
            for i, c in enumerate(covs):
                result[i] = round(c / total * 100, 1)
        return result
    return result
