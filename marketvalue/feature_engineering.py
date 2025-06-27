import networkx as nx
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import category_encoders as ce
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd
from pycaret.regression import *
from sklearn.preprocessing import MultiLabelBinarizer
import joblib
from marketvalue.config import CT_ENCODER_PATH, PCA_ENCODER_DIR, MLB_DIR, PCA_GROUPS_PATH
from pathlib import Path

cols_shooting = [
        "shooting_Gls", "shooting_SoT%", "shooting_Sh/90",
        "shooting_SoT/90","shooting_G/Sh" , "shooting_G/SoT", "shooting_Dist", 
        "shooting_xG",
        "shooting_npxG/Sh"
    ]
cols_passing = [
    "passing_Cmp", "passing_Att", "passing_Cmp%",
    "passing_Ast", "passing_xAG", "passing_xA",
    "passing_KP", "passing_PPA", "passing_PrgP"
]

cols_possession = [
    "possession_Touches",
    "possession_Carries",
    "possession_PrgC",
    "possession_PrgR",
    "possession_Rec",
    "possession_PrgDist",
    "possession_Mis",
    "possession_Dis",
    "possession_Att", "possession_Succ", "possession_Succ%", "possession_Tkld"
]

cols_playingtime = [
    "playingtime_MP",
    "playingtime_Min",
    "playingtime_Starts",
    "playingtime_Subs",
    "playingtime_PPM",
    "playingtime_+/-90", "playingtime_onxG", "playingtime_onxGA",
    "playingtime_xG+/-", "playingtime_xG+/-90"
]

cols_misc = [
    "misc_Age", "misc_CrdY", "misc_CrdR", "misc_Fls", "misc_Fld",
    "misc_Off", "misc_Recov", "misc_PKwon",
    "misc_PKcon", "misc_OG", "misc_Nation", "misc_Comp", "misc_Pos", "misc_Int"
]

cols_stats = [
    "stats_G-PK", "stats_npxG+xAG", "stats_Gls.1", "stats_Ast.1",
    "stats_xG.1", "stats_xAG.1"
]

columns_utiles = cols_misc + cols_passing + cols_playingtime + cols_possession + cols_shooting + cols_stats

def remove_unnecessary_cols(df, columns_utiles, train):
    columnas_disponibles = [col for col in columns_utiles if col in df.columns]
    if train:
        columnas_disponibles = columnas_disponibles + ["fee"]

    return df[columnas_disponibles]


def apply_pca_to_group(df, columns, group_name, train=True):
    scaler_path = PCA_ENCODER_DIR / f"{group_name}_scaler.joblib"
    pca_path = PCA_ENCODER_DIR / f"{group_name}_pca.joblib"

    if train:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(df[columns])
        pca = PCA(n_components=1)
        pca_result = pca.fit_transform(scaled)

        joblib.dump(scaler, scaler_path)
        joblib.dump(pca, pca_path)
    else:
        if not scaler_path.exists() or not pca_path.exists():
            print(f"Saltando PCA para {group_name}, modelos no encontrados")
            return df
        scaler = joblib.load(scaler_path)
        pca = joblib.load(pca_path)

        try:
            scaled = scaler.transform(df[columns])
            pca_result = pca.transform(scaled)
        except KeyError:
            print(f"Saltando PCA para {group_name}, columnas faltantes")
            return df

    df[f"pca_{columns[0]}"] = pca_result[:, 0]
    df.drop(columns=columns, inplace=True)

    return df

def get_correlation_groups(df, threshold=0.80, train=True):
    numeric_df = df.select_dtypes(include='number').copy()

    if train:
        corr = numeric_df.corr().abs()
        G = nx.Graph()
        G.add_nodes_from(corr.columns)

        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                if corr.iloc[i, j] > threshold:
                    col1 = corr.columns[i]
                    col2 = corr.columns[j]
                    G.add_edge(col1, col2)

        groups = [list(g) for g in nx.connected_components(G) if len(g) >= 2]
        joblib.dump(groups, PCA_GROUPS_PATH)

    else:
        if not PCA_GROUPS_PATH.exists():
            print("No se encontró el archivo de grupos PCA")
            return df
        groups = joblib.load(PCA_GROUPS_PATH)

    for group in groups:
        group = [col for col in group if col in df.columns]
        if len(group) < 2:
            continue
        group_name = "_".join(group)
        df = apply_pca_to_group(df, group, group_name, train=train)

    return df


def encode_categorical_columns(
    df,
    multilabel_cols,
    categorical_cols,
    multilabel_sep=",",
    training=True,
    mlb_dir: Path = None,
    ct_path: Path = None
):
    df = df.copy()

    multilabel_dfs = []
    mlb_dict = {}

    # Multilabel encode
    for col in multilabel_cols:
        multilabel_data = df[col].fillna("").apply(lambda x: [v.strip() for v in x.split(multilabel_sep) if v.strip()])
        if training:
            mlb = MultiLabelBinarizer()
            onehot = mlb.fit_transform(multilabel_data)
            # Guardar mlb
            if mlb_dir is not None:
                joblib.dump(mlb, mlb_dir / f"mlb_{col}.joblib")
        else:
            # Cargar mlb
            if mlb_dir is not None:
                mlb = joblib.load(mlb_dir / f"mlb_{col}.joblib")
            else:
                raise ValueError("mlb_dir es necesario en modo producción")
            onehot = mlb.transform(multilabel_data)
        onehot_df = pd.DataFrame(onehot, columns=[f"{col}_{cls}" for cls in mlb.classes_], index=df.index)
        multilabel_dfs.append(onehot_df)
        mlb_dict[col] = mlb

    # One hot encode simple
    if categorical_cols:
        if training:
            ct = ColumnTransformer(
                transformers=[
                    ("onehot", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), categorical_cols)
                ],
                remainder="drop"
            )
            onehot_array = ct.fit_transform(df[categorical_cols])
            # Guardar ct
            if ct_path is not None:
                joblib.dump(ct, ct_path)
        else:
            if ct_path is not None:
                ct = joblib.load(ct_path)
            else:
                raise ValueError("ct_path es necesario en modo producción")
            onehot_array = ct.transform(df[categorical_cols])

        onehot_columns = ct.named_transformers_["onehot"].get_feature_names_out(categorical_cols)
        onehot_df = pd.DataFrame(onehot_array, columns=onehot_columns, index=df.index)
    else:
        onehot_df = pd.DataFrame(index=df.index)  # vacío si no hay cols

    # Eliminar columnas originales codificadas
    cols_to_drop = multilabel_cols + categorical_cols
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Concatenar resultados
    dfs_to_concat = [df] + multilabel_dfs + [onehot_df]
    df_final = pd.concat(dfs_to_concat, axis=1)

    return df_final



def build_dataset(transfer_data, train = True):
    df = remove_unnecessary_cols(transfer_data, columns_utiles, train)
    num_cols = df.select_dtypes(include = 'number').columns.tolist()
    df = df.dropna(subset = num_cols)
    df = get_correlation_groups(df, threshold=0.95, train= train)


    df_encoded = encode_categorical_columns(df, 
        multilabel_cols=["misc_Pos"],
        categorical_cols=["misc_Comp", "misc_Nation"],
        training=train,
        mlb_dir=MLB_DIR,
        ct_path=CT_ENCODER_PATH
    )
    Y = []
    if train:
        Y = df_encoded["fee"]
    
    num_cols = df_encoded.select_dtypes(include = 'number').columns.tolist()
    X = df_encoded[num_cols]
    X = X.dropna(subset = num_cols)

    return X, Y

def train_pycaret_model(df, target_col, model_path,  session_id=123):
    
    # Inicializar setup de PyCaret
    s = setup(
        data=df,
        target=target_col,
        session_id=session_id,
        normalize=True,
        fold=5,
        fold_strategy='kfold'
    )
    
    # Comparar modelos y elegir el mejor según R2
    best_model = compare_models(sort='R2')
    
    final_model = finalize_model(best_model)
    
    save_model(final_model, model_path)
    