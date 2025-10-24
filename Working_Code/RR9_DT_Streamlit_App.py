import pandas as pd
import streamlit as st
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401  # needed for IterativeImputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.ensemble import BaggingRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib
import os
from typing import Optional, Tuple

# Optional: torch for VAE tab
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

# Import VAE implementation from local GenAI module
try:
    from GenAI.VAE_RNASeq import VAE as RNASeqVAE
    VAE_AVAILABLE = True
except Exception:
    VAE_AVAILABLE = False


def set_page_default():
    st.set_page_config(layout='wide')


@st.cache_data(show_spinner=True)
def acquire_data():
    rnaseq = pd.read_csv(
        'https://osdr.nasa.gov/geode-py/ws/studies/OSD-255/download?source=datamanager&file=GLDS-255_rna_seq_Normalized_Counts.csv',
        index_col=0,
    ).transpose()

    zo1 = pd.read_csv(
        'https://osdr.nasa.gov/geode-py/ws/studies/OSD-568/download?source=datamanager&file=LSDS-5_immunostaining_microscopy_Zo-1tr_TRANSFORMED.csv'
    )

    tunel = pd.read_csv(
        'https://osdr.nasa.gov/geode-py/ws/studies/OSD-568/download?source=datamanager&file=LSDS-5_immunostaining_microscopy_TUNELtr_TRANSFORMED.csv'
    )

    pecam = pd.read_csv(
        'https://osdr.nasa.gov/geode-py/ws/studies/OSD-568/download?source=datamanager&file=LSDS-5_immunostaining_microscopy_PECAMtr_TRANSFORMED.csv'
    )  # ['Source Name']=['F15','F16','F17','F18','F19','F20','GC15','GC16','GC17','GC18','GC19']

    microct = pd.read_csv(
        'https://osdr.nasa.gov/geode-py/ws/studies/OSD-557/download?source=datamanager&file=LSDS-1_microCT_MicroCT_Transformed_Reusable_Results.csv'
    )  # NOTE: OSD-557 also has raw image files for immunostaining and H&E

    pna = pd.read_csv(
        'https://osdr.nasa.gov/geode-py/ws/studies/OSD-557/download?source=datamanager&file=LSDS-1_immunostaining_microscopy_PNAtr_Transformed_Reusable_Results.csv'
    )

    hne = pd.read_csv(
        'https://osdr.nasa.gov/geode-py/ws/studies/OSD-557/download?source=datamanager&file=LSDS-1_immunostaining_microscopy_HNEtr_Transformed_Reusable_Results.csv'
    )

    tonometry = pd.read_csv(
        'https://osdr.nasa.gov/geode-py/ws/studies/OSD-583/download?source=datamanager&file=LSDS-16_tonometry_maoTRANSFORMED.csv'
    )

    protein = pd.read_excel(
        'https://osdr.nasa.gov/geode-py/ws/studies/OSD-715/download?source=datamanager&file=GLDS-639_proteomics_01_Mao_RR9_Retina_092018_MQ_iBAQ.xlsx',
        sheet_name='Combat corrected',
        index_col=0,
    ).transpose()

    protein = (
        protein.drop(protein.columns[[0, 1, 2, 4, 5, 6]], axis=1)
        .rename(columns={"Sample": "Source Name"})
        .reset_index()
        .drop(columns=['index'])
    )  # remove some Excel formatting

    protein = protein.set_index('Source Name').apply(
        pd.to_numeric, errors='coerce'
    )  # convert to numeric instead of dtype "object"

    # Filter omics data:
    # - remove genes and proteins with greater than 1% zero values
    # - keep 1000 genes and 1000 proteins with highest variance

    rnaseq = rnaseq.loc[:, (rnaseq == 0).mean() < 0.01]  # Remove features with too many zeros
    print(rnaseq.shape)
    rnaseq = rnaseq.loc[:, rnaseq.var().nlargest(1000).index]  # Take top 1000 features by variance

    protein = protein.loc[:, (protein == 0).mean() < 0.01]  # Remove features with too many zeros
    print(protein.shape)
    protein = protein.loc[
        :, protein.var().nlargest(1000).index
    ]  # Take top 1000 features by variance

    all_genes = rnaseq.columns.tolist()  # get all gene names for later
    all_proteins = protein.columns.tolist()  # get all protein names for later

    # Add Source Name to all the dfs that are missing it
    # Rename some Source Names that already exist

    # Fviv = CC1 (confirmed in OSD-583 metadata)
    # GViv = CC2?

    gsm_dict = {
        "GSM3932702": "F11",
        "GSM3932703": "F15",
        "GSM3932704": "F16",
        "GSM3932705": "F17",
        "GSM3932706": "F18",
        "GSM3932707": "F19",
        "GSM3932708": "F20",
        "GSM3932694": "GC11",
        "GSM3932695": "GC15",
        "GSM3932696": "GC16",
        "GSM3932697": "GC17",
        "GSM3932698": "GC18",
        "GSM3932699": "GC19",
        "GSM3932700": "GC20",
        "GSM3932693": "GC9",
        "GSM3932701": "F9",
    }

    rnaseq['Source Name'] = rnaseq.index.map(gsm_dict)

    zo1['Source Name'] = [
        'F15',
        'F16',
        'F17',
        'F18',
        'F19',
        'F20',
        'GC15',
        'GC16',
        'GC17',
        'GC18',
        'GC19',
        'GC20',
    ]

    tunel['Source Name'] = [
        'F15',
        'F16',
        'F17',
        'F18',
        'F19',
        'F20',
        'GC15',
        'GC16',
        'GC17',
        'GC18',
        'GC19',
        'GC20',
        'CC2_15',
        'CC2_16',
        'CC2_17',
        'CC2_18',
        'CC2_20',
        'Viv15',
        'Viv16',
        'Viv17',
        'Viv18',
        'Viv19',
        'Viv20',
    ]  # rename VG to CC2 and V to Viv for consistency

    pecam['Source Name'] = [
        'F15',
        'F16',
        'F17',
        'F18',
        'F19',
        'F20',
        'GC15',
        'GC16',
        'GC17',
        'GC18',
        'GC19',
    ]

    microct['Source Name'] = [
        'F10',
        'F12',
        'F13',
        'F14',
        'Viv10',
        'Viv12',
        'Viv13',
        'Viv14',
        'GC10',
        'GC11',
        'GC13',
        'GC14',
    ]  # rename V to Viv

    hne['Source Name'] = [
        "F15",
        "F16",
        "F17",
        "F18",
        "F19",
        "F20",
        "GC15",
        "GC16",
        "GC17",
        "GC18",
        "GC19",
        "Viv15",
        "Viv16",
        "Viv17",
        "Viv18",
        "Viv19",
        "Viv20",
        "CC2_15",
        "CC2_16",
        "CC2_17",
        "CC2_18",
        "CC2_19",
        "CC2_20",
    ]

    tonometry['Source Name'] = [
        "F1",
        "F2",
        "F3",
        "F4",
        "F5",
        "F6",
        "F7",
        "F8",
        "F9",
        "F10",
        "F11",
        "F12",
        "F13",
        "F14",
        "F15",
        "F16",
        "F17",
        "F18",
        "F19",
        "F20",
        "CC1_1",
        "CC1_2",
        "CC1_3",
        "CC1_4",
        "CC1_5",
        "CC1_6",
        "CC1_7",
        "CC1_8",
        "CC1_9",
        "CC1_10",
        "CC1_11",
        "CC1_12",
        "CC1_13",
        "CC1_14",
        "CC1_15",
        "CC1_16",
        "CC1_17",
        "CC1_18",
        "CC1_19",
        "CC1_20",
        "GC1",
        "GC2",
        "GC3",
        "GC4",
        "GC5",
        "GC6",
        "GC7",
        "GC8",
        "GC9",
        "GC10",
        "GC11",
        "GC12",
        "GC13",
        "GC14",
        "GC15",
        "GC16",
        "GC17",
        "GC18",
        "GC19",
        "GC20",
        "Viv1",
        "Viv2",
        "Viv3",
        "Viv4",
        "Viv5",
        "Viv6",
        "Viv7",
        "Viv8",
        "Viv9",
        "Viv10",
        "Viv11",
        "Viv12",
        "Viv13",
        "Viv14",
        "Viv15",
        "Viv16",
        "Viv17",
        "Viv18",
        "Viv19",
        "Viv20",
        "CC2_1",
        "CC2_2",
        "CC2_3",
        "CC2_4",
        "CC2_5",
        "CC2_6",
        "CC2_7",
        "CC2_8",
        "CC2_9",
        "CC2_10",
        "CC2_11",
        "CC2_12",
        "CC2_13",
        "CC2_14",
        "CC2_15",
        "CC2_16",
        "CC2_17",
        "CC2_18",
        "CC2_19",
        "CC2_20",
    ]  # rename "FViv" to "CC1"  for consistency

    protein['Source Name'] = protein.index
    protein.reset_index(drop=True, inplace=True)

    # Drop "Sample Name" and a couple other irrelevant columns
    zo1.drop(columns=['Sample_Name'], inplace=True)
    tunel.drop(columns=['Sample_Name'], inplace=True)
    pecam.drop(columns=['Sample_Name'], inplace=True)
    microct.drop(columns=['Sample Name', 'Treatment'], inplace=True)
    pna.drop(columns=['Sample Name', 'Treatment'], inplace=True)
    hne.drop(columns=['Sample Name'], inplace=True)
    tonometry.drop(
        columns=['Sample Name', 'Factor Value: Spaceflight', 'time_Start', 'Time_End'], inplace=True
    )

    # Add suffix to all physiological data column names to avoid duplicates (rnaseq and protein should be unique)
    zo1.columns = [col + '_zo1' if col != 'Source Name' else col for col in zo1.columns]
    tunel.columns = [col + '_tunel' if col != 'Source Name' else col for col in tunel.columns]
    pecam.columns = [col + '_pecam' if col != 'Source Name' else col for col in pecam.columns]
    microct.columns = [col + '_microct' if col != 'Source Name' else col for col in microct.columns]
    pna.columns = [col + '_pna' if col != 'Source Name' else col for col in pna.columns]
    hne.columns = [col + '_hne' if col != 'Source Name' else col for col in hne.columns]
    tonometry.columns = [
        col + '_tonometry' if col != 'Source Name' else col for col in tonometry.columns
    ]

    dfs = [rnaseq, protein, zo1, tunel, pecam, microct, pna, hne, tonometry]

    # Convert 'Source Name' to string in all DataFrames to avoid dtype conflicts
    for df in dfs:
        df['Source Name'] = df['Source Name'].astype(str)

    # Merge all DataFrames on "Source Name" column
    merged_df = dfs[0]
    for df in dfs[1:]:
        merged_df = pd.merge(merged_df, df, on="Source Name", how="outer")

    # Move source name column to the front
    column_to_move = merged_df.pop("Source Name")
    merged_df.insert(0, "Source Name", column_to_move)

    pd.set_option('display.max_rows', None)

    all_phys = [
        x
        for x in merged_df.drop(columns='Source Name').columns
        if x not in all_genes and x not in all_proteins
    ]

    # Add a group column
    merged_df['Group'] = [
        "CC1",
        "CC1",
        "CC1",
        "CC1",
        "CC1",
        "CC1",
        "CC1",
        "CC1",
        "CC1",
        "CC1",
        "CC1",
        "CC1",
        "CC1",
        "CC1",
        "CC1",
        "CC1",
        "CC1",
        "CC1",
        "CC1",
        "CC1",
        "CC2",
        "CC2",
        "CC2",
        "CC2",
        "CC2",
        "CC2",
        "CC2",
        "CC2",
        "CC2",
        "CC2",
        "CC2",
        "CC2",
        "CC2",
        "CC2",
        "CC2",
        "CC2",
        "CC2",
        "CC2",
        "CC2",
        "CC2",
        "F",
        "F",
        "F",
        "F",
        "F",
        "F",
        "F",
        "F",
        "F",
        "F",
        "F",
        "F",
        "F",
        "F",
        "F",
        "F",
        "F",
        "F",
        "F",
        "F",
        "GC",
        "GC",
        "GC",
        "GC",
        "GC",
        "GC",
        "GC",
        "GC",
        "GC",
        "GC",
        "GC",
        "GC",
        "GC",
        "GC",
        "GC",
        "GC",
        "GC",
        "GC",
        "GC",
        "GC",
        "Viv",
        "Viv",
        "Viv",
        "Viv",
        "Viv",
        "Viv",
        "Viv",
        "Viv",
        "Viv",
        "Viv",
        "Viv",
        "Viv",
        "Viv",
        "Viv",
        "Viv",
        "Viv",
        "Viv",
        "Viv",
        "Viv",
        "Viv",
    ]

    return merged_df, all_phys, all_genes, all_proteins


@st.cache_data(show_spinner=True)
def scale_data(merged_df):
    merged_df.drop('Group', axis=1).set_index('Source Name').max().max()

    # Min-Max scale the data
    scaler = MinMaxScaler()
    df = merged_df.drop('Group', axis=1).set_index('Source Name')
    merged_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)

    merged_scaled.max().max()

    # Select only numeric columns and flatten them into a single list, ignoring NaN values
    numeric_values = merged_scaled.select_dtypes(include='number').values.flatten()
    numeric_values = numeric_values[~np.isnan(numeric_values)]  # Remove NaN values

    # Add back in the "Group" column

    merged_scaled['Group'] = merged_df.set_index('Source Name')["Group"]

    return merged_scaled, numeric_values


def _build_vae(input_dim: int, latent_dim: int = 32, hidden: int = 256):
    """
    Construct a simple VAE using the existing RNASeqVAE class with MLP encoder/decoder.
    """
    if not (TORCH_AVAILABLE and VAE_AVAILABLE):
        raise RuntimeError("PyTorch and/or VAE class not available")

    encoder_sizes = [input_dim, hidden]
    decoder_sizes = [hidden, input_dim]
    model = RNASeqVAE(
        encoder_layer_sizes=encoder_sizes,
        latent_size=latent_dim,
        decoder_layer_sizes=decoder_sizes,
        conditional=False,
        num_labels=0,
    )
    return model


def _train_vae_on_masked_data(X_np: np.ndarray, epochs: int = 300, lr: float = 1e-3, beta: float = 1.0, hidden: int = 256, latent: int = 32, device: str | None = None):
    """
    Train a VAE to reconstruct X with missing values (NaNs). Loss is computed on observed entries only.
    Returns trained model and reconstruction as numpy array in [0,1] scale (same scale as input X_np).
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available. Please install torch.")
    if not VAE_AVAILABLE:
        raise RuntimeError("VAE implementation not found. Ensure GenAI/VAE_RNASeq.py is present.")

    # Mask for observed entries
    mask = ~np.isnan(X_np)
    # Initialize missing with column means (safer than zeros)
    col_means = np.nanmean(X_np, axis=0)
    X_init = np.where(mask, X_np, col_means)

    x = torch.tensor(X_init, dtype=torch.float32)
    m = torch.tensor(mask.astype(np.float32))

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    x = x.to(device)
    m = m.to(device)

    model = _build_vae(input_dim=X_np.shape[1], latent_dim=latent, hidden=hidden).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)

    progress = st.progress(0, text="Training VAE...")
    loss_chart = st.empty()
    losses = []
    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        recon, mu, log_var, _ = model(x)
        # Reconstruction loss only on observed entries
        rec_loss = ((recon - x) ** 2 * m).sum() / (m.sum() + 1e-8)
        # KL divergence
        kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / x.shape[0]
        loss = rec_loss + beta * kl
        loss.backward()
        opt.step()

        losses.append(loss.detach().item())
        if ep % max(1, epochs // 100) == 0 or ep == epochs:
            progress.progress(min(1.0, ep / epochs), text=f"Training VAE... epoch {ep}/{epochs} | loss={losses[-1]:.4f}")
            # lightweight loss plot
            try:
                loss_fig = px.line(y=losses, labels={"x": "epoch", "y": "loss"})
                loss_fig.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10))
                loss_chart.plotly_chart(loss_fig, use_container_width=True)
            except Exception:
                pass

    model.eval()
    with torch.no_grad():
        recon, _, _, _ = model(x)
    X_recon = recon.detach().cpu().numpy()
    progress.empty()
    loss_chart.empty()
    return model, X_recon


def _impute_scaled_block(merged_scaled: pd.DataFrame, merged_df_original: pd.DataFrame, feature_cols: list[str], epochs: int, lr: float, beta: float, hidden: int, latent: int):
    """
    Run VAE on scaled feature block and return an imputed copy of merged_df (in original scale).
    """
    # Prepare data matrix in [0,1]
    X_scaled = merged_scaled[feature_cols].to_numpy(dtype=float)
    # Train VAE
    _, X_recon_scaled = _train_vae_on_masked_data(
        X_scaled, epochs=epochs, lr=lr, beta=beta, hidden=hidden, latent=latent
    )
    # Impute: keep observed, replace NaNs with reconstruction
    mask = ~np.isnan(X_scaled)
    X_imputed_scaled = np.where(mask, X_scaled, X_recon_scaled)

    # Inverse scale back to original using per-column min/max from original merged_df
    orig_block = merged_df_original.set_index('Source Name')[feature_cols]
    col_min = orig_block.min(skipna=True)
    col_max = orig_block.max(skipna=True)
    # Avoid division by zero: if max==min, treat scale as 1
    scale = (col_max - col_min).replace(0, 1.0)
    # Broadcast
    X_imputed_orig = X_imputed_scaled * scale.values + col_min.values

    # Build resulting dataframe
    imputed_df = merged_df_original.copy()
    # Fill only NaNs in selected columns
    for i, col in enumerate(feature_cols):
        col_vals = imputed_df[col].to_numpy(dtype=float)
        nan_idx = np.isnan(col_vals)
        # Set NaNs to imputed original-scale values
        col_vals[nan_idx] = X_imputed_orig[:, i][nan_idx]
        imputed_df[col] = col_vals
    return imputed_df


def plot_heatmap(merged_scaled, all_phys, all_genes, all_proteins):
    """
    Create an interactive heatmap using Plotly to display which samples have which data types.
    """
    # Select the relevant columns and rename them
    heatmap_data = merged_scaled[all_phys + [all_genes[0], all_proteins[0]]].rename(
        columns={all_genes[0]: 'RNAseq', all_proteins[0]: 'Protein'}
    )

    # Create the Plotly heatmap
    fig = px.imshow(
        heatmap_data.T,
        labels={'x': 'Samples', 'y': 'Data Types'},
        color_continuous_scale="RdBu",  # Use a valid Plotly colorscale
        title="Data Types Heatmap",
    )

    # Figure layout
    fig.update_layout(
        width=1200,
        height=1200,
        xaxis=dict(title='Samples', tickangle=45),
        yaxis=dict(title='Data Types'),
        margin=dict(l=200, r=200, t=100, b=100),
    )

    # Display the heatmap in Streamlit
    st.plotly_chart(fig, use_container_width=True)


# =============== Imputation helpers (rr9_imputation-inspired) ===============

class RandomSampleImputerCompat:
    """
    A simple random-sample imputer compatible with sklearn's Transformer API.
    For each column, during transform, missing values are replaced by random samples
    drawn (with replacement) from the observed values in that column from the fit data.
    """

    def __init__(self, random_state: Optional[int] = 2):
        self.random_state = random_state
        self._rng = np.random.default_rng(self.random_state)
        self._observed_values = None

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self._observed_values = {}
        for col in X.columns:
            vals = X[col].dropna().to_numpy()
            # If a column is entirely NA, fall back to zeros
            if vals.size == 0:
                vals = np.array([0.0])
            self._observed_values[col] = vals
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        X_out = X.copy()
        for col in X.columns:
            vals = self._observed_values[col]
            mask = X_out[col].isna()
            if mask.any():
                n_missing = mask.sum()
                sampled = self._rng.choice(vals, size=n_missing, replace=True)
                X_out.loc[mask, col] = sampled
        return X_out.values


def _load_rr9_reference_csv(path: str) -> Optional[pd.DataFrame]:
    try:
        if os.path.exists(path):
            return pd.read_csv(path)
    except Exception:
        pass
    return None


def _select_imputation_dataframe(source: str) -> Tuple[pd.DataFrame, str]:
    """
    Return a dataframe and label based on user's selection.
    - source: one of 'session', 'full', 'flight', 'non_flight'
    """
    if source == 'session':
        if 'merged_df' not in st.session_state:
            raise RuntimeError('Please load data in the "Initial Data Loading" tab first.')
        return st.session_state['merged_df'].copy(), 'session_merged_df'

    base_paths = [
        '/home/gojian/Notebooks/GeneLab/AI-ML_AWG/Working_Code/rr9_imputation',
        '/home/gojian/Notebooks/GeneLab/AI-ML_AWG/Manuscript_Code/rr9_imputation',
    ]
    filename_map = {
        'full': 'full_df.csv',
        'flight': 'merged_flight_data.csv',
        'non_flight': 'merged_non_flight_data.csv',
    }
    fname = filename_map[source]
    for base in base_paths:
        df = _load_rr9_reference_csv(os.path.join(base, fname))
        if df is not None:
            return df, fname
    raise FileNotFoundError(f"Could not find {fname} in rr9_imputation folders.")


def _ensure_id_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, list[str]]:
    """Ensure 'Source Name' and 'Group' are present if possible. Return df and list of feature columns."""
    cols = df.columns.tolist()
    id_cols = [c for c in ['Source Name', 'Group'] if c in cols]
    feature_cols = [c for c in cols if c not in id_cols]
    return df, feature_cols


def _build_knn_pipeline(n_neighbors: int = 2, weights: str = 'distance') -> Pipeline:
    scaler = StandardScaler()
    knn = KNNImputer(n_neighbors=n_neighbors, weights=weights)
    return Pipeline([('scaler', scaler), ('knn', knn)])


def _build_rsi_pipeline(random_state: int = 2) -> Pipeline:
    scaler = StandardScaler()
    rsi = RandomSampleImputerCompat(random_state=random_state)
    return Pipeline([('scaler', scaler), ('rsi', rsi)])


def _build_mice_bagger_pipeline(
    include_group: bool,
    bag_n_estimators: int = 10,
    max_iter: int = 10,
    tol: float = 0.01,
    random_state: int = 2,
    n_jobs: int = -1,
    imputation_order: str = 'roman',
) -> Tuple[Pipeline, bool]:
    """
    When include_group=True, returns a Pipeline that encodes 'Group' and imputes features jointly.
    Otherwise, returns a Pipeline that imputes only numeric features (excluding id columns).
    Returns (pipeline, expects_pre_split_features_only)
    """
    bagger = BaggingRegressor(random_state=random_state, n_estimators=bag_n_estimators, n_jobs=n_jobs, warm_start=True)
    itera = IterativeImputer(
        random_state=random_state,
        initial_strategy='median',
        estimator=bagger,
        max_iter=max_iter,
        tol=tol,
        imputation_order=imputation_order,
        skip_complete=True,
        verbose=0,
    )

    if include_group:
        # This pipeline expects input X that includes the 'Group' column
        prep = ColumnTransformer([
            ('features', StandardScaler(with_mean=True, with_std=True), lambda df: [c for c in df.columns if c not in ['Source Name', 'Group']]),
            ('group', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), ['Group'])
        ])
        pipe = Pipeline([('prep', prep), ('imputer', itera)])
        return pipe, False
    else:
        # Only features, no group encoding
        pipe = Pipeline([('scaler', StandardScaler()), ('imputer', itera)])
        return pipe, True


def _run_impute_pipeline(
    df: pd.DataFrame,
    method: str,
    params: dict,
    save_prefix: str,
    reuse_if_exists: bool = True,
) -> Tuple[pd.DataFrame, Optional[str], Optional[str]]:
    """
    Run selected imputation. Returns (imputed_df, saved_csv_path, saved_pipeline_path).
    """
    df_in, feature_cols = _ensure_id_columns(df)
    id_cols = [c for c in ['Source Name', 'Group'] if c in df_in.columns]

    # Build output paths in Working_Code/rr9_imputation by default
    out_dir = '/home/gojian/Notebooks/GeneLab/AI-ML_AWG/Working_Code/rr9_imputation'
    os.makedirs(out_dir, exist_ok=True)

    csv_out = os.path.join(out_dir, f'{save_prefix}.csv')
    pipe_out = os.path.join(out_dir, f'{save_prefix}_pipeline.joblib')

    if reuse_if_exists and os.path.exists(csv_out):
        try:
            cached = pd.read_csv(csv_out)
            return cached, csv_out, pipe_out if os.path.exists(pipe_out) else None
        except Exception:
            pass

    X = df_in.drop(columns=[c for c in ['Source Name'] if c in df_in.columns])

    if method == 'knn':
        pipe = _build_knn_pipeline(
            n_neighbors=int(params.get('n_neighbors', 2)),
            weights=params.get('weights', 'distance'),
        )
        X_imp = pipe.fit_transform(X.drop(columns=[c for c in ['Group'] if c in X.columns]))
        # inverse scale then rebuild df
        X_imp = pipe.named_steps['scaler'].inverse_transform(X_imp)
        feat_used = [c for c in X.columns if c != 'Group']
        imputed_df = pd.DataFrame(X_imp, columns=feat_used, index=df_in.index)
        # Add back id columns
        for c in id_cols:
            imputed_df[c] = df_in[c].values
        imputed_df = imputed_df[[c for c in df_in.columns]]

    elif method == 'rsi':
        pipe = _build_rsi_pipeline(random_state=int(params.get('random_state', 2)))
        X_imp = pipe.fit_transform(X.drop(columns=[c for c in ['Group'] if c in X.columns]))
        X_imp = pipe.named_steps['scaler'].inverse_transform(X_imp)
        feat_used = [c for c in X.columns if c != 'Group']
        imputed_df = pd.DataFrame(X_imp, columns=feat_used, index=df_in.index)
        for c in id_cols:
            imputed_df[c] = df_in[c].values
        imputed_df = imputed_df[[c for c in df_in.columns]]

    elif method == 'mice_bagger':
        include_group = bool(params.get('include_group', True))
        # If Group is not available, force disable include_group
        if 'Group' not in X.columns:
            include_group = False
        pipe, expects_feats_only = _build_mice_bagger_pipeline(
            include_group=include_group,
            bag_n_estimators=int(params.get('bag_n_estimators', 10)),
            max_iter=int(params.get('max_iter', 10)),
            tol=float(params.get('tol', 0.01)),
            random_state=int(params.get('random_state', 2)),
            n_jobs=int(params.get('n_jobs', -1)),
            imputation_order=params.get('imputation_order', 'roman'),
        )

        # Fit/transform
        if include_group:
            pipe.fit(X)
            X_imp = pipe.transform(X)
            # Recover original feature names from transformer components
            prep = pipe.named_steps['prep']
            # Numeric features used by lambda in ColumnTransformer are not directly named; rebuild order
            feat_names = [c for c in X.columns if c not in ['Source Name', 'Group']]
            group_encoder = prep.named_transformers_['group']
            group_names = list(group_encoder.get_feature_names_out(['Group']))
            all_out_cols = feat_names + group_names
            X_imp_df = pd.DataFrame(X_imp, columns=all_out_cols, index=df_in.index)
            # Keep original schema: drop encoded group columns, keep original Group
            imputed_df = X_imp_df[feat_names].copy()
            for c in id_cols:
                imputed_df[c] = df_in[c].values
            imputed_df = imputed_df[[c for c in df_in.columns]]
        else:
            X_feats = X.drop(columns=[c for c in ['Group'] if c in X.columns])
            pipe.fit(X_feats)
            X_imp = pipe.transform(X_feats)
            # inverse scaling
            X_imp = pipe.named_steps['scaler'].inverse_transform(X_imp)
            feat_used = list(X_feats.columns)
            imputed_df = pd.DataFrame(X_imp, columns=feat_used, index=df_in.index)
            for c in id_cols:
                imputed_df[c] = df_in[c].values
            imputed_df = imputed_df[[c for c in df_in.columns]]
    else:
        raise ValueError(f"Unknown method: {method}")

    # Save artifacts
    try:
        imputed_df.to_csv(csv_out, index=False)
        joblib.dump(pipe, pipe_out)
        saved_csv, saved_pipe = csv_out, pipe_out
    except Exception:
        saved_csv, saved_pipe = None, None

    return imputed_df, saved_csv, saved_pipe


# Streamlit app setup
def main():
    set_page_default()

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            'Initial Data Loading',
            'Imputation',
            'Data Generation GAN',
            'Data Generation VAE',
        ]
    )

    tab1.subheader('Data Loading NOW')
    tab2.subheader('Imputation')
    tab3.subheader('Data Generation GAN')
    tab4.subheader('Data Generation VAE')

    # Sidebar for data loading
    with tab1:

        merged_df, all_phys, all_genes, all_proteins = acquire_data()
        merged_scaled, numeric_values = scale_data(merged_df)

        print(numeric_values)

        plot_heatmap(merged_scaled, all_phys, all_genes, all_proteins)

        # Persist to session state for use in other tabs
        st.session_state['merged_df'] = merged_df
        st.session_state['merged_scaled'] = merged_scaled
        st.session_state['all_phys'] = all_phys
        st.session_state['all_genes'] = all_genes
        st.session_state['all_proteins'] = all_proteins

    # Imputation Tab (rr9_imputation-inspired pipelines)
    with tab2:
        st.markdown("Manage imputation using KNN, Random-Sample, or MICE+Bagging. Long runs can be reused from cached CSVs.")

        # Dataset selection
        ds_col1, ds_col2 = st.columns([2, 1])
        with ds_col1:
            ds_source = st.selectbox(
                'Dataset source',
                options=[
                    ('Use data from "Initial Data Loading" tab', 'session'),
                    ('rr9_imputation/full_df.csv', 'full'),
                    ('rr9_imputation/merged_flight_data.csv', 'flight'),
                    ('rr9_imputation/merged_non_flight_data.csv', 'non_flight'),
                ],
                format_func=lambda x: x[0],
            )[1]
        with ds_col2:
            reuse_cached = st.checkbox('Reuse cached CSV if exists', value=True,
                                       help='If a matching output CSV exists, load it instead of recomputing.')

        # Method selection and params
        method = st.selectbox('Imputation method', options=['KNN', 'Random Sample (RSI)', 'MICE + Bagging'])

        params = {}
        save_prefix_default = ''
        if method == 'KNN':
            c1, c2 = st.columns(2)
            with c1:
                n_neighbors = st.slider('Neighbors (k)', min_value=1, max_value=15, value=2, step=1)
            with c2:
                weights = st.selectbox('Weights', options=['uniform', 'distance'], index=1)
            params.update({'n_neighbors': n_neighbors, 'weights': weights})
            save_prefix_default = {
                'session': 'session_imputed_knn',
                'full': 'full_df_knn2',
                'flight': 'flight_imputed_knn2',
                'non_flight': 'non_flight_imputed_knn2',
            }.get(ds_source, 'imputed_knn')

        elif method == 'Random Sample (RSI)':
            random_state = st.number_input('Random seed', min_value=0, value=2, step=1)
            params.update({'random_state': random_state})
            save_prefix_default = {
                'session': 'session_imputed_rsi',
                'full': 'full_df_rsi',
                'flight': 'flight_imputed_rsi',
                'non_flight': 'non_flight_imputed_rsi',
            }.get(ds_source, 'imputed_rsi')

        else:  # MICE + Bagging
            c1, c2, c3 = st.columns(3)
            with c1:
                max_iter = st.slider('Max Iterations', min_value=1, max_value=50, value=10)
            with c2:
                bag_estimators = st.slider('Bagging n_estimators', min_value=5, max_value=200, value=10, step=5)
            with c3:
                tol = st.number_input('Tolerance (tol)', min_value=0.0, value=0.01, step=0.01, format='%f')
            include_group = st.checkbox('Include Group (one-hot) in imputation', value=True,
                                        help='Recommended for full dataset to borrow strength across groups.')
            params.update({
                'include_group': include_group,
                'bag_n_estimators': bag_estimators,
                'max_iter': max_iter,
                'tol': tol,
            })
            save_prefix_default = {
                'session': 'session_imputed_mice_bagger',
                'full': 'full_imputed_mice_bagger_with_group' if include_group else 'full_imputed_mice_bagger',
                'flight': 'flight_imputed_mice_bagger',
                'non_flight': 'non_flight_imputed_mice_bagger',
            }.get(ds_source, 'imputed_mice_bagger')

        # Output naming
        save_prefix = st.text_input('Output prefix (CSV and pipeline will be saved in Working_Code/rr9_imputation)',
                                    value=save_prefix_default)

        # Run
        if st.button('Run Imputation', type='primary'):
            try:
                with st.spinner('Preparing data...'):
                    df_in, src_label = _select_imputation_dataframe(ds_source)

                st.info(f'Input shape: {df_in.shape}. Columns: {len(df_in.columns)}.')

                method_key = {'KNN': 'knn', 'Random Sample (RSI)': 'rsi', 'MICE + Bagging': 'mice_bagger'}[method]

                # Progress placeholder
                progress = st.progress(0.0, text='Running imputation...')
                # Execute
                imputed_df, saved_csv, saved_pipe = _run_impute_pipeline(
                    df=df_in,
                    method=method_key,
                    params=params,
                    save_prefix=save_prefix,
                    reuse_if_exists=reuse_cached,
                )
                progress.progress(1.0, text='Done')

                st.success('Imputation complete.')
                st.dataframe(imputed_df.head(20))

                # Cache in session and offer download
                st.session_state['imputed_df'] = imputed_df
                csv_bytes = imputed_df.to_csv(index=False).encode('utf-8')
                st.download_button('Download imputed CSV', data=csv_bytes, file_name=f'{save_prefix}.csv', mime='text/csv')

                # Correlation heatmap (top-left NxN)
                try:
                    N = st.slider('Correlation preview size (NxN)', min_value=20, max_value=200, value=100, step=10)
                    # Exclude id cols
                    idc = [c for c in ['Source Name', 'Group'] if c in imputed_df.columns]
                    corr = imputed_df.drop(columns=idc).corr(numeric_only=True)
                    corr = corr.iloc[:N, :N]
                    fig = px.imshow(corr, title=f'Correlation Matrix (Top {N}x{N})', color_continuous_scale='RdBu')
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.info(f'Correlation preview skipped: {e}')

                # Paths
                if saved_csv:
                    st.caption(f'Saved CSV: {saved_csv}')
                if saved_pipe:
                    st.caption(f'Saved pipeline: {saved_pipe}')

            except Exception as e:
                st.exception(e)

    # VAE Tab: Impute missing values in RNAseq genes using a VAE
    with tab4:
        if 'merged_df' not in st.session_state:
            st.info('Load data in the "Initial Data Loading" tab first.')
        else:
            merged_df = st.session_state['merged_df']
            merged_scaled = st.session_state['merged_scaled']
            all_genes = st.session_state['all_genes']
            all_proteins = st.session_state['all_proteins']
            all_phys = st.session_state['all_phys']

            st.markdown("Select VAE hyperparameters and impute missing RNAseq gene values.")

            # Hyperparameters UI
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                latent = st.slider('Latent dim', min_value=8, max_value=128, value=32, step=8)
            with c2:
                hidden = st.selectbox('Hidden size', options=[128, 256, 512], index=1)
            with c3:
                epochs = st.slider('Epochs', min_value=50, max_value=2000, value=500, step=50)
            with c4:
                lr = st.selectbox('Learning rate', options=[1e-2, 5e-3, 1e-3, 5e-4, 1e-4], index=2)

            beta = st.slider('KL weight (beta)', min_value=0.1, max_value=5.0, value=1.0, step=0.1)

            # Choose feature subset
            feature_set = st.selectbox(
                'Feature subset to impute',
                options=['RNAseq genes', 'Proteins', 'Physiological', 'All numeric'],
                index=0,
                help='Select which columns to pass to the VAE for imputation. Data is scaled to [0,1] before training.'
            )
            if feature_set == 'RNAseq genes':
                feature_cols = all_genes
            elif feature_set == 'Proteins':
                feature_cols = all_proteins
            elif feature_set == 'Physiological':
                feature_cols = all_phys
            else:
                # All numeric columns excluding identifier and group
                feature_cols = [
                    c for c in merged_scaled.columns
                    if c not in ['Group']
                ]

            st.caption(f"Feature block: {feature_set} ({len(feature_cols)} features)")

            disabled_reason = None
            if not TORCH_AVAILABLE:
                disabled_reason = 'PyTorch is not installed. Please install torch to use this tab.'
            elif not VAE_AVAILABLE:
                disabled_reason = 'VAE implementation not found (GenAI/VAE_RNASeq.py).'

            if disabled_reason:
                st.error(disabled_reason)
            else:
                if st.button('Train VAE and Impute', type='primary'):
                    with st.spinner('Imputing with VAE...'):
                        try:
                            imputed_df = _impute_scaled_block(
                                merged_scaled=merged_scaled,
                                merged_df_original=merged_df,
                                feature_cols=feature_cols,
                                epochs=int(epochs),
                                lr=float(lr),
                                beta=float(beta),
                                hidden=int(hidden),
                                latent=int(latent),
                            )

                            st.success('Imputation complete! Preview below:')
                            st.dataframe(imputed_df.head(20))

                            # Store in session for reuse
                            st.session_state['imputed_df_vae'] = imputed_df

                            # Download
                            csv = imputed_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label='Download imputed dataframe (CSV)',
                                data=csv,
                                file_name='merged_df_imputed_vae.csv',
                                mime='text/csv'
                            )
                        except Exception as e:
                            st.exception(e)


if __name__ == '__main__':
    main()
