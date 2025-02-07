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


def set_page_default():
    st.set_page_config(layout='wide')


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


# Streamlit app setup
def main():
    set_page_default()

    tab1, tab2, tab3 = st.tabs(
        [
            'Initial Data Loading',
            'Imputation',
            'Data Generation GAN',
        ]
    )

    tab1.subheader('Data Loading')
    tab2.subheader('Imputation')
    tab3.subheader('Data Generation GAN')

    # Sidebar for data loading
    with tab1:

        merged_df, all_phys, all_genes, all_proteins = acquire_data()
        merged_scaled, numeric_values = scale_data(merged_df)

        print(numeric_values)

        plot_heatmap(merged_scaled, all_phys, all_genes, all_proteins)


if __name__ == '__main__':
    main()
