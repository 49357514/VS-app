import pandas as pd
import streamlit as app
import joblib
import subprocess
from rdkit import Chem
from rdkit.Chem import Lipinski, Descriptors
import numpy as np
import os
app.set_page_config(page_title = 'Virtual Screenig Tool for Aromatase Receptor',
                   layout = 'wide')

app.write("""
         # Virtual Screenig Tool for Aromatase Receptor
         """)

with app.sidebar.header('Upload for Prediction:'):
    uploaded_file = app.sidebar.file_uploader("Upload your smile in CSV format:", type=['csv'])

# model = joblib.load('SVR.joblib')

def calculate_lipinski(smiles):
    mol_data = []
    for smile in smiles:
        mol = Chem.MolFromSmiles(smile)
        mol_data.append(mol)
    baseData = np.arange(1,1)
    i = 0
    for molecule in mol_data:
        mol_Wt = Descriptors.MolWt(molecule)
        mol_logP = Descriptors.MolLogP(molecule)
        mol_H_Donors = Descriptors.NumHDonors(molecule)
        mol_H_Acceptors = Descriptors.NumHAcceptors(molecule)

        data = np.array([mol_Wt, mol_logP, mol_H_Donors, mol_H_Acceptors])
        if(i == 0):
            baseData = data
        else:
            baseData = np.vstack([baseData, data])
        i = i+1
    descriptors = pd.DataFrame(data = baseData, columns = ['MW', 'LogP', 'NumHDonors', 'NumHAcceptors'])
    return descriptors

def fingerprint_calculator(smiles_df):
    smiles_df.to_csv('molecule.smi', sep = '\t', index = False, header = False)

    ## convert smiles dataset to fingerprint
    try:
        placeholder = app.empty()
        placeholder.info('Please wait while we calculating molecular fingerprints ...')
        subprocess.run(['bash', './padel/padel.sh'], check=True, )
        placeholder.info('Molecular Fingerprints Calculated Successfully. ')
        os.remove('molecule.smi')
        placeholder.empty()
        print('molecule.smi removed successfully ...')
    except subprocess.CalledProcessError as err:
        app.write('Error while calculating fingerprint')
        print(err)
    return

# def make_pred(df):
#     prediction = model.predict(df)
#     app.write(prediction)
#     return

def feature_selection(fingerprint_df):
    
    std_df = pd.read_csv('dataset.csv')
    X_test = fingerprint_df[list(std_df.columns)] ## getting columns of the training dataset
    return

def web_app(uploaded_csv):
    ## Calculate Lipinski's RO5
    lipinski_df = calculate_lipinski(uploaded_csv['canonical_smiles'])

    ## Join Lipsnski dataframe and uploaded smile csv dataframe
    uploaded_csv.reset_index(inplace=True, drop=True)
    lipinski_df.reset_index(inplace=True, drop=True)

    lipinski_joined = pd.concat([uploaded_csv, lipinski_df], axis=1)
    app.subheader('Calculated Lipnski RO5 of uploaded dataset :')
    app.write(lipinski_joined)

    ## Calculate fingerprint
    fingerprint_calculator(uploaded_csv)
    fingerprint_df = pd.read_csv('descriptors_output.csv')
    fingerprint_df.rename(columns={'Name':'molecule_chembl_id'}, inplace=True)
    os.remove('descriptors_output.csv') # to avoid multiple file creation

    ## join existing lipinski_joined df and fingerprint df
    fingerprint_df.reset_index(inplace=True, drop=True)
    lipinski_joined.reset_index(inplace=True, drop=True)
    fingerprint_joined = pd.merge(lipinski_joined, fingerprint_df, on=['molecule_chembl_id'], how='right')
    app.subheader('Calculated Fingerprint of uploaded dataset :')
    app.write(fingerprint_joined)
    feature_selection(fingerprint_df=fingerprint_joined)
    return


if uploaded_file is not None:
    uploaded_csv = pd.read_csv(uploaded_file)
    app.write(uploaded_csv)

    
else:
    app.info('Please upload dataset in specified CSV format to predict ...')
    app.subheader('Example of CSV format :')
    example_smile_csv = pd.read_csv('molecules.csv')
    app.write(example_smile_csv.head(10))
    if app.button('Click to use this Example Dataset'):
        uploaded_csv = pd.read_csv('molecules.csv', nrows=20)
        web_app(uploaded_csv)