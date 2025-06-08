import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import tqdm
import os

from .utils import extract_evidence

def extract_fragment_data(
        precursor_df: pd.DataFrame,
        msms_df: pd.DataFrame,
        msmsScan_df: pd.DataFrame,
        frag_types: list,
        is_included_other_mods: bool # Different from phospho mods
    ):
    msms_df.fillna('', inplace=True)
    precursor_df['nce'] = 0
    frag_dict = {frag_type: idx for idx, frag_type in enumerate(frag_types)}

    start_idx = 0
    mz_data_list = []
    intensity_data_list = []

    for idx, peptide in tqdm.tqdm(precursor_df.iterrows(), total=len(precursor_df), desc="Processing peptides"):  
        msms_ids = peptide['msms_ids'].split(';') # Contain peptide with multiple msms id
        msms_data_mapping = msms_df[(msms_df['id'].isin(msms_ids)) & (peptide['scan_num'] == msms_df['Scan number'])][['Masses', 'Intensities', 'Matches']].reset_index()
        nces = msmsScan_df[(msmsScan_df['MS/MS IDs'].isin(msms_ids)) & (peptide['scan_num'] == msmsScan_df['Scan number'])]['Collision energy'].values
        precursor_df.at[idx, 'nce'] = nces[0] if len(nces) > 0 else 0
        if msms_data_mapping.shape[0] != 1: # Abnormal case
            continue

        masses = msms_data_mapping.loc[0, 'Masses'].split(';')
        intensities = msms_data_mapping.loc[0, 'Intensities'].split(';')
        match_ions = msms_data_mapping.loc[0, 'Matches'].split(';')

        masses = np.array(masses, dtype=float)
        intensities = np.array(intensities, dtype=float)
        maxIonIdx = len(peptide['sequence']) - 1
        masses_data = np.zeros(shape=(maxIonIdx, len(frag_types)), dtype=float)
        intensities_data = np.zeros(shape=(maxIonIdx, len(frag_types)), dtype=float)
        for i, ion in enumerate(match_ions):
            charge = 1 if '2+' not in ion else 2
            if ion == 'pY':
                continue
            if '-' in ion:
                if is_included_other_mods:
                    typeIonNth, loss_type = ion.split('-')
                    mid_type = f'_{loss_type}_z'
                    fr_type, num = typeIonNth[0], int(typeIonNth[1:])
                else: continue
                
            elif '*' in ion: # mod_loss
                mid_type = '_modloss_z'
                fr_type, num = ion[0], int(ion[1:-1])
            else:
                mid_type = '_z'
                fr_type, num = ion[0], int(ion[1:]) if charge == 1 else int(ion[1:-4])
            
            if fr_type not in ('b', 'y') or num < 1 or num > maxIonIdx:
                continue
            
            columns_idx = frag_dict[fr_type + mid_type + str(charge)]
            if fr_type == 'b':
                masses_data[num - 1, columns_idx] = float(masses[i])
                intensities_data[num - 1, columns_idx] = float(intensities[i])
            elif fr_type == 'y':
                masses_data[maxIonIdx - num, columns_idx] = float(masses[i])
                intensities_data[maxIonIdx - num , columns_idx] = float(intensities[i])
                    
        if np.max(intensities_data) > 0:
            intensities_data /= np.max(intensities_data)
        mz_data_list.extend(masses_data.tolist())
        intensity_data_list.extend(intensities_data.tolist())

        precursor_df.at[idx, 'frag_start_idx'] = start_idx
        precursor_df.at[idx, 'frag_stop_idx'] = start_idx + maxIonIdx
        start_idx += maxIonIdx

    fragment_mz_df = pd.DataFrame(np.vstack(mz_data_list), columns=frag_types)
    fragment_intensity_df = pd.DataFrame(np.vstack(intensity_data_list), columns=frag_types)
    return fragment_mz_df, fragment_intensity_df

def extract_peptide_info(
        path: str,
        out_dir: str,
        frag_types: list,
        is_included_other_mods: bool,
        pep_threshold: float = 0.01,
        score_threshold: float = 60.0,
        delta_score_threshold: float = 10.0,
        instrument: str = 'Lumos'
    ):
    print('Start: ...')
    psm_df = extract_evidence(path=path)
    psm_df = psm_df[(psm_df['pep'] <= pep_threshold) & (psm_df['score'] >= score_threshold)
                                & (psm_df['delta_score'] >= delta_score_threshold)]
    print('Evidence reading: Done')
    msms_df = pd.read_csv(fr'{path}\msms.txt', sep='\t')
    peptides = pd.read_csv(fr'{path}\peptides.txt', sep='\t')
    proteinGroupDf = pd.read_csv(fr'{path}\proteinGroups.txt', sep='\t')
    msmsScansDf = pd.read_csv(fr'{path}\msmsScans.txt', sep='\t')

    psm_df.loc[:, 'frag_start_idx'] = -1
    psm_df.loc[:, 'frag_stop_idx'] = -1
    msms_df['id'] = msms_df['id'].astype(str)
    msmsScansDf['MS/MS IDs'] = msmsScansDf['MS/MS IDs'].astype(str)

    print('Start extract fragment')
    fragment_mz_df, fragment_intensity_df = extract_fragment_data(psm_df, msms_df, msmsScansDf, frag_types, is_included_other_mods)

    # NTerm, CTerm
    psm_df['NTerm'] = psm_df['mods'].apply(lambda mod: 1 if 'Acetyl@Protein_N-term' in mod else 0)
    psm_df['CTerm'] = 0

    # Decoy
    psm_df['decoy'] = psm_df['decoy'].apply(lambda decoy: 1 if decoy == '+' else 0)

    # Merge with peptides
    psm_df_final = psm_df[['charge', 'frag_stop_idx', 'frag_start_idx', 'mod_sites', 'mods', 'precursor_mz', 'proteins',
                        'raw_name', 'rt', 'rt_norm', 'scan_num', 'score', 'sequence', 'spec_idx', 'genes', 'protein_names',
                        'peptide_id', 'CTerm', 'NTerm', 'protein_group_id', 'nce', 'decoy']]
    psm_df_final = psm_df_final.merge(right=peptides[['id', 'Unique (Proteins)']], left_on='peptide_id', right_on='id')
    psm_df_final.drop(columns=['id'], inplace=True)

    # Merge with proteinGroup
    proteinGroupDf['id'] = proteinGroupDf['id'].astype(str)
    for i, row in tqdm.tqdm(psm_df_final.iterrows(), total=len(psm_df_final)):
        psm_df_final.loc[i, 'ProteinGroup'] = ';'.join(list(proteinGroupDf[proteinGroupDf['id'].isin(row['protein_group_id'].split(';'))]['Protein IDs']))
    psm_df_final.drop(columns=['protein_group_id'], inplace=True)

    # Rename column to serve train/predict
    psm_df_final.rename(columns={"Unique (Proteins)": "Proteotypic", "proteins": "UniprotID",
    "protein_names": "ProteinName", "Protein IDs": "ProteinGroup"}, inplace=True)
    psm_df_final['Proteotypic'] = psm_df_final['Proteotypic'].apply(lambda x: int(x == 'yes'))

    psm_df_final['instrument'] = instrument
    psm_df_final['nAA'] = psm_df_final['sequence'].apply(lambda x: len(x))

    psm_df_final.to_csv(fr'{out_dir}\psm_df.csv', index=False)
    fragment_mz_df.to_csv(fr'{out_dir}\fragment_mz_df.csv', index=False)
    fragment_intensity_df.to_csv(fr'{out_dir}\fragment_intensity_df.csv', index=False)

def extract_peptide_info_to_separated_raw_file(
        path: str,
        out_dir: str,
        frag_types: list,
        is_included_other_mods: bool,
        raw_names: list = [],
        pep_threshold: float = 0.01,
        score_threshold: float = 60.0,
        delta_score_threshold: float = 10.0,
        instrument: str = 'Lumos'
    ):
    print('Start: ...')
    precursor_df = extract_evidence(path=path)
    precursor_df = precursor_df[(precursor_df['pep'] <= pep_threshold) & (precursor_df['score'] >= score_threshold)
                                & (precursor_df['delta_score'] >= delta_score_threshold)]
    print('Evidence reading: Done')
    msms_df = pd.read_csv(fr'{path}\msms.txt', sep='\t')
    peptides = pd.read_csv(fr'{path}\peptides.txt', sep='\t')
    proteinGroupDf = pd.read_csv(fr'{path}\proteinGroups.txt', sep='\t')
    msmsScansDf = pd.read_csv(fr'{path}\msmsScans.txt', sep='\t')

    precursor_df.loc[:, 'frag_start_idx'] = -1
    precursor_df.loc[:, 'frag_stop_idx'] = -1
    msms_df['id'] = msms_df['id'].astype(str)
    msmsScansDf['MS/MS IDs'] = msmsScansDf['MS/MS IDs'].astype(str)

    list_raw_name = precursor_df['raw_name'].unique()
    if len(raw_names) == 0:
        raw_names = list_raw_name
    for raw_name in raw_names:
        if raw_name not in list_raw_name:
            print(f"Raw name {raw_name} not found")
            continue
        print(f"Extract raw name: {raw_name}")
        psm_df = precursor_df[precursor_df['raw_name'] == raw_name]
        specified_msms_df = msms_df[msms_df['Raw file'] == raw_name]
        specified_msmsScansDf = msmsScansDf[msmsScansDf['Raw file'] == raw_name]
        fragment_mz_df, fragment_intensity_df = extract_fragment_data(psm_df, specified_msms_df, specified_msmsScansDf, frag_types, is_included_other_mods)

        # NTerm, CTerm
        psm_df['NTerm'] = psm_df['mods'].apply(lambda mod: 1 if 'Acetyl@Protein_N-term' in mod else 0)
        psm_df['CTerm'] = 0

        # Decoy
        psm_df['decoy'] = psm_df['decoy'].apply(lambda decoy: 1 if decoy == '+' else 0)

        # Merge with peptides
        psm_df_final = psm_df[['charge', 'frag_stop_idx', 'frag_start_idx', 'mod_sites', 'mods', 'precursor_mz', 'proteins',
                            'raw_name', 'rt', 'rt_norm', 'scan_num', 'score', 'sequence', 'spec_idx', 'genes', 'protein_names',
                            'peptide_id', 'CTerm', 'NTerm', 'protein_group_id', 'nce', 'decoy']]
        psm_df_final = psm_df_final.merge(right=peptides[['id', 'Unique (Proteins)']], left_on='peptide_id', right_on='id')
        psm_df_final.drop(columns=['id'], inplace=True)

        # Merge with proteinGroup
        proteinGroupDf['id'] = proteinGroupDf['id'].astype(str)
        for i, row in tqdm.tqdm(psm_df_final.iterrows(), total=len(psm_df_final)):
            psm_df_final.loc[i, 'ProteinGroup'] = ';'.join(list(proteinGroupDf[proteinGroupDf['id'].isin(row['protein_group_id'].split(';'))]['Protein IDs']))
        psm_df_final.drop(columns=['protein_group_id'], inplace=True)

        # Rename column to serve train/predict
        psm_df_final.rename(columns={"Unique (Proteins)": "Proteotypic", "proteins": "UniprotID",
        "protein_names": "ProteinName", "Protein IDs": "ProteinGroup"}, inplace=True)
        psm_df_final['Proteotypic'] = psm_df_final['Proteotypic'].apply(lambda x: int(x == 'yes'))

        psm_df_final['instrument'] = instrument
        psm_df_final['nAA'] = psm_df_final['sequence'].apply(lambda x: len(x))

        out_dir_cb = fr'{out_dir}\{raw_name}'
        if not os.path.exists(out_dir_cb):
            os.makedirs(out_dir_cb)
        psm_df_final.to_csv(fr'{out_dir_cb}\psm_df.csv', index=False)
        fragment_mz_df.to_csv(fr'{out_dir_cb}\fragment_mz_df.csv', index=False)
        fragment_intensity_df.to_csv(fr'{out_dir_cb}\fragment_intensity_df.csv', index=False)