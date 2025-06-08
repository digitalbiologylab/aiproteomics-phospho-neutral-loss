from peptdeep.pretrained_models import ModelManager
import pandas as pd
import numpy as np
import tqdm

def generate_spectral_lib(
    output_file: str,
    model_mgr: ModelManager,
    psm_df: pd.DataFrame,
    predict_items: list, # ['mobility', 'rt', 'ms2']
    frag_types: list,
    multiprocessing: bool
) -> None:
    # Preprocessing
    psm_df['mod_sites'] = psm_df['mod_sites'].astype(str)
    psm_df['mods'] = psm_df['mods'].astype(str)
    psm_df.fillna({'mod_sites': ''}, inplace=True)
    psm_df.fillna({'mods': ''}, inplace=True)

    pep_df = model_mgr.predict_all(
        precursor_df=psm_df,
        predict_items=predict_items,
        frag_types=frag_types,
        multiprocessing=multiprocessing
    )

    precursor_df = pep_df['precursor_df']
    fragment_mz_df = pep_df['fragment_mz_df']
    fragment_intensity_df = pep_df['fragment_intensity_df']
    assert max(precursor_df['frag_stop_idx']) == len(fragment_intensity_df)

    precursor_df['idx'] = precursor_df.index
    output_df = precursor_df[['idx']]
    output_df['ModifiedPeptide'] = precursor_df.apply(modify_peptide, axis = 1) # Modify peptide sequence
    rename_dict = {'raw_name': 'Run', 'sequence': 'PeptideSequence','charge': 'PrecursorCharge', 'precursor_mz': 'PrecursorMz',
        'rt_norm': 'Tr_recalibrated',
    'genes': 'Genes'}
    selected_cols = ['raw_name', 'idx', 'sequence', 'charge', 'precursor_mz', 'rt_norm', 'frag_start_idx',
            'frag_stop_idx', 'UniprotID', 'CTerm', 'NTerm', 'ProteinName', 'Proteotypic',
            'genes', 'ProteinGroup', 'decoy']
    if 'mobility' in predict_items:
        selected_cols.append('mobility_pred')
        rename_dict.update({'mobility_pred':'IonMobility'})
    
    output_df = pd.merge(left=output_df,
        right=precursor_df[selected_cols],
        on='idx'
    ).rename(columns=rename_dict)

    fragment_types = fragment_mz_df.columns
    output_list = []

    # Add fragment information
    for i, row in tqdm.tqdm(output_df.iterrows(), total=len(output_df)):
        # if( i < 100): 
        #     continue
        intens = fragment_intensity_df[row['frag_start_idx']:row['frag_stop_idx']]
        mzs = fragment_mz_df[row['frag_start_idx']:row['frag_stop_idx']]
        for fr_type in fragment_types:
            ion_type = fr_type[0]
            ion_charge = fr_type[-1]
            series_number = 0 if ion_type == 'b' else len(mzs) + 1
            for mz, intensity in zip(mzs[fr_type],intens[fr_type]):
                series_number += 1 if ion_type == 'b' else -1
                if(mz <= 0 or intensity <= 0): continue
                frag_loss_type = 'noloss' if 'modloss' not in fr_type else 'H3PO4'
                output_list.append(list(row) + [mz, intensity, int(ion_charge), ion_type, series_number, frag_loss_type])
    
    output_df_col = list(output_df.columns.values) + ['ProductMz' ,'LibraryIntensity', 'FragmentCharge', 'FragmentType', 'FragmentSeriesNumber', 'FragmentLossType']
    output_df = pd.DataFrame(data=output_list, columns=output_df_col)
    output_df.drop(columns=['idx', 'frag_start_idx', 'frag_stop_idx'], inplace=True)
    
    output_df.to_csv(output_file, sep='\t', index=False)

def modify_peptide(row: pd.Series) -> str:
    mod_mapping = {
        'Phospho': 'UniMod:21',
        'Carbamidomethyl': 'UniMod:4',
        'Acetyl@Protein_N-te': 'UniMod:1',
        'Oxidation': 'UniMod:35',
        'Deamidated': 'UniMod:7'
    }
    sequence, mods, mod_sites = row['sequence'], row['mods'], row['mod_sites']
    if not len(mod_sites):
        return '_' + sequence + '_'

    modified_sequence = ''
    last_mod = 0
    for mod, mod_site in zip(mods.split(';'), mod_sites.split(';')):
        modified_sequence += sequence[last_mod:int(mod_site)] + '[' + mod_mapping[mod[:-2]] + ']'
        last_mod = int(mod_site)
    modified_sequence = '_' + modified_sequence + sequence[last_mod:] + '_'
    return modified_sequence

def combine_spectral_lib(lib_paths = []):
    libs = []
    cols = []
    for i, path in enumerate(lib_paths):
        lib_df = pd.read_csv(path, sep='\t')
        libs.extend(lib_df.values)
        if i == 0:
            cols = lib_df.columns
        
    return pd.DataFrame(np.vstack(libs), columns=cols)