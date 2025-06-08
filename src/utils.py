from alphabase.psm_reader import psm_reader_provider
from peptdeep.model.ms2 import calc_ms2_similarity
import pandas as pd

def extract_evidence(path):
    column_mapping = {
        'sequence': 'Sequence',                           # Chuỗi peptide (không chứa biến đổi)
        'modified_sequence': 'Modified sequence',         # Chuỗi peptide đã chỉnh sửa (bao gồm các biến đổi)
        'length': 'Length',                                # Độ dài chuỗi peptide
        'modifications': 'Modifications',                 # Thông tin các biến đổi
        'phospho_probabilities': 'Phospho (STY) Probabilities', # Xác suất phosphorylation trên Tyrosine
        'oxidation_probabilities': 'Oxidation (M) Probabilities', # Xác suất oxidation trên Methionine
        'phospho_score_diffs': 'Phospho (STY) Score Diffs',  # Điểm số khác biệt của phosphorylation
        'oxidation_score_diffs': 'Oxidation (M) Score Diffs', # Điểm số khác biệt của oxidation
        'phospho_sites': 'Phospho (STY)',                   # Vị trí phosphorylation trên Tyrosine
        'oxidation_sites': 'Oxidation (M)',               # Vị trí oxidation trên Methionine
        'missed_cleavages': 'Missed cleavages',           # Số lần cắt bỏ sót
        'proteins': 'Proteins',                           # Danh sách các protein liên quan
        'leading_protein': 'Leading razor protein',       # Protein chính liên kết với peptide
        'type': 'Type',                                   # Loại peptide (ví dụ: unique, razor)
        'raw_name': 'Raw file',                           # Tên file thô LC-MS/MS
        'experiment': 'Experiment',                       # Thông tin thí nghiệm
        'msms_mz': 'MS/MS m/z',                           # Giá trị m/z của MS/MS
        'charge': 'Charge',                               # Điện tích peptide
        'mz': 'm/z',                                      # Giá trị m/z
        'mass': 'Mass',                                   # Khối lượng peptide
        'resolution': 'Resolution',                       # Độ phân giải của phân tích khối phổ
        'mass_error_ppm': 'Mass Error [ppm]',             # Sai số khối lượng tính bằng ppm
        'mass_error_da': 'Mass Error [Da]',               # Sai số khối lượng tính bằng Dalton
        'rt': 'Retention time',                           # Thời gian lưu trữ
        'score': 'Score',                                 # Điểm số xác định peptide
        'delta_score': 'Delta score',                     # Chênh lệch điểm số
        'pep': 'PEP',                                     # Posterior Error Probability (PEP)
        'intensity': 'Intensity',
        'intensities': 'Intensities',                     # Cường độ tín hiệu
        'decoy': 'Reverse',                             # Cờ đánh dấu trình tự đảo ngược (decoy)
        'potential_contaminant': 'Potential contaminant', # Đánh dấu peptide là chất gây nhiễu hay không
        'peptide_id': 'Peptide ID',                       # ID duy nhất cho peptide
        'mod_peptide_id': 'Mod. peptide ID',              # ID peptide bao gồm biến đổi
        'msms_ids': 'MS/MS IDs',                         # ID của các MS/MS liên qua
        'scan_num': 'MS/MS scan number',                  # Số scan của MS/MS
        'experiment': "Experiment",
        'genes': 'Gene names',
        'protein_names': 'Protein names',
        'peptide_id': 'Peptide ID',
        'protein_group_id': 'Protein group IDs'
    }

    modification_mapping = {
        'Oxidation@M': ['Oxidation (M)', 'M(ox)'],
        'Phospho@Y': ['Phospho (Y))','Y(ph)'],
        'Phospho@STY': ['Phospho (STY)', 'STY(ph)'],
        'Acetyl@N-term': ['Acetyl (N-term)', 'N-term(ac)']
    }

    ccs = fr'{path}\evidence.txt'
    mq_reader = psm_reader_provider.get_reader(
        reader_type='maxquant',
        column_mapping=column_mapping,
        modification_mapping = modification_mapping,
        keep_decoy=True
    )
    mq_reader.load(ccs)

    return mq_reader.psm_df

def assess_model(model_mgr, psm_df, intensity_df):
    pred_intensities = model_mgr.ms2_model.predict(psm_df)
    
    result_psm, metrics = calc_ms2_similarity(
        psm_df, predict_intensity_df=pred_intensities, fragment_intensity_df=intensity_df
    )
    return result_psm, metrics

def convert_psm_data_to_hdf(path: str, out_dir: str):
    name = path.split('\\')[-1]
    with pd.HDFStore(fr'{out_dir}\{name}.hdf', 'w') as store:
        psm_df = pd.read_csv(fr'{path}\psm_df.csv')
        fragment_mz_df = pd.read_csv(fr'{path}\fragment_mz_df.csv')
        fragment_intensity_df = pd.read_csv(fr'{path}\fragment_intensity_df.csv')
        store['psm_df'] = psm_df
        store['fragment_mz_df'] = fragment_mz_df
        store['fragment_intensity_df'] = fragment_intensity_df