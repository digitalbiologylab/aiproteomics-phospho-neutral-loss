from peptdeep.mass_spec.match import match_centroid_mz
from peptdeep.model.ms2 import spectral_angle

import torch
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from plotly.subplots import make_subplots
import plotly.graph_objects as go


def match_frag_spec_df(
    spec_df: pd.DataFrame,
    frag_df: pd.DataFrame,
    ppm_tol=20
) -> pd.DataFrame:
    tols = spec_df.mz_values.values*ppm_tol*1e-6
    matched_idxes = match_centroid_mz(
        spec_df.mz_values.values,
        frag_df.mz_values.values, 
        tols
    )
    matched_intens = spec_df.intensity_values.values[matched_idxes]
    matched_intens[matched_idxes==-1] = 0
    matched_mzs = spec_df.mz_values.values[matched_idxes]
    mass_errs = (
        matched_mzs-frag_df.mz_values.values
    )/frag_df.mz_values.values*1e6
    mass_errs[matched_idxes==-1] = -1
    matched_mzs[matched_idxes==-1] = 0
    frag_df['intensity_matched'] = matched_intens


    matched_df = frag_df.copy()
    matched_df['mz_values'] = matched_mzs
    matched_df['intensity_values'] = matched_intens
    
    
    frag_df['intensity_values'] *= -matched_intens.max()
    matched_df['mass_dev_ppm'] = mass_errs

    return matched_df

def plot_all_peaks(fig, spec_df: pd.DataFrame, color: str):
    fig.add_trace(
        go.Scatter(
            x=spec_df.mz_values,
            y=spec_df.intensity_values,
            mode='markers',
            marker=dict(color=color, size=1),
            hovertemplate='<b>m/z:</b> %{x}<br><b>Intensity:</b> %{y}',
            name='',
            showlegend=False
        )
    )

def plot_ion(fig, df: pd.DataFrame, color: str):
    fig.add_trace(
        go.Scatter(
            x=df.mz_values,
            y=df.intensity_values,
            mode='markers',
            marker=dict(color=color, size=1),
            hovertext=df.ions,
            hovertemplate='<b>m/z:</b> %{x}<br><b>Intensity:</b> %{y}<br><b>Ion:</b> %{hovertext}.',
            name='',
            showlegend=False
        )
    )

def add_mass_error_scatter(fig_common, df: pd.DataFrame, color: str, name: str, row: int, col: int):
    df = df[df.intensity_values>0]
    fig_common.add_trace(
        go.Scatter(
            x=df.mz_values,
            y=df.mass_dev_ppm.round(4),
            hovertext=df.ions,
            hovertemplate='<b>m/z:</b> %{x};<br><b>Mass error(ppm):</b> %{y};<br><b>Ion:</b> %{hovertext}.',
            mode='markers',
            marker=dict(
                color=color,
                size=6
            ),
            name=name
        ),
        row=row, col=col
    )

def scatters_to_vlines(fig, 
    plot_df: pd.DataFrame, color_map: dict, 
    spectrum_line_width: float, height: int,
    title: str, template, metrics
):
    fig.update_layout(
        template=template,
        shapes=[
            dict(
                type='line',
                xref='x',
                yref='y',
                x0=plot_df.loc[i, 'mz_values'],
                y0=0,
                x1=plot_df.loc[i, 'mz_values'],
                y1=plot_df.loc[i, 'intensity_values'],
                line=dict(
                    color=color_map[plot_df.loc[i, 'ions'][0]],
                    width=spectrum_line_width
                )
            ) for i in plot_df.index
        ],
        yaxis=dict(
            title='intensity',
        ),
        legend=dict(
            orientation="h",
            x=1,
            xanchor="right",
            yanchor="bottom",
            y=1.01
        ),
        hovermode="closest",
        height=height,
        title=dict(
            text=f"{title} ({', '.join([f"{key}={value:.3f}" for key, value in metrics.items()])})",
            yanchor='bottom'
        )
    )

def mirror_plot(spec_df, frag_df, title, ppm_tol=20,
    spectrum_line_width=1.5, height=520,
    color_map={'b':'blue','y':'red','unmatched':'lightgrey'},
    template="plotly_white",
    metrics = ['PCC']
    ):
    color_map['-'] = color_map['unmatched']
    matched_df = match_frag_spec_df(spec_df, frag_df, ppm_tol)
    scores = {}
    if 'PCC' in metrics:
        pcc = torch.nn.functional.cosine_similarity(
            torch.tensor(frag_df.intensity_values.values-frag_df.intensity_values.values.mean()), 
            -torch.tensor(matched_df.intensity_values-matched_df.intensity_values.mean()), 
            dim=0
        )
        scores['PCC'] = pcc
    if 'COS' in metrics or 'SA' in metrics:
        cos = torch.cosine_similarity(torch.tensor(frag_df.intensity_values.values), torch.tensor(matched_df.intensity_values), dim=0)
        scores['COS'] = cos
        if 'SA' in metrics:
            sa = spectral_angle(cos)
            scores['SA'] = sa
    
    matched_df = matched_df[matched_df.mz_values>0]
    spec_df['ions'] = "-"
    plot_df = pd.concat(
        [
            spec_df[["ions","mz_values","intensity_values"]],
            matched_df,
            frag_df,
        ]
    ).reset_index(drop=True)

    fig = go.Figure()
    plot_all_peaks(fig, spec_df, color_map['-'])
    plot_ion(fig, plot_df[plot_df.ions.str.startswith('y')], color_map['y'])
    plot_ion(fig, plot_df[plot_df.ions.str.startswith('b')], color_map['b'])

    scatters_to_vlines(fig, 
        plot_df, color_map, spectrum_line_width,
        height, title, template, scores
    )

    fig_common = make_subplots(
        rows=6, cols=3, shared_xaxes=True,
        figure=fig,
        specs=[
          [{"rowspan": 4, "colspan": 3}, None, None],
          [None, None, None],
          [None, None, None],
          [None, None, None],
          [{"colspan": 3}, None, None],
          [{}, {}, {}]
        ],
        vertical_spacing=0.07,
        column_widths=[0.25, 0.5, 0.25]
    )

    add_mass_error_scatter(fig_common, 
        plot_df[plot_df.ions.str.startswith('y')],
        color=color_map['y'], name='y ions', row=5, col=1
    )

    add_mass_error_scatter(fig_common, 
        plot_df[plot_df.ions.str.startswith('b')],
        color=color_map['b'], name='b ions', row=5, col=1
    )

    fig_common.update_yaxes(
        title_text=r"ppm", row=5, col=1, 
        range=[-ppm_tol*1.1, ppm_tol*1.1]
    )
    fig_common.update_xaxes(
        title_text=r"m/z", row=5, col=1,
    )
    fig_common.update_xaxes(matches='x')

    return fig_common

def predict_one_peptide(peptide_df: pd.DataFrame, model_mgr: object):

    def get_frag_type(column, idx, nAA):
        if column[0] in "abc":
            ion_num = idx+1
        else:
            ion_num = nAA-idx-1
        charge = int(column[-1])
        return f"{column[0]}{ion_num}{column[1:-3]}{'+'*charge}", ion_num
    
    assert peptide_df.shape[0] == 1

    peptide = peptide_df.loc[0].to_dict()
    predict_dict = model_mgr.predict_all(
        peptide_df, predict_items=['mobility','rt','ms2'],
    )
    frag_mz_df = predict_dict['fragment_mz_df']
    frag_inten_df = predict_dict['fragment_intensity_df']
    nAA = peptide['nAA']

    frags = {}
    intens = {}
    frag_nums = {}
    for column in frag_mz_df.columns.values:
        for i,(mz,inten) in enumerate(zip(
            frag_mz_df[column].values,frag_inten_df[column].values
        )):
            if mz < 10: continue
            frag_type, ion_num = get_frag_type(column, i, nAA)
            frags[frag_type] = mz
            intens[frag_type] = inten
            frag_nums[frag_type] = ion_num
    
    peptide['fragments'] = frags
    peptide['intensities_pred'] = intens
    peptide['fragment_numbers'] = frag_nums
    
    return peptide

def get_frag_to_plot(peptide_dict: dict):
    return pd.concat([
    pd.DataFrame().from_dict(
        peptide_dict['intensities_pred'], orient='index', 
        columns=['intensity_values']
    ),
    pd.DataFrame().from_dict(
        peptide_dict['fragments'], orient='index', 
        columns=['mz_values']
    ),
    pd.DataFrame().from_dict(
        peptide_dict['fragment_numbers'], orient='index', 
        columns=['fragment_numbers']
    ),
    ], axis=1).reset_index().rename(columns={'index':'ions'})

def compare_metrics(
    first_metrics: pd.DataFrame,
    second_metrics: pd.DataFrame,
    first_label_title = 'Pretrained model',
    second_label_title = 'Tuning model',
    compare_values = ['mean', '25%', '50%', '75%', '>0.90', '>0.75'],
    figsize=(10,5),
    cmap=ListedColormap(['#aec7e8', '#1f77b4']),
    evaluations = ['PCC']

):
    for evaluation in evaluations:
        df = pd.DataFrame([[value] + [round(first_metrics.loc[value, evaluation], 2), round(second_metrics.loc[value, evaluation], 2)]  for value in compare_values],
                            columns=['Aspect', first_label_title, second_label_title])
        ax = df.plot(x='Aspect',
                kind='bar',
                figsize=figsize,
                legend=True,
                width=0.5,
                colormap=cmap)
        
        for p in ax.containers:
            ax.bar_label(p, fontsize=9)

        plt.title(evaluation, fontsize=15)
        plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
        plt.show()