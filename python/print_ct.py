import pandas as pd
import abracatabra as abt
import matplotlib.pyplot as plt

# Define the data for each I0 setting
data = {
    '100k': {
        'fat': {'HU': -17.1, 'mu': 0.188, 'SNR': 0.32, 'HU_CNR_vs_muscle': 5.47, 'mu_CNR_vs_muscle': 5.62, 'Contrast_vs_muscle': 0.945},
        'muscle': {'HU': -314.6, 'mu': 0.130, 'SNR': 5.79, 'Contrast_vs_muscle': 0.0},
        'tumor': {'HU': 228.3, 'mu': 0.234, 'SNR': 4.20, 'HU_CNR_vs_muscle': 9.99, 'mu_CNR_vs_muscle': 10.07, 'Contrast_vs_muscle': 1.726},
        'nerve': {'HU': -419.9, 'mu': 0.110, 'SNR': 7.73, 'HU_CNR_vs_muscle': 1.94, 'mu_CNR_vs_muscle': 1.94, 'Contrast_vs_muscle': 0.334},
        'tumor-to-nerve HU_CNR': {'HU_CNR_vs_muscle': 11.93, 'mu_CNR_vs_muscle': 12.01},
        'tumor-to-nerve mu_CNR': {'mu_CNR_vs_muscle': 12.01},
        'tumor-to-nerve contrast': {'Contrast_vs_muscle': 1.544}
    },
    '500k': {
        'fat': {'HU': -17.2, 'mu': 0.188, 'SNR': 0.66, 'HU_CNR_vs_muscle': 11.39, 'mu_CNR_vs_muscle': 11.69, 'Contrast_vs_muscle': 0.945},
        'muscle': {'HU': -314.7, 'mu': 0.130, 'SNR': 12.05, 'Contrast_vs_muscle': 0.0},
        'tumor': {'HU': 228.2, 'mu': 0.234, 'SNR': 8.74, 'HU_CNR_vs_muscle': 20.79, 'mu_CNR_vs_muscle': 20.96, 'Contrast_vs_muscle': 1.725},
        'nerve': {'HU': -419.7, 'mu': 0.110, 'SNR': 16.07, 'HU_CNR_vs_muscle': 4.02, 'mu_CNR_vs_muscle': 4.03, 'Contrast_vs_muscle': 0.334},
        'tumor-to-nerve HU_CNR': {'HU_CNR_vs_muscle': 24.81, 'mu_CNR_vs_muscle': 24.99},
        'tumor-to-nerve mu_CNR': {'mu_CNR_vs_muscle': 24.99},
        'tumor-to-nerve contrast': {'Contrast_vs_muscle': 1.544}
    },
    '1M': {
        'fat': {'HU': -17.2, 'mu': 0.188, 'SNR': 0.86, 'HU_CNR_vs_muscle': 14.84, 'mu_CNR_vs_muscle': 15.23, 'Contrast_vs_muscle': 0.945},
        'muscle': {'HU': -314.7, 'mu': 0.130, 'SNR': 15.70, 'Contrast_vs_muscle': 0.0},
        'tumor': {'HU': 228.2, 'mu': 0.234, 'SNR': 11.38, 'HU_CNR_vs_muscle': 27.08, 'mu_CNR_vs_muscle': 27.31, 'Contrast_vs_muscle': 1.725},
        'nerve': {'HU': -419.7, 'mu': 0.110, 'SNR': 20.94, 'HU_CNR_vs_muscle': 5.24, 'mu_CNR_vs_muscle': 5.25, 'Contrast_vs_muscle': 0.334},
        'tumor-to-nerve HU_CNR': {'HU_CNR_vs_muscle': 32.32, 'mu_CNR_vs_muscle': 32.56},
        'tumor-to-nerve mu_CNR': {'mu_CNR_vs_muscle': 32.56},
        'tumor-to-nerve contrast': {'Contrast_vs_muscle': 1.544}
    },
    # 'None': {
    #     'fat': {'HU': -17.2, 'mu': 0.188, 'SNR': -1.58, 'HU_CNR_vs_muscle': 27.38, 'mu_CNR_vs_muscle': 28.10, 'Contrast_vs_muscle': 0.945},
    #     'muscle': {'HU': -314.7, 'mu': 0.130, 'SNR': -28.96, 'Contrast_vs_muscle': 0.0},
    #     'tumor': {'HU': 228.2, 'mu': 0.234, 'SNR': 21.01, 'HU_CNR_vs_muscle': 49.97, 'mu_CNR_vs_muscle': 50.38, 'Contrast_vs_muscle': 1.725},
    #     'nerve': {'HU': -419.7, 'mu': 0.110, 'SNR': -38.63, 'HU_CNR_vs_muscle': 9.66, 'mu_CNR_vs_muscle': 9.69, 'Contrast_vs_muscle': 0.334},
    #     'tumor-to-nerve HU_CNR': {'HU_CNR_vs_muscle': 59.63, 'mu_CNR_vs_muscle': 60.07},
    #     'tumor-to-nere mu_CNR': {'mu_CNR_vs_muscle': 60.07},
    #     'tumor-to-nere contrast': {'Contrast_vs_muscle': 1.544}
    # }
}

# Build flat DataFrame
rows = []
for I0, tissues in data.items():
    for tissue, metrics in tissues.items():
        row = {'I0': I0, 'Tissue': tissue}
        row.update(metrics)
        rows.append(row)
df = pd.DataFrame(rows)

# Ensure I0 ordering
df['I0'] = pd.Categorical(df['I0'],
    categories=['100k', '500k', '1M', 'None'], ordered=True)

metrics_list  = ['HU','mu','SNR','HU_CNR_vs_muscle',
                 'mu_CNR_vs_muscle','Contrast_vs_muscle']
main_tissues  = ['fat','muscle','tumor','nerve']
tn_mapping    = {
    'tumor-to-nerve HU_CNR':  'HU_CNR_vs_muscle',
    'tumor-to-nerve mu_CNR':  'mu_CNR_vs_muscle',
    'tumor-to-nerve contrast':'Contrast_vs_muscle'
}

# ——————————————————————————————————————————————
# 2) Create one TabbedPlotWindow and add a figure for each metric
# ——————————————————————————————————————————————
window = abt.TabbedPlotWindow(
    window_id='CT Metrics',  # an identifier to let you reuse/update this window
    ncols=2                  # arrange tabs in 2 columns
)

# Main-tissue plots
for metric in metrics_list:
    fig = window.add_figure_tab(metric)     # tab named like 'HU'
    ax  = fig.add_subplot()
    pivot = (df[df['Tissue'].isin(main_tissues)]
             .pivot(index='I0', columns='Tissue', values=metric))
    pivot.plot(ax=ax, marker='o')
    ax.set_title(f'{metric} vs I0 for Main Tissues')
    ax.set_xlabel('I0')
    ax.set_ylabel(metric)

# Tumor-to-nerve plots
for label, col in tn_mapping.items():
    fig = window.add_figure_tab(label)
    ax  = fig.add_subplot()
    series = (df[df['Tissue']==label]
              .set_index('I0')[col])
    series.plot(ax=ax, marker='o')
    ax.set_title(f'{label} ({col}) vs I0')
    ax.set_xlabel('I0')
    ax.set_ylabel(col)

# Tidy up layout and show once
window.apply_tight_layout()
abt.show_all_windows()