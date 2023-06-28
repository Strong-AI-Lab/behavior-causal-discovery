
import pandas as pd
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')

from format_data import PandasFormatter

from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
from tigramite import plotting as tp
from tigramite import data_processing as pp



# Read data
data = pd.read_csv('cam2_2022_10_20_GH200023.csv')
print(data)


# Format data
formatter = PandasFormatter(data)
sequences = formatter.format()
# print(sequences)


# Filter unuasable data
# sequences = sequences[6:8]
# print(sequences[0])
sequences = [seq for seq in sequences if len(seq) > 9]


# Run causal discovery algorithm
tau_max = 5
realisations = len(sequences)
alpha_level = 0.05
variables = formatter.get_formatted_columns()
N = len(variables) # behaviours of individual, close neighbour, distant neighbour, zones of individual, close neighbour, distant neighbour

p_matrices = {'PCMCI':np.ones((realisations, N, N, tau_max+1))}
val_matrices = {'PCMCI':np.zeros((realisations, N, N, tau_max+1))}  

for i in tqdm.tqdm(range(realisations)):
    data = pp.DataFrame(sequences[i], var_names=variables)
    pcmci = PCMCI(dataframe=data, cond_ind_test=ParCorr())
    results = pcmci.run_pcmci(tau_max=tau_max, pc_alpha=0.2)
    p_matrices['PCMCI'][i] = results['p_matrix']
    val_matrices['PCMCI'][i] = results['val_matrix']

sig_links = {'PCMCI':(p_matrices['PCMCI'] <= alpha_level).mean(axis=0)}
ave_val_matrices = {'PCMCI':val_matrices['PCMCI'].mean(axis=0)}


# Visualise results
min_sig = 0.2
vminmax = 0.4
graph = (sig_links['PCMCI'] > min_sig)
tp.plot_graph(val_matrix=ave_val_matrices['PCMCI'],
              graph=graph, var_names=variables,
              link_width=sig_links['PCMCI'],
              vmin_edges=-vminmax,
              vmax_edges=vminmax)

plt.show()