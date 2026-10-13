Test MAE of the MLP (mean +- std over the seeds, number of seeds), in the units of the table

| Target | HDF | HDF w/o size, diameter | Morgan | Morgan + size, diameter |
|---|---|---|---|---|
| HOPV15 PCE [\%] | 1.741 +- 0.33 (10) | 1.589 +- 0.33 (10) | 1.463 +- 0.33 (10) | 1.446 +- 0.27 (10) |
| HOPV15 $V_{oc}$ [V] | 0.1098 +- 0.023 (10) | 0.1147 +- 0.018 (10) | 0.1003 +- 0.023 (10) | 0.1046 +- 0.02 (10) |
| FreeSolv $\Delta G$ [kcal/mol] | 0.7996 +- 0.12 (10) | 0.6862 +- 0.14 (10) | 1.134 +- 0.21 (10) | 1.157 +- 0.19 (10) |
| BACE IC50 [pIC50] | 0.5486 +- 0.051 (10) | 0.5791 +- 0.034 (10) | 0.5123 +- 0.031 (10) | 0.5127 +- 0.037 (10) |
| Lipophilicity logD | 0.4928 +- 0.015 (10) | 0.496 +- 0.017 (10) | 0.5646 +- 0.024 (10) | 0.5599 +- 0.025 (10) |
| AqSolDB logS [log\,M] | 0.5997 +- 0.017 (10) | 0.6861 +- 0.017 (10) | 0.7492 +- 0.027 (10) | 0.6884 +- 0.021 (10) |
| AqSolDB ClogP | 0.1804 +- 0.013 (10) | 0.4256 +- 0.018 (10) | 0.5434 +- 0.02 (10) | 0.4058 +- 0.017 (10) |
| COMPAS-3X $U_0$ [eV] | 0.8484 +- 0.12 (10) | 10.64 +- 0.93 (10) | 97.69 +- 1.5 (10) | 2.662 +- 0.21 (10) |
| COMPAS-3X Gap [mHa] | 5.82 +- 0.094 (10) | 4.448 +- 0.14 (10) | 3.262 +- 0.039 (10) | 3.167 +- 0.038 (10) |
| COMPAS-3X $\mu$ [D] | 0.04794 +- 0.0023 (10) | 0.04575 +- 0.0031 (10) | 0.03497 +- 0.00035 (10) | 0.03413 +- 0.00043 (10) |
| QM9 $\mu$ [D] | 0.5586 +- 0.0042 (10) | 0.5596 +- 0.0025 (10) | 0.5734 +- 0.0068 (10) | 0.5715 +- 0.0057 (10) |
| QM9 $C_v$ [cal/(mol\,K)] | 0.3281 +- 0.0028 (10) | 0.5307 +- 0.0095 (10) | 0.8142 +- 0.0087 (10) | 0.4364 +- 0.0072 (10) |
| QM9 ZPVE [mHa] | 0.8156 +- 0.018 (10) | 2.246 +- 0.082 (10) | 4.955 +- 0.044 (10) | 3.183 +- 0.044 (10) |
| QM9 $\alpha$ [$a_0^3$] | 0.6263 +- 0.01 (10) | 1.142 +- 0.032 (10) | 1.844 +- 0.018 (10) | 0.9631 +- 0.011 (10) |
| QM9 $\Delta H$ [Ha] | 0.5466 +- 0.036 (10) | 3.936 +- 0.15 (10) | 7.249 +- 0.19 (10) | 2.477 +- 0.099 (10) |
| QM9 $U_0$ [Ha] | 0.5224 +- 0.03 (10) | 3.905 +- 0.11 (10) | 7.281 +- 0.15 (10) | 2.44 +- 0.11 (10) |
| QM9 Gap [mHa] | 7.272 +- 0.031 (10) | 7.349 +- 0.059 (10) | 7.183 +- 0.07 (10) | 7.06 +- 0.077 (10) |

Paired change of the test MAE of the first arm relative to the second (negative = lower error); Holm over the four contrasts of a target

| Target | Contrast | n | Change (mean, 95% CI) | first arm better in | Wilcoxon p | Holm p |
|---|---|---|---|---|---|---|
| HOPV15 PCE [\%] | hdf_noglobal vs. hdf | 10 | -7.7% [-19.3, +3.9] | 6/10 | 0.232 | 0.465 |
| HOPV15 PCE [\%] | morgan_desc vs. morgan | 10 | -0.1% [-4.9, +4.7] | 5/10 | 0.846 | 0.846 |
| HOPV15 PCE [\%] | morgan_desc vs. hdf | 10 | -15.9% [-24.9, -7.0] | 9/10 | 0.004 | 0.016 |
| HOPV15 PCE [\%] | hdf_noglobal vs. morgan | 10 | +10.8% [-4.8, +26.3] | 2/10 | 0.131 | 0.393 |
| HOPV15 $V_{oc}$ [V] | hdf_noglobal vs. hdf | 10 | +8.0% [-8.9, +25.0] | 4/10 | 0.492 | 0.984 |
| HOPV15 $V_{oc}$ [V] | morgan_desc vs. morgan | 10 | +6.9% [-9.4, +23.1] | 6/10 | 0.846 | 0.984 |
| HOPV15 $V_{oc}$ [V] | morgan_desc vs. hdf | 10 | +0.9% [-28.4, +30.2] | 8/10 | 0.232 | 0.697 |
| HOPV15 $V_{oc}$ [V] | hdf_noglobal vs. morgan | 10 | +17.6% [+2.4, +32.9] | 1/10 | 0.027 | 0.109 |
| FreeSolv $\Delta G$ [kcal/mol] | hdf_noglobal vs. hdf | 10 | -13.9% [-24.7, -3.1] | 9/10 | 0.006 | 0.012 |
| FreeSolv $\Delta G$ [kcal/mol] | morgan_desc vs. morgan | 10 | +2.7% [-2.7, +8.0] | 3/10 | 0.432 | 0.432 |
| FreeSolv $\Delta G$ [kcal/mol] | morgan_desc vs. hdf | 10 | +46.5% [+27.6, +65.3] | 0/10 | 0.002 | 0.008 |
| FreeSolv $\Delta G$ [kcal/mol] | hdf_noglobal vs. morgan | 10 | -39.4% [-44.9, -33.8] | 10/10 | 0.002 | 0.008 |
| BACE IC50 [pIC50] | hdf_noglobal vs. hdf | 10 | +6.1% [+0.3, +11.9] | 2/10 | 0.049 | 0.098 |
| BACE IC50 [pIC50] | morgan_desc vs. morgan | 10 | +0.0% [-2.0, +2.1] | 5/10 | 1.000 | 1.000 |
| BACE IC50 [pIC50] | morgan_desc vs. hdf | 10 | -6.3% [-9.3, -3.3] | 10/10 | 0.002 | 0.008 |
| BACE IC50 [pIC50] | hdf_noglobal vs. morgan | 10 | +13.1% [+9.9, +16.4] | 0/10 | 0.002 | 0.008 |
| Lipophilicity logD | hdf_noglobal vs. hdf | 10 | +0.7% [-1.3, +2.6] | 4/10 | 0.557 | 0.863 |
| Lipophilicity logD | morgan_desc vs. morgan | 10 | -0.8% [-2.6, +1.0] | 5/10 | 0.432 | 0.863 |
| Lipophilicity logD | morgan_desc vs. hdf | 10 | +13.7% [+10.1, +17.2] | 0/10 | 0.002 | 0.008 |
| Lipophilicity logD | hdf_noglobal vs. morgan | 10 | -12.0% [-15.3, -8.7] | 10/10 | 0.002 | 0.008 |
| AqSolDB logS [log\,M] | hdf_noglobal vs. hdf | 10 | +14.5% [+11.2, +17.8] | 0/10 | 0.002 | 0.008 |
| AqSolDB logS [log\,M] | morgan_desc vs. morgan | 10 | -8.1% [-9.8, -6.3] | 10/10 | 0.002 | 0.008 |
| AqSolDB logS [log\,M] | morgan_desc vs. hdf | 10 | +14.8% [+12.3, +17.3] | 0/10 | 0.002 | 0.008 |
| AqSolDB logS [log\,M] | hdf_noglobal vs. morgan | 10 | -8.3% [-11.2, -5.4] | 10/10 | 0.002 | 0.008 |
| AqSolDB ClogP | hdf_noglobal vs. hdf | 10 | +136.6% [+127.1, +146.0] | 0/10 | 0.002 | 0.008 |
| AqSolDB ClogP | morgan_desc vs. morgan | 10 | -25.3% [-27.2, -23.4] | 10/10 | 0.002 | 0.008 |
| AqSolDB ClogP | morgan_desc vs. hdf | 10 | +125.6% [+116.9, +134.2] | 0/10 | 0.002 | 0.008 |
| AqSolDB ClogP | hdf_noglobal vs. morgan | 10 | -21.6% [-24.3, -18.9] | 10/10 | 0.002 | 0.008 |
| COMPAS-3X $U_0$ [eV] | hdf_noglobal vs. hdf | 10 | +1190.0% [+978.8, +1401.2] | 0/10 | 0.002 | 0.008 |
| COMPAS-3X $U_0$ [eV] | morgan_desc vs. morgan | 10 | -97.3% [-97.4, -97.1] | 10/10 | 0.002 | 0.008 |
| COMPAS-3X $U_0$ [eV] | morgan_desc vs. hdf | 10 | +219.2% [+187.0, +251.4] | 0/10 | 0.002 | 0.008 |
| COMPAS-3X $U_0$ [eV] | hdf_noglobal vs. morgan | 10 | -89.1% [-89.8, -88.4] | 10/10 | 0.002 | 0.008 |
| COMPAS-3X Gap [mHa] | hdf_noglobal vs. hdf | 10 | -23.6% [-24.9, -22.3] | 10/10 | 0.002 | 0.008 |
| COMPAS-3X Gap [mHa] | morgan_desc vs. morgan | 10 | -2.9% [-3.8, -2.0] | 10/10 | 0.002 | 0.008 |
| COMPAS-3X Gap [mHa] | morgan_desc vs. hdf | 10 | -45.6% [-46.4, -44.7] | 10/10 | 0.002 | 0.008 |
| COMPAS-3X Gap [mHa] | hdf_noglobal vs. morgan | 10 | +36.3% [+33.6, +39.1] | 0/10 | 0.002 | 0.008 |
| COMPAS-3X $\mu$ [D] | hdf_noglobal vs. hdf | 10 | -4.6% [-6.0, -3.3] | 10/10 | 0.002 | 0.008 |
| COMPAS-3X $\mu$ [D] | morgan_desc vs. morgan | 10 | -2.4% [-2.9, -1.9] | 10/10 | 0.002 | 0.008 |
| COMPAS-3X $\mu$ [D] | morgan_desc vs. hdf | 10 | -28.7% [-30.8, -26.5] | 10/10 | 0.002 | 0.008 |
| COMPAS-3X $\mu$ [D] | hdf_noglobal vs. morgan | 10 | +30.8% [+24.4, +37.2] | 0/10 | 0.002 | 0.008 |
| QM9 $\mu$ [D] | hdf_noglobal vs. hdf | 10 | +0.2% [-0.4, +0.8] | 4/10 | 0.625 | 0.750 |
| QM9 $\mu$ [D] | morgan_desc vs. morgan | 10 | -0.3% [-1.1, +0.5] | 7/10 | 0.375 | 0.750 |
| QM9 $\mu$ [D] | morgan_desc vs. hdf | 10 | +2.3% [+1.5, +3.1] | 0/10 | 0.002 | 0.008 |
| QM9 $\mu$ [D] | hdf_noglobal vs. morgan | 10 | -2.4% [-3.3, -1.5] | 9/10 | 0.004 | 0.012 |
| QM9 $C_v$ [cal/(mol\,K)] | hdf_noglobal vs. hdf | 10 | +61.8% [+59.2, +64.4] | 0/10 | 0.002 | 0.008 |
| QM9 $C_v$ [cal/(mol\,K)] | morgan_desc vs. morgan | 10 | -46.4% [-47.2, -45.6] | 10/10 | 0.002 | 0.008 |
| QM9 $C_v$ [cal/(mol\,K)] | morgan_desc vs. hdf | 10 | +33.0% [+31.0, +35.1] | 0/10 | 0.002 | 0.008 |
| QM9 $C_v$ [cal/(mol\,K)] | hdf_noglobal vs. morgan | 10 | -34.8% [-36.0, -33.7] | 10/10 | 0.002 | 0.008 |
| QM9 ZPVE [mHa] | hdf_noglobal vs. hdf | 10 | +175.5% [+166.8, +184.2] | 0/10 | 0.002 | 0.008 |
| QM9 ZPVE [mHa] | morgan_desc vs. morgan | 10 | -35.8% [-36.5, -35.0] | 10/10 | 0.002 | 0.008 |
| QM9 ZPVE [mHa] | morgan_desc vs. hdf | 10 | +290.5% [+283.0, +298.0] | 0/10 | 0.002 | 0.008 |
| QM9 ZPVE [mHa] | hdf_noglobal vs. morgan | 10 | -54.7% [-55.8, -53.5] | 10/10 | 0.002 | 0.008 |
| QM9 $\alpha$ [$a_0^3$] | hdf_noglobal vs. hdf | 10 | +82.3% [+79.3, +85.2] | 0/10 | 0.002 | 0.008 |
| QM9 $\alpha$ [$a_0^3$] | morgan_desc vs. morgan | 10 | -47.8% [-48.4, -47.2] | 10/10 | 0.002 | 0.008 |
| QM9 $\alpha$ [$a_0^3$] | morgan_desc vs. hdf | 10 | +53.8% [+51.3, +56.3] | 0/10 | 0.002 | 0.008 |
| QM9 $\alpha$ [$a_0^3$] | hdf_noglobal vs. morgan | 10 | -38.1% [-39.3, -36.9] | 10/10 | 0.002 | 0.008 |
| QM9 $\Delta H$ [Ha] | hdf_noglobal vs. hdf | 10 | +623.5% [+578.9, +668.1] | 0/10 | 0.002 | 0.008 |
| QM9 $\Delta H$ [Ha] | morgan_desc vs. morgan | 10 | -65.8% [-67.1, -64.6] | 10/10 | 0.002 | 0.008 |
| QM9 $\Delta H$ [Ha] | morgan_desc vs. hdf | 10 | +355.3% [+326.0, +384.7] | 0/10 | 0.002 | 0.008 |
| QM9 $\Delta H$ [Ha] | hdf_noglobal vs. morgan | 10 | -45.7% [-47.0, -44.4] | 10/10 | 0.002 | 0.008 |
| QM9 $U_0$ [Ha] | hdf_noglobal vs. hdf | 10 | +650.0% [+613.0, +687.1] | 0/10 | 0.002 | 0.008 |
| QM9 $U_0$ [Ha] | morgan_desc vs. morgan | 10 | -66.5% [-67.6, -65.4] | 10/10 | 0.002 | 0.008 |
| QM9 $U_0$ [Ha] | morgan_desc vs. hdf | 10 | +368.4% [+344.7, +392.0] | 0/10 | 0.002 | 0.008 |
| QM9 $U_0$ [Ha] | hdf_noglobal vs. morgan | 10 | -46.3% [-47.9, -44.7] | 10/10 | 0.002 | 0.008 |
| QM9 Gap [mHa] | hdf_noglobal vs. hdf | 10 | +1.1% [+0.5, +1.6] | 1/10 | 0.006 | 0.008 |
| QM9 Gap [mHa] | morgan_desc vs. morgan | 10 | -1.7% [-2.3, -1.1] | 10/10 | 0.002 | 0.008 |
| QM9 Gap [mHa] | morgan_desc vs. hdf | 10 | -2.9% [-3.6, -2.2] | 10/10 | 0.002 | 0.008 |
| QM9 Gap [mHa] | hdf_noglobal vs. morgan | 10 | +2.3% [+1.7, +2.9] | 0/10 | 0.002 | 0.008 |
