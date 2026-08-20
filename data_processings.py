import numpy as np
import pandas as pd

from astropy.table import Table

df_p = pd.read_csv("data/catalogs/preprocessed_cat/PS_p_RGZ_samples.csv")
T = Table.from_pandas(df_p)
# def extended_or_compact(total_flux, e_total_f)

for i in range(10):
    row = T[i]
    tflux = row['Total_flux']
    pflux = row['Peak_flux']
    e_tflux = row['E_Total_flux']
    e_pflux = row['E_Peak_flux']
    t1 = np.log(tflux / pflux)
    t2 = (e_tflux / tflux) ** 2 + (e_pflux / pflux) ** 2
    if t1 > 2 * np.sqrt(t2):
        print("Extended",
              f"{row['VLASS_component_name']}, DC_Maj: {row['DC_Maj']}, DC_Min: {row['DC_Min']}, DC_PA: {row['DC_PA']}")
    else:
        print("Compact",
              f"{row['VLASS_component_name']}, DC_Maj: {row['DC_Maj']}, DC_Min: {row['DC_Min']}, DC_PA: {row['DC_PA']}")
