import joblib

import CONFIG
import numpy as np
import pandas as pd
import statsmodels.api as sm

dta =sm.datasets.fair.load_pandas().data

dta['affairs']=dta['affairs'].map(lambda x:1 if x >0 else 0)
dta.to_csv(CONFIG.DATA,index=False)

print(dta.columns)
