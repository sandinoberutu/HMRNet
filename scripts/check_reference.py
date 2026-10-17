"""Validate archived results. This does not rerun model training."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

root=Path(__file__).resolve().parents[1]
r=root/'results/reference'
runs=pd.read_excel(r/'experiment_08v2_final_10seed_runs.xlsx')
pairs=pd.read_csv(r/'experiment_08v2_final_10seed_pairwise.csv')
config=json.loads((r/'experiment_08v2_best_config.json').read_text())
assert len(runs)==10 and sorted(runs['Seed'].tolist())==list(range(1,11))
assert len(pairs)==45
expected=config['final_10seed']
checks={
    'Mean_C_v':float(runs['C_v'].mean()),
    'Std_C_v':float(runs['C_v'].std(ddof=1)),
    'Mean_Topic_Diversity':float(runs['Topic_Diversity'].mean()),
    'Mean_Topic_Redundancy':float(runs['Topic_Redundancy'].mean()),
    'Mean_TS':float(pairs['Topic_Stability'].mean()),
}
for key,value in checks.items():
    assert np.isclose(value,expected[key],atol=1e-10,rtol=0),(key,value,expected[key])
assert round(checks['Mean_C_v'],4)==0.7095
assert round(checks['Std_C_v'],4)==0.0257
print(json.dumps({'status':'PASS','scope':'Archived reference metrics only','n_seeds':10,'n_pairs':45,'metrics':checks},indent=2))
