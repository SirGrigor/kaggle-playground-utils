import sys
sys.path.insert(0, "/home/ilgrig/IdeaProjects/kaggle/playground-s6e4/repro")
import pandas as pd
import ablation_s6e4 as A
from sklearn.model_selection import KFold

N = int(sys.argv[1]) if len(sys.argv) > 1 else 500
cfg = A.VARIANTS["V0_full"]
data = A.prepare_data(cfg, sample_n=None, nrows=N)
X, y = data["X"], data["y"]
FEATURES, CATS = data["FEATURES"], data["CATS"]
kf = KFold(n_splits=A.N_FOLDS, shuffle=True, random_state=A.KF_SEED)
tr, va = next(iter(kf.split(X)))
X_train, y_train, X_val = X.iloc[tr], y.iloc[tr], X.iloc[va]

te = A.OrderedTE()
fd = pd.concat((X_train, y_train), axis=1); fd["weight"] = 1.0
bagged = [te.fit(fd.sample(frac=1, random_state=A.KF_SEED + i),
                 category_cols=FEATURES, target_col=A.TARGET_COL)
          for i in range(cfg["te_bagging"])]
Xtr = pd.concat(bagged).drop([A.TARGET_COL, "weight"], axis=1).drop(CATS, axis=1)
Xva = te.transform(X_val.copy()).drop(CATS, axis=1)

Xva = Xva.loc[:, ~Xva.columns.duplicated()]
nan_cols = Xva.columns[Xva.isna().any()].tolist()
out = [f"N={N}",
       f"X_val n_nan_cols={len(nan_cols)}",
       f"nan_cols(sample)={nan_cols[:8]}"]
# are the NaN columns the raw base feature columns (not TE)? show a couple
for c in nan_cols[:4]:
    out.append(f"  {c}: dtype={Xva[c].dtype} n_nan={int(Xva[c].isna().sum())}")
with open("/tmp/_nan_probe_out.txt", "w") as f:
    f.write("\n".join(out) + "\n")
print("written")
