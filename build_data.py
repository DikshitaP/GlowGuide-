"""Convert the notebook's `skincare_recsys_artifacts.joblib` into the portable `data/` folder the app reads.

Run once (and again whenever you retrain):
    python build_data.py skincare_recsys_artifacts.joblib

Output is plain CSV + NumPy, so it does not depend on the pandas / scikit-learn versions used to train.
Needs scikit-learn only here (to rebuild the ingredient-similarity matrix); the app itself does not.
"""
import json, os, re, sys, __main__

import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics.pairwise import cosine_similarity


# The notebook's TF-IDF vectoriser was saved with these two helpers, so they must exist before loading.
def norm(t): return re.sub(r"\s+", " ", str(t).lower().replace("\u00a0", " "))


def tokenize_inci(text):
    out = []
    for x in re.split(r"[,;]|\.\s", norm(text)):
        x = re.sub(r"\([^)]*\)", "", x); x = re.sub(r"[\d.]+\s*%", "", x); x = re.sub(r"[^a-z0-9\- /]", "", x).strip()
        if len(x) > 2:
            out.append(x)
    return out


__main__.norm, __main__.tokenize_inci = norm, tokenize_inci

src = sys.argv[1] if len(sys.argv) > 1 else "skincare_recsys_artifacts.joblib"
out_dir = sys.argv[2] if len(sys.argv) > 2 else "data"
a = joblib.load(src)
kb, prod = a["kb"].reset_index(drop=True), a["prod"].reset_index(drop=True)
os.makedirs(out_dir, exist_ok=True)

# ingredient knowledge base
kb_out = kb[["common_name", "irritancy_lvl", "comedogenic", "pregnancy_safe", "vegan", "is_active", "evidence_level",
             "time_of_use", "pairs_well_with", "avoid_with", "limitations"]].copy()
kb_out["skin_types_set"] = kb["skin_types_set"].apply(lambda s: ";".join(sorted(s)))
kb_out.to_csv(f"{out_dir}/kb.csv", index=False)

# products
p_out = prod[["product_name", "brand", "subcategory", "price", "rating", "noofratings", "has_inci", "quality", "title_href"]].copy()
p_out["declared_types"] = prod["declared_types"].apply(lambda d: "null" if d is None else json.dumps(sorted(d)))
p_out.to_csv(f"{out_dir}/products.csv", index=False)

# arrays (ML_PCT and the item-item similarity are rebuilt exactly as in the notebook)
ML_PCT = np.column_stack([rankdata(a["ML_PROB"][:, k]) / len(prod) for k in range(a["ML_PROB"].shape[1])])
tf = a["tfidf"].transform(prod["ingredients"].where(prod["has_inci"], ""))
mask = prod["has_inci"].values[:, None]
X_tfidf = tf.multiply(mask).tocsr()                       # rows without an ingredient list stay all-zero
ITEM_SIM = cosine_similarity(X_tfidf).astype(np.float32)
np.savez_compressed(f"{out_dir}/arrays.npz", P=a["P"], E=a["E"], EVIDENCE_W=a["evidence_w"], ML_PCT=ML_PCT,
                    ST_PROB=a["ST_PROB"], ITEM_SIM=ITEM_SIM, AVOID=a["avoid_matrix"], ML_WEIGHT=np.array(a["ML_WEIGHT"]))
print(f"Wrote {out_dir}/ (kb.csv, products.csv, arrays.npz) for {len(prod)} products, {len(kb)} ingredients")
