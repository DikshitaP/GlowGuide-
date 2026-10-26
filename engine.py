"""Recommendation engine for the skincare app.

A direct port of Section 6 of the notebook (Skincare_Recommender_v2), reading the
`data/` folder (made by build_data.py) instead of the notebook's global variables.
Scoring logic is unchanged, so the app returns the same results as the notebook.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
from scipy.stats import rankdata


CONCERNS = ["acne", "dark_spots", "dryness", "redness", "aging", "oiliness", "dullness"]
CONCERN_LABEL = {"acne": "Acne & breakouts", "dark_spots": "Dark spots / uneven tone", "dryness": "Dryness / dehydration",
                 "redness": "Redness & sensitivity", "aging": "Fine lines & wrinkles", "oiliness": "Excess oil / large pores",
                 "dullness": "Dullness & rough texture"}
SKIN_TYPES = ["dry", "oily", "combination", "normal", "sensitive"]
CAT_FIT = {   # domain prior from notebook Section 6
 "acne": {"cleanser": 1, "face wash": 1, "toner": .9, "serum": 1, "mask": .8, "moisturizer": .6, "eye treatment": .1, "spray": .4},
 "dark_spots": {"serum": 1, "moisturizer": .7, "mask": .6, "toner": .6, "cleanser": .3, "face wash": .3, "eye treatment": .5, "spray": .2},
 "dryness": {"moisturizer": 1, "serum": .9, "mask": .8, "spray": .6, "toner": .6, "cleanser": .4, "face wash": .3, "eye treatment": .5},
 "redness": {"moisturizer": 1, "serum": .9, "mask": .8, "cleanser": .6, "toner": .7, "face wash": .5, "spray": .6, "eye treatment": .3},
 "aging": {"serum": 1, "eye treatment": .9, "moisturizer": .9, "mask": .6, "toner": .4, "cleanser": .2, "face wash": .2, "spray": .3},
 "oiliness": {"cleanser": 1, "face wash": 1, "toner": 1, "serum": .8, "mask": .8, "moisturizer": .6, "spray": .5, "eye treatment": .1},
 "dullness": {"toner": 1, "serum": 1, "mask": .9, "cleanser": .6, "face wash": .6, "moisturizer": .5, "eye treatment": .2, "spray": .3}}
ROUTINE_STEPS = [("Cleanse", ["cleanser", "face wash"]), ("Tone", ["toner"]), ("Treat (serum)", ["serum"]),
                 ("Moisturise", ["moisturizer"]), ("Eye care (optional)", ["eye treatment"]), ("Weekly mask (optional)", ["mask"])]

DEFAULT_W = dict(rel=0.55, skin=0.15, quality=0.15, value=0.05, cat=0.10)
NAME_ONLY_DISCOUNT = 0.85


@dataclass
class UserProfile:
    skin_type: str = "combination"
    concerns: list = field(default_factory=lambda: ["acne"])   # priority order
    budget: float = 3000.0                                     # max price per product (INR)
    sensitive: bool = False
    pregnant: bool = False
    vegan: bool = False
    fragrance_free: bool = False
    avoid_ingredients: list = field(default_factory=list)
    product_types: list | None = None

    @property
    def is_sensitive(self) -> bool:
        return self.skin_type == "sensitive" or self.sensitive


class Recommender:
    def __init__(self, data_dir: str = "data"):
        kb = pd.read_csv(f"{data_dir}/kb.csv")
        kb["skin_types_set"] = kb["skin_types_set"].fillna("").apply(lambda s: {x for x in s.split(";") if x})
        for c in ["pregnancy_safe", "vegan"]:
            kb[c] = kb[c].fillna("safe" if c == "pregnancy_safe" else "yes")
        kb["is_active"] = kb["is_active"].astype(bool)
        self.kb = kb
        prod = pd.read_csv(f"{data_dir}/products.csv")
        prod["declared_types"] = prod["declared_types"].apply(lambda s: set(json.loads(s)) if isinstance(s, str) and s != "null" else None)
        prod["has_inci"] = prod["has_inci"].astype(bool)
        self.prod = prod
        z = np.load(f"{data_dir}/arrays.npz")
        self.P, self.E, self.EVIDENCE_W = z["P"], z["E"], z["EVIDENCE_W"]
        self.ML_PCT, self.ST_PROB, self.ITEM_SIM, self.AVOID = z["ML_PCT"], z["ST_PROB"], z["ITEM_SIM"], z["AVOID"]
        self.ML_WEIGHT = float(z["ML_WEIGHT"])
        self.CONCERNS, self.CONCERN_LABEL, self.SKIN_TYPES = CONCERNS, CONCERN_LABEL, SKIN_TYPES
        self.CAT_FIT, self.ROUTINE_STEPS = CAT_FIT, ROUTINE_STEPS
        self.SUBCATS = sorted(prod["subcategory"].dropna().unique())

        self.N_KB = len(kb)
        self.NAMES = kb["common_name"].values
        self.IRR, self.COMED = kb["irritancy_lvl"].values, kb["comedogenic"].values
        self.PREG, self.VEG = kb["pregnancy_safe"].values, kb["vegan"].values
        self.I_FRAG = int(np.where(self.NAMES == "Fragrance")[0][0])
        self.I_ALC = int(np.where(self.NAMES == "Denatured Alcohol")[0][0])
        self.ACTIVE = kb["is_active"].values.astype(float)
        self.SUB = prod["subcategory"].fillna("").str.lower().values
        self.BRAND = prod["brand"].str.lower().values
        self.PRICE = prod["price"].values
        self.HAS_INCI = prod["has_inci"].values
        self.QUALITY = prod["quality"].values
        # per-concern category-fit vectors, computed once
        self._catfit_vec = {c: np.array([self.CAT_FIT[c].get(s, 0.5) for s in self.SUB]) for c in self.CONCERNS}

    # ---------- helpers for the UI ----------
    @property
    def active_ingredients(self) -> list[str]:
        return sorted(self.kb.loc[self.kb.is_active, "common_name"])

    # ---------- Step A: profile -> ingredient preference vector ----------
    def concern_weights(self, p: UserProfile) -> dict:
        w = np.array([1 / (1 + 0.5 * i) for i in range(len(p.concerns))])
        w /= w.sum()
        return dict(zip(p.concerns, w))

    def _blocked(self, p: UserProfile) -> set:
        bad = {a.lower() for a in p.avoid_ingredients}
        return {j for j, n in enumerate(self.NAMES) if n.lower() in bad}

    def ingredient_vector(self, p: UserProfile) -> np.ndarray:
        cw = self.concern_weights(p)
        suit_types = {p.skin_type, "all"} | ({"acne-prone"} if "acne" in p.concerns else set()) \
            | ({"mature"} if "aging" in p.concerns else set())
        suit = np.array([1.0 if (s & suit_types) else 0.6 for s in self.kb["skin_types_set"]])
        safe = np.ones(self.N_KB)
        if p.is_sensitive:
            safe *= np.where(self.IRR == 2, 0.3, np.where(self.IRR == 1, 0.7, 1.0))
        if p.skin_type in ("oily", "combination") or {"acne", "oiliness"} & set(p.concerns):
            safe *= np.where(self.COMED >= 3, 0.2, np.where(self.COMED == 2, 0.7, 1.0))
        if p.pregnant:
            safe *= np.where(self.PREG == "avoid", 0.0, np.where(self.PREG == "caution", 0.5, 1.0))
        if p.vegan:
            safe *= np.where(self.VEG == "no", 0.0, np.where(self.VEG == "varies", 0.7, 1.0))
        for j in self._blocked(p):
            safe[j] = 0.0
        eff = sum(cw[c] * self.E[:, self.CONCERNS.index(c)] for c in p.concerns)
        return eff * self.EVIDENCE_W * suit * safe * self.kb["is_active"].values

    def recommend_ingredients(self, p: UserProfile, top_k: int = 8) -> pd.DataFrame:
        u, kb = self.ingredient_vector(p), self.kb
        rows = []
        for j in np.argsort(-u)[:top_k]:
            if u[j] <= 0:
                continue
            helps = [self.CONCERN_LABEL[c] for c in p.concerns if self.E[j, self.CONCERNS.index(c)] >= 0.3]
            rows.append(dict(
                ingredient=self.NAMES[j], helps_with=", ".join(helps), match_score=round(float(u[j]), 2),
                evidence=kb.evidence_level[j], best_time=kb.time_of_use[j],
                pairs_well_with=kb.pairs_well_with[j] if pd.notna(kb.pairs_well_with[j]) else "",
                dont_mix_with=kb.avoid_with[j] if pd.notna(kb.avoid_with[j]) else "",
                watch_out=kb.limitations[j] if pd.notna(kb.limitations[j]) else ""))
        return pd.DataFrame(rows)

    # ---------- Step B: product scoring ----------
    def _exclusion_mask(self, p: UserProfile) -> np.ndarray:
        ok = np.isfinite(self.PRICE) & (self.PRICE > 0) & (self.PRICE <= p.budget)
        if p.product_types:
            ok &= np.isin(self.SUB, [t.lower() for t in p.product_types])
        present = self.P > 0
        if p.pregnant or p.vegan or p.fragrance_free or p.avoid_ingredients:
            ok &= self.HAS_INCI                      # safety can't be verified without a full list
        if p.pregnant:
            ok &= ~(present & (self.PREG == "avoid")).any(1)
        if p.vegan:
            ok &= ~(present & (self.VEG == "no")).any(1)
        if p.fragrance_free:
            ok &= ~present[:, self.I_FRAG]
        for j in self._blocked(p):
            ok &= ~present[:, j]
        return ok

    def score_products(self, p: UserProfile, weights: dict | None = None):
        W = {**DEFAULT_W, **(weights or {})}
        cw, u, present, P = self.concern_weights(p), self.ingredient_vector(p), self.P > 0, self.P
        dot = P @ u
        scale = np.median(dot[dot > 0]) if (dot > 0).any() else 1.0
        ing_score = 1 - np.exp(-dot / scale)
        ml_rel = sum(cw[c] * self.ML_PCT[:, self.CONCERNS.index(c)] for c in p.concerns)
        content = np.where(self.HAS_INCI, (1 - self.ML_WEIGHT) * ing_score + self.ML_WEIGHT * ml_rel,
                           NAME_ONLY_DISCOUNT * ing_score)
        k = self.SKIN_TYPES.index(p.skin_type)
        declared = np.array([(1.0 if p.skin_type in d else 0.25) if d is not None else np.nan
                             for d in self.prod["declared_types"]])
        predicted = 0.35 + 0.35 * (rankdata(self.ST_PROB[:, k]) / len(self.prod))
        base = np.where(np.isnan(declared), predicted, declared)
        suit_types = {p.skin_type, "all"}
        suit = np.array([1.0 if (s & suit_types) else 0.6 for s in self.kb["skin_types_set"]])
        act = P * self.ACTIVE[None, :]
        mass = act.sum(1)
        ing_fit = np.where(mass > 0, (act @ suit) / np.where(mass > 0, mass, 1), 0.8)
        skin_fit = 0.65 * base + 0.35 * ing_fit
        cat_fit = sum(cw[c] * self._catfit_vec[c] for c in p.concerns)
        value = np.nan_to_num((1 - self.PRICE / p.budget).clip(0, 1))
        pen = np.zeros(len(self.prod))
        if p.is_sensitive:
            pen += 0.12 * (present & (self.IRR == 2)).any(1) + 0.12 * present[:, self.I_FRAG] \
                + 0.08 * present[:, self.I_ALC]
        if p.skin_type in ("oily", "combination") or {"acne", "oiliness"} & set(p.concerns):
            pen += 0.12 * (present & (self.COMED >= 3)).any(1)
        if p.pregnant:
            pen += 0.08 * (present & (self.PREG == "caution")).any(1)
        score = (W["rel"] * content + W["skin"] * skin_fit + W["quality"] * self.QUALITY
                 + W["value"] * value + W["cat"] * cat_fit - pen)
        parts = pd.DataFrame(dict(content=content, skin_fit=skin_fit, quality=self.QUALITY,
                                  cat_fit=cat_fit, value=value, penalty=pen))
        return score, parts, u

    def _explain(self, i: int, u: np.ndarray, p: UserProfile) -> list[str]:
        contrib = self.P[i] * u
        out = []
        for j in [j for j in np.argsort(-contrib)[:4] if contrib[j] > 0]:
            helps = [c.replace("_", " ") for c in p.concerns if self.E[j, self.CONCERNS.index(c)] >= 0.3]
            out.append(f"{self.NAMES[j]}" + (f" ({', '.join(helps)})" if helps else ""))
        return out

    # ---------- Step C: diversity re-ranking ----------
    def recommend_products(self, p: UserProfile, top_n: int = 10, diversify: bool = True,
                           max_per_brand: int = 2, pool: int = 60) -> pd.DataFrame:
        score, parts, u = self.score_products(p)
        cand = np.where(self._exclusion_mask(p) & (self.P.sum(1) > 0))[0]
        if len(cand) == 0:
            return pd.DataFrame()
        order = cand[np.argsort(-score[cand])]
        if not diversify:
            chosen = list(order[:top_n])
        else:
            poolix = list(order[:pool])
            s = score[poolix]
            s = (s - s.min()) / (np.ptp(s) + 1e-9)
            sn, chosen, brand_n = dict(zip(poolix, s)), [], {}
            while poolix and len(chosen) < top_n:
                best, best_v = None, -9
                for i in poolix:
                    if brand_n.get(self.BRAND[i], 0) >= max_per_brand:
                        continue
                    red = max((self.ITEM_SIM[i, j] for j in chosen), default=0.0)
                    v = 0.8 * sn[i] - 0.2 * red
                    if v > best_v:
                        best, best_v = i, v
                if best is None:
                    break
                chosen.append(best)
                poolix.remove(best)
                brand_n[self.BRAND[best]] = brand_n.get(self.BRAND[best], 0) + 1
        out = self.prod.loc[chosen, ["product_name", "brand", "subcategory", "price", "rating",
                                     "noofratings", "title_href"]].copy()
        out["score"] = score[chosen].round(3)
        out["key_actives"] = [self._explain(i, u, p) for i in chosen]
        out["ingredient_source"] = np.where(self.HAS_INCI[chosen], "full ingredient list", "product name only")
        out["skin_fit"] = parts.loc[chosen, "skin_fit"].round(2).values
        W = DEFAULT_W
        for col, key, w in [("c_ingredients", "content", W["rel"]), ("c_skin", "skin_fit", W["skin"]),
                            ("c_quality", "quality", W["quality"]), ("c_category", "cat_fit", W["cat"]),
                            ("c_value", "value", W["value"])]:
            out[col] = (w * parts.loc[chosen, key].values).round(4)
        out["c_penalty"] = (-parts.loc[chosen, "penalty"].values).round(4)
        out.insert(0, "idx", chosen)
        out.insert(0, "rank", range(1, len(out) + 1))
        return out.reset_index(drop=True)

    # ---------- item-to-item ("find a dupe") ----------
    def similar_products(self, idx: int, k: int = 5, cheaper: bool = False) -> pd.DataFrame:
        sims = self.ITEM_SIM[idx].astype(float).copy()
        sims[idx] = -1
        if cheaper:
            sims[~(self.PRICE < self.PRICE[idx])] = -1
        top = [i for i in np.argsort(-sims)[:k] if sims[i] > -1]
        out = self.prod.loc[top, ["product_name", "brand", "subcategory", "price", "rating", "title_href"]].copy()
        out["similarity"] = sims[top].round(2)
        return out.reset_index(drop=True)

    # ---------- routine + conflict check ----------
    def build_routine(self, p: UserProfile, steps=("Cleanse", "Treat (serum)", "Moisturise")):
        rows, picked = [], []
        for name, subs in self.ROUTINE_STEPS:
            if name not in steps:
                continue
            r = self.recommend_products(replace(p, product_types=subs), top_n=1, diversify=False)
            if len(r):
                picked.append((name, int(r.idx[0])))
                rows.append(dict(step=name, **r.iloc[0][["product_name", "brand", "subcategory", "price", "rating", "key_actives", "title_href",
                                                         "c_ingredients", "c_skin", "c_quality", "c_category", "c_value", "c_penalty"]].to_dict()))
        warns = []
        for a in range(len(picked)):
            for b in range(a + 1, len(picked)):
                if picked[a][0] == "Cleanse" or picked[b][0] == "Cleanse":
                    continue                                   # rinse-off: short contact time
                ia = np.where((self.P[picked[a][1]] > 0) & (self.ACTIVE > 0))[0]
                ib = np.where((self.P[picked[b][1]] > 0) & (self.ACTIVE > 0))[0]
                for x in ia:
                    for y in ib:
                        if x != y and self.AVOID[x, y]:
                            warns.append(f"{self.NAMES[x]} ({picked[a][0]}) + {self.NAMES[y]} ({picked[b][0]}): "
                                         "best not layered together. Use on alternate nights, or AM vs PM.")
        return pd.DataFrame(rows), sorted(set(warns))
