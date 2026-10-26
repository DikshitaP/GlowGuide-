"""Glow Guide: Streamlit frontend for the skincare recommender.

Run:  streamlit run app.py
Needs the data/ folder (created by build_data.py from the notebook's joblib file).
Sign-up / login / logout handled by auth.py (users stored in users.db).
"""
import html
import os
import time

import altair as alt
import pandas as pd
import streamlit as st

import auth
from engine import DEFAULT_W, Recommender, UserProfile

st.set_page_config(page_title="Glow Guide · Skincare picks", page_icon="🌸", layout="wide")
DATA_DIR = os.environ.get("DATA_DIR", "data")


@st.cache_resource(show_spinner="Loading the model…")
def load_engine():
    return Recommender(DATA_DIR)


if not os.path.exists(f"{DATA_DIR}/arrays.npz"):
    st.error(f"Couldn't find the `{DATA_DIR}/` folder. Run `python build_data.py skincare_recsys_artifacts.joblib` first.")
    st.stop()

rec = load_engine()

# ------------------------------------------------------------------ styling
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=DM+Sans:wght@400;500;600;700&display=swap');
html, body, .stApp, [class*="css"] {font-family: 'DM Sans', system-ui, sans-serif;}
.stApp {background: linear-gradient(180deg, #FFF4F7 0%, #FFF9FB 40%, #FBF6FF 100%);}
.block-container {padding-top: 1.4rem; max-width: 1140px;}
header[data-testid="stHeader"] {background: transparent;}
#MainMenu, footer {visibility: hidden;}

/* hero */
.hero {background: linear-gradient(120deg, #FFD3E1 0%, #F9B8CE 40%, #E3C6F5 100%); color:#4A2236;
       border-radius: 26px; padding: 36px 40px; margin-bottom: 22px; position: relative; overflow: hidden;
       box-shadow: 0 12px 34px rgba(217,82,127,.18);}
.hero .blob {position:absolute; border-radius:50%; filter: blur(2px);}
.hero .b1 {width:240px; height:240px; right:-50px; top:-70px; background: radial-gradient(circle, rgba(255,255,255,.75), rgba(255,255,255,0) 70%);}
.hero .b2 {width:180px; height:180px; right:190px; bottom:-80px; background: radial-gradient(circle, rgba(255,226,209,.9), rgba(255,255,255,0) 70%);}
.hero .b3 {width:120px; height:120px; left:46%; top:-40px; background: radial-gradient(circle, rgba(255,255,255,.5), rgba(255,255,255,0) 70%);}
.hero h1 {font-family:'Playfair Display', serif; font-size: 2.9rem; line-height:1.05; margin:0 0 8px 0; color:#4A2236; font-weight:700; position:relative;}
.hero p {margin:0; font-size:1.05rem; color:#6B3A52; max-width: 600px; position:relative;}
.hero .stats {margin-top:18px; display:flex; gap:10px; flex-wrap:wrap; position:relative; z-index:1;}
.hero .stat {background: rgba(255,255,255,.62); border:1px solid rgba(255,255,255,.9); padding:6px 14px;
             border-radius:999px; font-size:.86rem; font-weight:500; color:#6B2D4A;}

/* sidebar */
section[data-testid="stSidebar"] {background:#FFFFFF; border-right:1px solid #F6DCE5;}
section[data-testid="stSidebar"] h2 {font-family:'Playfair Display', serif; font-size:1.6rem; color:#4A2236;}
div[data-testid="stForm"] {border:none; padding:0;}
.stFormSubmitButton > button {background: linear-gradient(135deg,#E8789A,#D9527F); color:#fff; border:none; border-radius:14px;
  padding:.7rem 1rem; font-weight:700; box-shadow:0 8px 18px rgba(217,82,127,.32);}
.stFormSubmitButton > button:hover {filter: brightness(1.05); transform: translateY(-1px); color:#fff;}

/* tabs */
.stTabs [data-baseweb="tab-list"] {gap: 6px; border-bottom: 1px solid #F3D8E2;}
.stTabs [data-baseweb="tab"] {border-radius: 12px 12px 0 0; padding: 10px 18px; font-weight:600; color:#8B6275;}
.stTabs [aria-selected="true"] {color:#C23A6C !important; background:#FFE6EE;}
.stTabs [data-baseweb="tab-highlight"] {background-color:#D9527F;}

/* chips */
.chips {margin: 4px 0 18px 0;}
.chip {display:inline-block; background:#fff; border:1px solid #F3CFDC; color:#B03562; padding:5px 14px;
       border-radius:999px; font-size:.85rem; font-weight:500; margin:0 8px 8px 0;}
.chip.accent {background:#EFE6FD; border-color:#DACBF7; color:#6B4BB8;}

/* cards */
.card {background:#FFFCFD; border:1px solid #EBDDE3; border-radius:14px; padding:20px 22px; margin-bottom:14px;
       display:flex; gap:24px; align-items:flex-start;}
.body {flex:1; min-width:0;}
.title {font-weight:600; font-size:1.05rem; color:#3A2433; margin-bottom:10px;}
.sub {color:#8B7480; font-size:.92rem; margin-bottom:14px;}
.pill {display:inline-block; padding:5px 14px; margin:0 8px 6px 0; border-radius:999px; background:#FBE9EF; color:#9A3B5E; font-size:.84rem;}
.pill.warn {background:#FFF1D6; color:#8A5A00;}
.pill.red {background:#FDE4E4; color:#9B2C2C;}
.pill.green {background:#DDF5E6; color:#1B6B3A;}
.pill.amber {background:#FFF1D6; color:#8A5A00;}
.pill.grey {background:#F1ECEF; color:#6E5A65;}
.side {flex:0 0 250px;}
.price {font-size:1.3rem; font-weight:700; color:#3A2433; margin-bottom:12px;}
.btn {display:block; text-align:center; padding:13px 18px; border-radius:12px; background:#fff; border:1px solid #E7D8DF;
      color:#3A2433 !important; font-size:.95rem; text-decoration:none !important;}
.btn:hover {border-color:#D9527F; color:#C23A6C !important;}
.kv {font-size:.86rem; color:#7A5A6B; margin-top:2px;}
.kv b {color:#3A2433;}

.userbox {background:#FFF0F5; border:1px solid #F3CFDC; border-radius:12px; padding:9px 14px; margin-bottom:8px; font-size:.9rem; color:#6B2D4A;}

/* landing */
.how {display:grid; grid-template-columns: repeat(3, 1fr); gap:16px; margin-top:6px;}
.how .step {background:#fff; border:1px solid #F5DEE7; border-radius:22px; padding:24px; box-shadow:0 3px 14px rgba(217,82,127,.07);}
.how .num {width:36px; height:36px; border-radius:12px; background:linear-gradient(135deg,#FFD3E1,#E3C6F5); color:#8A2D55; font-weight:700;
           display:flex; align-items:center; justify-content:center; margin-bottom:12px; font-family:'Playfair Display', serif;}
.how h4 {margin:0 0 6px 0; font-size:1.02rem; color:#3A2433;}
.how p {margin:0; color:#8B6275; font-size:.9rem;}
.total {background:#FFE6EE; border-radius:16px; padding:12px 18px; font-weight:700; color:#B03562; display:inline-block;}
.explain {background:#fff; border:1px solid #F5DEE7; border-radius:18px; padding:16px 20px; margin-bottom:14px; color:#6E5A65; font-size:.92rem;}
.explain b {color:#3A2433;}
.notice {color:#A58A98; font-size:.8rem; margin-top:26px; padding-top:14px; border-top:1px solid #F5DEE7;}
@media (max-width: 760px) {.how {grid-template-columns: 1fr;} .card {flex-wrap: wrap;} .side {text-align:left; flex-basis:100%;}
  .hero {padding: 24px 22px;} .hero h1 {font-size: 2.1rem;}}
</style>
""",
    unsafe_allow_html=True,
)

esc = html.escape
COMPS = [("c_ingredients", "Ingredient match", "#E0568A"), ("c_skin", "Skin-type fit", "#F2A7C3"),
         ("c_quality", "Product quality", "#B79AEA"), ("c_category", "Category fit", "#FFB89A"),
         ("c_value", "Value for money", "#8FD3C1")]
WEIGHT_OF = dict(zip([c[0] for c in COMPS], [DEFAULT_W["rel"], DEFAULT_W["skin"], DEFAULT_W["quality"], DEFAULT_W["cat"], DEFAULT_W["value"]]))


def inr(x):
    return f"₹{x:,.0f}" if pd.notna(x) else "n/a"


def pills(items, cls=""):
    return "".join(f"<span class='pill {cls}'>{esc(str(i))}</span>" for i in items)


def link_btn(url):
    return (f"<a class='btn' href='{esc(url, quote=True)}' target='_blank' rel='noopener'>View product ↗</a>"
            if isinstance(url, str) and url.startswith("http") else "")


def product_card(row, match=None, show_actives=True):
    sub, rating = getattr(row, "subcategory", None), getattr(row, "rating", None)
    meta = esc(str(row.brand)) + (f" · {esc(str(sub).title())}" if isinstance(sub, str) else "")
    meta += f" · ★ {rating:.1f}" if rating is not None and pd.notna(rating) else ""
    actives = getattr(row, "key_actives", None) if show_actives else None
    flag = pills(["Matched from name only: check the full ingredient list"], "warn") \
        if getattr(row, "ingredient_source", None) == "product name only" else ""
    st.markdown(
        f"<div class='card'><div class='body'><div class='title'>{esc(str(row.product_name))}</div>"
        f"<div class='sub'>{meta}</div>{pills(actives) if actives else ''}{flag}</div>"
        f"<div class='side'><div class='price'>{inr(row.price)}</div>{link_btn(getattr(row, 'title_href', None))}</div></div>",
        unsafe_allow_html=True)


EVID_CLS = {"strong": "green", "moderate": "amber", "limited": "red"}

# ------------------------------------------------------------------ hero
def render_hero():
    st.markdown(
        f"""
    <div class="hero"><div class="blob b1"></div><div class="blob b2"></div><div class="blob b3"></div>
      <h1>Glow Guide</h1>
      <p>Skincare picks matched to your skin, your goals and your budget. Ingredient science first, products second.</p>
      <div class="stats"><span class="stat">🧴 {len(rec.prod):,} products</span>
      <span class="stat">🧪 {len(rec.kb)} ingredients profiled</span>
      <span class="stat">🏷️ {rec.prod.brand.nunique()} brands</span></div>
    </div>
    """,
        unsafe_allow_html=True)


# ------------------------------------------------------------------ login / logout gate
st.session_state.setdefault("auth_user", None)
user = st.session_state["auth_user"]

if user is None:
    st.markdown("<style>section[data-testid='stSidebar'], [data-testid='stSidebarCollapsedControl'], "
                "[data-testid='collapsedControl'] {display:none;}</style>", unsafe_allow_html=True)
    render_hero()
    _, mid, _ = st.columns([1, 1.5, 1])
    with mid:
        if st.session_state.get("signed_out"):
            st.markdown("<div class='card' style='display:block;text-align:center'><div class='title'>You've been signed out</div>"
                        "<div class='sub' style='margin-bottom:0'>Thanks for visiting Glow Guide. Your session has ended.</div></div>",
                        unsafe_allow_html=True)
            if st.button("Sign in again", type="primary", width="stretch"):
                st.session_state["signed_out"] = False
                st.rerun()
            st.stop()

        t_login, t_signup = st.tabs(["Log in", "Create account"])
        with t_login:
            with st.form("login_form"):
                lu = st.text_input("Username", key="login_user")
                lp = st.text_input("Password", type="password", key="login_pw")
                do_login = st.form_submit_button("Log in", type="primary", width="stretch")
            if do_login:
                if time.time() < st.session_state.get("lock_until", 0):
                    st.error("Too many failed attempts. Please wait a few seconds and try again.")
                else:
                    name = auth.verify_user(lu, lp) if lu and lp else None
                    if name:
                        st.session_state.update(auth_user=name, signed_out=False, login_fails=0)
                        st.rerun()
                    fails = st.session_state.get("login_fails", 0) + 1
                    st.session_state["login_fails"] = fails
                    if fails >= 5:
                        st.session_state.update(lock_until=time.time() + 30, login_fails=0)
                    st.error("Incorrect username or password.")
        with t_signup:
            with st.form("signup_form"):
                su = st.text_input("Choose a username", key="su_user")
                sp = st.text_input("Choose a password (8+ characters)", type="password", key="su_pw")
                sp2 = st.text_input("Confirm password", type="password", key="su_pw2")
                do_signup = st.form_submit_button("Create account", type="primary", width="stretch")
            if do_signup:
                if sp != sp2:
                    st.error("Passwords don't match.")
                else:
                    err = auth.create_user(su, sp)
                    if err:
                        st.error(err)
                    else:
                        st.session_state.update(auth_user=su.strip(), signed_out=False)
                        st.rerun()
    st.stop()

render_hero()

# ------------------------------------------------------------------ sidebar form
with st.sidebar:
    st.markdown(f"<div class='userbox'>👤 Signed in as <b>{esc(user)}</b></div>", unsafe_allow_html=True)
    if st.button("Log out", width="stretch"):
        st.session_state["auth_user"] = None
        st.session_state["signed_out"] = True
        st.session_state.pop("user_profile", None)
        st.rerun()
    st.header("Your skin")
    with st.form("profile"):
        skin = st.selectbox("Skin type", rec.SKIN_TYPES, index=rec.SKIN_TYPES.index("combination"), format_func=str.capitalize)
        labels = {c: rec.CONCERN_LABEL[c] for c in rec.CONCERNS}
        c1 = st.selectbox("Main concern", rec.CONCERNS, format_func=labels.get)
        c2 = st.selectbox("Second concern (optional)", [""] + rec.CONCERNS, format_func=lambda c: labels.get(c, "None"))
        c3 = st.selectbox("Third concern (optional)", [""] + rec.CONCERNS, format_func=lambda c: labels.get(c, "None"))
        budget = st.slider("Max price per product (₹)", 300, 10000, 2500, step=100)
        types = st.multiselect("Product types (empty = all)", rec.SUBCATS)
        st.markdown("**Safety & preferences**")
        sensitive = st.checkbox("Also sensitive")
        pregnant = st.checkbox("Pregnant or breast-feeding")
        vegan = st.checkbox("Vegan only")
        frag_free = st.checkbox("Fragrance-free")
        avoid = st.multiselect("Ingredients to avoid", rec.active_ingredients)
        go = st.form_submit_button("✨ Get my recommendations", type="primary", width="stretch")

if go:
    concerns = list(dict.fromkeys([c for c in (c1, c2, c3) if c]))
    st.session_state["user_profile"] = UserProfile(
        skin_type=skin, concerns=concerns, budget=float(budget), sensitive=sensitive, pregnant=pregnant,
        vegan=vegan, fragrance_free=frag_free, avoid_ingredients=avoid, product_types=types or None)

p = st.session_state.get("user_profile")
if p is None:
    st.markdown(
        """
<div class="how">
  <div class="step"><div class="num">1</div><h4>Tell us about your skin</h4>
  <p>Pick your skin type, up to three concerns in order of priority, and your budget in the sidebar.</p></div>
  <div class="step"><div class="num">2</div><h4>We match the ingredients</h4>
  <p>Your concerns are scored against a knowledge base of actives, with safety, evidence and irritation built in.</p></div>
  <div class="step"><div class="num">3</div><h4>Get products and a routine</h4>
  <p>Ranked products with a score breakdown, a ready-made routine with conflict warnings, and cheaper look-alikes.</p></div>
</div>
""",
        unsafe_allow_html=True)
    st.stop()

# ------------------------------------------------------------------ profile chips
chips = [f"{p.skin_type.capitalize()} skin" + (" + sensitive" if p.sensitive and p.skin_type != "sensitive" else "")]
chips += [rec.CONCERN_LABEL[c] for c in p.concerns]
chip_html = "".join(f"<span class='chip'>{esc(c)}</span>" for c in chips) + f"<span class='chip'>Up to {inr(p.budget)}</span>"
chip_html += "".join(f"<span class='chip accent'>{n}</span>" for n, on in
                     [("Pregnancy-safe", p.pregnant), ("Vegan", p.vegan), ("Fragrance-free", p.fragrance_free)] if on)
chip_html += "".join(f"<span class='chip accent'>No {esc(a)}</span>" for a in p.avoid_ingredients)
st.markdown(f"<div class='chips'>{chip_html}</div>", unsafe_allow_html=True)

tab_prod, tab_why, tab_ing, tab_routine, tab_sim = st.tabs(
    ["🧴 Products", "📊 Why these?", "🧪 Ingredients", "🗓️ Routine", "🔍 Find similar"])

# ------------------------------------------------------------------ products
recs = pd.DataFrame()
with tab_prod:
    n = st.slider("How many products", 3, 15, 8, key="n_products")
    recs = rec.recommend_products(p, n)
    if recs.empty:
        st.warning("No products match all your filters. Try a higher budget or fewer safety options.")
    else:
        for r in recs.itertuples():
            product_card(r)
        with st.expander("Scores table (for the report)"):
            st.dataframe(recs[["rank", "product_name", "brand", "price", "score", "skin_fit"]],
                         hide_index=True, width="stretch")

# ------------------------------------------------------------------ score breakdown charts
with tab_why:
    if recs.empty:
        st.info("Get some recommendations first.")
    else:
        w = DEFAULT_W
        st.markdown(
            "<div class='explain'>Every product gets a score out of 1, built from five parts. "
            f"<b>Ingredient match</b> ({w['rel']:.0%}): how well its actives fit your concerns. "
            f"<b>Skin-type fit</b> ({w['skin']:.0%}). <b>Product quality</b> ({w['quality']:.0%}): ratings, adjusted for how many people rated. "
            f"<b>Category fit</b> ({w['cat']:.0%}): e.g. a cleanser vs a serum for acne. <b>Value</b> ({w['value']:.0%}): how far under your budget it is. "
            "Irritation, pore-clogging or pregnancy-caution ingredients subtract points.</div>", unsafe_allow_html=True)

        comps = COMPS + ([("c_penalty", "Penalty", "#9C8A94")] if (recs.c_penalty < 0).any() else [])
        recs = recs.assign(label=recs["rank"].astype(str) + ". " + recs.product_name.str.slice(0, 34))
        order = recs.label.tolist()
        long = recs.melt(id_vars=["label", "rank"], value_vars=[c[0] for c in comps], var_name="key", value_name="points")
        long["Part"] = long.key.map({k: n_ for k, n_, _ in comps})
        st.markdown("##### What makes up each product's score")
        chart = (alt.Chart(long).mark_bar(cornerRadiusEnd=3).encode(
            y=alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelLimit=300, labelColor="#3A2433", labelOverlap=False)),
            x=alt.X("points:Q", title="Score (out of 1)", stack="zero"),
            color=alt.Color("Part:N", scale=alt.Scale(domain=[c[1] for c in comps], range=[c[2] for c in comps]),
                            legend=alt.Legend(orient="bottom", title=None)),
            tooltip=["label", "Part", alt.Tooltip("points:Q", format=".3f")])
            .properties(height=max(140, 40 * len(recs))).configure_view(strokeWidth=0))
        st.altair_chart(chart, width="stretch")

        st.markdown("##### Look at one product")
        pick = st.selectbox("Product", order, label_visibility="collapsed")
        row = recs[recs.label == pick].iloc[0]
        d = pd.DataFrame([dict(Part=name, **{"How well it does (0 to 1)": min(1.0, float(row[k]) / WEIGHT_OF[k]),
                                             "Weight": WEIGHT_OF[k]}) for k, name, _ in COMPS])
        c1_, c2_ = st.columns([3, 2])
        with c1_:
            bars = (alt.Chart(d).mark_bar(cornerRadiusEnd=5, size=22).encode(
                y=alt.Y("Part:N", sort=[c[1] for c in COMPS], title=None, axis=alt.Axis(labelColor="#3A2433")),
                x=alt.X("How well it does (0 to 1):Q", scale=alt.Scale(domain=[0, 1]), title="How well it does (0 to 1)"),
                color=alt.Color("Part:N", scale=alt.Scale(domain=[c[1] for c in COMPS], range=[c[2] for c in COMPS]), legend=None),
                tooltip=["Part", alt.Tooltip("How well it does (0 to 1):Q", format=".2f"), alt.Tooltip("Weight:Q", format=".0%")])
                .properties(height=210).configure_view(strokeWidth=0))
            st.altair_chart(bars, width="stretch")
        with c2_:
            best = d.loc[d["How well it does (0 to 1)"].idxmax()]
            st.markdown(f"**Total score: {row.score:.2f}**")
            st.markdown(f"Strongest part: **{best.Part}** ({best['How well it does (0 to 1)']:.2f}).")
            if row.c_penalty < 0:
                st.markdown(f"Penalty: **{row.c_penalty:.2f}** for ingredients that can irritate or clog pores for your profile.")
            if row.key_actives:
                st.markdown("Key actives: " + ", ".join(row.key_actives))

# ------------------------------------------------------------------ ingredients
with tab_ing:
    st.caption("Look for these on the label. They are ranked by how well they fit your skin and concerns.")
    ing = rec.recommend_ingredients(p, 8)
    if ing.empty:
        st.warning("No ingredients fit this combination. Try removing an avoid-ingredient or a safety option.")
    for r in ing.itertuples():
        evid = str(r.evidence)
        helps = pills([h.strip() for h in r.helps_with.split(",") if h.strip()])
        extra = ""
        if r.pairs_well_with:
            extra += f"<div class='kv'>Pairs well with <b>{esc(str(r.pairs_well_with))}</b></div>"
        if r.dont_mix_with:
            extra += f"<div class='kv'>Don't mix with <b>{esc(str(r.dont_mix_with))}</b></div>"
        if r.watch_out:
            extra += f"<div class='kv'>Watch out: {esc(str(r.watch_out))}</div>"
        st.markdown(
            f"<div class='card'><div class='body'><div class='title'>{esc(r.ingredient)}</div>"
            f"<div class='sub'>Best time: {esc(str(r.best_time))}</div>{helps}{extra}</div>"
            f"<div class='side'><span class='pill {EVID_CLS.get(evid.lower(), 'grey')}'>Evidence: {esc(evid)}</span></div></div>",
            unsafe_allow_html=True)

# ------------------------------------------------------------------ routine
with tab_routine:
    steps = st.multiselect("Routine steps", [s for s, _ in rec.ROUTINE_STEPS], default=["Cleanse", "Treat (serum)", "Moisturise"])
    routine, warns = rec.build_routine(p, tuple(steps))
    if routine.empty:
        st.warning("Couldn't build a routine within these filters.")
    else:
        for i, r in enumerate(routine.itertuples(), 1):
            st.markdown(f"<div class='matchlbl' style='margin:6px 0 4px 4px'>STEP {i} · {esc(r.step.upper())}</div>", unsafe_allow_html=True)
            product_card(r)
        st.markdown(f"<div class='total'>Total routine cost: {inr(routine.price.sum())}</div>", unsafe_allow_html=True)
        st.write("")
        if warns:
            for w_ in warns:
                st.warning(w_)
        else:
            st.success("No ingredient conflicts found between these products.")

# ------------------------------------------------------------------ similar
with tab_sim:
    st.caption("Pick a product you already use and find ones with a similar ingredient list, handy for cheaper alternatives.")
    names = rec.prod.apply(lambda r: f"{r.product_name} ({r.brand})", axis=1)
    pick_i = st.selectbox("Product", names.index, format_func=lambda i: names[i], index=None, placeholder="Start typing a product name")
    cheaper = st.checkbox("Only cheaper options")
    if pick_i is not None:
        st.markdown(f"Reference: **{names[pick_i]}**, {inr(rec.prod.price[pick_i])}")
        sim = rec.similar_products(int(pick_i), 5, cheaper)
        if sim.empty:
            st.info("Nothing cheaper with a similar ingredient list.")
        for r in sim.itertuples():
            product_card(r, show_actives=False)
            st.caption(f"Ingredient similarity: {r.similarity:.2f}")

st.markdown(
    "<div class='notice'>Informational tool, not medical advice. If you are pregnant or have a skin condition, check ingredients "
    "with your doctor. Recommendations are based on ingredient lists, not clinical testing.</div>",
    unsafe_allow_html=True)
