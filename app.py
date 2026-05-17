# app.py — BookMind Streamlit Dashboard

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from model import DataLoader, SVDRecommender, CBFRecommender, HybridRecommender

st.set_page_config(
    page_title="BookMind",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# THEME & CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;0,700;1,400&family=Crimson+Pro:wght@300;400;600&display=swap');

:root {
  --bg:     #080C18;
  --bg2:    #0D1222;
  --bg3:    #111827;
  --bg4:    #162032;
  --amber:  #E8A320;
  --amber2: #F5C355;
  --teal:   #2DD4BF;
  --purple: #8B5CF6;
  --text:   #CCC4B0;
  --text2:  #7A7060;
  --text3:  #484038;
  --white:  #EDE5CF;
  --border: #1C2840;
}

/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Crimson Pro', Georgia, serif !important;
}
h1, h2, h3, h4 {
    font-family: 'Playfair Display', serif !important;
    color: var(--white) !important;
    letter-spacing: -0.01em;
}
p { line-height: 1.7; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: var(--bg2) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── Sidebar Nav — Style B ── */
[data-testid="stSidebar"] [data-testid="stRadio"] {
    background: var(--bg3) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 6px !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] > div {
    display: flex !important;
    flex-direction: column !important;
    gap: 2px !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    background: transparent !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 10px 14px !important;
    cursor: pointer !important;
    font-size: 14px !important;
    font-family: 'Crimson Pro', serif !important;
    color: var(--text2) !important;
    transition: background .15s, color .15s !important;
    width: 100% !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: var(--bg4) !important;
    color: var(--white) !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb] {
    background: #1C1408 !important;
    color: var(--amber) !important;
    border-left: 3px solid var(--amber) !important;
    padding-left: 11px !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] input { display: none !important; }
[data-testid="stSidebar"] [data-testid="stRadio"] div[data-testid="stMarkdownContainer"] p {
    margin: 0 !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px !important;
    background: var(--bg3) !important;
    padding: 6px !important;
    border-radius: 8px !important;
    border: 1px solid var(--border) !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text2) !important;
    border-radius: 5px !important;
    font-family: 'Crimson Pro', serif !important;
    font-size: 13px !important;
    padding: 6px 16px !important;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    background: #1C1408 !important;
    color: var(--amber2) !important;
    border-bottom: 2px solid var(--amber) !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 20px !important; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #E8A320, #F5C355) !important;
    color: #0A0A0A !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'Crimson Pro', serif !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    padding: 10px 28px !important;
    letter-spacing: .02em !important;
}
.stButton > button:hover { opacity: .9 !important; }

/* ── Inputs ── */
.stTextInput input, .stSelectbox > div > div, .stNumberInput input {
    background: var(--bg3) !important;
    border: 1px solid var(--border) !important;
    color: var(--white) !important;
    border-radius: 6px !important;
    font-family: 'Crimson Pro', serif !important;
    font-size: 14px !important;
}
.stSlider [data-baseweb="slider"] { padding: 0 !important; }

/* ── Cards ── */
.metric-card {
    background: var(--bg3);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 24px 20px;
    text-align: center;
}
.metric-val {
    font-family: 'Playfair Display', serif;
    font-size: 32px;
    font-weight: 700;
    color: var(--amber);
    line-height: 1.1;
}
.metric-label {
    font-size: 11px;
    color: var(--text2);
    text-transform: uppercase;
    letter-spacing: .08em;
    margin-top: 6px;
}

.rec-card {
    background: var(--bg3);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 18px 20px;
    margin-bottom: 12px;
    transition: border-color .2s;
}
.rec-card:hover { border-color: #2C3E60; }
.rec-title {
    font-family: 'Playfair Display', serif;
    font-size: 16px;
    color: var(--white);
    margin-bottom: 4px;
    line-height: 1.3;
}
.rec-author { font-size: 13px; color: var(--text2); margin-bottom: 10px; }
.rec-score  { font-size: 12px; color: var(--amber); font-weight: 600; }

/* rating bar */
.rbar-wrap { margin-top: 10px; }
.rbar-row  { display: flex; align-items: center; gap: 8px; margin-bottom: 3px; }
.rbar-label { font-size: 11px; color: var(--text2); width: 18px; text-align: right; flex-shrink: 0; }
.rbar-bg    { flex: 1; height: 5px; background: #1C2840; border-radius: 3px; overflow: hidden; }
.rbar-fill  { height: 100%; border-radius: 3px; }
.rbar-count { font-size: 10px; color: var(--text3); width: 40px; }

.tag {
    display: inline-block;
    font-size: 11px;
    border-radius: 10px;
    padding: 2px 10px;
    font-weight: 600;
    margin-bottom: 8px;
}

hr { border-color: var(--border) !important; margin: 24px 0 !important; }
.section-label {
    font-size: 11px;
    color: var(--text3);
    text-transform: uppercase;
    letter-spacing: .1em;
    margin-bottom: 16px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Memuat dataset & melatih model...")
def load_models():
    loader         = DataLoader(min_rating_pengguna=5, min_rating_buku=10)
    ratings, books = loader.load()
    svd            = SVDRecommender(n_factors=50).fit(ratings)
    cbf            = CBFRecommender().fit(books)
    hybrid         = HybridRecommender(cf_weight=0.5)
    hybrid.cf      = svd
    hybrid.cbf     = cbf
    hybrid.ratings = ratings
    hybrid.books   = books
    return ratings, books, svd, cbf, hybrid

ratings, books, svd, cbf, hybrid = load_models()

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:20px 4px 24px'>
        <div style='display:flex;align-items:center;gap:12px'>
            <div style='width:38px;height:38px;background:linear-gradient(135deg,#E8A320,#F5C355);
                        border-radius:8px;display:flex;align-items:center;justify-content:center;
                        font-size:18px;font-weight:800;color:#0A0A0A;font-family:Playfair Display,serif;
                        flex-shrink:0'>B</div>
            <div>
                <div style='font-family:Playfair Display,serif;font-size:18px;color:#EDE5CF;
                            font-weight:700;line-height:1'>BookMind</div>
                <div style='font-size:10px;color:#484038;text-transform:uppercase;
                            letter-spacing:.1em;margin-top:3px'>Hybrid Engine</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio("nav", ["◈  Beranda", "◉  Rekomendasi", "◆  Analitik", "◇  Tentang Model"],
                    label_visibility="collapsed")

    st.markdown("<hr style='border-color:#1C2840;margin:16px 0'>", unsafe_allow_html=True)
    st.markdown("<div class='section-label'>Dataset</div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='display:flex;flex-direction:column;gap:6px'>
        <div style='display:flex;justify-content:space-between;font-size:13px'>
            <span style='color:#7A7060'>Buku</span>
            <span style='color:#CCC4B0;font-weight:600'>{len(books):,}</span>
        </div>
        <div style='display:flex;justify-content:space-between;font-size:13px'>
            <span style='color:#7A7060'>Ratings</span>
            <span style='color:#CCC4B0;font-weight:600'>{len(ratings):,}</span>
        </div>
        <div style='display:flex;justify-content:space-between;font-size:13px'>
            <span style='color:#7A7060'>Pengguna</span>
            <span style='color:#CCC4B0;font-weight:600'>{ratings['user_id'].nunique():,}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def rating_bars_html(r1, r2, r3, r4, r5):
    vals   = [r1, r2, r3, r4, r5]
    total  = sum(vals) or 1
    colors = ["#F43F5E", "#F97316", "#EAB308", "#84CC16", "#E8A320"]
    rows   = ""
    for i, (v, c) in enumerate(zip(vals, colors)):
        pct = v / total * 100
        rows += f"""
        <div class="rbar-row">
            <div class="rbar-label">★{i+1}</div>
            <div class="rbar-bg"><div class="rbar-fill" style="width:{pct:.1f}%;background:{c}"></div></div>
            <div class="rbar-count">{v:,}</div>
        </div>"""
    return f"<div class='rbar-wrap'>{rows}</div>"

def render_rec_card(row, rank, tag_label, tag_color, score_col):
    score_val = row.get(score_col, None)
    score_str = f"{score_val:.3f}" if score_val is not None and not pd.isna(score_val) else "—"
    rating    = row.get("rating", 0)
    author    = row.get("author", row.get("authors", ""))
    image_url = row.get("image", "")
    r1 = int(row.get("ratings_1", 0) or 0)
    r2 = int(row.get("ratings_2", 0) or 0)
    r3 = int(row.get("ratings_3", 0) or 0)
    r4 = int(row.get("ratings_4", 0) or 0)
    r5 = int(row.get("ratings_5", 0) or 0)
    total_r   = r1 + r2 + r3 + r4 + r5

    img_html  = f"<img src='{image_url}' style='width:56px;height:80px;object-fit:cover;border-radius:6px;flex-shrink:0'>" if image_url else \
                "<div style='width:56px;height:80px;background:#1C2840;border-radius:6px;flex-shrink:0'></div>"

    bars = rating_bars_html(r1, r2, r3, r4, r5) if total_r > 0 else ""

    st.markdown(f"""
    <div class="rec-card">
        <div style='display:flex;gap:16px'>
            {img_html}
            <div style='flex:1;min-width:0'>
                <div style='display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:8px'>
                    <span class="tag" style='background:{tag_color}18;color:{tag_color};border:1px solid {tag_color}33'>
                        #{rank} {tag_label}
                    </span>
                    <span class="rec-score">⭐ {rating:.2f} &nbsp;·&nbsp; {score_col.replace('_',' ')}: {score_str}</span>
                </div>
                <div class="rec-title">{row.get('title','')}</div>
                <div class="rec-author">{author}</div>
                {bars}
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

def plotly_dark(fig, height=None):
    layout = dict(
        plot_bgcolor="#111827", paper_bgcolor="#0D1222",
        font_color="#CCC4B0", font_family="Crimson Pro",
        title_font_family="Playfair Display",
        margin=dict(t=48, b=32, l=16, r=16),
    )
    if height: layout["height"] = height
    fig.update_layout(**layout)
    fig.update_xaxes(gridcolor="#1C2840", showgrid=True)
    fig.update_yaxes(gridcolor="#1C2840", showgrid=True)
    return fig

# ─────────────────────────────────────────────
# PAGE: BERANDA
# ─────────────────────────────────────────────
if "Beranda" in page:
    st.markdown("""
    <div style='text-align:center;padding:48px 0 32px'>
        <div style='font-size:11px;letter-spacing:.2em;text-transform:uppercase;
                    color:#E8A320;margin-bottom:16px;font-family:Crimson Pro,serif'>
            Book Recommendation Engine
        </div>
        <h1 style='font-size:clamp(32px,5vw,56px);line-height:1.05;margin-bottom:20px;font-weight:700'>
            Find your next<br><em style='color:#E8A320'>favourite book</em>
        </h1>
        <p style='font-size:16px;color:#7A7060;max-width:480px;margin:0 auto 0;line-height:1.8'>
            Combining <strong style='color:#CCC4B0'>Content-Based</strong> and
            <strong style='color:#CCC4B0'>Collaborative Filtering</strong>
            to surface books you'll actually love.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Stats
    c1, c2, c3, c4 = st.columns(4)
    for col, val, label in zip(
        [c1, c2, c3, c4],
        [f"{len(books):,}", f"{len(ratings):,}", f"{ratings['user_id'].nunique():,}", f"{books['rating'].mean():.2f}"],
        ["Books", "Total Ratings", "Readers", "Avg Rating"]
    ):
        with col:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-val">{val}</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:40px'></div>", unsafe_allow_html=True)

    # Method cards
    st.markdown("<div class='section-label' style='text-align:center'>How it works</div>", unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    for col, icon, title, desc, color in zip(
        [m1, m2, m3],
        ["◈", "◉", "◆"],
        ["Content-Based", "Collaborative Filtering", "Weighted Hybrid"],
        [
            "TF-IDF on title & author. Cosine similarity finds books with similar content.",
            "Truncated SVD on the user-item matrix. Discovers hidden patterns from thousands of readers.",
            "Final score = 0.5 × CF + 0.5 × CBF. Best of both worlds.",
        ],
        ["#8B5CF6", "#2DD4BF", "#E8A320"]
    ):
        with col:
            st.markdown(f"""<div class="metric-card" style='text-align:left;padding:28px 24px'>
                <div style='font-size:22px;color:{color};margin-bottom:14px;font-family:monospace'>{icon}</div>
                <div style='font-family:Playfair Display,serif;font-size:16px;color:#EDE5CF;
                            margin-bottom:10px;font-weight:600'>{title}</div>
                <div style='font-size:13px;color:#7A7060;line-height:1.7'>{desc}</div>
            </div>""", unsafe_allow_html=True)

    # Top books
    st.markdown("<div style='height:40px'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-label' style='text-align:center'>Most popular</div>", unsafe_allow_html=True)
    top5 = books.nlargest(5, "ratings_count")[["title","author","rating","ratings_count","image"]].reset_index(drop=True)
    cols = st.columns(5)
    for col, row in zip(cols, top5.itertuples()):
        with col:
            if row.image:
                st.markdown(f"<img src='{row.image}' style='width:100%;border-radius:8px;margin-bottom:12px;display:block'>", unsafe_allow_html=True)
            st.markdown(f"""
            <div style='font-family:Playfair Display,serif;font-size:13px;color:#EDE5CF;
                        margin-bottom:5px;line-height:1.4'>{row.title[:38]}{'…' if len(row.title)>38 else ''}</div>
            <div style='font-size:11px;color:#7A7060;margin-bottom:4px'>{row.author[:28]}</div>
            <div style='font-size:12px;color:#E8A320'>⭐ {row.rating:.2f}
                <span style='color:#484038;margin-left:4px'>{int(row.ratings_count/1000)}k ratings</span>
            </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PAGE: REKOMENDASI
# ─────────────────────────────────────────────
elif "Rekomendasi" in page:
    st.markdown("<h2 style='margin-bottom:6px'>Cari Buku Serupa</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#7A7060;margin-bottom:28px'>Ketik judul buku yang kamu suka — sistem akan merekomendasikan buku serupa.</p>", unsafe_allow_html=True)

    search_query = st.text_input("Cari judul buku", placeholder="e.g. Harry Potter, The Hobbit, Pride and Prejudice",
                                  label_visibility="collapsed")
    c1, c2 = st.columns([2, 1])
    with c1:
        method = st.radio("Metode", ["Hybrid", "Content-Based (TF-IDF)", "Collaborative (SVD)"], horizontal=True)
    with c2:
        n_similar = st.slider("Jumlah rekomendasi", 5, 15, 8, label_visibility="visible")

    if search_query:
        matches = books[books["title"].str.contains(search_query, case=False, na=False)]
        if matches.empty:
            st.warning("Tidak ada buku ditemukan.")
        else:
            selected_title = st.selectbox("Pilih dari hasil pencarian:", matches["title"].tolist(),
                                          label_visibility="collapsed")
            book_row = matches[matches["title"] == selected_title].iloc[0]
            book_id  = book_row["book_id"]
            img      = book_row.get("image", "")
            r1,r2,r3,r4,r5 = [int(book_row.get(f"ratings_{i}", 0) or 0) for i in range(1,6)]

            st.markdown(f"""
            <div style='background:var(--bg3);border:1px solid #243258;border-radius:12px;
                         padding:20px 24px;margin:16px 0 24px;display:flex;gap:20px;align-items:flex-start'>
                {'<img src="' + img + '" style="width:72px;height:100px;object-fit:cover;border-radius:6px;flex-shrink:0">' if img else ''}
                <div style='flex:1'>
                    <div class="section-label" style='margin-bottom:8px'>Buku Dipilih</div>
                    <div style='font-family:Playfair Display,serif;font-size:20px;color:#EDE5CF;
                                margin-bottom:6px'>{book_row.get('title','')}</div>
                    <div style='font-size:14px;color:#E8A320;margin-bottom:8px'>{book_row.get('author','')}</div>
                    <div style='font-size:13px;color:#7A7060'>⭐ {book_row.get('rating',0):.2f}
                        &nbsp;·&nbsp; {int(book_row.get('ratings_count',0)):,} ratings</div>
                    {rating_bars_html(r1,r2,r3,r4,r5)}
                </div>
            </div>""", unsafe_allow_html=True)

            if st.button("Cari Rekomendasi"):
                with st.spinner("Mencari buku serupa..."):
                    if "Hybrid" in method:
                        cbf_recs = cbf.recommend(book_id, n=n_similar * 2)
                        cf_ids   = [int(i) for i in svd.item_cf_similarity(n_similar=n_similar*2).get(str(book_id), [])]
                        if not cbf_recs.empty:
                            cbf_recs["cf_boost"]     = cbf_recs["book_id"].apply(lambda b: 0.3 if b in cf_ids else 0.0)
                            cbf_recs["hybrid_score"] = 0.7 * cbf_recs["cbf_score"] + cbf_recs["cf_boost"]
                            recs = cbf_recs.sort_values("hybrid_score", ascending=False).head(n_similar)
                        else:
                            recs = pd.DataFrame()
                        tag_label, tag_color, score_col = "Hybrid", "#E8A320", "hybrid_score"

                    elif "Content" in method:
                        recs = cbf.recommend(book_id, n=n_similar)
                        tag_label, tag_color, score_col = "CBF", "#8B5CF6", "cbf_score"

                    else:
                        cf_ids = [int(i) for i in svd.item_cf_similarity(n_similar=n_similar).get(str(book_id), [])]
                        recs   = books[books["book_id"].isin(cf_ids)][
                            ["book_id","title","author","rating","ratings_count","image",
                             "ratings_1","ratings_2","ratings_3","ratings_4","ratings_5"]
                        ].copy()
                        recs["cf_score"] = recs["book_id"].apply(
                            lambda b: 1.0 - cf_ids.index(b)/len(cf_ids) if b in cf_ids else 0)
                        recs = recs.sort_values("cf_score", ascending=False)
                        tag_label, tag_color, score_col = "SVD CF", "#2DD4BF", "cf_score"

                    if recs is None or len(recs) == 0:
                        st.warning("Tidak ada rekomendasi ditemukan.")
                    else:
                        # merge ratings breakdown if not present
                        rating_cols = ["ratings_1","ratings_2","ratings_3","ratings_4","ratings_5"]
                        missing = [c for c in rating_cols if c not in recs.columns]
                        if missing:
                            recs = recs.merge(books[["book_id","image"] + rating_cols], on="book_id", how="left",
                                              suffixes=("","_y"))
                        if "image" not in recs.columns:
                            recs = recs.merge(books[["book_id","image"]], on="book_id", how="left")

                        st.markdown(f"<h3 style='margin-bottom:16px'>Top {len(recs)} Rekomendasi</h3>", unsafe_allow_html=True)
                        for i, row in enumerate(recs.itertuples()):
                            render_rec_card(row._asdict(), i+1, tag_label, tag_color, score_col)

# ─────────────────────────────────────────────
# PAGE: ANALITIK
# ─────────────────────────────────────────────
elif "Analitik" in page:
    st.markdown("<h2 style='margin-bottom:6px'>Analitik Dataset</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#7A7060;margin-bottom:24px'>Goodbooks-10k · 10,000 buku · 6 juta ratings</p>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["Distribusi Rating", "Deep Dive Buku", "Top Penulis", "Tren Publikasi"])

    # ── TAB 1: Distribusi ──
    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(books, x="rating", nbins=40,
                               title="Average Rating per Buku",
                               color_discrete_sequence=["#E8A320"])
            fig.update_traces(marker_line_width=0)
            st.plotly_chart(plotly_dark(fig), use_container_width=True)

        with c2:
            rating_counts = ratings["rating"].value_counts().sort_index().reset_index()
            rating_counts.columns = ["rating", "count"]
            fig2 = px.bar(rating_counts, x="rating", y="count",
                          title="Distribusi Rating Pengguna (1–5)",
                          color="rating",
                          color_continuous_scale=[[0,"#F43F5E"],[0.5,"#EAB308"],[1,"#E8A320"]])
            st.plotly_chart(plotly_dark(fig2), use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            rpu  = ratings.groupby("user_id").size().reset_index(name="count")
            fig3 = px.histogram(rpu, x="count", nbins=50,
                                title="Jumlah Rating per Pengguna",
                                color_discrete_sequence=["#8B5CF6"])
            fig3.update_traces(marker_line_width=0)
            st.plotly_chart(plotly_dark(fig3), use_container_width=True)

        with c4:
            rpb  = ratings.groupby("book_id").size().reset_index(name="count")
            fig4 = px.histogram(rpb, x="count", nbins=50,
                                title="Jumlah Rating per Buku",
                                color_discrete_sequence=["#2DD4BF"])
            fig4.update_traces(marker_line_width=0)
            st.plotly_chart(plotly_dark(fig4), use_container_width=True)

        # Insight box
        avg_per_user = rpu["count"].mean()
        avg_per_book = rpb["count"].mean()
        sparsity     = 1 - len(ratings) / (ratings["user_id"].nunique() * ratings["book_id"].nunique())
        st.markdown(f"""
        <div style='background:#111827;border:1px solid #1C2840;border-radius:12px;
                     padding:20px 24px;display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-top:8px'>
            <div>
                <div style='font-size:11px;color:#7A7060;text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px'>Avg ratings / user</div>
                <div style='font-size:24px;font-family:Playfair Display,serif;color:#E8A320'>{avg_per_user:.1f}</div>
            </div>
            <div>
                <div style='font-size:11px;color:#7A7060;text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px'>Avg ratings / buku</div>
                <div style='font-size:24px;font-family:Playfair Display,serif;color:#2DD4BF'>{avg_per_book:.1f}</div>
            </div>
            <div>
                <div style='font-size:11px;color:#7A7060;text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px'>Matrix sparsity</div>
                <div style='font-size:24px;font-family:Playfair Display,serif;color:#8B5CF6'>{sparsity:.2%}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── TAB 2: Deep Dive ──
    with tab2:
        search_book = st.text_input("Cari buku untuk deep dive", placeholder="e.g. The Great Gatsby",
                                    label_visibility="collapsed", key="dd_search")
        if search_book:
            dd_matches = books[books["title"].str.contains(search_book, case=False, na=False)]
            if dd_matches.empty:
                st.warning("Buku tidak ditemukan.")
            else:
                dd_title = st.selectbox("Pilih buku:", dd_matches["title"].tolist(), key="dd_select",
                                        label_visibility="collapsed")
                dd_row   = dd_matches[dd_matches["title"] == dd_title].iloc[0]
                r1,r2,r3,r4,r5 = [int(dd_row.get(f"ratings_{i}", 0) or 0) for i in range(1,6)]
                total_r  = r1+r2+r3+r4+r5

                img = dd_row.get("image","")
                c_img, c_info = st.columns([1, 3])
                with c_img:
                    if img:
                        st.markdown(f"<img src='{img}' style='width:100%;border-radius:8px'>", unsafe_allow_html=True)
                with c_info:
                    st.markdown(f"""
                    <div style='padding:4px 0'>
                        <div style='font-family:Playfair Display,serif;font-size:24px;color:#EDE5CF;
                                    margin-bottom:8px;line-height:1.2'>{dd_row.get('title','')}</div>
                        <div style='font-size:15px;color:#E8A320;margin-bottom:16px'>{dd_row.get('author','')}</div>
                        <div style='display:grid;grid-template-columns:repeat(3,1fr);gap:12px'>
                            <div style='background:#111827;border:1px solid #1C2840;border-radius:8px;padding:12px'>
                                <div style='font-size:10px;color:#7A7060;text-transform:uppercase;letter-spacing:.08em;margin-bottom:4px'>Avg Rating</div>
                                <div style='font-size:22px;font-family:Playfair Display,serif;color:#E8A320'>⭐ {dd_row.get('rating',0):.2f}</div>
                            </div>
                            <div style='background:#111827;border:1px solid #1C2840;border-radius:8px;padding:12px'>
                                <div style='font-size:10px;color:#7A7060;text-transform:uppercase;letter-spacing:.08em;margin-bottom:4px'>Total Ratings</div>
                                <div style='font-size:22px;font-family:Playfair Display,serif;color:#2DD4BF'>{total_r:,}</div>
                            </div>
                            <div style='background:#111827;border:1px solid #1C2840;border-radius:8px;padding:12px'>
                                <div style='font-size:10px;color:#7A7060;text-transform:uppercase;letter-spacing:.08em;margin-bottom:4px'>Tahun</div>
                                <div style='font-size:22px;font-family:Playfair Display,serif;color:#8B5CF6'>{int(dd_row.get('year',0)) if dd_row.get('year',0) else '—'}</div>
                            </div>
                        </div>
                    </div>""", unsafe_allow_html=True)

                st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
                c_bar, c_pie = st.columns(2)
                with c_bar:
                    fig_dd = go.Figure(go.Bar(
                        x=[f"★{i}" for i in range(1,6)],
                        y=[r1,r2,r3,r4,r5],
                        marker_color=["#F43F5E","#F97316","#EAB308","#84CC16","#E8A320"],
                        text=[f"{v:,}" for v in [r1,r2,r3,r4,r5]],
                        textposition="outside",
                    ))
                    fig_dd.update_layout(title="Breakdown Rating", showlegend=False)
                    st.plotly_chart(plotly_dark(fig_dd), use_container_width=True)

                with c_pie:
                    fig_pie = px.pie(
                        values=[r1,r2,r3,r4,r5],
                        names=["★1","★2","★3","★4","★5"],
                        title="Proporsi Rating",
                        color_discrete_sequence=["#F43F5E","#F97316","#EAB308","#84CC16","#E8A320"],
                        hole=0.4,
                    )
                    st.plotly_chart(plotly_dark(fig_pie), use_container_width=True)

                # Rank in dataset
                rank_by_rating = int(books["rating"].rank(ascending=False)[books["book_id"] == dd_row["book_id"]].values[0])
                rank_by_pop    = int(books["ratings_count"].rank(ascending=False)[books["book_id"] == dd_row["book_id"]].values[0])
                st.markdown(f"""
                <div style='background:#111827;border:1px solid #1C2840;border-radius:12px;
                             padding:16px 24px;display:flex;gap:24px;flex-wrap:wrap;margin-top:4px'>
                    <div>
                        <div style='font-size:11px;color:#7A7060;text-transform:uppercase;letter-spacing:.08em;margin-bottom:4px'>Ranking (Rating)</div>
                        <div style='font-size:20px;font-family:Playfair Display,serif;color:#E8A320'>#{rank_by_rating} dari {len(books):,}</div>
                    </div>
                    <div>
                        <div style='font-size:11px;color:#7A7060;text-transform:uppercase;letter-spacing:.08em;margin-bottom:4px'>Ranking (Popularitas)</div>
                        <div style='font-size:20px;font-family:Playfair Display,serif;color:#2DD4BF'>#{rank_by_pop} dari {len(books):,}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background:#111827;border:1px dashed #1C2840;border-radius:12px;
                         padding:48px;text-align:center;color:#484038'>
                Ketik judul buku di atas untuk melihat analitik detail
            </div>""", unsafe_allow_html=True)

    # ── TAB 3: Top Penulis ──
    with tab3:
        author_col  = "author" if "author" in books.columns else "authors"
        top_authors = (
            books.groupby(author_col)
            .agg(jumlah_buku=("book_id","count"),
                 avg_rating=("rating","mean"),
                 total_ratings=("ratings_count","sum"))
            .reset_index()
            .sort_values("total_ratings", ascending=False)
            .head(20)
        )
        fig_a = px.bar(top_authors, x="total_ratings", y=author_col,
                       orientation="h", title="Top 20 Penulis — Total Ratings",
                       color="avg_rating",
                       color_continuous_scale=[[0,"#6B4A10"],[0.5,"#E8A320"],[1,"#F5C355"]],
                       hover_data=["jumlah_buku","avg_rating"])
        fig_a.update_layout(yaxis={"categoryorder":"total ascending"})
        st.plotly_chart(plotly_dark(fig_a, height=580), use_container_width=True)

        # Scatter: avg_rating vs jumlah buku
        author_all = (
            books.groupby(author_col)
            .agg(jumlah_buku=("book_id","count"), avg_rating=("rating","mean"), total_ratings=("ratings_count","sum"))
            .reset_index()
        )
        fig_as = px.scatter(author_all[author_all["jumlah_buku"] >= 2],
                            x="jumlah_buku", y="avg_rating",
                            size="total_ratings", color="avg_rating",
                            hover_name=author_col,
                            title="Penulis: Jumlah Buku vs Avg Rating (penulis ≥2 buku)",
                            color_continuous_scale=[[0,"#6B4A10"],[1,"#F5C355"]],
                            opacity=0.8)
        st.plotly_chart(plotly_dark(fig_as), use_container_width=True)

    # ── TAB 4: Tren ──
    with tab4:
        decade_df = books[books["year"] > 0].copy()
        decade_df["decade"] = (decade_df["year"] // 10 * 10).astype(int)
        decade_agg = decade_df.groupby("decade").agg(
            count=("book_id","count"), avg_rating=("rating","mean"),
            total_ratings=("ratings_count","sum")
        ).reset_index()

        fig5 = go.Figure()
        fig5.add_trace(go.Bar(x=decade_agg["decade"], y=decade_agg["count"],
                              name="Jumlah Buku", marker_color="#1C2840"))
        fig5.add_trace(go.Scatter(x=decade_agg["decade"], y=decade_agg["avg_rating"],
                                  name="Avg Rating", mode="lines+markers",
                                  marker_color="#E8A320", line_color="#E8A320", yaxis="y2"))
        fig5.update_layout(
            title="Buku dan Rating berdasarkan Dekade",
            yaxis={"title":"Jumlah Buku","gridcolor":"#1C2840"},
            yaxis2={"title":"Avg Rating","overlaying":"y","side":"right","range":[3,5]},
            legend={"bgcolor":"#0D1222"},
        )
        st.plotly_chart(plotly_dark(fig5), use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            sample = books.sample(min(800, len(books)), random_state=42)
            fig6   = px.scatter(sample, x="ratings_count", y="rating",
                                title="Rating vs Popularitas",
                                color="rating",
                                color_continuous_scale=[[0,"#6B4A10"],[1,"#F5C355"]],
                                opacity=0.6, hover_data=["title"])
            st.plotly_chart(plotly_dark(fig6), use_container_width=True)

        with c2:
            fig7 = px.box(decade_df[decade_df["decade"] >= 1950],
                          x="decade", y="rating",
                          title="Distribusi Rating per Dekade (1950+)",
                          color_discrete_sequence=["#E8A320"])
            st.plotly_chart(plotly_dark(fig7), use_container_width=True)

# ─────────────────────────────────────────────
# PAGE: TENTANG MODEL
# ─────────────────────────────────────────────
elif "Tentang" in page:
    st.markdown("<h2 style='margin-bottom:6px'>Tentang Model</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#7A7060;margin-bottom:28px'>Arsitektur hybrid recommendation system yang digunakan BookMind.</p>", unsafe_allow_html=True)

    st.markdown("<h3>Alur Sistem</h3>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background:#111827;border:1px solid #1C2840;border-radius:12px;padding:32px;margin-bottom:28px'>
        <div style='display:flex;align-items:center;justify-content:center;gap:0;flex-wrap:wrap;row-gap:16px'>
            <div style='text-align:center;padding:16px 20px;background:#0D1222;border:1px solid #243258;border-radius:10px;min-width:120px'>
                <div style='font-size:11px;color:#7A7060;text-transform:uppercase;letter-spacing:.08em;margin-bottom:8px'>Input</div>
                <div style='font-size:15px;font-family:Playfair Display,serif;color:#EDE5CF'>Buku Pilihan</div>
            </div>
            <div style='color:#1C2840;font-size:24px;margin:0 12px'>→</div>
            <div style='display:flex;flex-direction:column;gap:10px'>
                <div style='text-align:center;padding:12px 20px;background:#1C0D3A;border:1px solid #4C2D8C;border-radius:8px;min-width:170px'>
                    <div style='font-size:11px;color:#8B5CF6;font-weight:600;text-transform:uppercase;letter-spacing:.06em'>Content-Based</div>
                    <div style='font-size:11px;color:#7A7060;margin-top:4px'>TF-IDF + Cosine Similarity</div>
                </div>
                <div style='text-align:center;padding:12px 20px;background:#0D2030;border:1px solid #0E5E6F;border-radius:8px;min-width:170px'>
                    <div style='font-size:11px;color:#2DD4BF;font-weight:600;text-transform:uppercase;letter-spacing:.06em'>Collaborative</div>
                    <div style='font-size:11px;color:#7A7060;margin-top:4px'>Truncated SVD · k=50</div>
                </div>
            </div>
            <div style='color:#1C2840;font-size:24px;margin:0 12px'>→</div>
            <div style='text-align:center;padding:16px 20px;background:#1C1408;border:1px solid #6B4A10;border-radius:10px;min-width:150px'>
                <div style='font-size:11px;color:#7A7060;text-transform:uppercase;letter-spacing:.08em;margin-bottom:8px'>Weighted Sum</div>
                <div style='font-size:15px;font-family:Playfair Display,serif;color:#E8A320'>0.5 × CF + 0.5 × CBF</div>
            </div>
            <div style='color:#1C2840;font-size:24px;margin:0 12px'>→</div>
            <div style='text-align:center;padding:16px 20px;background:#0D1222;border:1px solid #243258;border-radius:10px;min-width:120px'>
                <div style='font-size:11px;color:#7A7060;text-transform:uppercase;letter-spacing:.08em;margin-bottom:8px'>Output</div>
                <div style='font-size:15px;font-family:Playfair Display,serif;color:#EDE5CF'>Top-N Books</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<h3>Perbandingan Metode</h3>", unsafe_allow_html=True)
    comparison = pd.DataFrame({
        "Metode":       ["Content-Based (TF-IDF)", "Collaborative Filtering (SVD)", "Hybrid (50/50)"],
        "Tipe":         ["Memory-Based", "Model-Based", "Hybrid"],
        "Cold Start":   ["✅ Item baru OK", "❌ Perlu data historis", "⚠️ Partial"],
        "Serendipity":  ["Rendah", "Tinggi", "Sedang"],
        "Keunggulan":   [
            "Transparan, tidak butuh data user lain",
            "Menemukan pola tersembunyi antar pembaca",
            "Gabungkan kelebihan keduanya",
        ],
    })
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    st.markdown("<h3 style='margin-top:28px'>Cara Kerja SVD</h3>", unsafe_allow_html=True)
    for step, desc in [
        ("User-Item Matrix", "Bangun matrix R (users × books). Mayoritas kosong — disimpan sebagai sparse matrix."),
        ("Mean-Centering", "Kurangi rata-rata rating tiap user. Menghilangkan bias skala antar pengguna."),
        ("Truncated SVD", "Dekomposisi: R ≈ U × Σ × Vt. Ambil k=50 faktor terbesar — buang noise."),
        ("Rekonstruksi", "Prediksi = U[u] × Σ × Vt + mean[u]. Hanya satu baris per request, hemat RAM."),
        ("Top-N", "Ranking semua kandidat buku, ambil N tertinggi yang belum pernah dirating."),
    ]:
        st.markdown(f"""
        <div style='display:flex;gap:16px;margin-bottom:12px;padding:14px 18px;
                     background:#111827;border:1px solid #1C2840;border-radius:8px'>
            <div style='width:6px;background:linear-gradient(180deg,#E8A320,#F5C355);
                        border-radius:3px;flex-shrink:0'></div>
            <div>
                <div style='font-size:13px;color:#EDE5CF;font-weight:600;margin-bottom:4px'>{step}</div>
                <div style='font-size:13px;color:#7A7060;line-height:1.6'>{desc}</div>
            </div>
        </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div style='margin-top:64px;padding:24px;border-top:1px solid #1C2840;
             display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px'>
    <div style='font-family:Playfair Display,serif;font-size:15px;color:#484038'>BookMind</div>
    <div style='font-size:12px;color:#484038'>goodbooks-10k · SVD + TF-IDF Hybrid</div>
</div>
""", unsafe_allow_html=True)