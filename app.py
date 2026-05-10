# app.py — BookMind Streamlit Dashboard
# pip install streamlit pandas numpy scipy scikit-learn plotly

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from model import DataLoader, SVDRecommender, CBFRecommender, HybridRecommender

st.set_page_config(
    page_title="BookMind — Hybrid Recommendation Engine",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# THEME
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=Crimson+Pro:wght@300;400;600&display=swap');

:root {
  --bg:    #080C18; --bg2: #0D1222; --bg3: #111827;
  --amber: #E8A320; --amber2: #F5C355; --teal: #2DD4BF;
  --text:  #CCC4B0; --text2: #7A7060; --white: #EDE5CF;
}
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Crimson Pro', Georgia, serif !important;
}
[data-testid="stSidebar"] { background-color: var(--bg2) !important; border-right: 1px solid #1C2840; }
[data-testid="stSidebar"] * { color: var(--text) !important; }
h1, h2, h3 { font-family: 'Playfair Display', serif !important; color: var(--white) !important; }

.metric-card {
    background: var(--bg3); border: 1px solid #1C2840;
    border-radius: 10px; padding: 20px; text-align: center;
}
.metric-val { font-family: 'Playfair Display', serif; font-size: 28px; font-weight: 700; color: var(--amber); }
.metric-label { font-size: 11px; color: var(--text2); text-transform: uppercase; letter-spacing: .06em; margin-top: 4px; }

.rec-card {
    background: var(--bg3); border: 1px solid #1C2840;
    border-radius: 10px; padding: 14px 16px; margin-bottom: 10px;
}
.rec-title { font-family: 'Playfair Display', serif; font-size: 15px; color: var(--white); margin-bottom: 3px; }
.rec-meta  { font-size: 12px; color: var(--text2); }
.rec-score { font-size: 12px; color: var(--amber); font-weight: 600; }

.stButton > button {
    background: linear-gradient(135deg, #E8A320, #F5C355) !important;
    color: #0A0A0A !important; border: none !important; border-radius: 6px !important;
    font-family: 'Crimson Pro', serif !important; font-size: 15px !important;
    font-weight: 600 !important; padding: 8px 24px !important;
}
.stTabs [data-baseweb="tab"] {
    background: var(--bg3) !important; color: var(--text2) !important;
    border-radius: 5px !important; font-family: 'Crimson Pro', serif !important;
}
.stTabs [aria-selected="true"] {
    background: #6B4A10 !important; color: var(--amber2) !important;
    border: 1px solid var(--amber) !important;
}
hr { border-color: #1C2840 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD MODELS (cached)
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
    <div style='display:flex;align-items:center;gap:10px;margin-bottom:24px'>
        <div style='width:36px;height:36px;background:linear-gradient(135deg,#E8A320,#F5C355);
                    border-radius:6px;display:flex;align-items:center;justify-content:center;
                    font-size:18px;font-weight:700;color:#0A0A0A'>B</div>
        <div>
            <div style='font-family:Playfair Display,serif;font-size:17px;color:#EDE5CF;font-weight:700'>BookMind</div>
            <div style='font-size:10px;color:#484038;text-transform:uppercase;letter-spacing:.06em'>Hybrid Engine</div>
        </div>
    </div>""", unsafe_allow_html=True)

    page = st.radio("Navigasi", ["🏠 Beranda", "🎯 Rekomendasi", "📊 Analitik", "🧠 Tentang Model"],
                    label_visibility="collapsed")
    st.divider()
    st.markdown("<div style='font-size:11px;color:#484038;text-transform:uppercase;letter-spacing:.06em'>Dataset</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='color:#CCC4B0;font-size:13px'>📚 {len(books):,} buku</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='color:#CCC4B0;font-size:13px'>⭐ {len(ratings):,} ratings</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='color:#CCC4B0;font-size:13px'>👤 {ratings['user_id'].nunique():,} pengguna</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def render_rec_card(row, rank, tag_label, tag_color, score_col):
    score_val = row.get(score_col, None)
    score_str = f"{score_val:.3f}" if score_val is not None and not pd.isna(score_val) else "—"
    rating    = row.get("rating", 0)
    author    = row.get("author", row.get("authors", ""))
    image_url = row.get("image", "")

    img_html = f"<img src='{image_url}' style='width:48px;height:68px;object-fit:cover;border-radius:4px;margin-right:14px;flex-shrink:0'>" if image_url else ""

    st.markdown(f"""
    <div class="rec-card">
        <div style='display:flex;align-items:flex-start'>
            {img_html}
            <div style='flex:1'>
                <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:6px'>
                    <span style='font-size:11px;background:{tag_color}22;color:{tag_color};
                                 border:1px solid {tag_color}44;border-radius:10px;
                                 padding:2px 8px;font-weight:600'>#{rank} {tag_label}</span>
                    <span class="rec-score">⭐ {rating:.2f} · score: {score_str}</span>
                </div>
                <div class="rec-title">{row.get('title','')}</div>
                <div class="rec-meta">{author}</div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

def plotly_dark(fig, height=None):
    layout = dict(
        plot_bgcolor="#111827", paper_bgcolor="#0D1222",
        font_color="#CCC4B0",
    )
    if height:
        layout["height"] = height
    fig.update_layout(**layout)
    fig.update_xaxes(gridcolor="#1C2840")
    fig.update_yaxes(gridcolor="#1C2840")
    return fig

# ─────────────────────────────────────────────
# PAGE: BERANDA
# ─────────────────────────────────────────────
if page == "🏠 Beranda":
    st.markdown("""
    <div style='text-align:center;padding:40px 0 20px'>
        <div style='font-size:11px;letter-spacing:.18em;text-transform:uppercase;color:#E8A320;margin-bottom:14px'>
            Statistical Data Mining — ITS Surabaya
        </div>
        <h1 style='font-size:clamp(28px,4vw,48px);line-height:1.1;margin-bottom:16px'>
            BookMind — <em style='color:#E8A320'>Hybrid</em><br>Recommendation Engine
        </h1>
        <p style='font-size:15px;color:#7A7060;max-width:500px;margin:0 auto 32px;line-height:1.7'>
            Menggabungkan <strong style='color:#CCC4B0'>Content-Based Filtering</strong> (TF-IDF)
            dan <strong style='color:#CCC4B0'>Collaborative Filtering</strong> (SVD)
            untuk rekomendasi buku yang personal.
        </p>
    </div>""", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, val, label in zip(
        [c1, c2, c3, c4],
        [f"{len(books):,}", f"{len(ratings):,}", f"{ratings['user_id'].nunique():,}", f"{books['rating'].mean():.2f}"],
        ["Total Buku", "Total Ratings", "Pengguna Aktif", "Avg Rating"]
    ):
        with col:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-val">{val}</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown("<h2 style='text-align:center;margin-bottom:8px'>Metodologi</h2>", unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    for col, icon, title, desc in zip(
        [m1, m2, m3],
        ["🧬", "👥", "⚡"],
        ["Content-Based Filtering", "Collaborative Filtering", "Weighted Hybrid"],
        [
            "TF-IDF pada judul & penulis, cosine similarity untuk menemukan buku dengan konten serupa. Solusi cold-start untuk item baru.",
            "Model-Based CF via truncated SVD (scipy). Dekomposisi matrix user-item untuk latent factors. Evaluasi: RMSE & MAE.",
            "Score = 0.5×CF + 0.5×CBF. Menutupi kelemahan masing-masing — cold-start dari CBF, serendipity dari CF.",
        ]
    ):
        with col:
            st.markdown(f"""<div class="metric-card" style='text-align:left'>
                <div style='font-size:24px;margin-bottom:10px'>{icon}</div>
                <div style='font-family:Playfair Display,serif;font-size:16px;color:#EDE5CF;margin-bottom:8px'>{title}</div>
                <div style='font-size:13px;color:#7A7060;line-height:1.6'>{desc}</div>
            </div>""", unsafe_allow_html=True)

    # Top 5 buku populer dengan cover
    st.divider()
    st.markdown("<h2 style='text-align:center;margin-bottom:16px'>Buku Terpopuler</h2>", unsafe_allow_html=True)
    top5 = books.nlargest(5, "ratings_count")[["title","author","rating","ratings_count","image"]].reset_index(drop=True)
    cols = st.columns(5)
    for i, (col, row) in enumerate(zip(cols, top5.itertuples())):
        with col:
            img = getattr(row, "image", "")
            if img:
                st.markdown(f"<img src='{img}' style='width:100%;border-radius:6px;margin-bottom:8px'>", unsafe_allow_html=True)
            st.markdown(f"""
            <div style='font-family:Playfair Display,serif;font-size:13px;color:#EDE5CF;margin-bottom:4px'>{row.title[:40]}{'...' if len(row.title)>40 else ''}</div>
            <div style='font-size:11px;color:#7A7060'>{row.author[:30]}</div>
            <div style='font-size:11px;color:#E8A320;margin-top:4px'>⭐ {row.rating:.2f}</div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PAGE: REKOMENDASI
# ─────────────────────────────────────────────
elif page == "🎯 Rekomendasi":
    st.markdown("<h2>Rekomendasi Buku</h2>", unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["👤 Berdasarkan User", "📖 Berdasarkan Buku"])

    # ── TAB 1: USER-BASED ──
    with tab1:
        c1, c2 = st.columns([2, 1])
        with c1:
            user_id = st.selectbox("Pilih User ID", sorted(ratings["user_id"].unique()), key="uid")
        with c2:
            n_recs = st.number_input("Jumlah rekomendasi", 5, 20, 10)

        method = st.radio("Metode", ["⚡ Hybrid", "👥 Collaborative Filtering (SVD)", "🧬 Content-Based (TF-IDF)"],
                          horizontal=True)

        if st.button("🔍 Cari Rekomendasi", key="btn_user"):
            with st.spinner("Menghitung rekomendasi..."):
                user_ratings = ratings[ratings["user_id"] == user_id]
                st.markdown(f"<div style='font-size:13px;color:#7A7060;margin-bottom:16px'>User <strong style='color:#E8A320'>{user_id}</strong> · {len(user_ratings)} buku dirating</div>", unsafe_allow_html=True)

                if "Hybrid" in method:
                    recs, score_col, tag_color, tag_label = hybrid.recommend(user_id, n=int(n_recs)), "hybrid_score", "#E8A320", "Hybrid"
                elif "Collaborative" in method:
                    recs, score_col, tag_color, tag_label = svd.recommend(user_id, ratings, books, n=int(n_recs)), "cf_score", "#2DD4BF", "SVD CF"
                else:
                    top_book = user_ratings.loc[user_ratings["rating"].idxmax(), "book_id"]
                    recs, score_col, tag_color, tag_label = cbf.recommend(top_book, n=int(n_recs)), "cbf_score", "#8B5CF6", "CBF"

                if recs is None or len(recs) == 0:
                    st.warning("Tidak ada rekomendasi.")
                else:
                    # merge image
                    if "image" not in recs.columns:
                        recs = recs.merge(books[["book_id","image"]], on="book_id", how="left")
                    st.markdown(f"<h3>Top {len(recs)} Rekomendasi</h3>", unsafe_allow_html=True)
                    for i, row in enumerate(recs.itertuples()):
                        render_rec_card(row._asdict(), i+1, tag_label, tag_color, score_col)

    # ── TAB 2: BOOK-BASED ──
    with tab2:
        search_query = st.text_input("🔍 Cari judul buku...", placeholder="e.g. Harry Potter, Pride and Prejudice")
        n_similar    = st.slider("Jumlah buku serupa", 5, 15, 6)

        if search_query:
            matches = books[books["title"].str.contains(search_query, case=False, na=False)]
            if matches.empty:
                st.warning("Tidak ada buku ditemukan.")
            else:
                title_options = matches["title"].tolist()
                selected_title = st.selectbox("Pilih dari hasil pencarian:", title_options)
                book_row  = matches[matches["title"] == selected_title].iloc[0]
                book_id   = book_row["book_id"]

                # Show selected book card
                img = book_row.get("image", "")
                st.markdown(f"""
                <div style='background:#111827;border:1px solid #243258;border-radius:10px;
                             padding:16px;margin:12px 0 20px;display:flex;gap:16px;align-items:flex-start'>
                    {'<img src="' + img + '" style="width:60px;border-radius:4px;flex-shrink:0">' if img else ''}
                    <div>
                        <div style='font-size:11px;color:#7A7060;text-transform:uppercase;letter-spacing:.06em;margin-bottom:4px'>Buku Referensi</div>
                        <div style='font-family:Playfair Display,serif;font-size:18px;color:#EDE5CF'>{book_row.get('title','')}</div>
                        <div style='font-size:13px;color:#E8A320;margin-top:4px'>{book_row.get('author','')}</div>
                        <div style='font-size:12px;color:#7A7060;margin-top:4px'>⭐ {book_row.get('rating',0):.2f} · {int(book_row.get('ratings_count',0)):,} ratings</div>
                    </div>
                </div>""", unsafe_allow_html=True)

                if st.button("🔍 Cari Buku Serupa", key="btn_book"):
                    with st.spinner("Mencari..."):
                        recs = cbf.recommend(book_id, n=n_similar)
                        if "image" not in recs.columns:
                            recs = recs.merge(books[["book_id","image"]], on="book_id", how="left")
                        st.markdown("<h3>Buku Serupa (Content-Based)</h3>", unsafe_allow_html=True)
                        for i, row in enumerate(recs.itertuples()):
                            render_rec_card(row._asdict(), i+1, "CBF", "#8B5CF6", "cbf_score")

# ─────────────────────────────────────────────
# PAGE: ANALITIK
# ─────────────────────────────────────────────
elif page == "📊 Analitik":
    st.markdown("<h2>Analitik Dataset</h2>", unsafe_allow_html=True)

    tab_dist, tab_breakdown, tab_author, tab_tren = st.tabs([
        "📚 Distribusi Rating", "⭐ Breakdown Rating", "✍️ Top Penulis", "📈 Tren Publikasi"
    ])

    with tab_dist:
        fig = px.histogram(books, x="rating", nbins=30,
                           title="Distribusi Average Rating per Buku",
                           color_discrete_sequence=["#E8A320"])
        st.plotly_chart(plotly_dark(fig), use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            fig2 = px.histogram(ratings, x="rating", nbins=5,
                                title="Distribusi Rating oleh User (1–5)",
                                color_discrete_sequence=["#2DD4BF"])
            st.plotly_chart(plotly_dark(fig2), use_container_width=True)
        with c2:
            rpu  = ratings.groupby("user_id").size().reset_index(name="count")
            fig3 = px.histogram(rpu, x="count", nbins=40,
                                title="Jumlah Rating per User",
                                color_discrete_sequence=["#8B5CF6"])
            st.plotly_chart(plotly_dark(fig3), use_container_width=True)

    with tab_breakdown:
        # Stacked bar: ratings_1..5 untuk top 20 buku terpopuler
        top20 = books.nlargest(20, "ratings_count")[
            ["title", "ratings_1","ratings_2","ratings_3","ratings_4","ratings_5"]
        ].copy()
        top20["title_short"] = top20["title"].str[:30]
        top20_melt = top20.melt(id_vars="title_short",
                                value_vars=["ratings_1","ratings_2","ratings_3","ratings_4","ratings_5"],
                                var_name="Bintang", value_name="Jumlah")
        top20_melt["Bintang"] = top20_melt["Bintang"].str.replace("ratings_","★")

        fig_br = px.bar(top20_melt, x="Jumlah", y="title_short", color="Bintang",
                        orientation="h", barmode="stack",
                        title="Breakdown Rating (★1–★5) — Top 20 Buku Terpopuler",
                        color_discrete_sequence=["#F43F5E","#F97316","#EAB308","#22C55E","#E8A320"])
        fig_br.update_layout(yaxis={"categoryorder":"total ascending"})
        st.plotly_chart(plotly_dark(fig_br, height=560), use_container_width=True)

    with tab_author:
        author_col  = "author" if "author" in books.columns else "authors"
        top_authors = (
            books.groupby(author_col)
            .agg(books=("book_id","count"), avg_rating=("rating","mean"), total_ratings=("ratings_count","sum"))
            .reset_index().sort_values("total_ratings", ascending=False).head(15)
        )
        fig4 = px.bar(top_authors, x="total_ratings", y=author_col, orientation="h",
                      title="Top 15 Penulis berdasarkan Total Ratings",
                      color="avg_rating",
                      color_continuous_scale=[[0,"#6B4A10"],[0.5,"#E8A320"],[1,"#F5C355"]])
        fig4.update_layout(yaxis={"categoryorder":"total ascending"})
        st.plotly_chart(plotly_dark(fig4, height=520), use_container_width=True)

    with tab_tren:
        decade_df = books[books["year"] > 0].copy()
        decade_df["decade"] = (decade_df["year"] // 10 * 10).astype(int)
        decade_agg = decade_df.groupby("decade").agg(
            count=("book_id","count"), avg_rating=("rating","mean")
        ).reset_index()

        fig5 = go.Figure()
        fig5.add_trace(go.Bar(x=decade_agg["decade"], y=decade_agg["count"],
                              name="Jumlah Buku", marker_color="#1C2840"))
        fig5.add_trace(go.Scatter(x=decade_agg["decade"], y=decade_agg["avg_rating"],
                                  name="Avg Rating", mode="lines+markers",
                                  marker_color="#E8A320", line_color="#E8A320", yaxis="y2"))
        fig5.update_layout(
            title="Buku dan Rating berdasarkan Dekade Publikasi",
            yaxis={"title":"Jumlah Buku","gridcolor":"#1C2840"},
            yaxis2={"title":"Avg Rating","overlaying":"y","side":"right","range":[3,5]},
            legend={"bgcolor":"#111827"}, xaxis_title="Dekade",
        )
        st.plotly_chart(plotly_dark(fig5), use_container_width=True)

        sample = books.sample(min(500, len(books)), random_state=42)
        fig6   = px.scatter(sample, x="ratings_count", y="rating",
                            title="Rating vs Popularitas (sample 500 buku)",
                            color="rating",
                            color_continuous_scale=[[0,"#6B4A10"],[1,"#F5C355"]],
                            opacity=0.7, hover_data=["title"])
        st.plotly_chart(plotly_dark(fig6), use_container_width=True)

# ─────────────────────────────────────────────
# PAGE: TENTANG MODEL
# ─────────────────────────────────────────────
elif page == "🧠 Tentang Model":
    st.markdown("<h2>Tentang Model</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#7A7060'>Arsitektur hybrid recommendation system yang digunakan BookMind.</p>", unsafe_allow_html=True)

    # Diagram alur
    st.markdown("<h3>Alur Hybrid System</h3>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background:#111827;border:1px solid #1C2840;border-radius:12px;padding:28px;margin-bottom:24px'>
        <div style='display:flex;align-items:center;justify-content:center;gap:0;flex-wrap:wrap'>

            <!-- Input -->
            <div style='text-align:center;padding:12px 20px;background:#0D1222;border:1px solid #243258;border-radius:8px;min-width:110px'>
                <div style='font-size:20px'>👤</div>
                <div style='font-size:12px;color:#E8A320;font-weight:600;margin-top:4px'>User ID</div>
                <div style='font-size:10px;color:#484038'>+ Rating History</div>
            </div>

            <div style='color:#484038;font-size:20px;margin:0 8px'>→</div>

            <!-- Split -->
            <div style='display:flex;flex-direction:column;gap:10px'>

                <!-- CBF -->
                <div style='text-align:center;padding:10px 18px;background:#1C0D3A;border:1px solid #4C2D8C;border-radius:8px;min-width:160px'>
                    <div style='font-size:11px;color:#8B5CF6;font-weight:600;text-transform:uppercase;letter-spacing:.05em'>Content-Based</div>
                    <div style='font-size:10px;color:#7A7060;margin-top:2px'>TF-IDF + Cosine Sim</div>
                    <div style='font-size:10px;color:#7A7060'>Anchor: top-rated book</div>
                </div>

                <!-- CF -->
                <div style='text-align:center;padding:10px 18px;background:#0D2030;border:1px solid #0E5E6F;border-radius:8px;min-width:160px'>
                    <div style='font-size:11px;color:#2DD4BF;font-weight:600;text-transform:uppercase;letter-spacing:.05em'>Collaborative</div>
                    <div style='font-size:10px;color:#7A7060;margin-top:2px'>Truncated SVD</div>
                    <div style='font-size:10px;color:#7A7060'>User-Item Matrix</div>
                </div>
            </div>

            <div style='color:#484038;font-size:20px;margin:0 8px'>→</div>

            <!-- Normalisasi -->
            <div style='text-align:center;padding:12px 18px;background:#111827;border:1px solid #1C2840;border-radius:8px;min-width:130px'>
                <div style='font-size:11px;color:#CCC4B0;font-weight:600'>Normalisasi</div>
                <div style='font-size:10px;color:#7A7060;margin-top:4px'>Min-Max [0, 1]</div>
                <div style='font-size:10px;color:#7A7060'>per score</div>
            </div>

            <div style='color:#484038;font-size:20px;margin:0 8px'>→</div>

            <!-- Weighted -->
            <div style='text-align:center;padding:12px 18px;background:#1C1408;border:1px solid #6B4A10;border-radius:8px;min-width:150px'>
                <div style='font-size:11px;color:#E8A320;font-weight:600'>Weighted Sum</div>
                <div style='font-size:10px;color:#7A7060;margin-top:4px'>0.5 × CF</div>
                <div style='font-size:10px;color:#7A7060'>+ 0.5 × CBF</div>
            </div>

            <div style='color:#484038;font-size:20px;margin:0 8px'>→</div>

            <!-- Output -->
            <div style='text-align:center;padding:12px 20px;background:#0D1222;border:1px solid #243258;border-radius:8px;min-width:110px'>
                <div style='font-size:20px'>📚</div>
                <div style='font-size:12px;color:#E8A320;font-weight:600;margin-top:4px'>Top-N</div>
                <div style='font-size:10px;color:#484038'>Rekomendasi</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Perbandingan metode
    st.markdown("<h3>Perbandingan Metode</h3>", unsafe_allow_html=True)
    comparison = pd.DataFrame({
        "Metode":       ["Content-Based (TF-IDF)", "Collaborative Filtering (SVD)", "Hybrid (Weighted 50/50)"],
        "Tipe":         ["Memory-Based CBF", "Model-Based CF", "Hybrid"],
        "Cold Start":   ["✅ Item baru OK", "❌ User & Item baru gagal", "⚠️ Partial (CBF fallback)"],
        "Serendipity":  ["🔴 Rendah", "🟢 Tinggi", "🟡 Sedang"],
        "Skalabilitas": ["🟢 Tinggi", "🟡 Sedang", "🟡 Sedang"],
        "Keunggulan":   [
            "Transparan, tidak perlu data user lain",
            "Temukan pola laten antar user",
            "Gabungkan kelebihan keduanya",
        ],
    })
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    # SVD explanation
    st.markdown("<h3>Cara Kerja SVD</h3>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background:#111827;border:1px solid #1C2840;border-radius:10px;padding:20px;line-height:1.9;font-size:14px;color:#CCC4B0'>
        <strong style='color:#EDE5CF'>1. User-Item Matrix</strong> — Bangun matrix R (users × books), mayoritas kosong (sparse).<br>
        <strong style='color:#EDE5CF'>2. Mean-Centering</strong> — Kurangi rata-rata rating tiap user supaya bias skala hilang.<br>
        <strong style='color:#EDE5CF'>3. Truncated SVD</strong> — Dekomposisi: R ≈ U × Σ × Vt, ambil k=50 faktor terbesar.<br>
        <strong style='color:#EDE5CF'>4. Rekonstruksi per User</strong> — Prediksi = U[u] × Σ × Vt + mean[u], hanya satu baris (hemat RAM).<br>
        <strong style='color:#EDE5CF'>5. Top-N</strong> — Ambil buku belum dirating dengan prediksi tertinggi.
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div style='margin-top:48px;padding:20px;border-top:1px solid #1C2840;
             text-align:center;font-size:12px;color:#484038'>
    BookMind · Statistical Data Mining · Institut Teknologi Sepuluh Nopember Surabaya<br>
    <span style='color:#7A7060'>goodbooks-10k · SVD + TF-IDF Hybrid</span>
</div>
""", unsafe_allow_html=True)