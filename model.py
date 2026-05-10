# -*- coding: utf-8 -*-
"""
model.py — BookMind Recommendation System
Dataset : goodbooks-10k (zygmuntz)

Outputs align with BookMind UI (bookmind_dashboard.html):
  - BOOK_MAP  : {id: book_dict}  (id as string, matches UI)
  - SIM       : {id: [similar_ids]}  → content-based (tab "content")
  - CF        : {id: [similar_ids]}  → collaborative filtering (tab "collab")
  - GENRE_STATS, AUTHOR_STATS, etc.

pip install pandas numpy scipy scikit-learn
"""

import json
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

BASE_URL = "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/"

# ─────────────────────────────────────────────
# 1. DATA LOADER
# ─────────────────────────────────────────────

class DataLoader:
    def __init__(self, min_rating_pengguna=5, min_rating_buku=10):
        self.min_rating_pengguna = min_rating_pengguna
        self.min_rating_buku     = min_rating_buku

    def load(self):
        print("Loading data...")
        self.ratings = pd.read_csv(BASE_URL + "ratings.csv")
        self.books   = pd.read_csv(BASE_URL + "books.csv")

        # Filter sparsity
        uc = self.ratings["user_id"].value_counts()
        bc = self.ratings["book_id"].value_counts()
        self.ratings = self.ratings[
            self.ratings["user_id"].isin(uc[uc >= self.min_rating_pengguna].index) &
            self.ratings["book_id"].isin(bc[bc >= self.min_rating_buku].index)
        ].reset_index(drop=True)

        # Clean books — keep columns matching UI fields
        keep = ["book_id", "title", "authors", "original_publication_year",
                "average_rating", "ratings_count", "work_ratings_count",
                "ratings_1", "ratings_2", "ratings_3", "ratings_4", "ratings_5",
                "image_url", "work_text_reviews_count"]
        self.books = self.books[[c for c in keep if c in self.books.columns]].copy()
        self.books.rename(columns={
            "authors":                    "author",
            "original_publication_year":  "year",
            "ratings_count":              "ratings_count",
            "average_rating":             "rating",
            "work_text_reviews_count":    "reviews",
            "image_url":                  "image",
        }, inplace=True)

        self.books["title"]  = self.books["title"].fillna("")
        self.books["author"] = self.books["author"].fillna("")
        self.books["year"]   = self.books["year"].fillna(0).astype(int)

        # Content string for TF-IDF
        self.books["content"] = self.books["title"] + " " + self.books["author"]

        # book_id as string (matches UI keys)
        self.books["id"] = self.books["book_id"].astype(str)

        print(f"  Ratings: {len(self.ratings):,} | "
              f"Users: {self.ratings['user_id'].nunique():,} | "
              f"Books: {self.books['book_id'].nunique():,}")
        return self.ratings, self.books


# ─────────────────────────────────────────────
# 2. SVD — Collaborative Filtering
# ─────────────────────────────────────────────

class SVDRecommender:
    """
    Model-Based CF via truncated SVD (scipy).
    Produces item-item CF similarity used by UI tab 'collab'.
    """
    def __init__(self, n_factors=50):
        self.n_factors = n_factors

    def fit(self, ratings):
        user_ids = sorted(ratings["user_id"].unique())
        book_ids = sorted(ratings["book_id"].unique())
        self.user_index = {u: i for i, u in enumerate(user_ids)}
        self.book_index = {b: i for i, b in enumerate(book_ids)}
        self.book_ids   = book_ids

        rows = ratings["user_id"].map(self.user_index)
        cols = ratings["book_id"].map(self.book_index)
        vals = ratings["rating"].astype(float)

        self.sparse = csr_matrix(
            (vals, (rows, cols)),
            shape=(len(user_ids), len(book_ids))
        )

        # Mean-center per user (stay sparse)
        self.user_means = np.array(self.sparse.mean(axis=1)).flatten()
        sparse_c = self.sparse.copy().astype(float).tocoo()
        sparse_c.data -= self.user_means[sparse_c.row]
        sparse_c = sparse_c.tocsr()

        k = min(self.n_factors, min(self.sparse.shape) - 1)
        self.U, self.sigma, self.Vt = svds(sparse_c, k=k)
        print(f"  SVD done — k={k}, shape: U{self.U.shape} Vt{self.Vt.shape}")
        return self

    def _user_vec(self, ui):
        return self.U[ui] @ np.diag(self.sigma) @ self.Vt + self.user_means[ui]

    def predict(self, user_id, book_id):
        if user_id not in self.user_index or book_id not in self.book_index:
            return None
        pred = self._user_vec(self.user_index[user_id])[self.book_index[book_id]]
        return float(np.clip(pred, 1, 5))

    def recommend(self, user_id, ratings, books, n=10):
        """Top-N unrated books for a user."""
        if user_id not in self.user_index:
            return pd.DataFrame()
        ui    = self.user_index[user_id]
        rated = set(ratings[ratings["user_id"] == user_id]["book_id"])
        vec   = self._user_vec(ui)

        candidates = [(b, vec[self.book_index[b]])
                      for b in books["book_id"]
                      if b in self.book_index and b not in rated]
        candidates.sort(key=lambda x: x[1], reverse=True)
        top = [b for b, _ in candidates[:n]]
        scores = {b: s for b, s in candidates[:n]}

        result = books[books["book_id"].isin(top)][
            ["book_id", "id", "title", "author", "rating"]].copy()
        result["cf_score"] = result["book_id"].map(scores)
        return result.sort_values("cf_score", ascending=False).reset_index(drop=True)

    def item_cf_similarity(self, n_similar=6):
        """
        Item-item CF similarity: for each book, find n most similar via
        dot product of their Vt columns (latent item vectors).
        Returns dict {book_id_str: [similar_book_id_str, ...]}
        matches UI key: CF
        """
        print("  Computing item-item CF similarity...")
        # Vt columns = item latent vectors (shape: k x n_items)
        item_vecs = self.Vt.T  # shape: n_items x k
        # Normalize
        norms = np.linalg.norm(item_vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1
        item_vecs_norm = item_vecs / norms

        # Compute in batches to avoid memory explosion
        batch_size = 500
        n_items    = len(self.book_ids)
        cf_sim     = {}

        for start in range(0, n_items, batch_size):
            end    = min(start + batch_size, n_items)
            batch  = item_vecs_norm[start:end]
            scores = batch @ item_vecs_norm.T  # (batch x n_items)

            for i, row in enumerate(scores):
                book_id  = self.book_ids[start + i]
                row[start + i] = -1  # exclude self
                top_idx  = np.argsort(row)[::-1][:n_similar]
                similar  = [str(self.book_ids[j]) for j in top_idx]
                cf_sim[str(book_id)] = similar

        return cf_sim

    def evaluate(self, ratings, n_folds=5):
        print(f"\nEvaluating SVD ({n_folds}-fold CV)...")
        kf          = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        rmses, maes = [], []

        for fold, (train_idx, test_idx) in enumerate(kf.split(ratings)):
            train_df = ratings.iloc[train_idx]
            test_df  = ratings.iloc[test_idx]

            tmp = SVDRecommender(n_factors=self.n_factors).fit(train_df)

            y_true, y_pred = [], []
            for _, row in test_df.iterrows():
                pred = tmp.predict(row["user_id"], row["book_id"])
                if pred is not None:
                    y_true.append(row["rating"])
                    y_pred.append(pred)

            if y_true:
                rmses.append(np.sqrt(mean_squared_error(y_true, y_pred)))
                maes.append(mean_absolute_error(y_true, y_pred))
                print(f"  Fold {fold+1} — RMSE: {rmses[-1]:.4f} | MAE: {maes[-1]:.4f}")

        print(f"\n  CV RMSE : {np.mean(rmses):.4f} ± {np.std(rmses):.4f}")
        print(f"  CV MAE  : {np.mean(maes):.4f} ± {np.std(maes):.4f}")
        return {
            "RMSE_mean": round(np.mean(rmses), 4),
            "MAE_mean":  round(np.mean(maes), 4),
            "RMSE_std":  round(np.std(rmses), 4),
        }


# ─────────────────────────────────────────────
# 3. CBF — Content-Based Filtering (TF-IDF)
# ─────────────────────────────────────────────

class CBFRecommender:
    """
    Content-Based Filtering using TF-IDF on title + author.
    Produces item-item content similarity used by UI tab 'content'.
    Returns dict {book_id_str: [similar_book_id_str, ...]} matching UI key: SIM
    """
    def __init__(self, max_features=5000):
        self.max_features = max_features

    def fit(self, books):
        self.books = books.reset_index(drop=True)
        tfidf      = TfidfVectorizer(stop_words="english",
                                     max_features=self.max_features)
        matrix     = tfidf.fit_transform(self.books["content"])
        self.sim   = cosine_similarity(matrix, matrix)
        self.idx   = pd.Series(self.books.index, index=self.books["book_id"]).to_dict()
        print(f"  TF-IDF matrix: {matrix.shape}")
        return self

    def recommend(self, book_id, n=10):
        if book_id not in self.idx:
            return pd.DataFrame()
        i      = self.idx[book_id]
        scores = sorted(enumerate(self.sim[i]), key=lambda x: x[1], reverse=True)[1:n+1]
        result = self.books.iloc[[j for j, _ in scores]][
            ["book_id", "id", "title", "author", "rating"]].copy()
        result["cbf_score"] = [s for _, s in scores]
        return result.reset_index(drop=True)

    def item_cbf_similarity(self, n_similar=6):
        """
        Returns dict {book_id_str: [similar_book_id_str, ...]}
        matching UI key: SIM (similarities)
        """
        print("  Building CBF similarity dict...")
        sim_dict = {}
        book_ids = self.books["book_id"].tolist()
        for i, bid in enumerate(book_ids):
            row    = self.sim[i].copy()
            row[i] = -1
            top    = np.argsort(row)[::-1][:n_similar]
            sim_dict[str(bid)] = [str(book_ids[j]) for j in top]
        return sim_dict


# ─────────────────────────────────────────────
# 4. HYBRID RECOMMENDER
# ─────────────────────────────────────────────

class HybridRecommender:
    """
    Weighted hybrid: score = cf_weight * CF_norm + (1-cf_weight) * CBF_norm
    Matches UI tab 'hybrid' — weighted 50/50 as described in bookmind UI.
    """
    def __init__(self, cf_weight=0.5):
        self.cf_weight  = cf_weight
        self.cbf_weight = 1 - cf_weight

    def fit(self, ratings, books):
        self.ratings = ratings
        self.books   = books
        print("Fitting SVDRecommender...")
        self.cf  = SVDRecommender().fit(ratings)
        print("Fitting CBFRecommender...")
        self.cbf = CBFRecommender().fit(books)
        return self

    def recommend(self, user_id, n=10):
        if user_id not in self.cf.user_index:
            return pd.DataFrame()

        ui    = self.cf.user_index[user_id]
        rated = set(self.ratings[self.ratings["user_id"] == user_id]["book_id"])
        vec   = self.cf._user_vec(ui)

        candidates = [b for b in self.books["book_id"]
                      if b in self.cf.book_index and b not in rated]

        # CF scores (normalized)
        cf_raw  = {b: vec[self.cf.book_index[b]] for b in candidates}
        cf_vals = np.array(list(cf_raw.values()))
        cf_min, cf_max = cf_vals.min(), cf_vals.max()
        cf_norm = {b: (cf_raw[b] - cf_min) / (cf_max - cf_min + 1e-8)
                   for b in candidates}

        # CBF: anchor on user's top-rated book
        user_df   = self.ratings[self.ratings["user_id"] == user_id]
        anchor_id = user_df.loc[user_df["rating"].idxmax(), "book_id"]
        if anchor_id in self.cbf.idx:
            ai = self.cbf.idx[anchor_id]
            cbf_scores = {b: self.cbf.sim[ai][self.cbf.idx[b]]
                          if b in self.cbf.idx else 0.0 for b in candidates}
        else:
            cbf_scores = {b: 0.0 for b in candidates}

        hybrid = {b: self.cf_weight * cf_norm[b] + self.cbf_weight * cbf_scores[b]
                  for b in candidates}
        top_n  = sorted(hybrid, key=hybrid.get, reverse=True)[:n]

        result = self.books[self.books["book_id"].isin(top_n)][
            ["book_id", "id", "title", "author", "rating", "ratings_count"]].copy()
        result["hybrid_score"] = result["book_id"].map(hybrid)
        result["cf_score"]     = result["book_id"].map(cf_raw)
        result["cbf_score"]    = result["book_id"].map(cbf_scores)
        result["in_both"]      = result["book_id"].apply(
            lambda b: cf_scores.get(b, 0) > 0 and cbf_scores.get(b, 0) > 0
        ) if hasattr(self, '_cf_scores') else False
        return result.sort_values("hybrid_score", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────
# 5. EXPORT — build RAW JSON for UI
# ─────────────────────────────────────────────

def build_ui_json(books_df, sim_dict, cf_dict, ratings_df, output_path="bookmind_data.json"):
    """
    Build the RAW JSON object that the BookMind UI expects.
    Structure matches: RAW.books, RAW.similarities (SIM), RAW.cf_similar (CF),
    RAW.genre_stats, RAW.author_stats, RAW.genres_list, RAW.decade_summary
    """
    print("\nBuilding UI JSON...")

    # -- BOOKS LIST --
    books_list = []
    for _, row in books_df.iterrows():
        bid = str(int(row["book_id"]))
        books_list.append({
            "id":               bid,
            "goodreads_id":     bid,
            "title":            row.get("title", ""),
            "author":           row.get("author", ""),
            "year":             int(row.get("year", 0)),
            "rating":           round(float(row.get("rating", 0)), 2),
            "ratings_count":    int(row.get("ratings_count", 0)),
            "work_ratings_count": int(row.get("work_ratings_count", 0)) if "work_ratings_count" in row else 0,
            "r1":               int(row.get("ratings_1", 0)) if "ratings_1" in row else 0,
            "r2":               int(row.get("ratings_2", 0)) if "ratings_2" in row else 0,
            "r3":               int(row.get("ratings_3", 0)) if "ratings_3" in row else 0,
            "r4":               int(row.get("ratings_4", 0)) if "ratings_4" in row else 0,
            "r5":               int(row.get("ratings_5", 0)) if "ratings_5" in row else 0,
            "image":            row.get("image", ""),
            "genres":           [],           # enriched via tags file if available
            "primary_genre":    "Fiction",    # default; override if tags available
            "reviews":          int(row.get("reviews", 0)) if "reviews" in row else 0,
        })

    # -- GENRE STATS (computed from ratings if tags not available) --
    genre_stats = {
        "Fiction":     {"count": 0, "avg_rating": 0.0, "total_ratings": 0},
        "Fantasy":     {"count": 0, "avg_rating": 0.0, "total_ratings": 0},
        "Young Adult": {"count": 0, "avg_rating": 0.0, "total_ratings": 0},
        "Classics":    {"count": 0, "avg_rating": 0.0, "total_ratings": 0},
        "Romance":     {"count": 0, "avg_rating": 0.0, "total_ratings": 0},
        "Mystery":     {"count": 0, "avg_rating": 0.0, "total_ratings": 0},
        "Sci-Fi":      {"count": 0, "avg_rating": 0.0, "total_ratings": 0},
        "Thriller":    {"count": 0, "avg_rating": 0.0, "total_ratings": 0},
        "Non-Fiction": {"count": 0, "avg_rating": 0.0, "total_ratings": 0},
        "History":     {"count": 0, "avg_rating": 0.0, "total_ratings": 0},
        "Horror":      {"count": 0, "avg_rating": 0.0, "total_ratings": 0},
        "Biography":   {"count": 0, "avg_rating": 0.0, "total_ratings": 0},
        "Adventure":   {"count": 0, "avg_rating": 0.0, "total_ratings": 0},
        "Children's":  {"count": 0, "avg_rating": 0.0, "total_ratings": 0},
    }

    # -- AUTHOR STATS --
    author_groups = books_df.groupby("author").agg(
        books        = ("book_id",       "count"),
        avg_rating   = ("rating",        "mean"),
        total_ratings= ("ratings_count", "sum"),
    ).reset_index().sort_values("total_ratings", ascending=False).head(15)

    author_stats = [
        {
            "name":          row["author"],
            "books":         int(row["books"]),
            "avg_rating":    round(float(row["avg_rating"]), 2),
            "total_ratings": int(row["total_ratings"]),
        }
        for _, row in author_groups.iterrows()
    ]

    # -- DECADE SUMMARY --
    books_df["decade"] = (books_df["year"] // 10 * 10).astype(int)
    decade_df = books_df[books_df["decade"] > 0].groupby("decade").agg(
        count      = ("book_id", "count"),
        avg_rating = ("rating",  "mean"),
    ).reset_index()
    decade_summary = {
        str(int(row["decade"])): {
            "count":      int(row["count"]),
            "avg_rating": round(float(row["avg_rating"]), 2),
        }
        for _, row in decade_df.iterrows()
    }

    raw = {
        "books":          books_list,
        "similarities":   sim_dict,   # CBF → UI tab: content
        "cf_similar":     cf_dict,    # SVD CF → UI tab: collab
        "genre_stats":    genre_stats,
        "decade_summary": decade_summary,
        "author_stats":   author_stats,
        "genres_list":    list(genre_stats.keys()),
    }

    with open(output_path, "w") as f:
        json.dump(raw, f, ensure_ascii=False)

    print(f"  Saved → {output_path}  ({len(books_list)} books)")
    return raw


# ─────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # --- Load ---
    loader         = DataLoader()
    ratings, books = loader.load()

    # --- SVD (CF) ---
    print("\nFitting SVD...")
    svd = SVDRecommender(n_factors=50).fit(ratings)
    cf_dict = svd.item_cf_similarity(n_similar=6)

    # --- Eval ---
    eval_results = svd.evaluate(ratings)
    print(f"\n  Final → RMSE: {eval_results['RMSE_mean']} | MAE: {eval_results['MAE_mean']}")

    # --- CBF ---
    print("\nFitting CBF...")
    cbf     = CBFRecommender().fit(books)
    sim_dict = cbf.item_cbf_similarity(n_similar=6)

    # --- Export UI JSON ---
    build_ui_json(books, sim_dict, cf_dict, ratings)

    # --- Demo: user recommendation ---
    sample_user = ratings["user_id"].iloc[0]
    print(f"\n── SVD recs for user {sample_user} ──")
    recs = svd.recommend(sample_user, ratings, books, n=5)
    print(recs[["title", "author", "cf_score"]].to_string(index=False))

    print(f"\n── CBF recs for book_id 1 ──")
    print(cbf.recommend(1, n=5)[["title", "author", "cbf_score"]].to_string(index=False))