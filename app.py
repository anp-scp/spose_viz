"""
SPoSE Embedding Explorer — THINGS 1854 Objects
Three tabs: by object, by concept, overall heatmap.
"""

import io
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import dropbox
from PIL import Image

# ── Paths ─────────────────────────────────────────────────────────────────────
EMB_PATH    = "data/spose_embedding_66d_sorted.txt"
NAMES_PATH  = "data/unique_id.txt"
LABELS_PATH = "data/labels.txt"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SPoSE Explorer",
    page_icon="🧠",
    layout="wide",
)
st.title("SPoSE Embedding Explorer")


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading embeddings…")
def load_data():
    embeddings = np.loadtxt(EMB_PATH)

    with open(NAMES_PATH) as f:
        names = [ln.strip() for ln in f if ln.strip()]

    with open(LABELS_PATH) as f:
        labels = [ln.strip() for ln in f if ln.strip()]

    n_dims = embeddings.shape[1]
    if len(labels) < n_dims:
        labels += [f"Dim_{i}" for i in range(len(labels), n_dims)]
    labels = labels[:n_dims]

    return embeddings, names, labels


embeddings, names, labels = load_data()
n_objects, n_dims = embeddings.shape


# ── Dropbox client — refresh token never expires; SDK auto-renews access token ─
@st.cache_resource(show_spinner=False)
def _dropbox_client() -> dropbox.Dropbox:
    cfg = st.secrets["dropbox"]
    return dropbox.Dropbox(
        oauth2_refresh_token=cfg["refresh_token"],
        app_key=cfg["app_key"],
        app_secret=cfg["app_secret"],
    )


# ── Dropbox image fetch ───────────────────────────────────────────────────────
@st.cache_data(show_spinner="Fetching image…", ttl=3600)
def fetch_image(object_name: str) -> Image.Image | None:
    try:
        folder = st.secrets["dropbox"].get("folder_path", "")
        path = f"{folder}/{object_name}.jpg"
        _, res = _dropbox_client().files_download(path)
        return Image.open(io.BytesIO(res.content))
    except dropbox.exceptions.ApiError:
        return None
    except Exception as e:
        st.warning(f"Could not load image: {e}")
        return None


# ── Shared helpers ────────────────────────────────────────────────────────────
def top_concepts(vec: np.ndarray, threshold: float) -> list[tuple[str, float]]:
    """Return (label, value) pairs above threshold, sorted descending."""
    pairs = [(labels[i], float(vec[i])) for i in range(n_dims) if vec[i] > threshold]
    return sorted(pairs, key=lambda x: x[1], reverse=True)


def single_object_heatmap(vec: np.ndarray, object_name: str) -> go.Figure:
    """66×1 interactive heatmap for a single object's embedding (concepts on y-axis)."""
    z = vec.reshape(-1, 1)   # shape (66, 1)
    fig = go.Figure(go.Heatmap(
        z=z,
        x=[object_name],
        y=labels,
        colorscale="YlOrRd",
        colorbar=dict(title="Value"),
        hovertemplate="Concept: %{y}<br>Value: %{z:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title=f"SPoSE embedding — {object_name}",
        xaxis=dict(tickfont=dict(size=11), title="", side="top"),
        yaxis=dict(
            tickfont=dict(size=10),
            title="SPoSE Concept",
            autorange="reversed",
            tickmode="array",
            tickvals=labels,
            ticktext=labels,
        ),
        height=max(500, len(labels) * 22 + 100),
        margin=dict(l=220, r=20, t=50, b=40),
    )
    return fig


def render_object_view(object_name: str, threshold: float):
    """Render the full single-object panel (image | top concepts, then heatmap)."""
    toc_col, main_col = st.columns([1, 6])

    with toc_col:
        st.markdown(
            """**Contents**
- [🖼️ Image](#image)
- [📋 Top concepts](#top-active-concepts)
- [📊 Embedding heatmap](#embedding-heatmap)
""",
            unsafe_allow_html=False,
        )

    with main_col:
        img_col, concept_col = st.columns([1, 2])

        with img_col:
            st.subheader("Image", anchor="image")
            st.caption(object_name.replace("_", " ").title())
            img = fetch_image(object_name)
            if img is not None:
                st.image(img, use_container_width=True)
            else:
                st.info("Image not available in Dropbox.")

        with concept_col:
            st.subheader("Top active concepts", anchor="top-active-concepts")
            pairs = top_concepts(embeddings[names.index(object_name)], threshold)
            # with st.container(height=300):
            if pairs:
                st.dataframe(
                    {"Concept": [p[0] for p in pairs], "Score": [round(p[1], 4) for p in pairs]},
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info(f"No concepts above threshold {threshold:.2f}.")

        st.subheader("Embedding heatmap", anchor="embedding-heatmap")
        fig = single_object_heatmap(embeddings[names.index(object_name)], object_name)
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["🔍 By Object", "💡 By Concept", "📊 Overall Heatmap"])


# ── TAB 1 — Visualize by object ───────────────────────────────────────────────
with tab1:
    st.header("Visualize by Object")

    col_sel, col_thr = st.columns([3, 1])
    with col_sel:
        selected_object = st.selectbox(
            "Object",
            options=names,
            format_func=lambda n: n.replace("_", " ").title(),
            key="tab1_object",
        )
    with col_thr:
        threshold_1 = st.slider(
            "Concept threshold",
            min_value=0.0, max_value=1.0, value=0.2, step=0.01,
            key="tab1_threshold",
        )

    st.divider()

    if selected_object:
        render_object_view(selected_object, threshold_1)


# ── TAB 2 — Visualize by concept ─────────────────────────────────────────────
with tab2:
    st.header("Visualize by Concept")

    col_c, col_t = st.columns([3, 1])
    with col_c:
        selected_concept = st.selectbox(
            "Concept",
            options=labels,
            key="tab2_concept",
        )
    with col_t:
        threshold_2 = st.slider(
            "Concept threshold",
            min_value=0.0, max_value=1.0, value=0.2, step=0.01,
            key="tab2_threshold",
        )

    # Filter objects whose selected concept is above threshold
    concept_idx = labels.index(selected_concept)
    concept_vals = embeddings[:, concept_idx]
    filtered_mask = concept_vals > threshold_2
    filtered_names = [names[i] for i in np.where(filtered_mask)[0]]

    if not filtered_names:
        st.info(f"No objects with **{selected_concept}** > {threshold_2:.2f}.")
    else:
        st.caption(
            f"{len(filtered_names)} objects with **{selected_concept}** > {threshold_2:.2f}, "
            f"sorted by embedding value."
        )
        # Sort by concept value descending
        filtered_names = sorted(
            filtered_names, key=lambda n: concept_vals[names.index(n)], reverse=True
        )

        selected_object_2 = st.selectbox(
            "Object",
            options=filtered_names,
            format_func=lambda n: n.replace("_", " ").title(),
            key="tab2_object",
        )

        st.divider()
        
        if selected_object_2:
            render_object_view(selected_object_2, threshold_2)


# ── TAB 3 — Overall heatmap ───────────────────────────────────────────────────
with tab3:
    st.header("Overall Embedding Heatmap")

    col_n, col_seed = st.columns(2)
    with col_n:
        n_show = st.slider(
            "Number of randomly sampled objects to display",
            min_value=10, max_value=200, value=80, step=10,
            key="tab3_n_show",
            help="Randomly sample this many objects out of 1854 to show in the heatmap (for readability)."
        )
    

    st.caption(
            "When the number of objects to sample is large. Not all object names"
            "will fit on the x-axis, but hovering over columns will show the full name."
        )

    
    rng = np.random.default_rng(42)
    idx = np.sort(rng.choice(n_objects, size=n_show, replace=False))

    subset = embeddings[idx]          # (n_show, 66)
    snames = [names[i] for i in idx]
    short_names = [n.replace("_", " ") for n in snames]

    # Dimensions as rows, objects as columns — same orientation as original script
    fig3 = go.Figure(go.Heatmap(
        z=subset.T,                   # shape (66, n_show)
        x=short_names,
        y=labels,
        colorscale="YlOrRd",
        colorbar=dict(title="Value"),
        hovertemplate="Object: %{x}<br>Concept: %{y}<br>Value: %{z:.4f}<extra></extra>",
    ))
    fig3.update_layout(
        
        xaxis=dict(
            title="Objects (sampled)",
            tickangle=-70,
            tickfont=dict(size=9),
            side="top",
        ),
        yaxis=dict(
            title="SPoSE Concept",
            tickfont=dict(size=9),
            autorange="reversed",
            tickmode="array",
            tickvals=labels,
            ticktext=labels,
        ),
        height=max(500, len(labels) * 22 + 200),
        margin=dict(l=200, r=20, t=60, b=160),
    )
    st.plotly_chart(fig3, use_container_width=True)
