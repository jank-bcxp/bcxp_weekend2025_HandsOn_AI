import umap
import pandas as pd
import plotly.express as px
import numpy as np


OPENAI_API_KEY = ""


def plot_umap(user_embedding, menu_embeddings, menu, similarity_scores=None):
    """
    Visualisiert Embeddings (2D UMAP) inkl. Benutzeranfrage & Ähnlichkeiten.

    Args:
        user_embedding: 1D embedding (NumPy array or Torch Tensor)
        menu_embeddings: 2D array of embeddings (NumPy or Torch)
        menu: list of menu dicts with 'name'
        similarity_scores: list or array of cosine similarities (optional)
    """
    # 🔁 Torch → NumPy
    if hasattr(user_embedding, "detach"):
        user_embedding = user_embedding.detach().cpu().numpy()
    if hasattr(menu_embeddings, "detach"):
        menu_embeddings = menu_embeddings.detach().cpu().numpy()

    # ➕ Kombinieren
    all_embeddings = np.vstack([user_embedding, menu_embeddings])

    # 📉 UMAP-Projektion
    reducer = umap.UMAP(n_neighbors=5, min_dist=0.3, metric="cosine", random_state=42)
    embeddings_2d = reducer.fit_transform(all_embeddings)

    # 📊 DataFrame aufbauen
    df = pd.DataFrame(embeddings_2d, columns=["x", "y"])
    df["type"] = ["Anfrage"] + ["Gericht"] * len(menu)

    if similarity_scores is not None:
        # similarity_scores können float, torch.tensor oder np.float sein
        sim_scores = [
            s.item() if hasattr(s, "item") else float(s) for s in similarity_scores
        ]
        df["similarity"] = [None] + sim_scores
        df["label"] = ["Benutzeranfrage"] + [
            f"{item['name']} ({sim_scores[i]:.2f})" for i, item in enumerate(menu)
        ]
    else:
        df["similarity"] = [None] + [None] * len(menu)
        df["label"] = ["Benutzeranfrage"] + [item["name"] for item in menu]

    # 📈 Interaktives Plotly
    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="type",
        text="label",
        color_discrete_map={"Anfrage": "blue", "Gericht": "red"},
        hover_data={"similarity": True},
        title="🍕 Embedding-Projektion mit UMAP",
    )
    fig.update_traces(textposition="top center")
    fig.update_layout(
        height=600,
        legend=dict(orientation="h", y=1.05, x=1, xanchor="right", yanchor="bottom"),
    )
    fig.show()
