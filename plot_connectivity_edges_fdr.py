from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from nilearn import plotting as niplot


# Load centralized configuration
from config_loader import load_config

try:
    config = load_config()
    PROJECT_ROOT = config.project_root
    SCHEAFER_CENTROIDS = (
        PROJECT_ROOT
        / "eeg_pipeline"
        / "source_data"
        / "Schaefer2018"
        / "Schaefer2018_100Parcels_7Networks_order_FSLMNI152_2mm.Centroid_RAS.csv"
    )
except ImportError:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    SCHEAFER_CENTROIDS = (
        PROJECT_ROOT
        / "eeg_pipeline"
        / "source_data"
        / "Schaefer2018"
        / "Schaefer2018_100Parcels_7Networks_order_FSLMNI152_2mm.Centroid_RAS.csv"
    )

NETWORK_COLORS: Dict[str, str] = {
    "Vis": "#1f77b4",         # blue
    "SomMot": "#ff7f0e",      # orange
    "DorsAttn": "#2ca02c",    # green
    "SalVentAttn": "#d62728", # red
    "Limbic": "#9467bd",     # purple
    "Cont": "#8c564b",       # brown
    "Default": "#7f7f7f",    # gray
}


def _load_centroids(csv_path: Path) -> Dict[str, Tuple[float, float, float]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Schaefer2018 centroid CSV not found at {csv_path}")
    df = pd.read_csv(csv_path)
    # Expect columns: ROI Label, ROI Name, R, A, S
    name_col = "ROI Name" if "ROI Name" in df.columns else "ROI_Name"
    x_col = "R"; y_col = "A"; z_col = "S"
    mapping: Dict[str, Tuple[float, float, float]] = {}
    for _, row in df.iterrows():
        name = str(row[name_col])
        try:
            x, y, z = float(row[x_col]), float(row[y_col]), float(row[z_col])
            mapping[name] = (x, y, z)
        except Exception:
            continue
    return mapping


def _parse_network(label: str) -> str:
    # Examples: 7Networks_RH_DorsAttn_Post_3 -> network=DorsAttn
    parts = label.split("_")
    if len(parts) >= 3:
        return parts[2]
    return "Unknown"


def _parse_hemi(label: str) -> str:
    parts = label.split("_")
    if len(parts) >= 2:
        return parts[1]
    return "NA"


def build_adjacency_from_edges(df: pd.DataFrame,
                               centroids: Dict[str, Tuple[float, float, float]],
                               select: str = "reject",
                               weight: str = "r",
                               drop_missing: bool = True) -> Tuple[np.ndarray, List[str], List[Tuple[float,float,float]]]:
    """
    Parameters
    ----------
    df : DataFrame with columns node_i, node_j, r, p, fdr_reject, fdr_crit_p
    centroids : mapping ROI Name -> (x,y,z)
    select : 'reject' to use fdr_reject==True, or 'critp' to use p <= fdr_crit_p
    weight : 'r' or 'abs_r'
    drop_missing : if True, skip edges with unknown ROI coordinates
    Returns
    -------
    adj : (n_nodes, n_nodes) symmetric adjacency with weights
    names : list of ROI names in the same order as adj rows/cols
    coords : list of 3-tuples of node coordinates aligned with names
    """
    # Determine significant edges
    if select not in {"reject", "critp"}:
        raise ValueError("select must be 'reject' or 'critp'")

    if select == "reject":
        sig_mask = df.get("fdr_reject", pd.Series([False]*len(df))).astype(bool).to_numpy()
        crit = df.get("fdr_crit_p", pd.Series([np.nan]*len(df)))
    else:
        # Use p <= fdr_crit_p per-row (handles files that include top-K edges)
        if "p" not in df.columns or "fdr_crit_p" not in df.columns:
            raise ValueError("DataFrame must contain 'p' and 'fdr_crit_p' for select='critp'")
        sig_mask = (pd.to_numeric(df["p"], errors="coerce") <= pd.to_numeric(df["fdr_crit_p"], errors="coerce")).to_numpy()
        crit = df["fdr_crit_p"]

    df_sig = df.loc[sig_mask].copy()
    if df_sig.empty:
        raise ValueError("No edges pass the selected FDR criterion in the provided TSV.")

    # Gather unique nodes present
    node_set: List[str] = []
    for col in ["node_i", "node_j"]:
        if col not in df_sig.columns:
            raise ValueError("Input TSV must have columns 'node_i' and 'node_j'")
    for i, row in df_sig.iterrows():
        ni = str(row["node_i"]) ; nj = str(row["node_j"]) 
        for n in (ni, nj):
            if n not in node_set:
                node_set.append(n)

    # Filter nodes that have coordinates
    if drop_missing:
        node_set = [n for n in node_set if n in centroids]

    if not node_set:
        raise ValueError("None of the nodes in the significant edges have centroid coordinates.")

    n = len(node_set)
    name_to_idx = {name: i for i, name in enumerate(node_set)}
    adj = np.zeros((n, n), dtype=float)

    val_col = weight if weight in df_sig.columns else "r"
    vals = pd.to_numeric(df_sig[val_col], errors="coerce").to_numpy()

    for (_, row), w in zip(df_sig.iterrows(), vals):
        ni = str(row["node_i"]) ; nj = str(row["node_j"]) 
        if ni not in name_to_idx or nj not in name_to_idx:
            continue
        i = name_to_idx[ni] ; j = name_to_idx[nj]
        # Symmetrize
        adj[i, j] = w
        adj[j, i] = w

    coords = [centroids[name] for name in node_set]
    return adj, node_set, coords


def _pretty_title_from_stem(stem: str, select: str, weight: str, n_edges: int, n_nodes: int) -> str:
    """Create a readable title from a stats TSV stem.
    Attempts to parse patterns like 'corr_stats_edges_wpli_gamma_vs_rating_top20'.
    """
    toks = stem.split("_")
    try:
        e_idx = toks.index("edges")
    except ValueError:
        e_idx = -1
    measure = toks[e_idx + 1] if e_idx != -1 and e_idx + 1 < len(toks) else None
    band = toks[e_idx + 2] if e_idx != -1 and e_idx + 2 < len(toks) else None
    target = None
    if "vs" in toks:
        v_idx = toks.index("vs")
        target = toks[v_idx + 1] if v_idx + 1 < len(toks) else None
    parts: List[str] = []
    parts.append(f"FDR-significant edges ({select})")
    mb = " ".join([p for p in [measure, band] if p])
    if mb:
        parts.append(mb)
    if target:
        parts.append(f"vs {target}")
    parts.append(f"n_edges={n_edges}, n_nodes={n_nodes}, weight={weight}")
    return " — ".join(parts)


def plot_edges_connectome(tsv_path: Path,
                          output_base: Path | None = None,
                          select: str = "reject",
                          weight: str = "r",
                          node_color_mode: str = "network",
                          node_size_mode: str = "degree",
                          edge_cmap: str = "coolwarm",
                          edge_color: str = "#1f77b4",
                          edge_width: Optional[float] = None,
                          edge_alpha: float = 0.85,
                          node_outline_width: float = 0.6,
                          display_mode: str = "ortho",
                          legend: bool = False,
                          colorbar: bool = False,
                          custom_title: str | None = None,
                          centroids_csv: Path = SCHEAFER_CENTROIDS,
                          dpi: int = 300) -> List[Path]:
    tsv_path = tsv_path.resolve()
    df = pd.read_csv(tsv_path, sep="\t")

    centroids = _load_centroids(centroids_csv)
    adj, names, coords = build_adjacency_from_edges(df, centroids, select=select, weight=weight)

    # Node colors
    if node_color_mode == "network":
        nets = [_parse_network(n) for n in names]
        colors = [NETWORK_COLORS.get(net, "#aaaaaa") for net in nets]
    elif node_color_mode == "hemisphere":
        hemis = [_parse_hemi(n) for n in names]
        colors = ["#1f77b4" if h == "LH" else "#ff7f0e" for h in hemis]
    else:
        # No explicit coloring: let nilearn use defaults
        colors = None

    # Node sizes
    if node_size_mode == "degree":
        deg = np.count_nonzero(adj != 0, axis=0)
        # Scale: base 30 + 10*deg
        sizes = (30 + 10 * deg).astype(float)
    else:
        sizes = 30.0

    # Counts for title/summary
    n_nodes = adj.shape[0]
    n_edges = int(np.count_nonzero(np.triu(adj != 0, 1)))
    # Title
    title = custom_title or _pretty_title_from_stem(tsv_path.stem, select, weight, n_edges, n_nodes)

    vmax = np.nanmax(np.abs(adj)) if np.isfinite(adj).any() else 1.0

    display = niplot.plot_connectome(
        adjacency_matrix=adj,
        node_coords=coords,
        node_color=colors,
        node_size=sizes,
        edge_threshold=None,  # plot all non-zero edges
        edge_cmap=edge_cmap,
        title=None,  # we'll set a clean suptitle instead of black box
        display_mode=display_mode,
        annotate=False,
        colorbar=False,
        edge_kwargs={**({"linewidth": float(edge_width)} if edge_width is not None else {}),
                     "alpha": float(edge_alpha), "color": edge_color},
        node_kwargs={"linewidths": float(node_outline_width), "edgecolors": "k"} if node_outline_width > 0 else {},
    )

    # Matplotlib figure handle
    fig = display.frame_axes.figure

    # Optional colorbar (off by default)
    if colorbar:
        vmin = 0.0 if weight == "abs_r" else -vmax
        norm = Normalize(vmin=vmin, vmax=vmax)
        sm = ScalarMappable(norm=norm)
        sm.set_cmap(edge_cmap)
        try:
            cbar = fig.colorbar(sm, ax=fig.axes, fraction=0.03, pad=0.04, aspect=30)
        except Exception:
            cbar = fig.colorbar(sm, fraction=0.03, pad=0.04, aspect=30)
        cbar_label = "Edge weight (r)" if weight == "r" else "Edge weight (|r|)"
        try:
            cbar.set_label(cbar_label)
        except Exception:
            try:
                cbar.ax.set_ylabel(cbar_label)
            except Exception:
                pass

    # Clean title as suptitle
    fig.suptitle(title, y=0.99, fontsize=12)

    # Optional legend explaining node colors
    if legend:
        if node_color_mode == "network":
            cats = []
            for n in names:
                cats.append(_parse_network(n))
            cats_present = []
            for c in cats:
                if c not in cats_present:
                    cats_present.append(c)
            handles = [
                Line2D([0], [0], marker='o', color='w', label=cat,
                       markerfacecolor=NETWORK_COLORS.get(cat, "#aaaaaa"),
                       markeredgecolor='k', markeredgewidth=0.3, markersize=6)
                for cat in cats_present
            ]
            if handles:
                fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(0.985, 0.985),
                           fontsize=8, title='Node color: Yeo-7 network', frameon=True, framealpha=0.8)
        elif node_color_mode == "hemisphere":
            handles = [
                Line2D([0], [0], marker='o', color='w', label='LH',
                       markerfacecolor="#1f77b4", markeredgecolor='k', markeredgewidth=0.3, markersize=6),
                Line2D([0], [0], marker='o', color='w', label='RH',
                       markerfacecolor="#ff7f0e", markeredgecolor='k', markeredgewidth=0.3, markersize=6),
            ]
            fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(0.985, 0.985),
                       fontsize=8, title='Node color: hemisphere', frameon=True, framealpha=0.8)

    # Determine output base path
    if output_base is None:
        # Default: next to the TSV, under a sibling 'plots' directory if available
        # e.g., .../eeg/stats/foo.tsv -> .../eeg/plots/<same_stem>
        if tsv_path.parent.name == "stats" and tsv_path.parent.parent.name == "eeg":
            plots_dir = tsv_path.parent.parent / "plots" / "04_behavior_feature_analysis"
            plots_dir.mkdir(parents=True, exist_ok=True)
            output_base = plots_dir / tsv_path.stem
        else:
            output_base = tsv_path.with_suffix("")

    saved: List[Path] = []
    for ext in (".png", ".svg"):
        out = Path(f"{output_base}{ext}")
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        saved.append(out)
    plt.close(fig)
    return saved


def main():
    p = argparse.ArgumentParser(description="Plot FDR-significant connectivity edges from a TSV using nilearn plot_connectome.")
    p.add_argument("--tsv", type=str, required=True, help="Path to corr_stats_edges_*.tsv.")
    p.add_argument("--select", type=str, default="reject", choices=["reject", "critp"],
                   help="Selection rule: 'reject' = fdr_reject==True; 'critp' = p <= fdr_crit_p.")
    p.add_argument("--weight", type=str, default="r", choices=["r", "abs_r"],
                   help="Edge weight to visualize: signed r or absolute abs_r.")
    p.add_argument("--centroids", type=str, default=str(SCHEAFER_CENTROIDS),
                   help="Path to Schaefer2018 centroid CSV (ROI Name,R,A,S).")
    p.add_argument("--node-color", type=str, default="network", choices=["network", "hemisphere", "none"],
                   help="Color nodes by Yeo network, hemisphere, or no coloring.")
    p.add_argument("--node-size", type=str, default="degree", choices=["degree", "const"],
                   help="Size nodes by graph degree or as a constant size.")
    p.add_argument("--edge-cmap", type=str, default="coolwarm", help="Matplotlib colormap for edges (only used if colorbar is enabled).")
    p.add_argument("--edge-color", type=str, default="#1f77b4", help="Uniform edge color (hex or named color).")
    p.add_argument("--edge-width", type=float, default=None, help="Edge line width (omit to use nilearn default).")
    p.add_argument("--edge-alpha", type=float, default=0.85, help="Edge transparency (0-1).")
    p.add_argument("--node-outline-width", type=float, default=0.6, help="Outline width for node markers (0 to disable).")
    p.add_argument("--display-mode", type=str, default="ortho", help="nilearn display_mode, e.g. 'ortho', 'x', 'y', 'z', 'xz', 'yz'.")
    p.add_argument("--no-legend", action="store_true", help="Disable legend explaining node colors.")
    p.add_argument("--colorbar", action="store_true", help="Enable colorbar showing edge weight scale.")
    p.add_argument("--title", type=str, default=None, help="Custom title; otherwise derived from filename.")
    p.add_argument("--output", type=str, default=None,
                   help="Optional output base path (without extension). Defaults near the TSV.")
    args = p.parse_args()

    saved = plot_edges_connectome(
        tsv_path=Path(args.tsv),
        output_base=Path(args.output) if args.output else None,
        select=args.select,
        weight=args.weight,
        node_color_mode=args.node_color,
        node_size_mode=args.node_size,
        edge_cmap=args.edge_cmap,
        edge_color=args.edge_color,
        edge_width=args.edge_width,
        edge_alpha=args.edge_alpha,
        node_outline_width=args.node_outline_width,
        display_mode=args.display_mode,
        legend=not args.no_legend and False,
        colorbar=args.colorbar,
        custom_title=args.title,
        centroids_csv=Path(args.centroids),
    )
    print("Saved:")
    for pth in saved:
        print(" -", pth)


if __name__ == "__main__":
    main()
