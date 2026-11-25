"""
SBL.MDS_plots — load and plot outputs of the sbl-mds C++ driver.

Classes
-------
MDSPlotStyle  — figure size, dpi, and styling for MDS plots
MDSResult     — holds spectra + 2D/3D embeddings loaded from CSVs
MDSPlotter    — produces scree, 2D / 3D embedding
"""

import csv
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)


# ---------------------------------------------------------------------------
# Plot style configuration
# ---------------------------------------------------------------------------

class MDSPlotStyle:
    """Figure size, dpi and font settings for MDS plots.

    Parameters
    ----------
    figsize : tuple of float, optional
        Explicit (width, height) in inches. Defaults to (6, 4) for the scree
        plot and (6, 6) for the embeddings when None.
    dpi : int
        Resolution for saved figures.
    sample_marker_size, sample_color, sample_alpha
        Style for ordinary (non-landmark) embedded points.
    landmark_marker, landmark_marker_size, landmark_color
        Style for seeding points.
    landmark_label_prefix : str
        Prefix used for sequential landmark annotations in embedding plots.
    reference_marker, reference_marker_size, reference_color
        Style for named non-landmark experiment points.
    xlim, ylim : tuple(float, float), str, or None
        Optional axis limits for the 2D embedding plot. Strings may use
        "min,max" or "min:max".
    font_size, title_size, label_size, tick_size : float
        Font sizes (label/title/tick default to font_size when unset).
    """

    def __init__(self,
                 figsize=None,
                 dpi=300,
                 sample_marker_size=18,
                 sample_color="C0",
                 sample_alpha=0.7,
                 landmark_marker="*",
                 landmark_marker_size=70,
                 landmark_color="C3",
                 landmark_label_prefix="S",
                 reference_marker="D",
                 reference_marker_size=55,
                 reference_color="C2",
                 xlim=None,
                 ylim=None,
                 font_size=10.0,
                 title_size=None,
                 label_size=None,
                 tick_size=None):
        self.figsize = figsize
        self.dpi = dpi
        self.sample_marker_size = sample_marker_size
        self.sample_color = sample_color
        self.sample_alpha = sample_alpha
        self.landmark_marker = landmark_marker
        self.landmark_marker_size = landmark_marker_size
        self.landmark_color = landmark_color
        self.landmark_label_prefix = landmark_label_prefix
        self.reference_marker = reference_marker
        self.reference_marker_size = reference_marker_size
        self.reference_color = reference_color
        self.xlim = _normalize_axis_limit(xlim, "xlim")
        self.ylim = _normalize_axis_limit(ylim, "ylim")
        self.font_size = font_size
        self.title_size = title_size if title_size is not None else font_size
        self.label_size = label_size if label_size is not None else font_size
        self.tick_size = tick_size if tick_size is not None else max(6.0, font_size - 2.0)

    def apply_to_axes(self, ax):
        ax.title.set_fontsize(self.title_size)
        ax.xaxis.label.set_fontsize(self.label_size)
        ax.yaxis.label.set_fontsize(self.label_size)
        for t in ax.get_xticklabels() + ax.get_yticklabels():
            t.set_fontsize(self.tick_size)


def _normalize_axis_limit(value, name):
    if value is None:
        return None
    if isinstance(value, str):
        parts = value.replace(":", ",").split(",")
    else:
        try:
            parts = list(value)
        except TypeError as exc:
            raise ValueError(
                f"{name} must be None, a two-value sequence, or 'min,max'"
            ) from exc
    if len(parts) != 2:
        raise ValueError(
            f"{name} must contain exactly two values: min and max"
        )
    try:
        lo, hi = (float(part) for part in parts)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} values must be numeric") from exc
    if not (np.isfinite(lo) and np.isfinite(hi)):
        raise ValueError(f"{name} values must be finite")
    if lo >= hi:
        raise ValueError(f"{name} minimum must be smaller than maximum")
    return (lo, hi)


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def _read_numeric_vector(path):
    with open(path, "r") as f:
        reader = csv.reader(f)
        next(reader, None)  # header
        values = [float(row[0]) for row in reader if row]
    return np.array(values)


def _read_singular_values(path):
    return _read_numeric_vector(path)


def _read_eigenvalues(path):
    return _read_numeric_vector(path)


def _read_embedding(path):
    """Return (labels, X, is_landmark or None)."""
    labels = []
    coords = []
    is_lm = []
    with open(path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        has_lm = "is_landmark" in header
        n_coord_cols = len(header) - 1 - (1 if has_lm else 0)
        for row in reader:
            if not row:
                continue
            labels.append(row[0])
            coords.append([float(row[1 + j]) for j in range(n_coord_cols)])
            if has_lm:
                is_lm.append(int(row[-1]))
    return labels, np.asarray(coords), (np.asarray(is_lm) if is_lm else None)


# ---------------------------------------------------------------------------
# MDS result container
# ---------------------------------------------------------------------------

class MDSResult:
    """In-memory bundle of the CSVs produced by sbl-mds.exe.

    Attributes
    ----------
    singular_values : np.ndarray
    eigenvalues : np.ndarray
        Signed centered-Gram eigenvalues when `eigenvalues.csv` is present;
        otherwise the legacy fallback `singular_values ** 2`.
    eigenvalues_are_signed : bool
        True when signed eigenvalues were loaded from `eigenvalues.csv`.
    labels_2d : list[str]
    X2 : np.ndarray, shape (n, 2)
    is_landmark_2d : np.ndarray or None
    labels_3d : list[str] or None
    X3 : np.ndarray or None, shape (n, 3)
    is_landmark_3d : np.ndarray or None
    """

    def __init__(self,
                 singular_values,
                 eigenvalues,
                 eigenvalues_are_signed,
                 labels_2d, X2, is_landmark_2d,
                 labels_3d=None, X3=None, is_landmark_3d=None):
        self.singular_values = singular_values
        self.eigenvalues = eigenvalues
        self.eigenvalues_are_signed = eigenvalues_are_signed
        self.labels_2d = labels_2d
        self.X2 = X2
        self.is_landmark_2d = is_landmark_2d
        self.labels_3d = labels_3d
        self.X3 = X3
        self.is_landmark_3d = is_landmark_3d

    @property
    def has_3d(self):
        return self.X3 is not None

    @classmethod
    def from_directory(cls, in_dir):
        sv_path = os.path.join(in_dir, "singular_values.csv")
        ev_path = os.path.join(in_dir, "eigenvalues.csv")
        e2_path = os.path.join(in_dir, "embedding_2d.csv")
        e3_path = os.path.join(in_dir, "embedding_3d.csv")
        if not os.path.isfile(sv_path):
            raise FileNotFoundError(f"Missing {sv_path}")
        if not os.path.isfile(e2_path):
            raise FileNotFoundError(f"Missing {e2_path}")
        return cls.from_files(sv_path, e2_path,
                              e3_path if os.path.isfile(e3_path) else None,
                              ev_path=ev_path if os.path.isfile(ev_path) else None)

    @classmethod
    def from_files(cls, sv_path, e2_path, e3_path=None, ev_path=None):
        sigmas = _read_singular_values(sv_path)
        eigenvalues_are_signed = ev_path is not None
        eigenvalues = (_read_eigenvalues(ev_path)
                       if eigenvalues_are_signed else sigmas ** 2)
        labels2, X2, lm2 = _read_embedding(e2_path)
        if e3_path is not None:
            labels3, X3, lm3 = _read_embedding(e3_path)
        else:
            labels3, X3, lm3 = None, None, None
        return cls(sigmas, eigenvalues, eigenvalues_are_signed,
                   labels2, X2, lm2, labels3, X3, lm3)


# ---------------------------------------------------------------------------
# Plotter
# ---------------------------------------------------------------------------

def _point_masks(labels, is_lm, reference_labels=None):
    n = len(labels)
    landmark_mask = (is_lm.astype(bool) if is_lm is not None
                     else np.zeros(n, dtype=bool))
    reference_set = set(reference_labels or [])
    reference_mask = np.array([lab in reference_set for lab in labels], dtype=bool)
    reference_mask &= ~landmark_mask
    sample_mask = ~(landmark_mask | reference_mask)
    return sample_mask, landmark_mask, reference_mask


def _point_group(index, landmark_mask, reference_mask):
    if landmark_mask[index]:
        return "seeding"
    if reference_mask[index]:
        return "experiment"
    return "sample"


def _plot_label(label, index, landmark_mask, reference_mask, style):
    if landmark_mask[index]:
        landmark_index = int(np.count_nonzero(landmark_mask[:index + 1]))
        return f"{style.landmark_label_prefix}{landmark_index}"
    if reference_mask[index]:
        return label
    return None


def _apply_2d_aspect_and_limits(ax, style):
    limit_adjustment = ("box" if style.xlim is not None or style.ylim is not None
                        else "datalim")
    ax.set_aspect("equal", adjustable=limit_adjustment)
    if style.xlim is not None:
        ax.set_xlim(*style.xlim)
    if style.ylim is not None:
        ax.set_ylim(*style.ylim)


def _histogram_range(X, style):
    x_range = style.xlim
    y_range = style.ylim
    if x_range is None:
        x_min = float(np.min(X[:, 0]))
        x_max = float(np.max(X[:, 0]))
        if x_min == x_max:
            x_min -= 0.5
            x_max += 0.5
        x_range = (x_min, x_max)
    if y_range is None:
        y_min = float(np.min(X[:, 1]))
        y_max = float(np.max(X[:, 1]))
        if y_min == y_max:
            y_min -= 0.5
            y_max += 0.5
        y_range = (y_min, y_max)
    return [x_range, y_range]

class MDSPlotter:
    """Produce scree, embedding, and replot data from an MDSResult."""

    def __init__(self, result, style=None, reference_labels=None):
        self.result = result
        self.style = style if style is not None else MDSPlotStyle()
        self.reference_labels = set(reference_labels or [])

    def _axis_label(self, index):
        label = f"x{index + 1}"
        eigenvalues = self.result.eigenvalues
        if index >= len(eigenvalues):
            return label
        positive = eigenvalues[eigenvalues > 0]
        total = positive.sum()
        value = eigenvalues[index]
        if value <= 0 or total <= 0 or not np.isfinite(total) or not np.isfinite(value):
            return label
        return f"{label} ({100.0 * value / total:.1f}%)"

    def _plot_signed_eigenvalue_scree(self, out_path):
        st = self.style
        eigenvalues = self.result.eigenvalues
        n = len(eigenvalues)
        xs = np.arange(1, n + 1)
        fig, ax1 = plt.subplots(figsize=st.figsize or (6, 4), dpi=st.dpi)

        colors = np.where(eigenvalues >= 0, "C0", "C3")
        ax1.bar(xs, eigenvalues, color=colors, alpha=0.8)
        ax1.axhline(0.0, color="0.25", linewidth=0.8)
        ax1.set_xlabel("component index")
        ax1.set_ylabel("centered-Gram eigenvalue")

        positive = np.where(eigenvalues > 0, eigenvalues, 0.0)
        total_positive = positive.sum()
        if total_positive > 0 and np.isfinite(total_positive):
            cum = np.cumsum(positive) / total_positive
            ax2 = ax1.twinx()
            ax2.plot(xs, cum, "s--", color="C2", label="cumulative positive eig.")
            ax2.set_ylabel("cumulative positive eigenvalue fraction", color="C2")
            ax2.set_ylim(0, 1.02)
            ax2.tick_params(axis="y", labelcolor="C2")

        denom = np.abs(eigenvalues).sum()
        #if denom > 0 and np.isfinite(denom):
        #    negative_mass = np.abs(eigenvalues[eigenvalues < 0].sum()) / denom
        #    ax1.text(0.02, 0.02,
        #             f"negative mass: {100.0 * negative_mass:.2f}%",
        #             transform=ax1.transAxes, fontsize=st.tick_size,
        #             va="bottom", ha="left")

        st.apply_to_axes(ax1)
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)

    # -- individual plots ----------------------------------------------------

    def plot_scree(self, out_path):
        if self.result.eigenvalues_are_signed:
            self._plot_signed_eigenvalue_scree(out_path)
            return

        st = self.style
        sigmas = self.result.singular_values
        n = len(sigmas)
        xs = np.arange(1, n + 1)
        fig, ax1 = plt.subplots(figsize=st.figsize or (6, 4), dpi=st.dpi)
        ax1.plot(xs, sigmas, "o-", color="C0", label="singular value")
        ax1.set_xlabel("component index")
        ax1.set_ylabel("singular value", color="C0")
        ax1.tick_params(axis="y", labelcolor="C0")
        if (sigmas > 0).any():
            ax1.set_yscale("log")

        cum = np.cumsum(sigmas ** 2)
        total = cum[-1] if cum[-1] > 0 else 1.0
        cum = cum / total
        ax2 = ax1.twinx()
        ax2.plot(xs, cum, "s--", color="C3", label="cumulative positive eig.")
        ax2.set_ylabel("cumulative positive eigenvalue fraction", color="C3")
        ax2.set_ylim(0, 1.02)
        ax2.tick_params(axis="y", labelcolor="C3")

        st.apply_to_axes(ax1)
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)

    def plot_embedding_2d(self, out_path, show_labels=False):
        st = self.style
        labels, X, is_lm = self.result.labels_2d, self.result.X2, self.result.is_landmark_2d
        fig, ax = plt.subplots(figsize=st.figsize or (6, 6), dpi=st.dpi)
        sample_mask, landmark_mask, reference_mask = _point_masks(
            labels, is_lm, self.reference_labels
        )
        pts = X[sample_mask]
        lms = X[landmark_mask]
        refs = X[reference_mask]
        if pts.size:
            ax.scatter(pts[:, 0], pts[:, 1],
                       s=st.sample_marker_size, c=st.sample_color,
                       alpha=st.sample_alpha, label="sample")
        if lms.size:
            ax.scatter(lms[:, 0], lms[:, 1],
                       s=st.landmark_marker_size, c=st.landmark_color,
                       marker=st.landmark_marker,
                       edgecolors="k", linewidths=0.5, label="seeding")
        if refs.size:
            ax.scatter(refs[:, 0], refs[:, 1],
                       s=st.reference_marker_size, c=st.reference_color,
                       marker=st.reference_marker,
                       edgecolors="k", linewidths=0.5, label="experiment")
        if show_labels:
            label_mask = landmark_mask | reference_mask
            for index, (lab, (x, y), tag) in enumerate(zip(labels, X, label_mask)):
                if tag:
                    display_label = _plot_label(
                        lab, index, landmark_mask, reference_mask, st
                    )
                    ax.annotate(display_label, (x, y), fontsize=7,
                                xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel(self._axis_label(0))
        ax.set_ylabel(self._axis_label(1))
        _apply_2d_aspect_and_limits(ax, st)
        if lms.size or refs.size:
            ax.legend(loc="best", fontsize=8)
        st.apply_to_axes(ax)
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)

    def plot_embedding_3d(self, out_path, show_labels=False):
        if not self.result.has_3d:
            return
        st = self.style
        labels, X, is_lm = self.result.labels_3d, self.result.X3, self.result.is_landmark_3d
        fig = plt.figure(figsize=st.figsize or (6, 6), dpi=st.dpi)
        ax = fig.add_subplot(111, projection="3d")
        sample_mask, landmark_mask, reference_mask = _point_masks(
            labels, is_lm, self.reference_labels
        )
        pts = X[sample_mask]
        lms = X[landmark_mask]
        refs = X[reference_mask]
        if pts.size:
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                       s=max(st.sample_marker_size - 10, 1),
                       c=st.sample_color,
                       alpha=min(st.sample_alpha, 0.25),
                       depthshade=False,
                       label="sample")
        if lms.size:
            ax.scatter(lms[:, 0], lms[:, 1], lms[:, 2],
                       s=st.landmark_marker_size, c=st.landmark_color,
                       marker=st.landmark_marker,
                       edgecolors="k", linewidths=0.8,
                       depthshade=False,
                       label="seeding")
        if refs.size:
            ax.scatter(refs[:, 0], refs[:, 1], refs[:, 2],
                       s=st.reference_marker_size, c=st.reference_color,
                       marker=st.reference_marker,
                       edgecolors="k", linewidths=0.8,
                       depthshade=False,
                       label="experiment")
        if show_labels:
            label_mask = landmark_mask | reference_mask
            for index, (lab, (x, y, z), tag) in enumerate(zip(labels, X, label_mask)):
                if tag:
                    display_label = _plot_label(
                        lab, index, landmark_mask, reference_mask, st
                    )
                    ax.text(x, y, z, display_label, fontsize=7)
        ax.set_xlabel(self._axis_label(0))
        ax.set_ylabel(self._axis_label(1))
        ax.set_zlabel(self._axis_label(2))
        if lms.size or refs.size:
            ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)


    def _write_embedding_plot_data(self, out_path, labels, X, is_lm):
        sample_mask, landmark_mask, reference_mask = _point_masks(
            labels, is_lm, self.reference_labels
        )
        with open(out_path, "w", newline="") as fh:
            writer = csv.writer(fh)
            axis_columns = [f"x{i + 1}" for i in range(X.shape[1])]
            writer.writerow(["label", "group", *axis_columns,
                             "is_landmark", "is_reference"])
            for index, label in enumerate(labels):
                writer.writerow([
                    label,
                    _point_group(index, landmark_mask, reference_mask),
                    *[f"{value:.17g}" for value in X[index]],
                    int(landmark_mask[index]),
                    int(reference_mask[index]),
                ])

    def _write_scree_plot_data(self, out_path):
        with open(out_path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["component_index", "singular_value", "eigenvalue"])
            n = max(len(self.result.singular_values), len(self.result.eigenvalues))
            for index in range(n):
                sigma = (self.result.singular_values[index]
                         if index < len(self.result.singular_values) else "")
                eigenvalue = (self.result.eigenvalues[index]
                              if index < len(self.result.eigenvalues) else "")
                writer.writerow([
                    index + 1,
                    f"{sigma:.17g}" if sigma != "" else "",
                    f"{eigenvalue:.17g}" if eigenvalue != "" else "",
                ])

    def write_replot_data(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        self._write_scree_plot_data(
            os.path.join(out_dir, "scree_plot_data.csv")
        )
        self._write_embedding_plot_data(
            os.path.join(out_dir, "embedding_2d_plot_data.csv"),
            self.result.labels_2d,
            self.result.X2,
            self.result.is_landmark_2d,
        )
        sample_mask, _, _ = _point_masks(
            self.result.labels_2d,
            self.result.is_landmark_2d,
            self.reference_labels,
        )
        metadata = {
            "legend_labels": {
                "sample": "sample",
                "landmark": "seeding",
                "reference": "experiment",
            },
            "reference_labels": sorted(self.reference_labels),
            "axis_labels_2d": [self._axis_label(0), self._axis_label(1)],
            "scree_plot_data": "scree_plot_data.csv",
            "embedding_2d_plot_data": "embedding_2d_plot_data.csv",
            "style": {
                "figsize": self.style.figsize,
                "dpi": self.style.dpi,
                "sample_marker_size": self.style.sample_marker_size,
                "sample_alpha": self.style.sample_alpha,
                "landmark_marker": self.style.landmark_marker,
                "landmark_marker_size": self.style.landmark_marker_size,
                "reference_marker": self.style.reference_marker,
                "reference_marker_size": self.style.reference_marker_size,
                "xlim": self.style.xlim,
                "ylim": self.style.ylim,
            },
        }
        if self.result.has_3d:
            self._write_embedding_plot_data(
                os.path.join(out_dir, "embedding_3d_plot_data.csv"),
                self.result.labels_3d,
                self.result.X3,
                self.result.is_landmark_3d,
            )
            metadata["axis_labels_3d"] = [
                self._axis_label(0), self._axis_label(1), self._axis_label(2)
            ]
            metadata["embedding_3d_plot_data"] = "embedding_3d_plot_data.csv"
        with open(os.path.join(out_dir, "mds_plot_metadata.json"), "w") as fh:
            json.dump(metadata, fh, indent=2, sort_keys=True)

    # -- batch ---------------------------------------------------------------

    def plot_all(self, out_dir, fmt="pdf", show_labels=False):
        os.makedirs(out_dir, exist_ok=True)
        self.plot_scree(os.path.join(out_dir, f"scree.{fmt}"))
        self.plot_embedding_2d(os.path.join(out_dir, f"embedding_2d.{fmt}"),
                               show_labels=show_labels)
        
        if self.result.has_3d:
            self.plot_embedding_3d(os.path.join(out_dir, f"embedding_3d.{fmt}"),
                                   show_labels=show_labels)
        self.write_replot_data(out_dir)
