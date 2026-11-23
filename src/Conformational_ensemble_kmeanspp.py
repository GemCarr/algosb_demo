"""
SBL.Conformational_ensemble_kmeanspp -- Kmeans++ seeding based clustering of
a conformational ensemble, and its embedding via multi-dimensional scaling.

This module drives two executables:

- ``sbl-conf-ensemble-kmeanspp.exe`` (package Conformational_ensemble_analysis):
  Kmeans++ seeding (EGD), optionally followed by EGD reselection (EGD-EGD), of
  the conformations stored in a multi-model PDB file, using the squared lRMSD
  or the squared Euclidean distance as cost.  Each conformation is assigned to
  its nearest seed, which defines the clusters.
- ``sbl-mds.exe`` (package Dimensionality_reduction): classical or landmark
  multi-dimensional scaling, the seeds being used as landmarks.

Classes
-------
ConformationalEnsembleKmeansppRunner
    ``run_seeding`` / ``analyze_seeding`` (k-scan, elbow detection, seeds,
    cluster assignments) and ``run_mds`` / ``analyze_mds`` (scree plot and
    2D/3D scatter plots).
"""

from __future__ import annotations

import csv
import html
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from IPython.display import HTML, Image, Markdown, display


DEFAULT_PLOT_STYLE: dict = {
    "figsize":          "6x4",
    "dpi":              150,
    "font_size":        8,
    "title_size":       10,
    "label_size":       8,
    "tick_size":        8,
    "marker_size":      2,
    "line_width":       0.5,
    "hide_xticklabels": True,
}


class ConformationalEnsembleKmeansppRunner:
    """Kmeans++ seeding based clustering and MDS embedding of an ensemble.

    Parameters
    ----------
    root_results : path-like or None
        Base directory for output sub-directories.  Defaults to a
        ``results/`` folder in the current working directory.
    plot_style : dict or None
        Default plot-style overrides (see :data:`DEFAULT_PLOT_STYLE`).
    verbose : bool
        Print stage banners and output paths.
    display_notebook : bool
        Display tables and figures with IPython.
    """

    KMEANSPP_EXE = "sbl-conf-ensemble-kmeanspp.exe"
    MDS_EXE = "sbl-mds.exe"

    def __init__(
        self,
        root_results: Optional[os.PathLike] = None,
        plot_style: Optional[dict] = None,
        verbose: bool = True,
        display_notebook: bool = True,
    ) -> None:
        self.root_results = Path(root_results).resolve() if root_results else (
            Path.cwd() / "results"
        )
        self.root_results.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        self.display_notebook = display_notebook

        self.plot_style: dict = dict(DEFAULT_PLOT_STYLE)
        if plot_style:
            self.plot_style.update(plot_style)

        # ``clustering_exe`` keeps the attribute name used by LoopSamplingRunner.
        self.clustering_exe: Optional[str] = self._find_exe(
            self.KMEANSPP_EXE, required=True
        )
        self.mds_exe: Optional[str] = self._find_exe(
            self.MDS_EXE, required=True
        )

    # -------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------

    @staticmethod
    def _find_exe(name: str, required: bool = True) -> Optional[str]:
        if required:
            return Path(os.getcwd() + "/../src/" + name).resolve()
        else:
            return None

    def _print_stage(self, stage: str) -> None:
        if self.verbose:
            print(f"\nStage: {stage}")

    def _print_output(self, label: str, path: os.PathLike) -> None:
        if self.verbose:
            print(f"  {label}: {Path(path)}")

    def _display(self, obj) -> None:
        if self.display_notebook:
            display(obj)

    def _effective_style(self, override: Optional[dict]) -> dict:
        """Merge instance default style with a per-call override."""
        merged = dict(self.plot_style)
        if override:
            merged.update(override)
        return merged

    @staticmethod
    def _normalize_figure_formats(figure_formats) -> tuple[str, ...]:
        """Normalize requested figure suffixes while preserving order."""
        formats = tuple(figure_formats or ("png",))
        normalized: list[str] = []
        for fmt in formats:
            suffix = str(fmt).lower().lstrip(".")
            if suffix not in ("png", "svg", "pdf"):
                raise ValueError(
                    f"Unsupported figure format {fmt!r}; expected png, svg, or pdf"
                )
            if suffix not in normalized:
                normalized.append(suffix)
        return tuple(normalized)

    # -------------------------------------------------------------------
    # Kmeans++ seeding based clustering
    # -------------------------------------------------------------------

    def run_seeding(
        self,
        ifpath: os.PathLike,
        resids: str,
        *,
        odpath: Optional[os.PathLike] = None,
        k: Optional[int] = None,
        scan_k_max: int = 20,
        method: str = "EGD-EGD",
        metric: str = "lrmsd",
        random_seed: int = 42,
        n_local_trials: int = 8,
        save_pdbs: bool = False,
        atoms: Optional[str] = None,
        chainid: Optional[str] = None,
        log_file: Optional[str] = "sbl-conf-ensemble-kmeanspp.log",
        skip_seeding: bool = False,
    ) -> dict:
        """Run ``sbl-conf-ensemble-kmeanspp.exe``.

        ``scan_k_max`` writes the elbow-curve files.  ``k`` writes the
        selected-seed files (``seeding_seeds.csv``, ``seeding_seed_indices.txt``,
        ``seeding_assignments.csv``
        and optional seed PDBs).  Provide both to scan and select in one run.

        *atoms* (e.g. ``"N,CA,C,O"``) and *chainid* are forwarded to
        ``--atoms`` / ``--chainid``; by default the executable uses the
        backbone atoms ``N,CA,C`` of all chains.  Seed PDBs only contain the
        selected atoms, so use the atom set of the MDS ``atom_mode`` when the
        seeds serve as MDS landmarks (``"N,CA,C,O"`` for ``"backbone"``).

        When *skip_seeding* is True, no seeding executable is launched.  The
        method instead reuses the seeding outputs already present in the
        expected output directory.  This is useful after restarting a notebook
        kernel when the seeding step has already completed.

        Returns
        -------
        dict with keys ``odir``, ``stdout``, ``log_file``, ``skipped``.
        """
        ifpath = Path(ifpath).resolve()
        if odpath is None:
            odpath = self.root_results / f"seeding_{ifpath.stem}_{resids}"
        odpath = Path(odpath).resolve()

        log_path: Optional[Path] = None
        if log_file:
            log_path = Path(log_file)
            if not log_path.is_absolute():
                log_path = odpath / log_path

        if skip_seeding:
            scan_csv = odpath / "seeding_k_scan.csv"
            if scan_k_max != 0 and not scan_csv.is_file():
                raise RuntimeError(
                    f"Cannot skip seeding; expected cached k-scan CSV not found: {scan_csv}"
                )
            if k is not None:
                seeds_csv = odpath / "seeding_seeds.csv"
                seed_indices = odpath / "seeding_seed_indices.txt"
                missing = [p for p in (seeds_csv, seed_indices) if not p.is_file()]
                if missing:
                    raise RuntimeError(
                        "Cannot skip seeding; expected cached selected-seed file(s) not found:\n"
                        + "\n".join(str(p) for p in missing)
                    )
            self._print_stage("seeding")
            self._print_output("reused directory", odpath)
            return {
                "odir": odpath,
                "stdout": "",
                "log_file": log_path,
                "skipped": True,
            }

        if self.clustering_exe is None:
            raise RuntimeError(
                "sbl-conf-ensemble-kmeanspp.exe not found. "
                "Build the target or check clustering_exe."
            )

        odpath.mkdir(parents=True, exist_ok=True)

        cmd = [
            self.clustering_exe,
            "--ifpath",          str(ifpath),
            "--odpath",          str(odpath),
            "--scan-k-max",      str(scan_k_max),
            "--method",          method,
            "--metric",          metric,
            "--random-seed",     str(random_seed),
            "--n-local-trials",  str(n_local_trials),
            "--resids",          resids,
        ]
        if k is not None:
            cmd += ["--k", str(k)]
        if atoms is not None:
            cmd += ["--atoms", atoms]
        if chainid is not None:
            cmd += ["--chainid", chainid]
        if not save_pdbs:
            cmd.append("--no-save-pdbs")

        if log_file:
            log_path.parent.mkdir(parents=True, exist_ok=True)

        self._print_stage("seeding")
        self._print_output("output directory", odpath)
        if log_path is not None:
            self._print_output("log", log_path)
        res = subprocess.run(cmd, cwd=str(odpath), capture_output=True, text=True)
        if log_path is not None:
            log_path.write_text(res.stdout + res.stderr)
        if res.returncode != 0:
            print(res.stdout)
            print(res.stderr)
            raise RuntimeError(f"sbl-conf-ensemble-kmeanspp.exe failed (rc={res.returncode})")

        self._print_output("directory", odpath)
        return {
            "odir": odpath,
            "stdout": res.stdout,
            "log_file": log_path,
            "skipped": False,
        }

    @staticmethod
    def _detect_elbow_chord(ks: list[int], values: list[float]) -> Optional[int]:
        """Return the elbow k via max perpendicular distance from the chord.

        Implements the geometric core of the Kneedle algorithm:
        normalise (k, value) to the unit square, draw the chord between the
        first and last points, and pick the k whose point lies farthest from
        that chord on the convex side.

        Assumes *values* is (approximately) monotonically decreasing in *ks*.
        Returns *None* when fewer than 3 points are available.
        """
        n = len(ks)
        if n < 3 or len(values) != n:
            return None
        k_min, k_max = ks[0], ks[-1]
        v_min, v_max = min(values), max(values)
        dk = k_max - k_min
        dv = v_max - v_min
        if dk == 0 or dv == 0:
            return None
        # Normalise to [0, 1].
        x = [(ki - k_min) / dk for ki in ks]
        y = [(vi - v_min) / dv for vi in values]
        # Chord goes from (0, y[0]) to (1, y[-1]); perpendicular distance is
        # |(y[-1]-y[0])*x_i - (x[-1]-x[0])*y_i + x[-1]*y[0] - y[-1]*x[0]| / L.
        # With x[0]=0, x[-1]=1 this simplifies; we keep the general form.
        x0, y0 = x[0], y[0]
        x1, y1 = x[-1], y[-1]
        denom = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        if denom == 0:
            return None
        best_i, best_d = None, -1.0
        for i in range(1, n - 1):
            num = abs((y1 - y0) * x[i] - (x1 - x0) * y[i] + x1 * y0 - y1 * x0)
            d = num / denom
            if d > best_d:
                best_d, best_i = d, i
        return ks[best_i] if best_i is not None else None

    @staticmethod
    def _linear_fit_sse(xs: list[float], ys: list[float]) -> float:
        """Return least-squares residual sum of squares for ``y = a*x + b``."""
        n = len(xs)
        if n != len(ys) or n == 0:
            raise ValueError("xs and ys must have the same nonzero length")
        x_mean = sum(xs) / n
        y_mean = sum(ys) / n
        sxx = sum((x - x_mean) ** 2 for x in xs)
        if sxx == 0.0:
            return sum((y - y_mean) ** 2 for y in ys)
        sxy = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
        slope = sxy / sxx
        intercept = y_mean - slope * x_mean
        return sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))

    @classmethod
    def _detect_elbow_piecewise_linear(
        cls,
        ks: list[int],
        values: list[float],
    ) -> tuple[Optional[int], dict[int, float]]:
        """Return the k minimizing two-piece linear-fit SSE.

        For each interior point k_i, fit one line to points ``[:i + 1]`` and
        another line to points ``[i:]``.  The elbow is the k with the smallest
        total residual sum of squares across the two fits.  The endpoint is
        included in both neighboring segments so the candidate point is the
        shared break point.
        """
        n = len(ks)
        if n < 3 or len(values) != n:
            return None, {}

        x = [float(k) for k in ks]
        y = [float(v) for v in values]
        losses: dict[int, float] = {}
        for i in range(1, n - 1):
            left_loss = cls._linear_fit_sse(x[:i + 1], y[:i + 1])
            right_loss = cls._linear_fit_sse(x[i:], y[i:])
            losses[ks[i]] = left_loss + right_loss
        if not losses:
            return None, {}
        best_k = min(losses, key=lambda k: (losses[k], k))
        return best_k, losses

    def analyze_seeding(
        self,
        seeding_result: dict,
        *,
        plot_k_scan: bool = True,
        auto_elbow: bool = True,
        style: Optional[dict] = None,
        figure_formats=("png",),
    ) -> dict:
        """Parse and display the outputs of :meth:`run_seeding`.

        Reads ``seeding_k_scan.csv`` and, when present, ``seeding_seeds.csv``
        from *seeding_result["odir"]*.  It prints a summary table of the
        selected seeds when ``run_seeding(..., k=...)`` was used, and
        optionally plots the k-scan elbow curve with matplotlib.

        Parameters
        ----------
        seeding_result : dict
            Return value of :meth:`run_seeding`.
        plot_k_scan : bool
            When *True* (default) plot total distance vs k.
        auto_elbow : bool
            When *True* (default) detect the elbow of the k-scan curve by
            fitting two linear segments for every candidate k and selecting
            the k with the smallest total residual sum of squares.  The
            suggested k is printed, marked on the plot, and returned in the
            result dict.
        style : dict or None
            Per-call plot-style overrides.

        Returns
        -------
        dict with keys:

        ``seeds_df``
            List of selected-seed rows, or *None* if no fixed-k selection was
            run.  The key name is kept for notebook compatibility.
        ``scan_df``
            List of k-scan rows.  The key name is kept for notebook
            compatibility.
        ``assignments``
            Rows of ``seeding_assignments.csv`` (conformation index, seed
            rank, seed conformation index, distance to the seed), or *None*.
        ``cluster_sizes``
            Dict seed rank -> number of conformations in its cluster, or
            *None*.
        ``suggested_k``
            Auto-detected elbow k (int) or *None* when *auto_elbow* is False
            or the curve has too few points.
        ``fig``
            :class:`matplotlib.figure.Figure` if *plot_k_scan* is True,
            else *None*.
        """
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker

        def read_csv_rows(csv_path: Path) -> list[dict[str, str]]:
            with csv_path.open(newline="") as in_file:
                return list(csv.DictReader(in_file))

        def display_rows(rows: list[dict[str, str]]) -> None:
            if not rows:
                self._display(Markdown("(no rows)"))
                return
            columns = list(rows[0].keys())
            header = "".join(f"<th>{html.escape(col)}</th>" for col in columns)
            body = "".join(
                "<tr>" + "".join(
                    f"<td>{html.escape(str(row.get(col, '')))}</td>"
                    for col in columns
                ) + "</tr>"
                for row in rows
            )
            self._display(HTML(
                "<table>"
                f"<thead><tr>{header}</tr></thead>"
                f"<tbody>{body}</tbody>"
                "</table>"
            ))

        odir = Path(seeding_result["odir"])

        # ------------------------------------------------------------------
        # 1. Seeds summary
        # ------------------------------------------------------------------
        seeds_csv = odir / "seeding_seeds.csv"
        seeds_df = None
        if seeds_csv.is_file():
            seeds_df = read_csv_rows(seeds_csv)
            self._display(Markdown(f"**Selected seeds** (k = {len(seeds_df)}) — `{seeds_csv}`"))
            display_rows(seeds_df)
        else:
            self._display(Markdown(
                "**Selected seeds** were not written. "
                "Rerun `run_seeding(..., k=<number_of_seeds>)` to produce "
                "`seeding_seeds.csv`."
            ))

        # ------------------------------------------------------------------
        # 1b. Clusters: each conformation assigned to its nearest seed
        # ------------------------------------------------------------------
        assignments_csv = odir / "seeding_assignments.csv"
        assignments = None
        cluster_sizes = None
        if assignments_csv.is_file():
            assignments = read_csv_rows(assignments_csv)
            cluster_sizes = {}
            for row in assignments:
                seed = int(row["seed"])
                cluster_sizes[seed] = cluster_sizes.get(seed, 0) + 1
            size_rows = []
            for seed in sorted(cluster_sizes):
                dists = [float(r["distance"]) for r in assignments
                         if int(r["seed"]) == seed]
                size_rows.append({
                    "seed": seed,
                    "seed_index": next(r["seed_index"] for r in assignments
                                       if int(r["seed"]) == seed),
                    "cluster_size": cluster_sizes[seed],
                    "mean_distance_to_seed": f"{sum(dists) / len(dists):.3f}",
                    "max_distance_to_seed": f"{max(dists):.3f}",
                })
            self._display(Markdown(
                f"**Clusters** (nearest-seed assignment) — `{assignments_csv}`"
            ))
            display_rows(size_rows)

        # ------------------------------------------------------------------
        # 2. k-scan curve
        # ------------------------------------------------------------------
        scan_csv = odir / "seeding_k_scan.csv"
        if not scan_csv.is_file():
            raise FileNotFoundError(f"k-scan CSV not found: {scan_csv}")
        scan_df = read_csv_rows(scan_csv)

        # ------------------------------------------------------------------
        # 3. Auto-detect elbow (two-piece linear fit)
        # ------------------------------------------------------------------
        suggested_k: Optional[int] = None
        elbow_losses: dict[int, float] = {}
        elbow_method = "piecewise_linear_sse"
        if auto_elbow:
            try:
                ks_all = [int(row["k"]) for row in scan_df]
                vals_all = [float(row["total_distance"]) for row in scan_df]
                suggested_k, elbow_losses = self._detect_elbow_piecewise_linear(
                    ks_all,
                    vals_all,
                )
            except (KeyError, ValueError):
                suggested_k = None
                elbow_losses = {}
            if suggested_k is not None:
                self._display(Markdown(
                    f"**Suggested k (piecewise-linear SSE elbow): {suggested_k}**  "
                    f"— rerun `run_seeding(..., k={suggested_k}, save_pdbs=True)` "
                    "to materialise the seed PDBs."
                ))

        figure_formats = self._normalize_figure_formats(figure_formats)
        chosen_k = len(seeds_df) if seeds_df is not None else None
        fig = None
        figures: list[Path] = []
        if plot_k_scan:
            plot_data_path = odir / "seeding_k_scan_plot_data.csv"
            with plot_data_path.open("w", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow([
                    "k",
                    "total_distance",
                    "is_chosen_k",
                    "is_suggested_k",
                    "piecewise_linear_sse",
                ])
                for row in scan_df:
                    k_value = int(row["k"])
                    loss = elbow_losses.get(k_value)
                    writer.writerow([
                        k_value,
                        row["total_distance"],
                        int(chosen_k is not None and k_value == chosen_k),
                        int(suggested_k is not None and k_value == suggested_k),
                        "" if loss is None else f"{loss:.17g}",
                    ])
            metadata = {
                "plot_data": "seeding_k_scan_plot_data.csv",
                "source_data": "seeding_k_scan.csv",
                "chosen_k": chosen_k,
                "suggested_k": suggested_k,
                "elbow_method": elbow_method,
                "figure_formats": list(figure_formats),
            }
            with (odir / "seeding_k_scan_plot_metadata.json").open("w") as fh:
                json.dump(metadata, fh, indent=2, sort_keys=True)

            eff_style = self._effective_style(style)
            # Parse "WxH" figsize string
            figsize_str = str(eff_style.get("figsize", "6x4"))
            try:
                fw, fh = (float(v) for v in figsize_str.lower().split("x"))
            except ValueError:
                fw, fh = 6.0, 4.0

            dpi        = float(eff_style.get("dpi", 150))
            font_size  = float(eff_style.get("font_size", 8))
            title_size = float(eff_style.get("title_size", 10))
            label_size = float(eff_style.get("label_size", 8))
            tick_size  = float(eff_style.get("tick_size", 8))
            lw         = float(eff_style.get("line_width", 1.0))
            ms         = float(eff_style.get("marker_size", 4))

            plt.rcParams.update({
                "font.size":        font_size,
                "axes.titlesize":   title_size,
                "axes.labelsize":   label_size,
                "xtick.labelsize":  tick_size,
                "ytick.labelsize":  tick_size,
            })

            fig, ax = plt.subplots(figsize=(fw, fh), dpi=dpi)

            k = [int(row["k"]) for row in scan_df]
            total_dist = [float(row["total_distance"]) for row in scan_df]

            ax.plot(k, total_dist, marker="o", linewidth=lw, markersize=ms,
                    color="steelblue", label="total distance")
            ax.set_xlabel("k  (number of seeds)")
            ax.set_ylabel("Sum of nearest-seed distances  (Å)")
            ax.set_title("Seeding k-scan")
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            ax.legend(fontsize=font_size)
            ax.grid(True, linewidth=0.3, alpha=0.6)

            # Mark the chosen k (= number of rows in seeds_df) when available.
            if chosen_k is not None and 1 <= chosen_k <= len(k):
                if chosen_k in k:
                    ax.axvline(x=chosen_k, color="tomato", linewidth=lw,
                               linestyle="--", alpha=0.8,
                               label=f"chosen k={chosen_k}")
                    ax.legend(fontsize=font_size)

            # Mark the auto-detected elbow.
            if suggested_k is not None and suggested_k != chosen_k:
                ax.axvline(x=suggested_k, color="seagreen", linewidth=lw,
                           linestyle=":", alpha=0.9,
                           label=f"suggested k={suggested_k}")
                ax.legend(fontsize=font_size)

            fig.tight_layout()

            # Save alongside the CSV.
            for fmt in figure_formats:
                out_fig = odir / f"seeding_k_scan.{fmt}"
                fig.savefig(out_fig, dpi=dpi, bbox_inches="tight")
                figures.append(out_fig)
                self._display(Markdown(f"**k-scan plot** saved to `{out_fig}`"))
                if fmt == "png":
                    self._display(Image(filename=str(out_fig)))
            plt.close(fig)

        return {"seeds_df": seeds_df, "scan_df": scan_df,
                "assignments": assignments,
                "cluster_sizes": cluster_sizes,
                "suggested_k": suggested_k,
                "elbow_method": elbow_method,
                "elbow_losses": elbow_losses,
                "fig": fig, "figures": figures}

    # -------------------------------------------------------------------
    # Embedding via MDS
    # -------------------------------------------------------------------

    @staticmethod
    def _split_multimodel_pdb(
        ifpath: os.PathLike,
        odir: os.PathLike,
        *,
        prefix: Optional[str] = None,
    ) -> Path:
        """Split a multi-model PDB into one file per MODEL inside *odir*.

        Output files are named ``<prefix>_model_<NNNNN>.pdb`` (5-digit
        zero-padded) so they sort naturally and produce stable labels for
        downstream tools.  The split is cached: if *odir* already contains
        matching files the function returns immediately.

        Returns the output directory.
        """
        ifpath = Path(ifpath).resolve()
        odir = Path(odir).resolve()
        if prefix is None:
            prefix = ifpath.stem
        odir.mkdir(parents=True, exist_ok=True)

        existing = sorted(odir.glob(f"{prefix}_model_*.pdb"))
        if existing:
            return odir

        header_lines: list[str] = []
        current: list[str] = []
        in_model = False
        n_written = 0
        any_model = False

        def flush(idx: int) -> None:
            nonlocal n_written
            if not current:
                return
            out_path = odir / f"{prefix}_model_{idx:05d}.pdb"
            with out_path.open("w") as out:
                out.writelines(header_lines)
                out.writelines(current)
                out.write("END\n")
            n_written += 1

        with ifpath.open("r") as in_f:
            model_idx = 0
            for line in in_f:
                if line.startswith("MODEL"):
                    in_model = True
                    any_model = True
                    model_idx += 1
                    current = []
                    continue
                if line.startswith("ENDMDL"):
                    in_model = False
                    flush(model_idx)
                    current = []
                    continue
                if line.startswith("END") and not line.startswith("ENDMDL"):
                    continue
                if in_model:
                    current.append(line)
                else:
                    # Pre-MODEL header records (REMARK, CRYST1, …).
                    if any_model:
                        # Stray lines between models — skip.
                        continue
                    header_lines.append(line)
            # File with no MODEL records: write the body as a single model.
            if not any_model and current:
                flush(1)

        if n_written == 0:
            raise RuntimeError(
                f"_split_multimodel_pdb: no models written from {ifpath}"
            )
        return odir

    @staticmethod
    def _seed_pdbs_dir(seeding_run: dict) -> Path:
        """Return the directory holding ``seed_*.pdb`` files for a seeding run.

        ``sbl-conf-ensemble-kmeanspp.exe`` writes them as
        ``seed_<i>_index_<idx>.pdb`` inside a ``seeds/`` subdirectory
        of the seeding output directory (unless ``--no-save-pdbs`` was
        passed).  Also accept older root-level seed PDB layouts.
        """
        odir = Path(seeding_run["odir"]).resolve()
        candidates = [odir / "seeds", odir]
        for candidate in candidates:
            seeds = sorted(candidate.glob("seed_*_index_*.pdb"))
            if seeds:
                return candidate
        else:
            raise RuntimeError(
                "No seed PDB files (seed_*_index_*.pdb) found in:\n"
                + "\n".join(str(path) for path in candidates)
                + "\nRerun run_seeding(..., k=<int>, save_pdbs=True) "
                  "to create <odir>/seeds/seed_*_index_*.pdb files."
            )

    @staticmethod
    def _parse_resid_range(resids: str) -> tuple:
        """Parse a ``"<lo>-<hi>"`` resid string into ``(lo, hi)`` ints."""
        if resids is None:
            raise ValueError("resids must be a 'START-END' string")
        parts = str(resids).split("-")
        if len(parts) != 2:
            raise ValueError(f"resids must be 'START-END', got {resids!r}")
        return (int(parts[0]), int(parts[1]))

    @staticmethod
    def _extract_loop_pdb(
        src: os.PathLike,
        dst: os.PathLike,
        chainid: str,
        resid_range: tuple,
        atom_names: Optional[set] = None,
    ) -> Path:
        """Copy *src* into *dst* keeping only ``ATOM`` records on *chainid*
        whose residue number lies in ``[resid_range[0], resid_range[1]]``.
        If *atom_names* is given, only ATOM records whose atom name is in
        the set are kept.

        Pure-stdlib PDB filter; columns follow the fixed PDB format
        (chain id at column 22 / index 21, resSeq at columns 23-26,
        atom name at columns 13-16).
        """
        src = Path(src)
        dst = Path(dst)
        if not src.is_file():
            raise FileNotFoundError(f"landmark PDB not found: {src}")
        lo, hi = int(resid_range[0]), int(resid_range[1])
        dst.parent.mkdir(parents=True, exist_ok=True)
        n_kept = 0
        seen_atom_keys = set()
        with src.open() as fin, dst.open("w") as fout:
            for line in fin:
                if not line.startswith("ATOM"):
                    continue
                if len(line) < 26:
                    continue
                if line[21] != chainid:
                    continue
                try:
                    resseq = int(line[22:26])
                except ValueError:
                    continue
                if not (lo <= resseq <= hi):
                    continue
                if atom_names is not None:
                    if line[12:16].strip() not in atom_names:
                        continue
                alt_loc = line[16].strip()
                if alt_loc and alt_loc != "A":
                    continue
                atom_key = (
                    line[21],
                    line[17:20].strip(),
                    line[22:26].strip(),
                    line[26].strip(),
                    line[12:16].strip(),
                )
                if atom_key in seen_atom_keys:
                    continue
                seen_atom_keys.add(atom_key)
                fout.write(line)
                n_kept += 1
            fout.write("END\n")
        if n_kept == 0:
            raise RuntimeError(
                f"_extract_loop_pdb: no ATOM records on chain {chainid!r} "
                f"with resids in [{lo}, {hi}]"
                + (f" and atom names in {sorted(atom_names)}"
                   if atom_names is not None else "")
                + f" found in {src}"
            )
        return dst

    @classmethod
    def _normalize_landmark_lists(
        cls,
        landmarks,
        labels=None,
        chainids=None,
    ):
        """Validate and normalize landmark / labels / chainids lists.

        Returns ``(landmarks_paths, labels, chainids)`` all of equal length.
        """
        if not landmarks:
            raise ValueError("landmarks list is empty")
        lm_paths = [Path(p).resolve() for p in landmarks]
        n = len(lm_paths)

        if labels is None:
            labels = [p.stem for p in lm_paths]
        else:
            labels = list(labels)
        if len(labels) != n:
            raise ValueError(
                f"landmark_labels has {len(labels)} entries, "
                f"expected {n}"
            )
        if len(set(labels)) != n:
            raise ValueError(f"landmark_labels must be unique, got {labels}")

        if chainids is None:
            raise ValueError(
                "landmark_chainids is required (single chain id "
                "or a list with one entry per landmark)"
            )
        if isinstance(chainids, str):
            chainids = [chainids] * n
        else:
            chainids = list(chainids)
        if len(chainids) != n:
            raise ValueError(
                f"landmark_chainids has {len(chainids)} entries, "
                f"expected {n}"
            )
        return lm_paths, labels, chainids

    @classmethod
    def _stage_landmarks_sliced(
        cls,
        landmarks,
        labels,
        chainids,
        resids: str,
        odir: os.PathLike,
        atom_names: Optional[set] = None,
    ) -> Path:
        """Slice each landmark to ``(chainid, resids)`` (optionally also
        restricted to *atom_names*) and write ``odir/<label>.pdb``.
        Returns *odir*.

        Used for MDS: ``sbl-mds.exe`` requires landmark PDBs whose atom
        count matches the sampled conformations.
        """
        lm_paths, labels, chainids = cls._normalize_landmark_lists(
            landmarks, labels, chainids
        )
        resid_range = cls._parse_resid_range(resids)
        odir = Path(odir).resolve()
        odir.mkdir(parents=True, exist_ok=True)
        for src, label, chain in zip(lm_paths, labels, chainids):
            dst = odir / f"{label}.pdb"
            cls._extract_loop_pdb(src, dst, chain, resid_range, atom_names)
        return odir

    @staticmethod
    def _sample_atom_names(split_dir: os.PathLike) -> set:
        """Return the set of atom names present in the first per-model PDB
        of *split_dir* (used to match landmark slicing to what the sampler
        actually wrote)."""
        first = next(iter(sorted(Path(split_dir).glob("*.pdb"))), None)
        if first is None:
            raise RuntimeError(f"No PDB files in {split_dir}")
        names = set()
        with first.open() as fin:
            for line in fin:
                if line.startswith("ATOM") and len(line) >= 16:
                    names.add(line[12:16].strip())
        return names

    def run_mds(
        self,
        ifpath: os.PathLike,
        *,
        mode: str = "landmark-user",
        landmarks_dir: Optional[os.PathLike] = None,
        landmarks: Optional[list] = None,
        landmark_labels: Optional[list] = None,
        landmark_chainids=None,
        landmark_resids: Optional[str] = None,
        reference_landmarks: Optional[list] = None,
        reference_labels: Optional[list] = None,
        reference_chainids=None,
        reference_resids: Optional[str] = None,
        seeding_run: Optional[dict] = None,
        odpath: Optional[os.PathLike] = None,
        target_dim: int = 3,
        atom_mode: str = "backbone",
        skip_mds: bool = False,
        log_file: Optional[str] = "sbl-mds.log",
    ) -> dict:
        """Run ``sbl-mds.exe`` on the conformations in a multi-model PDB.

        Parameters
        ----------
        ifpath : path-like
            Multi-model PDB holding the conformational ensemble (e.g. a loop
            sampling output ``*all-embeddings*.pdb``).
        mode : {"classical", "landmark-user", "landmark-seeding"}
            ``"classical"`` runs classical MDS on the sample only.
            ``"landmark-user"`` adds *landmarks_dir* PDBs as landmarks.
            ``"landmark-seeding"`` uses the seed PDBs from a prior
            :meth:`run_seeding` call passed via *seeding_run*.
        landmarks_dir : path-like, optional
            For ``mode="landmark-user"`` only.  Directory of single-model
            PDBs whose atom count already matches the sampled
            conformations.  Mutually exclusive with *landmarks*.
        landmarks : list[path-like], optional
            For ``mode="landmark-user"`` only.  Explicit list of landmark
            PDB files (may be full structures); each is sliced to
            ``(landmark_chainids[i], landmark_resids)`` and staged into
            ``<odir>/_landmarks_staged/<landmark_labels[i]>.pdb``.
        landmark_labels : list[str], optional
            One label per *landmarks* entry.  Used as the staged file
            basename and as the MDS plot label.  Defaults to each
            landmark's filename stem.
        landmark_chainids : str | list[str], optional
            Required when *landmarks* is given.  Single chain id applied
            to all landmarks, or one per landmark.
        landmark_resids : str, optional
            ``"START-END"`` residue range used to slice landmark PDBs.
            Required when *landmarks* is given.
        reference_landmarks : list[path-like], optional
            Explicit list of PDB files to project as named non-landmark
            reference points.  These are sliced to
            ``(reference_chainids[i], reference_resids)`` and copied into
            the sample model directory, but not passed via ``--landmarks``.
            This is useful for showing experimental structures in a
            seed-landmark MDS embedding without changing the landmark basis.
        reference_labels : list[str], optional
            One label per *reference_landmarks* entry.  Defaults to each
            reference PDB's filename stem.
        reference_chainids : str | list[str], optional
            Required when *reference_landmarks* is given.  Single chain id
            applied to all references, or one per reference.
        reference_resids : str, optional
            ``"START-END"`` residue range used to slice reference PDBs.
            Required when *reference_landmarks* is given.
        seeding_run : dict, optional
            Required for ``mode="landmark-seeding"``.  Return value of
            :meth:`run_seeding` with ``k=<int>`` and ``save_pdbs=True``.
        odpath : path-like, optional
            Output directory (default ``<root_results>/mds_<stem>_<mode>``).
        target_dim : int
            Target embedding dimension (must be >= 2 for the plotter).
        atom_mode : str
            Forwarded to ``--atom-mode`` (e.g. ``"backbone"``, ``"CA"``).
        skip_mds : bool
            When *True*, reuse the cached MDS outputs under *odpath* if
            ``singular_values.csv`` is present (mirrors ``skip_sampling`` /
            ``skip_seeding``).

        Returns
        -------
        dict with keys ``odir``, ``stdout``, ``log_file``, ``skipped``,
        ``mode``, ``landmarks_dir`` (resolved path or None), and optional
        reference point metadata.
        """
        if mode not in ("classical", "landmark-user", "landmark-seeding"):
            raise ValueError(
                "mode must be one of 'classical', 'landmark-user', "
                f"'landmark-seeding' (got {mode!r})"
            )

        ifpath = Path(ifpath).resolve()
        if not ifpath.is_file():
            raise FileNotFoundError(f"Input PDB not found: {ifpath}")

        if odpath is None:
            tag = {
                "classical":         "classical",
                "landmark-user":     "landmark-user",
                "landmark-seeding":  "landmark-seeding",
            }[mode]
            odpath = self.root_results / f"mds_{ifpath.stem}_{tag}"
        odpath = Path(odpath).resolve()

        # Resolve landmarks
        resolved_landmarks: Optional[Path] = None
        _stage_after_split = False  # for explicit-list MDS landmarks
        resolved_references: Optional[Path] = None
        normalized_reference_labels: list[str] = []
        if mode == "landmark-user":
            if landmarks is not None and landmarks_dir is not None:
                raise ValueError(
                    "mode='landmark-user': pass either landmarks_dir "
                    "or landmarks, not both"
                )
            if landmarks is not None:
                if landmark_resids is None:
                    raise ValueError(
                        "mode='landmark-user' with landmarks=[...] "
                        "requires landmark_resids='START-END'"
                    )
                # Validate eagerly; actual slicing happens after sample split
                # so we can intersect atom names with the sampled file.
                self._normalize_landmark_lists(
                    landmarks, landmark_labels, landmark_chainids
                )
                _stage_after_split = True
                resolved_landmarks = Path(odpath) / "_landmarks_staged"
            elif landmarks_dir is not None:
                resolved_landmarks = Path(landmarks_dir).resolve()
                if not resolved_landmarks.is_dir():
                    raise FileNotFoundError(
                        f"landmarks_dir not found: {resolved_landmarks}"
                    )
                if not list(resolved_landmarks.glob("*.pdb")):
                    raise RuntimeError(
                        f"No *.pdb files in landmarks_dir {resolved_landmarks}"
                    )
            else:
                raise ValueError(
                    "mode='landmark-user' requires landmarks_dir "
                    "or landmarks=[...]"
                )
        elif mode == "landmark-seeding":
            if seeding_run is None:
                raise ValueError(
                    "mode='landmark-seeding' requires seeding_run "
                    "(return value of run_seeding(..., k=K, save_pdbs=True))"
                )
            resolved_landmarks = self._seed_pdbs_dir(seeding_run)

            if reference_landmarks is not None:
                if reference_resids is None:
                    raise ValueError(
                        "reference_landmarks=[...] requires "
                        "reference_resids='START-END'"
                    )
                _, normalized_reference_labels, _ = self._normalize_landmark_lists(
                    reference_landmarks, reference_labels, reference_chainids
                )
                resolved_references = Path(odpath) / "_references_staged"
            elif any(v is not None for v in (
                reference_labels, reference_chainids, reference_resids
            )):
                raise ValueError(
                    "reference_labels/reference_chainids/reference_resids require "
                    "reference_landmarks=[...]"
                )

        log_path: Optional[Path] = None
        if log_file:
            log_path = Path(log_file)
            if not log_path.is_absolute():
                log_path = odpath / log_path

        sv_csv = odpath / "singular_values.csv"
        if skip_mds:
            if not sv_csv.is_file():
                raise RuntimeError(
                    f"Cannot skip MDS; expected cached output not found: {sv_csv}"
                )
            self._print_stage("MDS")
            self._print_output("reused directory", odpath)
            return {
                "odir":          odpath,
                "stdout":        "",
                "log_file":      log_path,
                "skipped":       True,
                "mode":          mode,
                "landmarks_dir": resolved_landmarks,
                "reference_dir": resolved_references,
                "reference_labels": normalized_reference_labels,
            }

        if self.mds_exe is None:
            raise RuntimeError(
                "sbl-mds.exe not found on PATH. Install the SBL command first."
            )

        odpath.mkdir(parents=True, exist_ok=True)

        # Split sampling output into per-model PDBs for --pdb-dir.
        split_dir = odpath / "_models"
        self._split_multimodel_pdb(ifpath, split_dir, prefix="sample")

        wanted = None
        if _stage_after_split or resolved_references is not None:
            sample_names = self._sample_atom_names(split_dir)
            atom_mode_names = (
                {"N", "CA", "C", "O"} if atom_mode == "backbone" else {"CA"}
            )
            wanted = sample_names & atom_mode_names
            if not wanted:
                raise RuntimeError(
                    f"No atoms matching atom_mode={atom_mode!r} found in "
                    f"sample (sample atoms: {sorted(sample_names)})"
                )

        # Stage explicit-list landmarks now, restricted to the atom-name
        # set actually present in the sample (e.g. sampler may write only
        # N/CA/C even though atom_mode='backbone' nominally includes O).
        if _stage_after_split:
            self._stage_landmarks_sliced(
                landmarks=landmarks,
                labels=landmark_labels,
                chainids=landmark_chainids,
                resids=landmark_resids,
                odir=resolved_landmarks,
                atom_names=wanted,
            )

        if resolved_references is not None:
            self._stage_landmarks_sliced(
                landmarks=reference_landmarks,
                labels=reference_labels,
                chainids=reference_chainids,
                resids=reference_resids,
                odir=resolved_references,
                atom_names=wanted,
            )

        # For landmark modes, copy landmark PDBs into the same --pdb-dir so
        # they participate in the embedding (sbl-mds matches landmarks by
        # basename against the files listed via --pdb-dir).
        if resolved_landmarks is not None:
            for lm in sorted(resolved_landmarks.glob("*.pdb")):
                dst = split_dir / lm.name
                shutil.copy2(lm, dst)

        if resolved_references is not None:
            reserved_labels = {
                path.stem for path in split_dir.glob("sample_model_*.pdb")
            }
            if resolved_landmarks is not None:
                reserved_labels.update(
                    path.stem for path in resolved_landmarks.glob("*.pdb")
                )
            for ref in sorted(resolved_references.glob("*.pdb")):
                dst = split_dir / ref.name
                if ref.stem in reserved_labels:
                    raise RuntimeError(
                        f"Reference label conflicts with an existing MDS "
                        f"model basename: {ref.stem!r}"
                    )
                shutil.copy2(ref, dst)

        cmd = [
            self.mds_exe,
            "--pdb-dir",    str(split_dir),
            "--out-dir",    str(odpath),
            "--target-dim", str(target_dim),
            "--atom-mode",  atom_mode,
        ]
        if resolved_landmarks is not None:
            cmd += ["--landmarks", str(resolved_landmarks)]

        self._print_stage("MDS")
        self._print_output("output directory", odpath)
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self._print_output("log", log_path)

        res = subprocess.run(cmd, cwd=str(odpath),
                             capture_output=True, text=True)
        if log_path is not None:
            log_path.write_text(res.stdout + res.stderr)
        if res.returncode != 0:
            print(res.stdout)
            print(res.stderr)
            raise RuntimeError(f"sbl-mds.exe failed (rc={res.returncode})")
        self._print_output("directory", odpath)

        return {
            "odir":          odpath,
            "stdout":        res.stdout,
            "log_file":      log_path,
            "skipped":       False,
            "mode":          mode,
            "landmarks_dir": resolved_landmarks,
            "reference_dir": resolved_references,
            "reference_labels": normalized_reference_labels,
        }

    def analyze_mds(
        self,
        mds_run: dict,
        *,
        show_labels: bool = True,
        fmt: str = "png",
        figure_formats=None,
        style: Optional[dict] = None,
    ) -> dict:
        """Load ``sbl-mds`` outputs and produce scree + 2D/3D scatter plots.

        Uses :class:`SBL.MDS_plots.MDSResult` / :class:`MDSPlotter`.  Figures
        are written into ``mds_run["odir"]`` and inlined in the notebook via
        ``IPython.display.Image``.

        Returns a dict ``{"result": MDSResult, "plotter": MDSPlotter,
        "odir": Path, "figures": [Path, ...]}``.
        """
        from SBL.MDS_plots import MDSResult, MDSPlotter, MDSPlotStyle

        odir = Path(mds_run["odir"]).resolve()
        result = MDSResult.from_directory(str(odir))

        eff_style = self._effective_style(style)
        # Parse "WxH" figsize string.
        figsize_str = str(eff_style.get("figsize", "6x4"))
        try:
            fw, fh = (float(v) for v in figsize_str.lower().split("x"))
            figsize = (fw, fh)
        except ValueError:
            figsize = None

        plot_style = MDSPlotStyle(
            figsize=figsize,
            dpi=int(eff_style.get("dpi", 150)),
            font_size=float(eff_style.get("font_size", 10)),
            sample_marker_size=18,
            landmark_marker_size=70,
            reference_marker_size=60,
            xlim=eff_style.get("xlim"),
            ylim=eff_style.get("ylim"),
        )
        reference_labels = mds_run.get("reference_labels") or []
        plotter = MDSPlotter(result, plot_style,
                             reference_labels=reference_labels)
        formats = self._normalize_figure_formats(figure_formats or (fmt,))
        figures = []
        for figure_format in formats:
            plotter.plot_all(str(odir), fmt=figure_format,
                             show_labels=show_labels)
            figures.extend(sorted(odir.glob(f"*.{figure_format}")))
        self._display(Markdown(
            f"**MDS — mode `{mds_run.get('mode', '?')}`** "
            f"({len(result.labels_2d)} points"
            + (
                f", {int(result.is_landmark_2d.sum())} landmarks"
                if result.is_landmark_2d is not None else ""
            )
            + (
                f", {len(reference_labels)} references"
                if reference_labels else ""
            )
            + f") — `{odir}`"
        ))
        for f in figures:
            self._display(Markdown(f"**{f.name}**"))
            if f.suffix == ".png":
                self._display(Image(filename=str(f)))

        return {
            "result":  result,
            "plotter": plotter,
            "odir":    odir,
            "figures": figures,
        }
