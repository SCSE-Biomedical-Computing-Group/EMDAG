import os
import argparse
import hashlib
import pickle
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from tqdm.auto import tqdm

from emdag.datasets.sabdab import SAbDabDataset
from emdag.tools.eval.base import TaskScanner, EvalTask
from emdag.tools.eval.similarity import extract_reslist
from emdag.utils.protein import constants


# Helpers
def cdr_tag_to_type(tag: str) -> str:
    tag = tag.strip().upper()
    if tag.startswith("H_CDR"):
        return "H" + tag[-1]
    if tag.startswith("L_CDR"):
        return "L" + tag[-1]
    # already like H1/H2/H3/L1...
    return tag


def reslist_to_ca_array(reslist) -> np.ndarray:
    # CA-only;
    return np.asarray([res["CA"].get_coord() for res in reslist], dtype=np.float32)


def kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    """
    Compute RMSD between P and Q after optimal rigid superposition.
    Uses standard Kabsch algorithm matching Charnley's reference.
    """
    if P.shape != Q.shape:
        raise ValueError(f"Shape mismatch: P{P.shape} vs Q{Q.shape}")

    if P.shape[0] == 0:
        return float("nan")

    # Work in double precision
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)

    # Center both point clouds
    P_c = P - P.mean(axis=0)
    Q_c = Q - Q.mean(axis=0)

    # Cross-covariance matrix
    C = P_c.T @ Q_c

    # SVD: C = V @ diag(S) @ Wt
    V, S, Wt = np.linalg.svd(C)

    # Check for reflection (improper rotation)
    d = (np.linalg.det(V) * np.linalg.det(Wt)) < 0.0

    if d:
        # Flip sign of last column of V
        V[:, -1] = -V[:, -1]

    # Optimal rotation matrix
    R = V @ Wt

    # Apply rotation to P
    P_aligned = P_c @ R

    # Compute RMSD
    diff = P_aligned - Q_c
    msd = np.mean(np.sum(diff * diff, axis=1))
    rmsd = np.sqrt(msd)

    return float(rmsd)


def pairwise_rmsd_matrix_arrays(loops: List[np.ndarray]) -> np.ndarray:
    """Pairwise Kabsch RMSD matrix for a list of equal-length loops."""
    n = len(loops)
    D = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            dij = kabsch_rmsd(loops[i], loops[j])
            D[i, j] = D[j, i] = dij
    return D


def entropy_from_labels(labels: np.ndarray) -> float:
    if labels.size == 0:
        return 0.0
    counts = np.bincount(labels.astype(np.int64))
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


# Coordinate cache


def build_task_coord_cache(
    tasks: List[EvalTask], which: str = "gen"
) -> Dict[int, np.ndarray]:
    """
    Build CA coordinate cache for each EvalTask.
    Keyed by id(task) to avoid hashing issues.
    """
    assert which in ("gen", "ref")
    cache: Dict[int, np.ndarray] = {}

    for t in tqdm(tasks, desc=f"Coord cache ({which})", dynamic_ncols=True):
        model = (
            t.get_gen_biopython_model()
            if which == "gen"
            else t.get_ref_biopython_model()
        )
        reslist = extract_reslist(model, t.residue_first, t.residue_last)
        ca = reslist_to_ca_array(reslist)
        cache[id(t)] = ca
    return cache


# Clustering


def leader_clustering(
    loops: List[np.ndarray], tau: float
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Simple leader clustering using Kabsch RMSD:
    - First loop is first template
    - Assign to nearest template if within tau else create new template
    """
    templates: List[np.ndarray] = []
    labels = np.empty(len(loops), dtype=np.int32)

    for i, loop in enumerate(loops):
        if not templates:
            templates.append(loop)
            labels[i] = 0
            continue

        rmsds = [kabsch_rmsd(loop, tpl) for tpl in templates]
        j = int(np.argmin(rmsds))
        if rmsds[j] <= tau:
            labels[i] = j
        else:
            templates.append(loop)
            labels[i] = len(templates) - 1

    return templates, labels


def agglomerative_clustering(loops: List[np.ndarray], tau: float) -> np.ndarray:
    if len(loops) == 1:
        return np.zeros(1, dtype=np.int32)
    D = pairwise_rmsd_matrix_arrays(loops)
    return (
        AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=tau,
            linkage="average",
            metric="precomputed",
        )
        .fit_predict(D)
        .astype(np.int32)
    )


def cluster_loops(
    loops: List[np.ndarray], tau: float, method: str = "leader"
) -> Tuple[np.ndarray, Optional[List[np.ndarray]]]:
    if not loops:
        return np.array([], dtype=np.int32), None
    if method == "leader":
        tpls, lbls = leader_clustering(loops, tau)
        return lbls, tpls
    elif method == "agglomerative":
        lbls = agglomerative_clustering(loops, tau)
        return lbls, None
    else:
        raise ValueError(f"Unknown clustering method: {method}")


# Multimodal conformational coverage


def multimodal_conformational_coverage(
    tasks_eval: List[EvalTask],
    gen_coords_cache: Dict[int, np.ndarray],
    ref_coords_cache: Dict[int, np.ndarray],
    tau_cluster: float,
    tau_nat: float,
    clustering_method: str = "agglomerative",
    max_gen_per_group: Optional[int] = None,
) -> pd.DataFrame:
    """
    For each (structure, cdr, method), cluster N generated loops, compute:
    - num_clusters
    - entropy of cluster sizes
    - best (min) Kabsch RMSD to native among samples
    - nat_clusters: how many clusters have at least one sample within tau_nat of native
    """
    rows = []
    groups = defaultdict(list)
    for t in tasks_eval:
        groups[(t.structure, t.cdr, t.method)].append(t)

    for (structure, cdr, method), tasks in tqdm(
        groups.items(), desc="Multimodal coverage", dynamic_ncols=True
    ):
        if max_gen_per_group is not None and len(tasks) > max_gen_per_group:
            tasks = tasks[:max_gen_per_group]

        loops = [gen_coords_cache[id(t)] for t in tasks]
        native = ref_coords_cache[id(tasks[0])]

        # sanity: require same length
        Ls = [x.shape[0] for x in loops]
        if any(L != native.shape[0] for L in Ls):
            # skip inconsistent groups
            continue

        # cluster generated loops
        labels, _ = cluster_loops(loops, tau_cluster, method=clustering_method)
        if labels.size == 0:
            continue

        num_clusters = int(labels.max() + 1)
        ent = entropy_from_labels(labels)

        # nativeness: any sample in cluster within tau_nat
        # also compute best-of-N RMSD to native
        rmsd_to_native = np.asarray(
            [kabsch_rmsd(loop, native) for loop in loops], dtype=np.float32
        )
        bon_rmsd = float(np.nanmin(rmsd_to_native))

        nat_clusters = 0
        for k in range(num_clusters):
            idx = np.where(labels == k)[0]
            if idx.size == 0:
                continue
            if float(np.nanmin(rmsd_to_native[idx])) <= tau_nat:
                nat_clusters += 1

        rows.append(
            {
                "structure": structure,
                "cdr": cdr,
                "method": method,
                "length": int(native.shape[0]),
                "N": int(len(loops)),
                "tau_cluster": float(tau_cluster),
                "tau_nat": float(tau_nat),
                "clustering_method": clustering_method,
                "num_clusters": num_clusters,
                "entropy": ent,
                "nat_clusters": int(nat_clusters),
                "bon_rmsd": bon_rmsd,
            }
        )

    return pd.DataFrame(rows)


def _get_dataset_item_id(dataset, idx: int, item) -> str:
    """Best-effort identifier for a dataset item."""
    # Common dict keys
    if isinstance(item, dict):
        for k in ("pdb_id", "pdb", "structure", "complex_id", "id", "name"):
            v = item.get(k, None)
            if v is not None:
                return str(v)
    # Common dataset attributes
    for attr in ("pdb_ids", "structures", "ids", "names"):
        if hasattr(dataset, attr):
            seq = getattr(dataset, attr)
            try:
                return str(seq[idx])
            except Exception:
                pass
    # Fallback: deterministic index-based id
    return str(idx)


# Canonical family construction (training)


def extract_cdr_coords_from_dataset_item(
    item, chain_type: str
) -> Dict[str, np.ndarray]:
    """
    Extract CA coords for each CDR in a dataset item.
    This uses the dataset's own canonicalization implicitly via its processed tensors.
    (We keep it simple: use CA from pos_heavyatom and index via cdr_flag.)
    """
    chain = item.get(chain_type)
    if chain is None:
        return {}

    cdr_flag = chain["cdr_flag"].cpu().numpy()
    ca_all = chain["pos_heavyatom"][:, 1, :].cpu().numpy()  # CA is index 1

    out = {}
    prefix = "H" if chain_type == "heavy" else "L"
    for i in (1, 2, 3):
        cdr_type = f"{prefix}{i}"
        idx = np.where(cdr_flag == getattr(constants.CDR, cdr_type))[0]
        if idx.size:
            out[cdr_type] = ca_all[idx].astype(np.float32)
    return out


def build_canonical_families_from_dataset(
    dataset,
    tau_family: float,
    clustering_method: str = "leader",
    max_samples_per_key: Optional[int] = None,
) -> Tuple[
    Dict[Tuple[str, int], np.ndarray],
    Dict[Tuple[str, int], np.ndarray],
    Dict[Tuple[str, int], Dict[int, List[dict]]],
]:
    """
    Build empirical families from training natives, per (cdr_type, length).
    Uses leader clustering by default (scales), based on Kabsch RMSD.

    Returns:
      family_templates[key] = (K, L, 3)
      native_freqs[key]     = (K,)
      family_members[key][family_id] = list of member dicts with provenance fields:
         - pdb_id (best-effort; may fall back to dataset index)
         - dataset_index
         - chain_type ("heavy"/"light")
         - cdr_type (e.g., "H3")
         - length
    """
    # Store per-loop provenance so we can later recover which training structures belong to each family.
    native: Dict[Tuple[str, int], List[dict]] = defaultdict(list)

    for i in tqdm(
        range(len(dataset)), desc="Extracting CDRs (train)", dynamic_ncols=True
    ):
        item = dataset[i]
        pdb_id = _get_dataset_item_id(dataset, i, item)
        for ct in ("heavy", "light"):
            for cdr, coords in extract_cdr_coords_from_dataset_item(item, ct).items():
                key = (cdr, int(coords.shape[0]))
                native[key].append(
                    {
                        "coords": coords,
                        "pdb_id": pdb_id,
                        "dataset_index": int(i),
                        "chain_type": ct,
                        "cdr_type": cdr,
                        "length": int(coords.shape[0]),
                    }
                )

    family_templates: Dict[Tuple[str, int], np.ndarray] = {}
    native_freqs: Dict[Tuple[str, int], np.ndarray] = {}
    family_members: Dict[Tuple[str, int], Dict[int, List[dict]]] = {}

    for key, entries in tqdm(
        native.items(), desc="Clustering families (train)", dynamic_ncols=True
    ):
        if max_samples_per_key is not None and len(entries) > max_samples_per_key:
            entries = entries[:max_samples_per_key]

        loops = [e["coords"] for e in entries]

        # For families we want scalable, stable behavior -> leader
        labels, tpls = cluster_loops(loops, tau_family, method="leader")
        if labels.size == 0:
            continue
        K = int(labels.max() + 1)

        family_templates[key] = np.stack(tpls, axis=0).astype(np.float32)
        counts = np.bincount(labels, minlength=K).astype(np.float32)
        native_freqs[key] = counts / counts.sum()

        # NEW: provenance mapping: which training examples belong to each family id
        fam_map: Dict[int, List[dict]] = defaultdict(list)
        for lbl, e in zip(labels.tolist(), entries):
            fam_map[int(lbl)].append(
                {
                    "pdb_id": e["pdb_id"],
                    "dataset_index": int(e["dataset_index"]),
                    "chain_type": e["chain_type"],
                    "cdr_type": e["cdr_type"],
                    "length": int(e["length"]),
                }
            )
        family_members[key] = dict(fam_map)

    return family_templates, native_freqs, family_members


# Canonical family coverage (eval)


def canonical_family_coverage(
    tasks_eval: List[EvalTask],
    gen_coords_cache: Dict[int, np.ndarray],
    ref_coords_cache: Dict[int, np.ndarray],
    family_templates: Dict[Tuple[str, int], np.ndarray],
    native_freqs: Dict[Tuple[str, int], np.ndarray],
    num_other_families_for_overlay: int = 5,
) -> pd.DataFrame:
    rows = []
    groups = defaultdict(list)
    for t in tasks_eval:
        groups[(t.structure, t.cdr, t.method)].append(t)

    for (structure, cdr_tag, method), tasks in tqdm(
        groups.items(), desc="Canonical coverage", dynamic_ncols=True
    ):
        native = ref_coords_cache[id(tasks[0])]
        L = int(native.shape[0])
        key = (cdr_tag_to_type(cdr_tag), L)
        if key not in family_templates:
            continue

        tpls = family_templates[key]  # (K, L, 3)
        K = int(tpls.shape[0])

        # native -> family assignment + profile
        rmsds_nat = np.asarray(
            [kabsch_rmsd(native, tpls[k]) for k in range(K)], dtype=np.float32
        )
        native_family_id = int(np.argmin(rmsds_nat))
        native_family_rmsd = float(rmsds_nat[native_family_id])

        # choose some "other families" for overlay: farthest from native (visually distinct)
        order_desc = np.argsort(-rmsds_nat)  # descending
        other_ids = [int(i) for i in order_desc if int(i) != native_family_id][
            : max(0, int(num_other_families_for_overlay))
        ]
        other_rmsds = [float(rmsds_nat[i]) for i in other_ids]

        # existing: assign generated loops to families
        fam_ids = []
        for t in tasks:
            gen = gen_coords_cache[id(t)]
            if gen.shape[0] != L:
                continue
            rmsds = [kabsch_rmsd(gen, tpls[k]) for k in range(K)]
            fam_ids.append(int(np.argmin(rmsds)))

        if len(fam_ids) == 0:
            continue

        counts = np.bincount(np.asarray(fam_ids, dtype=np.int64), minlength=K).astype(
            np.float32
        )
        p_gen = counts / counts.sum()
        p_nat = native_freqs[key]
        if p_nat.shape[0] != p_gen.shape[0]:
            # Shouldn't happen if templates/freqs built together, but guard anyway
            minK = min(p_nat.shape[0], p_gen.shape[0])
            p_gen = p_gen[:minK]
            p_nat = p_nat[:minK]

        tv = float(0.5 * np.sum(np.abs(p_gen - p_nat)))

        # compact strings (keeps CSV simple and avoids storing big arrays)
        rmsd_profile = ",".join([f"{x:.4f}" for x in rmsds_nat.tolist()])
        other_ids_str = ",".join(map(str, other_ids))
        other_rmsds_str = ",".join([f"{x:.4f}" for x in other_rmsds])

        rows.append(
            {
                "structure": structure,
                "cdr": cdr_tag,
                "method": method,
                "length": L,
                "K_families": int(K),
                "num_families_used": int(np.count_nonzero(counts)),
                "tv_distance": tv,
                # NEW columns (for overlaying later)
                "native_family_id": native_family_id,
                "native_family_rmsd": native_family_rmsd,
                "native_to_family_rmsds": rmsd_profile,
                "top_other_family_ids": other_ids_str,
                "top_other_family_rmsds": other_rmsds_str,
            }
        )

    return pd.DataFrame(rows)


# Main


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=str, required=True, help="Root dir with generated results"
    )
    parser.add_argument(
        "--pfx", type=str, default="rosetta", help="Postfix for generated files"
    )

    parser.add_argument(
        "--train_summary", type=str, required=True, help="Path to training summary TSV"
    )
    parser.add_argument(
        "--train_chothia_dir",
        type=str,
        required=True,
        help="Path to training Chothia PDBs",
    )
    parser.add_argument(
        "--train_processed_dir",
        type=str,
        default="./data/processed",
        help="Cache dir for training",
    )

    parser.add_argument("--multimodal", action="store_true")
    parser.add_argument("--canon", action="store_true")

    parser.add_argument(
        "--clustering_method",
        type=str,
        default="agglomerative",
        choices=["leader", "agglomerative"],
    )
    parser.add_argument("--tau_cluster", type=float, default=2.0)
    parser.add_argument("--tau_nat", type=float, default=1.5)
    parser.add_argument("--tau_family", type=float, default=2.0)
    parser.add_argument("--max_gen_per_group", type=int, default=None)
    parser.add_argument("--max_train_samples", type=int, default=None)

    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--rebuild_families", action="store_true")

    args = parser.parse_args()

    run_multimodal = (
        args.multimodal or args.canon or (not args.multimodal and not args.canon)
    )
    run_canon = args.canon or (not args.multimodal and not args.canon)

    cache_dir = args.cache_dir or os.path.join(args.root, ".canon_cache")
    os.makedirs(cache_dir, exist_ok=True)

    print("Scanning generated results...")
    scanner_eval = TaskScanner(root=args.root, postfix=args.pfx, db=None)
    tasks_eval = scanner_eval.scan()
    if not tasks_eval:
        raise RuntimeError(f"No EvalTask under {args.root} with postfix={args.pfx}")
    print(f"Found {len(tasks_eval)} evaluation tasks")

    print("Building coordinate caches...")
    gen_coords_cache = build_task_coord_cache(tasks_eval, which="gen")
    ref_coords_cache = build_task_coord_cache(tasks_eval, which="ref")

    if run_multimodal:
        print("\n" + "=" * 60)
        print("MULTIMODAL CONFORMATIONAL COVERAGE")
        print("=" * 60)
        df_multi = multimodal_conformational_coverage(
            tasks_eval=tasks_eval,
            gen_coords_cache=gen_coords_cache,
            ref_coords_cache=ref_coords_cache,
            tau_cluster=args.tau_cluster,
            tau_nat=args.tau_nat,
            clustering_method=args.clustering_method,
            max_gen_per_group=args.max_gen_per_group,
        )
        out_multi = os.path.join(args.root, "multimodal_coverage.csv")
        df_multi.to_csv(out_multi, index=False, float_format="%.6f")
        print(f"\nWrote {len(df_multi)} rows to {out_multi}")
        print(f"    clustering_method={args.clustering_method}")

    if run_canon:
        print("\n" + "=" * 60)
        print("CANONICAL FAMILY COVERAGE")
        print("=" * 60)

        cache_key = hashlib.md5(
            f"{args.train_summary}|{args.tau_family}|leader".encode()
        ).hexdigest()
        fam_path = os.path.join(cache_dir, f"families_{cache_key}.pkl")

        if (not args.rebuild_families) and os.path.exists(fam_path):
            print(f"Loading cached families from {fam_path}...")
            with open(fam_path, "rb") as f:
                payload = pickle.load(f)
                if isinstance(payload, tuple) and len(payload) == 3:
                    family_templates, native_freqs, family_members = payload
                elif isinstance(payload, tuple) and len(payload) == 2:
                    # Backward-compatible cache (no family_members stored)
                    family_templates, native_freqs = payload
                    family_members = {}
                else:
                    raise ValueError("Unrecognized families cache format")
        else:
            print("Building canonical families from training set...")
            train_dataset = SAbDabDataset(
                summary_path=args.train_summary,
                chothia_dir=args.train_chothia_dir,
                processed_dir=args.train_processed_dir,
                split="train",
                transform=None,
            )

            family_templates, native_freqs, family_members = (
                build_canonical_families_from_dataset(
                    dataset=train_dataset,
                    tau_family=args.tau_family,
                    clustering_method="leader",
                    max_samples_per_key=args.max_train_samples,
                )
            )

            with open(fam_path, "wb") as f:
                pickle.dump(
                    (family_templates, native_freqs, family_members),
                    f,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            print(f"Cached families to {fam_path}")

        df_canon = canonical_family_coverage(
            tasks_eval=tasks_eval,
            gen_coords_cache=gen_coords_cache,
            ref_coords_cache=ref_coords_cache,
            family_templates=family_templates,
            native_freqs=native_freqs,
        )

        out_canon = os.path.join(args.root, "canonical_family_coverage.csv")
        df_canon.to_csv(out_canon, index=False, float_format="%.6f")
        print(f"\nWrote {len(df_canon)} rows to {out_canon}")


if __name__ == "__main__":
    main()
