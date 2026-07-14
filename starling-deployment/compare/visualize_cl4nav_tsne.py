#!/usr/bin/env python3
"""Visualize CL4Nav RGB/depth features on a held-out paired dataset."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Dataset


class ResNetCL4Nav(nn.Module):
    """Architecture used by the original CL4Nav ResNet checkpoint."""

    def __init__(self, base_model: str, out_dim: int):
        super().__init__()
        constructors = {"resnet18": models.resnet18, "resnet50": models.resnet50}
        if base_model not in constructors:
            raise ValueError(f"Unsupported CL4Nav backbone: {base_model}")
        self.backbone = constructors[base_model](weights=None, num_classes=out_dim)
        hidden_dim = self.backbone.fc.in_features
        final_fc = self.backbone.fc
        self.backbone.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), final_fc
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.backbone(image)


def frame_id(path: Path) -> str:
    return path.stem.rsplit("_", 1)[-1]


class PairedEvalDataset(Dataset):
    def __init__(
        self,
        root: Path,
        image_size: int,
        max_pairs: int | None,
        sample_pairs: int | None,
        seed: int,
    ):
        rgb_by_id = {frame_id(path): path for path in (root / "RGB").glob("*.png")}
        depth_by_id = {
            frame_id(path): path for path in (root / "Depth").glob("*.png")
        }
        if not rgb_by_id or not depth_by_id:
            raise FileNotFoundError(
                f"Expected PNG files under {root / 'RGB'} and {root / 'Depth'}"
            )
        missing_depth = sorted(rgb_by_id.keys() - depth_by_id.keys())
        missing_rgb = sorted(depth_by_id.keys() - rgb_by_id.keys())
        if missing_depth or missing_rgb:
            raise ValueError(
                "RGB/depth frame IDs do not match: "
                f"missing depth={missing_depth[:5]}, missing RGB={missing_rgb[:5]}"
            )
        ids = sorted(rgb_by_id, key=lambda value: int(value))
        if max_pairs is not None and sample_pairs is not None:
            raise ValueError("Use only one of --max-pairs and --sample-pairs")
        if max_pairs is not None:
            if max_pairs < 2:
                raise ValueError("--max-pairs must be at least 2")
            ids = ids[:max_pairs]
        elif sample_pairs is not None:
            if not 2 <= sample_pairs <= len(ids):
                raise ValueError(
                    f"--sample-pairs must be between 2 and {len(ids)}"
                )
            ids = sorted(
                random.Random(seed).sample(ids, sample_pairs),
                key=lambda value: int(value),
            )
        self.samples = [(value, rgb_by_id[value], depth_by_id[value]) for value in ids]
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.samples)

    def _tensor(self, path: Path, mode: str) -> torch.Tensor:
        with Image.open(path) as image:
            image = image.convert(mode)
            if image.size != (self.image_size, self.image_size):
                image = image.resize(
                    (self.image_size, self.image_size), Image.Resampling.BILINEAR
                )
            array = np.asarray(image, dtype=np.float32) / 255.0
        if mode == "L":
            array = np.repeat(array[..., None], 3, axis=2)
        return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))

    def __getitem__(self, index: int):
        pair_id, rgb_path, depth_path = self.samples[index]
        return pair_id, self._tensor(rgb_path, "RGB"), self._tensor(depth_path, "L")


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[nn.Module, int]:
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint has no state_dict: {checkpoint_path}")
    state_dict = checkpoint["state_dict"]
    output_weight = state_dict.get("backbone.fc.2.weight")
    if output_weight is None:
        raise ValueError("Checkpoint does not contain backbone.fc.2.weight")
    out_dim = int(output_weight.shape[0])
    model = ResNetCL4Nav(checkpoint.get("arch", "resnet50"), out_dim)
    model.load_state_dict(state_dict, strict=True)
    model.eval().to(device)
    return model, out_dim


def extract_features(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[list[str], np.ndarray, np.ndarray]:
    pair_ids: list[str] = []
    rgb_features: list[np.ndarray] = []
    depth_features: list[np.ndarray] = []
    with torch.inference_mode():
        for ids, rgb, depth in loader:
            rgb_output = model(rgb.to(device, non_blocking=True))
            depth_output = model(depth.to(device, non_blocking=True))
            pair_ids.extend(ids)
            rgb_features.append(rgb_output.cpu().numpy())
            depth_features.append(depth_output.cpu().numpy())
    return pair_ids, np.concatenate(rgb_features), np.concatenate(depth_features)


def l2_normalize(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norms, np.finfo(features.dtype).eps)


def save_plot(
    embedding: np.ndarray,
    pair_count: int,
    output_path: Path,
    perplexity: float,
    connect_pairs: bool,
    title: str,
) -> None:
    fig, axis = plt.subplots(figsize=(10, 8), constrained_layout=True)
    rgb_xy = embedding[:pair_count]
    depth_xy = embedding[pair_count:]
    if connect_pairs:
        for rgb_point, depth_point in zip(rgb_xy, depth_xy):
            axis.plot(
                [rgb_point[0], depth_point[0]],
                [rgb_point[1], depth_point[1]],
                color="0.75", linewidth=0.35, alpha=0.25, zorder=1,
            )
    axis.scatter(
        rgb_xy[:, 0], rgb_xy[:, 1], s=18, alpha=0.72, marker="o",
        color="#1f77b4", edgecolors="none", label=f"RGB (n={pair_count})", zorder=2,
    )
    axis.scatter(
        depth_xy[:, 0], depth_xy[:, 1], s=18, alpha=0.72, marker="^",
        color="#d62728", edgecolors="none", label=f"Depth (n={pair_count})", zorder=2,
    )
    axis.set_title(f"{title}\nt-SNE, perplexity={perplexity:g}")
    axis.set_xlabel("t-SNE 1")
    axis.set_ylabel("t-SNE 2")
    axis.legend(frameon=True)
    axis.grid(alpha=0.15)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run t-SNE on paired RGB/depth features from a frozen CL4Nav encoder."
    )
    parser.add_argument(
        "--dataset", type=Path,
        default=Path("/home/jason/CL4Nav/datasets/provided_eval"),
        help="Directory containing RGB/ and Depth/ subdirectories.",
    )
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path(
            "/home/jason/CL4Nav/runs_agile/0415_simulation_1000_epochs_test2/"
            "checkpoint_1000.pth.tar"
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).resolve().parent / "results" / "cl4nav_tsne",
    )
    parser.add_argument(
        "--output-stem", default="cl4nav_eval_rgb_vs_depth_tsne",
        help="Base filename used for the PNG, NPZ, and CSV outputs.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument(
        "--sample-pairs", type=int, default=None,
        help="Randomly sample this many paired frames using --seed.",
    )
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--title", default="CL4Nav features on held-out eval data",
        help="First line of the plot title.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--connect-pairs", action="store_true",
        help="Draw a faint line between each matched RGB/depth point.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.workers < 0:
        raise ValueError("--batch-size must be positive and --workers non-negative")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dataset = PairedEvalDataset(
        args.dataset, args.image_size, args.max_pairs, args.sample_pairs, args.seed
    )
    sample_count = 2 * len(dataset)
    if not 0 < args.perplexity < sample_count:
        raise ValueError(
            f"--perplexity must be between 0 and {sample_count}, got {args.perplexity}"
        )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=device.type == "cuda",
    )
    model, feature_dim = load_model(args.checkpoint, device)
    pair_ids, rgb_features, depth_features = extract_features(model, loader, device)

    # CL4Nav's InfoNCE loss compares L2-normalized projection-head outputs.
    joint_features = l2_normalize(np.concatenate([rgb_features, depth_features]))
    embedding = TSNE(
        n_components=2, perplexity=args.perplexity, learning_rate="auto",
        init="pca", max_iter=args.iterations, random_state=args.seed,
    ).fit_transform(joint_features)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if Path(args.output_stem).name != args.output_stem:
        raise ValueError("--output-stem must be a filename stem, not a path")
    plot_path = args.output_dir / f"{args.output_stem}.png"
    archive_path = args.output_dir / f"{args.output_stem}.npz"
    csv_path = args.output_dir / f"{args.output_stem}.csv"
    labels = np.asarray(["RGB"] * len(dataset) + ["Depth"] * len(dataset))
    ids = np.asarray(pair_ids + pair_ids)
    raw_features = np.concatenate([rgb_features, depth_features])
    np.savez_compressed(
        archive_path, embedding=embedding, features=raw_features,
        normalized_features=joint_features, labels=labels, pair_ids=ids,
    )
    with csv_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(("pair_id", "modality", "tsne_1", "tsne_2"))
        writer.writerows(zip(ids, labels, embedding[:, 0], embedding[:, 1]))
    save_plot(
        embedding, len(dataset), plot_path, args.perplexity,
        args.connect_pairs, args.title,
    )

    paired_cosine = np.sum(
        l2_normalize(rgb_features) * l2_normalize(depth_features), axis=1
    )
    print(f"device: {device}")
    print(f"pairs: {len(dataset)}; points: {sample_count}; feature dim: {feature_dim}")
    print(
        "paired RGB/depth cosine similarity: "
        f"mean={paired_cosine.mean():.4f}, std={paired_cosine.std():.4f}"
    )
    print(f"plot: {plot_path}")
    print(f"data: {archive_path}")
    print(f"csv:  {csv_path}")


if __name__ == "__main__":
    main()
