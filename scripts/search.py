"""检索 RAG 索引；本地索引缺失时自动从 GitHub Release 下载。

用法：
  python scripts/search.py "你的问题" -k 5
  python scripts/search.py "问题" --repo owner/name --tag index-latest

索引默认缓存到仓库根目录的 index/（已在 .gitignore 中排除）。
私有仓库下载依赖已登录的 gh CLI（gh auth login）。
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_TAG = "index-latest"
ASSET_NAME = "rag-index.tar.gz"
REPO_ROOT = Path(__file__).resolve().parent.parent


def detect_repo():
    """优先取 RAG_REPO 环境变量，否则从 git remote origin 推断。"""
    if os.environ.get("RAG_REPO"):
        return os.environ["RAG_REPO"]
    try:
        url = subprocess.check_output(
            ["git", "remote", "get-url", "origin"],
            cwd=REPO_ROOT,
            text=True,
        ).strip()
    except Exception:
        return None
    url = url.rstrip("/")
    if url.endswith(".git"):
        url = url[:-4]
    parts = re.split(r"[:/]", url)
    return "/".join(parts[-2:]) if len(parts) >= 2 else None


def download_index(repo, tag, index_dir):
    """用 gh CLI 下载并解包 Release 附件到 index_dir。"""
    index_dir.mkdir(parents=True, exist_ok=True)
    print(f"正在从 {repo}@{tag} 下载索引...", file=sys.stderr)
    subprocess.run(
        [
            "gh",
            "release",
            "download",
            tag,
            "--repo",
            repo,
            "--pattern",
            ASSET_NAME,
            "--dir",
            str(index_dir),
            "--clobber",
        ],
        check=True,
    )
    archive = index_dir / ASSET_NAME
    with tarfile.open(archive) as tar:
        tar.extractall(index_dir)
    archive.unlink()


def ensure_index(index_dir, repo, tag):
    """确保 index_dir 中存在可用索引，返回其路径。"""
    if (index_dir / "embeddings.npy").exists() and (index_dir / "chunks.jsonl").exists():
        return index_dir
    if not repo:
        print(
            "未找到索引，且无法确定仓库（请设置 RAG_REPO 或用 --repo 指定）。",
            file=sys.stderr,
        )
        raise SystemExit(1)
    download_index(repo, tag, index_dir)
    return index_dir


def load_index(index_dir):
    embeddings = np.load(index_dir / "embeddings.npy")
    chunks = [
        json.loads(line)
        for line in (index_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return embeddings, chunks


def search(query, k, index_dir, model_name):
    embeddings, chunks = load_index(index_dir)
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    vector = model.encode([query], normalize_embeddings=True)[0]
    scores = embeddings @ np.asarray(vector, dtype=np.float32)
    top = np.argsort(-scores)[:k]
    results = []
    for idx in top:
        record = dict(chunks[int(idx)])
        record["score"] = float(scores[int(idx)])
        results.append(record)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="检索问题")
    parser.add_argument("-k", type=int, default=5, help="返回片段数")
    parser.add_argument("--repo", default=None, help="owner/name，默认自动推断")
    parser.add_argument("--tag", default=DEFAULT_TAG, help="Release 标签")
    parser.add_argument("--index", default=str(REPO_ROOT / "index"), help="索引缓存目录")
    parser.add_argument("--model", default=MODEL_NAME)
    args = parser.parse_args()

    repo = args.repo or detect_repo()
    index_dir = ensure_index(Path(args.index), repo, args.tag)
    results = search(args.query, args.k, index_dir, args.model)

    for rank, record in enumerate(results, 1):
        print(f"[{rank}] 相似度 {record['score']:.4f} | {record['source']} 第 {record['page']} 页")
        print(record["text"])
        print("-" * 60)


if __name__ == "__main__":
    main()
