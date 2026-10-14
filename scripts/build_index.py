"""从 books/ 下的 PDF / EPUB 构建 RAG 向量索引。

产物写入 --out 目录（默认 index/）：
  chunks.jsonl    每行一个切片，含 id / source / page / text
  embeddings.npy  归一化后的向量矩阵 (float32)
  meta.json       构建元信息（模型名、切片数）

索引不提交到 Git，由 CI 打包为 Release 附件。
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


def read_pdf(path):
    """逐页产出 (页码, 文本)。"""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    for page_no, page in enumerate(reader.pages, 1):
        yield page_no, page.extract_text() or ""


def read_epub(path):
    """逐章产出 (章节序号, 文本)。"""
    import ebooklib
    from bs4 import BeautifulSoup
    from ebooklib import epub

    book = epub.read_epub(str(path))
    for index, item in enumerate(book.get_items_of_type(ebooklib.ITEM_DOCUMENT), 1):
        soup = BeautifulSoup(item.get_content(), "html.parser")
        yield index, soup.get_text("\n")


def iter_books(books_dir):
    """遍历 books/ 下受支持的文件，产出 (路径, 页/章迭代器)。"""
    for path in sorted(Path(books_dir).rglob("*")):
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            yield path, read_pdf(path)
        elif suffix == ".epub":
            yield path, read_epub(path)


def normalize(text):
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"[ \t\u3000]+", " ", text)
    return text.strip()


def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    text = normalize(text)
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--books", default="books", help="原文目录")
    parser.add_argument("--out", default="index", help="索引输出目录")
    parser.add_argument("--model", default=MODEL_NAME, help="嵌入模型名")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=CHUNK_OVERLAP)
    args = parser.parse_args()

    records = []
    for path, pages in iter_books(args.books):
        count = 0
        for page_no, page_text in pages:
            for piece in chunk_text(page_text, args.chunk_size, args.overlap):
                records.append(
                    {
                        "id": len(records),
                        "source": path.as_posix(),
                        "page": page_no,
                        "text": piece,
                    }
                )
                count += 1
        print(f"{path}: {count} chunks", file=sys.stderr)

    if not records:
        print("没有可索引的内容，请把 PDF / EPUB 放到 books/ 下。", file=sys.stderr)
        raise SystemExit(1)

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(args.model)
    embeddings = model.encode(
        [r["text"] for r in records],
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "chunks.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    np.save(out / "embeddings.npy", embeddings)
    (out / "meta.json").write_text(
        json.dumps({"model": args.model, "count": len(records)}, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"已生成 {len(records)} 个切片 -> {out}/", file=sys.stderr)


if __name__ == "__main__":
    main()
