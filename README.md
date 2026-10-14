# prodcuct_learning
老汪的产品学习空间

## RAG 知识库

把书籍原文（PDF / EPUB）放进 `books/`，GitHub Actions 会在云端解析、切块、生成向量索引，并把索引发布为 **Release 附件**（`rag-index.tar.gz`）。索引不提交到 Git 仓库，任意电脑登录同一账号后都能取用。

### 工作流

```
books/ 原文 ──push──► GitHub Actions 构建索引 ──► Release 附件 rag-index.tar.gz
                                                        │
任意电脑 Trae 打开仓库 ──► python scripts/search.py "问题" ──► top-k 片段 ──► Agent 作答
```

### 使用步骤

1. 把 PDF / EPUB 放到 `books/` 并推送到 `main`，Actions 自动构建索引（也可在 Actions 页面手动 Run workflow）。
2. 在任意电脑的 Trae 中打开本仓库，先确保已登录 GitHub CLI：
   ```bash
   gh auth login
   ```
3. 运行检索：
   ```bash
   python -m pip install -r requirements.txt
   python scripts/search.py "你的问题" -k 5
   ```
   首次运行会自动从最新 Release 下载索引到 `index/`（已在 `.gitignore` 中排除）。

### 配置项

- 嵌入模型：`paraphrase-multilingual-MiniLM-L12-v2`（多语言，零 API Key、零费用）。
- 索引标签：`index-latest`；下载命令：`gh release download index-latest --pattern rag-index.tar.gz`。
- 私有仓库下载索引依赖 `gh auth login`；也可用 `RAG_REPO=owner/name` 或 `--repo` 指定仓库。

### 目录

```
books/                原文（进 Git）
scripts/build_index.py  解析 + 切块 + 嵌入 + 生成索引
scripts/search.py       检索（自动下载缺失索引）
.github/workflows/build-index.yml  云构建工作流
index/                索引缓存（不进 Git，源自 Release 附件）
```
