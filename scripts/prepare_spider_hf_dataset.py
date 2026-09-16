"""One-time offline prep: turn the official Spider archive into an HF dataset
repo that lm_eval/tasks/spider/utils.py can pull from at eval time.

Usage:
    python scripts/prepare_spider_hf_dataset.py \
        --zip .cache/spider_raw/spider.zip \
        --repo-id <your-hf-username>/spider-eval \
        --push

Run without --push first to build .cache/spider_prepared/ locally and eyeball
it before uploading anything to the Hub.
"""

import argparse
import json
import shutil
import zipfile
from pathlib import Path


def build_schema_text(table: dict) -> str:
    table_names = table["table_names_original"]
    col_names = table["column_names_original"]
    col_types = table["column_types"]
    pk_idxs = set(table["primary_keys"])
    fks = table["foreign_keys"]

    # columns_by_table[table_idx] = list of (col_idx, col_name, col_type)
    columns_by_table: dict[int, list] = {i: [] for i in range(len(table_names))}
    for col_idx, (tbl_idx, col_name) in enumerate(col_names):
        if tbl_idx == -1:
            continue
        columns_by_table[tbl_idx].append((col_idx, col_name, col_types[col_idx]))

    fk_by_child_col = {child: parent for child, parent in fks}

    statements = []
    for tbl_idx, tbl_name in enumerate(table_names):
        lines = []
        for col_idx, col_name, col_type in columns_by_table[tbl_idx]:
            parts = [col_name, col_type]
            if col_idx in pk_idxs:
                parts.append("PRIMARY KEY")
            lines.append("  " + " ".join(parts))
            if col_idx in fk_by_child_col:
                parent_idx = fk_by_child_col[col_idx]
                parent_tbl_idx, parent_col_name = col_names[parent_idx]
                parent_tbl_name = table_names[parent_tbl_idx]
                lines.append(
                    f"  FOREIGN KEY ({col_name}) REFERENCES {parent_tbl_name}({parent_col_name})"
                )
        statements.append(f"CREATE TABLE {tbl_name} (\n" + ",\n".join(lines) + "\n);")

    return "\n".join(statements)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True, type=Path)
    ap.add_argument("--out", default=Path(".cache/spider_prepared"), type=Path)
    ap.add_argument("--repo-id", default=None, help="e.g. hl3816/spider-eval")
    ap.add_argument("--push", action="store_true")
    args = ap.parse_args()

    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)

    extract_dir = args.out / "_extracted"
    with zipfile.ZipFile(args.zip) as zf:
        zf.extractall(extract_dir)

    # The official archive extracts to a single top-level data folder, plus a
    # __MACOSX junk folder from the original macOS-created zip.
    root_candidates = [
        p for p in extract_dir.iterdir() if p.is_dir() and p.name != "__MACOSX"
    ]
    root = root_candidates[0] if len(root_candidates) == 1 else extract_dir

    tables = json.loads((root / "tables.json").read_text(encoding="utf-8"))
    schema_map = {t["db_id"]: build_schema_text(t) for t in tables}
    (args.out / "schema.json").write_text(
        json.dumps(schema_map, indent=2), encoding="utf-8"
    )

    train = json.loads((root / "train_spider.json").read_text(encoding="utf-8"))
    dev = json.loads((root / "dev.json").read_text(encoding="utf-8"))

    data_dir = args.out / "data"
    data_dir.mkdir()
    for split_name, split_data in [("train", train), ("validation", dev)]:
        rows = [
            {"db_id": d["db_id"], "question": d["question"], "query": d["query"]}
            for d in split_data
        ]
        (data_dir / f"{split_name}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows), encoding="utf-8"
        )

    # Only keep the .sqlite file per db_id -- the official archive also bundles
    # raw CSV dumps, .sql text dumps, and annotation.json per database, none of
    # which we need, and the CSVs trip up the Hub's binary-file upload check.
    db_src = root / "database"
    db_dst = args.out / "database"
    db_dst.mkdir()
    missing = []
    for db_id in schema_map:
        src_file = db_src / db_id / f"{db_id}.sqlite"
        if not src_file.exists():
            missing.append(db_id)
            continue
        dst_dir = db_dst / db_id
        dst_dir.mkdir()
        shutil.copy2(src_file, dst_dir / f"{db_id}.sqlite")
    if missing:
        print(f"WARNING: {len(missing)} db_ids had no .sqlite file: {missing}")

    shutil.rmtree(extract_dir)

    print(f"Prepared dataset at {args.out}")
    print(f"  schema.json: {len(schema_map)} databases")
    print(f"  train: {len(train)} examples, validation: {len(dev)} examples")
    print(f"  database/: {len(list(db_dst.iterdir()))} db folders")

    if args.push:
        if not args.repo_id:
            raise SystemExit("--push requires --repo-id")
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.repo_id, repo_type="dataset", exist_ok=True)
        api.upload_folder(
            folder_path=str(args.out),
            repo_id=args.repo_id,
            repo_type="dataset",
        )
        print(f"Pushed to https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
