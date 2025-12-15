#!/usr/bin/env python3
"""PDFからMethod/Observations/Analysis相当節を抽出して周辺テキストを表示するスクリプト"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path
from typing import NamedTuple


class MethodHit(NamedTuple):
    page: int
    keyword: str
    snippet: str


def extract_text_with_pdftotext(pdf_path: Path) -> list[str]:
    """pdftotextでPDFからテキストをページごとに抽出"""
    try:
        result = subprocess.run(
            ["pdftotext", "-layout", "-enc", "UTF-8", str(pdf_path), "-"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        raise RuntimeError(f"pdftotextの実行に失敗しました: {e}") from e

    # ページ区切り（form feed）で分割
    pages = result.stdout.split("\f")
    # 前後の空白を正規化
    return [re.sub(r"\s+", " ", page).strip() for page in pages]


def find_method_pages(pages: list[str]) -> list[MethodHit]:
    """Method/Observations/Analysisなどのキーワードを含むページを検索"""
    patterns = [
        (r"\bMETHODS?\b", "METHOD"),
        (r"\bOBSERVATIONS?\b", "OBSERVATIONS"),
        (r"\bDATA\b", "DATA"),
        (r"\bANALYSIS\b", "ANALYSIS"),
        (r"\bPROCEDURE\b", "PROCEDURE"),
        (r"\bFLARE\b", "FLARE"),
        (r"\bDETECTION\b", "DETECTION"),
        (r"\bSELECTION\b", "SELECTION"),
    ]

    hits: list[MethodHit] = []
    for idx, page_text in enumerate(pages, start=1):
        if not page_text:
            continue
        for pattern, keyword in patterns:
            if re.search(pattern, page_text, re.IGNORECASE):
                # 最初の350文字をスニペットとして保存
                snippet = page_text[:350] + "..." if len(page_text) > 350 else page_text
                hits.append(MethodHit(page=idx, keyword=keyword, snippet=snippet))
                break  # 同じページでは最初のマッチのみ記録
    return hits


def print_method_hits(pdf_name: str, hits: list[MethodHit], pages: list[str], max_pages: int = 40) -> None:
    """Method候補ページを整形して表示"""
    print(f"\n===== {pdf_name} =====")
    if not hits:
        print("キーワードに一致するページが見つかりませんでした")
        return
    print(f"キーワードヒット数: {len(hits)}")
    for hit in hits[:max_pages]:
        print(f"\n--- p.{hit.page} [{hit.keyword}] ---")
        # 該当ページの全文を表示
        if hit.page <= len(pages):
            full_text = pages[hit.page - 1]
            print(full_text[:2000] + "..." if len(full_text) > 2000 else full_text)
        else:
            print("ページ番号が範囲外です")
    if len(hits) > max_pages:
        print(f"\n...（{len(hits) - max_pages}件のヒットを省略）")


def main() -> None:
    parser = argparse.ArgumentParser(description="PDFからMethod/Observations/Analysis相当節を抽出")
    parser.add_argument(
        "pdfs",
        nargs="+",
        type=Path,
        help="対象PDFファイル（複数可）",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=40,
        help="表示する最大ページ数（既定: 40）",
    )
    parser.add_argument(
        "--page",
        type=int,
        help="特定ページ番号を指定して全文表示",
    )
    args = parser.parse_args()

    for pdf_path in args.pdfs:
        if not pdf_path.exists():
            print(f"[ERROR] ファイルが存在しません: {pdf_path}")
            continue
        if pdf_path.suffix.lower() != ".pdf":
            print(f"[ERROR] PDFファイルではありません: {pdf_path}")
            continue

        try:
            pages = extract_text_with_pdftotext(pdf_path)
            if args.page:
                # 特定ページを表示
                if 1 <= args.page <= len(pages):
                    print(f"\n===== {pdf_path.name} - p.{args.page} =====")
                    print(pages[args.page - 1])
                else:
                    print(f"[ERROR] ページ番号 {args.page} は範囲外です（1-{len(pages)}）")
            else:
                hits = find_method_pages(pages)
                print_method_hits(pdf_path.name, hits, pages, args.max_pages)
        except Exception as e:
            print(f"[ERROR] {pdf_path.name} の処理中にエラー: {e}")


if __name__ == "__main__":
    main()
