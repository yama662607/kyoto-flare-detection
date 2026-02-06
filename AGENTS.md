# AGENTS.md

このドキュメントは、AI エージェントが本プロジェクトで効率的に作業するためのガイドラインです。

## Justfile ガイド

本プロジェクトでは、Python コードの品質チェックと CI/CD を標準化するために **Justfile** を使用しています。

### 基本コマンド

| コマンド              | 説明                              | 使用タイミング            |
| -------------------- | ---------------------------------------- | ---------------------- |
| `just` / `just check` | 全品質チェックの実行 (format + lint + test) | 作業開始前、PR 送信前 |
| `just fix`           | 自動修正 (format + lint --fix)           | エラーへの最初の対応 |
| `just setup`         | 環境セットアップ (依存関係のインストール) | 新規環境、依存関係更新後 |

### テスト

| コマンド                    | 説明          | 例                             |
| -------------------------- | -------------------- | ----------------------------------- |
| `just test`                | すべてのテストを実行        | -                                   |
| `just test path/to/file.py` | ファイル指定でテストを実行 | `just test tests/test_detector.py` |

### 各種タスク

| コマンド         | 説明                  | 目的               |
| --------------- | ---------------------------- | --------------------- |
| `just fmt`      | コードの整形             | 手動での整形     |
| `just fmt-check`| 整形状態のチェック             | CI でのチェック              |
| `just lint`     | 静的解析の実行          | コード品質のチェック    |
| `just lint-fix` | 静的解析エラーの自動修正         | 軽微なエラーの修正     |
| `just clean`    | キャッシュ・生成物の削除      | クリーンアップ               |

### 開発・運用

| コマンド     | 説明          | 備考        |
| ----------- | -------------------- | ------------ |
| `just dev`  | 開発サーバーの起動     | 未設定 |
| `just build`| 本番用ビルド     | 未設定 |

## エージェントのワークフロー

### 1. 作業開始時

```bash
just check  # 現在の品質状態を確認
```

### 2. コード変更後

```bash
just fix
just check
```

### 3. PR 送信前

```bash
just check && just test
```

### 4. 問題発見時

```bash
just fix
# 解決しない場合は手動で修正
```

## プロジェクト構成

```
kyoto-flare-detection/
├── justfile              # タスクランナー
├── pyproject.toml        # Python プロジェクト設定
├── src/                  # ソースコード
│   ├── base_flare_detector.py  # 基底クラス
│   ├── flarepy_DS_Tuc_A.py     # DS Tuc A 用実装
│   ├── flarepy_EK_Dra.py       # EK Dra 用実装
│   └── flarepy_V889_Her.py     # V889 Her 用実装
├── tests/                # テストコード
├── docs/                 # ドキュメント
├── notebooks/            # Jupyter notebooks
│   ├── flare_create_graphs.ipynb # メインの解析・グラフ生成
│   └── flare_detect_*.ipynb      # 星ごとの検出・解析
├── outputs/              # 出力ファイル (図表など)
│   └── debug/            # 実行ごとのデバッグ出力 (YYYYMMDD_HHMMSS)
└── tools/                # ユーティリティスクリプト
```

## ツール構成

- **パッケージマネージャー**: uv
- **フォーマッター**: Ruff format
- **リンター**: Ruff check
- **テスト**: pytest (最小限の動作確認)
- **対象ディレクトリ**: `src/`

## CI/CD との連携

Justfile のタスクは CI/CD と連携しています：

- **品質ゲート**: `just check`
- **自動修正**: `just fix`
- **標準化**: すべてのエージェントで共通のコマンドを使用

## 備考

- **引数の渡し方**: `just test` には追加引数を渡せます
- **環境変数**: `.env` ファイルは自動的に読み込まれます
- **エラー処理**: 各タスクは適切にエラーを処理することが期待されます
- **未設定タスク**: `dev` / `build` はプレースホルダーです
