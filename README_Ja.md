# Kyoto Flare Detection Project

## 🚀 クイックスタート

1. `uv` と `just` をインストールする（[セットアップガイド](#-セットアップガイド-setup-guide)参照）
2. リポジトリをクローンする
3. `uv sync` を実行して依存関係をインストールする
4. TESS の FITS データを `data/TESS/<star_name>/` に配置する
5. `notebooks/` 内のノートブックで解析を開始する


## 概要

`japanese` ブランチに、文章やコメントが日本語版のコードベースがあります。master ブランチは英語版です。

このプロジェクトは、TESS (Transiting Exoplanet Survey Satellite) の光度曲線データから恒星のフレアを検出し、そのエネルギーや頻度を分析するための Python フレームワークです。

### 主な機能

- TESS の FITS ファイルから光度曲線データを読み込み
- データのギャップ補正、デトレンド処理
- フレア現象の自動検出
- フレアのエネルギー、継続時間、ピーク時刻などの物理量を推定
- Plotly および Matplotlib による結果の可視化

## 重要

- TESS 衛星のデータは非常に大きいため、Git の追跡対象から除外しています。`data/`ディレクトリに TESS の FITS ファイルを配置して使用してください。
- `data/tess-response-function-v1.0.csv` に TESS 応答関数ファイルを配置してください。

構成例（ファイル名はセクターや対象により異なります）

```
data/
└── TESS/
    ├── DS_Tuc_A/
    │   └── tess2018206045859-s0001-0000000410214986-0120-s_lc.fits
    ├── EK_Dra/
    │   └── tess2019198215352-s0014-0000000159613900-0150-s_lc.fits
    └── V889_Her/
        └── tess2022164095748-s0053-0000000471000657-0226-s_lc.fits
└── tess-response-function-v1.0.csv
```

## コードの構成

このプロジェクトは、オブジェクト指向の設計パターンを採用しており、共通の処理を行う「基底クラス」と、星ごとに特化した処理を行う「派生クラス」に分かれています。

### `src/base_flare_detector.py`

`BaseFlareDetector` クラスを定義しています。このクラスは、フレア検出のコアとなるエンジン部分です。

- **役割**: 星の種類によらない共通のアルゴリズム（データの読み込み、ギャップ補正、デトレンド、フレア検出、エネルギー計算など）を実装します。
- **主なメソッド**:
  - `process_data()`: データ処理のパイプライン全体を実行します。
  - `plot_flare()`: `Plotly` を使用して光度曲線と検出されたフレアをインタラクティブに表示します。
  - `plot_flare_matplotlib()`: `Matplotlib` を使用して静的な光度曲線のグラフを生成します。

### `src/*_detector.py` (例: `src/ds_tuc_a_detector.py`)

特定の恒星に特化した派生クラスを定義しています。これらは `BaseFlareDetector` を継承して作成されます。

- **役割**: 各恒星の物理的特性（半径、温度など）や、データに見られる特有のノイズ（例: トランジット）に対応するための設定ファイルとして機能します。
- **実装**:
  - `__init__`メソッドをオーバーライドし、`super()`を通じて基底クラスにその星固有のパラメータを渡します。
  - 必要に応じて、`remove()` (データ除去) や `detrend_flux()` (デトレンド) などのメソッドをオーバーライドし、その星に特化したアルゴリズムを実装します。

## プロジェクト構成

```
/
├── data/                       # データファイル (Gitの追跡対象外)
│   ├── TESS/                   # TESSの.fitsファイルを恒星ごとに格納
│   └── tess-response-function-v1.0.csv # TESSの応答関数
├── notebooks/                  # 分析用Jupyter Notebook
│   ├── flare_create_graphs.ipynb   # [MAIN] 統合グラフ生成・分析ノートブック
│   ├── flare_detect_*.ipynb    # [STAR] 各恒星のフレア検出・解析 (DS_Tuc_A, EK_Dra, V889_Her)
│   └── learning/               # 実験・学習・検証用ノートブック
├── outputs/                    # 生成されたグラフや結果の出力先
│   └── figures/                # [Thesis] 論文用PDF図版の出力先
├── src/                        # ソースコード
│   ├── base_flare_detector.py  # [CORE] フレア検出のメインロジック
│   ├── flarepy_*.py            # [STAR] 各恒星用設定・実装 (DS_Tuc_A, EK_Dra, V889_Her)
│   └── visualization/          # [NEW] 論文用可視化モジュール
│       ├── flare_plots.py      # グラフ描画関数群
│       └── paper_style.py      # Matplotlib スタイル設定
├── docs/                       # プロジェクトドキュメント
├── tools/                    # ユーティリティ・保守スクリプト
├── .gitignore                  # Git除外設定
├── pyproject.toml              # プロジェクト依存関係・設定
├── justfile                    # タスクランナー (CI/CD, 検証用)
└── uv.lock                     # 依存パッケージロックファイル
```


## 使用方法

### VS Code での Jupyter Notebook カーネル設定

`uv`で作成した仮想環境を VS Code の Notebook で認識させるには、以下の手順を実行します。

1. `uv sync` を実行して `.venv` ディレクトリが作成されていることを確認します。
2. `ipykernel` パッケージをインストールします（Jupyter がカーネルを認識するために必要です）。
   ```bash
   uv add ipykernel
   ```
3. VS Code で Jupyter Notebook ファイル (`.ipynb`) を開きます。
4. 右上の **「カーネルの選択」** (`Select Kernel`) をクリックします。
5. **「Python 環境...」** (`Python Environments...`) を選択します。
6. リストから、プロジェクトルートにある `.venv` フォルダ内の Python インタプリタを選択します。これにより、Notebook がプロジェクトの仮想環境で実行されるようになります。

### 分析の実行

主な分析は `notebooks/` ディレクトリ内の Jupyter Notebook から行います。

```python
import sys
from pathlib import Path

# --- 1. Python実行パスの設定 ---
# 現在の作業ディレクトリの絶対パスを取得
PROJECT_ROOT = Path().resolve()

# 'notebooks' や 'src' ディレクトリから実行している場合、プロジェクトルートへ移動
if PROJECT_ROOT.name in ['notebooks', 'src']:
    PROJECT_ROOT = PROJECT_ROOT.parent

# 'src' モジュールを正しくインポートできるように、プロジェクトルートをパスに追加
sys.path.insert(0, str(PROJECT_ROOT))

# --- 2. フレア検出の実行 ---
# 対象とする恒星（例: DS Tuc A）専用の検出クラスをインポート
from src.flarepy_DS_Tuc_A import FlareDetector_DS_Tuc_A

# 解析したい TESS FITS ファイルのパス（PROJECT_ROOT を基準に指定）
file_path = PROJECT_ROOT / "data/TESS/DS_Tuc_A/tess2018206045859-s0001-0000000410214986-0120-s_lc.fits"

# 検出器のインスタンスを作成し、データ処理（デトレンド、検出、エネルギー計算等）を実行
# process_data=True にすると、データの読み込みから解析までの一連のパイプラインが自動的に実行されます
detector = FlareDetector_DS_Tuc_A(file=file_path, process_data=True)

# Plotly を使用して、光度曲線と検出されたフレアをインタラクティブに表示・確認
detector.plot_flare()

# Matplotlib を使用して、エネルギー分布（発生頻度分布）をプロット
detector.plot_energy_matplotlib()
```


## Outputs

`docs/OUTPUTS.md` に生成物とデバッグ出力の場所が記載されています。

---

## 🔧 セットアップガイド (Setup Guide)

### 1. uv のインストール

Python の環境構築を高速かつ確実に行うために使用します。

-   **macOS / Linux:**
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
-   **Windows:**
    ```powershell
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```
-   **その他パッケージマネージャー:**
    -   macOS (Homebrew): `brew install uv`
    -   Windows (winget): `winget install astral-sh.uv`

### 2. Just のインストール

プロジェクト内の様々なコマンド（ビルド、チェック、サーバー起動など）を実行するために必要です。

-   **macOS (Homebrew):** `brew install just`
*   **Windows (winget):** `winget install casey.just`
*   **Linux (Ubuntu/Debian):** `sudo apt install just`

> [!TIP]
> インストール後、ターミナルを再起動して `just --version` および `uv --version` が動作することを確認してください。

> [!NOTE]
> **Windows との互換性:** このプロジェクトでは、すべてのファイルパス操作に `Pathlib` を使用しており、macOS、Linux、Windows 間での互換性を確保しています。`just` を使用する場合、`justfile` 内のユーティリティタスクを円滑に実行するために、標準的なコマンド（Git Bash など）をサポートするシェルを使用することをお勧めします。
