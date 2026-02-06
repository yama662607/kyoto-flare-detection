# Kyoto Flare Detection Project

## 概要

この日本語版は `japanese` ブランチにオリジナルを保持しています。日本語の内容を参照したい場合は `japanese` ブランチを確認してください。


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

## セットアップ

本プロジェクトは、Python のパッケージ管理ツールとして `uv` を使用します。

### 1. `uv` のインストール

`uv`がインストールされていない場合は、お使いの OS に応じて以下のコマンドを実行してください。

**macOS / Linux:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**

```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

インストール後、ターミナルを再起動してください。

### 2. 仮想環境の作成と依存関係のインストール

プロジェクトのルートディレクトリで以下のコマンドを実行すると、`.venv` という名前の仮想環境が作成され、`uv.lock`に基づいて依存パッケージがインストールされます。

```bash
uv sync
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

# プロジェクトルートの設定
PROJECT_ROOT = Path().resolve()
if PROJECT_ROOT.name in ['notebooks', 'src']:
    PROJECT_ROOT = PROJECT_ROOT.parent

sys.path.insert(0, str(PROJECT_ROOT))

from src.flarepy_DS_Tuc_A import FlareDetector_DS_Tuc_A

# 解析したいFITSファイルのパス
file_path = "data/TESS/DS_Tuc_A/tess2018206045859-s0001-0000000410214986-0120-s_lc.fits"

# インスタンスを作成し、データ処理を実行
# process_data=True にすると、データの読み込みからフレア検出までの一連の処理が自動的に実行されます。
detector = FlareDetector_DS_Tuc_A(file=file_path, process_data=True)

# 結果をプロット
detector.plot_flare() # Plotlyによる光度曲線プロット
detector.plot_energy_matplotlib() # Matplotlibによるエネルギー分布プロット
```


## Outputs

See `docs/OUTPUTS.md` for generated artifacts and debug output locations.
