# Kyoto Flare Detection Project

## 概要

このプロジェクトは、TESS (Transiting Exoplanet Survey Satellite) の光度曲線データから恒星のフレアを検出し、そのエネルギーや頻度を分析するためのPythonフレームワークです。

### 主な機能

- TESSのFITSファイルから光度曲線データを読み込み
- データのギャップ補正、デトレンド処理
- フレア現象の自動検出
- フレアのエネルギー、継続時間、ピーク時刻などの物理量を推定
- PlotlyおよびMatplotlibによる結果の可視化

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
├── data/                # データファイル (Gitの追跡対象外)
│   ├── TESS/            # TESSの.fitsファイルを恒星ごとに格納
│   └── tess-response-function-v1.0.csv # TESSの応答関数
├── notebooks/           # 分析用Jupyter Notebook
├── outputs/             # 生成されたグラフや結果の出力先
├── src/                 # ソースコード
│   ├── __init__.py
│   ├── archive/         # リファクタリング前の旧コード
│   ├── base_flare_detector.py   # フレア検出の共通ロジックを実装した基底クラス
│   ├── ds_tuc_a_detector.py     # DS Tuc Aに特化した派生クラス
│   ├── ek_dra_detector.py       # EK Draに特化した派生クラス
│   └── v889_her_detector.py     # V889 Herに特化した派生クラス
├── .gitignore           # Gitの追跡対象外ファイルを指定
├── pyproject.toml       # プロジェクトのメタデータと依存関係
└── uv.lock              # uv用の固定された依存関係ファイル
```

## セットアップ

本プロジェクトは、Pythonのパッケージ管理ツールとして `uv` を使用します。

### 1. `uv` のインストール

`uv`がインストールされていない場合は、お使いのOSに応じて以下のコマンドを実行してください。

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

### VS CodeでのJupyter Notebookカーネル設定

`uv`で作成した仮想環境をVS CodeのNotebookで認識させるには、以下の手順を実行します。

1. `uv sync` を実行して `.venv` ディレクトリが作成されていることを確認します。
2. `ipykernel` パッケージをインストールします（Jupyterがカーネルを認識するために必要です）。
   ```bash
   uv pip install ipykernel
   ```
3. VS CodeでJupyter Notebookファイル (`.ipynb`) を開きます。
4. 右上の **「カーネルの選択」** (`Select Kernel`) をクリックします。
5. **「Python 環境...」** (`Python Environments...`) を選択します。
6. リストから、プロジェクトルートにある `.venv` フォルダ内のPythonインタプリタを選択します。これにより、Notebookがプロジェクトの仮想環境で実行されるようになります。

### 分析の実行

主な分析は `notebooks/` ディレクトリ内のJupyter Notebookから行います。

```python
from src.ds_tuc_a_detector import DSTucAFlareDetector

# 解析したいFITSファイルのパス
file_path = "data/TESS/DS_Tuc_A/tess2018206045859-s0001-0000000410214986-0120-s_lc.fits"

# インスタンスを作成し、データ処理を実行
# process_data=True にすると、データの読み込みからフレア検出までの一連の処理が自動的に実行されます。
detector = DSTucAFlareDetector(file=file_path, process_data=True)

# 結果をプロット
detector.plot_flare() # Plotlyによる光度曲線プロット
detector.plot_energy_matplotlib() # Matplotlibによるエネルギー分布プロット
```