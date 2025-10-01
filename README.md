# Kyoto Flare Detection Project

## 概要

このプロジェクトは、TESS (Transiting Exoplanet Survey Satellite) の光度曲線データから恒星のフレアを検出し、そのエネルギーや頻度を分析するためのPythonフレームワークです。

主な機能:
- TESSのFITSファイルから光度曲線データを読み込み
- データのギャップ補正、デトレンド処理
- フレア現象の自動検出
- フレアのエネルギー、継続時間、ピーク時刻などの物理量を推定
- PlotlyおよびMatplotlibによる結果の可視化

## プロジェクト構成

```
/
├── data/                # データファイル (Gitの追跡対象外)
│   ├── TESS/            # TESSの.fitsファイルを恒星ごとに格納
│   └── tess-response-function-v1.0.csv # TESSの応答関数
├── notebooks/           # 分析用Jupyter Notebook
│   ├── flare_detect_DS_Tuc_A.ipynb
│   ├── flare_detect_EK_Dra.ipynb
│   └── flare_detect_V889_Her.ipynb
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

本プロジェクトは、Pythonのパッケージ管理ツールとして `uv` を使用しています。

1. **uvのインストール**
   `uv`がインストールされていない場合は、公式の指示に従ってインストールしてください。

2. **仮想環境の作成と依存関係のインストール**
   プロジェクトのルートディレクトリで以下のコマンドを実行すると、仮想環境が作成され、`uv.lock`に基づいて依存パッケージがインストールされます。
   ```bash
   uv sync
   ```

## 使用方法

主な分析は `notebooks/` ディレクトリ内のJupyter Notebookから行います。

1. **Jupyter Notebookの起動**
   仮想環境を有効にした後、Jupyterを起動します。
   ```bash
   source .venv/bin/activate  # macOS / Linux
   .venv\Scripts\activate    # Windows

   jupyter notebook
   ```

2. **Notebookの実行**
   `notebooks/` から対象の星のNotebookを開き、セルを実行してください。
   以下は、コードの基本的な使用例です。

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

## コード概要

### `base_flare_detector.py`

`BaseFlareDetector` クラスを定義しています。このクラスは、星の種類によらないフレア検出のコアロジックを実装しています。
- データの読み込みと正規化
- ギャップ補正とデトレンド
- フレアの検出と物理量の計算
- 描画メソッド (Plotly, Matplotlib)

### `*_detector.py` (例: `ds_tuc_a_detector.py`)

特定の恒星に特化した処理を実装するための派生クラスを定義しています。
- `BaseFlareDetector` を継承します。
- `__init__` メソッドで、その星固有の物理パラメータ（恒星半径、温度など）や、アルゴリズムのチューニングパラメータを設定します。
- トランジットデータの除去など、その星にのみ必要なデータ前処理がある場合は、`remove()` などのメソッドをオーバーライドして実装します。
