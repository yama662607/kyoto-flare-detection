# hirotoフォルダとsrcフォルダの比較分析

## 概要

このドキュメントは、`hiroto`フォルダと`src`フォルダの内容を詳細に比較し、違いを明確にすることを目的とします。

## フォルダ構造の比較

### hirotoフォルダ
```
hiroto/
├── flare_DS_Tuc_A.ipynb (9.7MB)
└── flarepy_improved.py (41KB)
```

### srcフォルダ
```
src/
├── __init__.py
├── __pycache__/
├── base_flare_detector.py (22KB)
├── flarepy_DS_Tuc_A.py.py (1.6KB)
├── flarepy_EK_Dra.py (785B)
├── flarepy_V889_Her.py (4.4KB)
└── archive/
    ├── flarepy_DS_Tuc_A.py (44KB)
    ├── flarepy_EK_Dra.py (43KB)
    ├── flarepy_EK_Dra_plotly.py (44KB)
    └── flarepy_V889_Her.py (52KB)
```

## 主要な違い

### 1. アーキテクチャの違い

#### hirotoフォルダ（単一クラス設計）
- **flarepy_improved.py**: 単一の`FlareDetector`クラスを含む
- 汎用的な設計で、コンストラクタで恒星パラメータを指定
- Jupyter Notebookで実行することを想定

#### srcフォルダ（オブジェクト指向設計）
- **base_flare_detector.py**: 基底クラス`BaseFlareDetector`を定義
- **個別クラス**: 各恒星ごとに専用クラスを継承して実装
  - `FlareDetector_DS_Tuc_A`
  - `FlareDetector_EK_Dra` 
  - `FlareDetector_V889_Her`
- モジュール性と再利用性を重視した設計

### 2. コード規模の違い

| ファイル | hirotoフォルダ | srcフォルダ |
|---------|----------------|-------------|
| 主要クラス | 41KB (1,086行) | 22KB (490行) + 個別クラス |
| Jupyter Notebook | 9.7MB (321,242行) | なし |
| トータル | 約9.8MB | 約166KB |

### 3. 機能の違い

#### hirotoフォルダの特徴
- **包括的なプロット機能**: 6段階のサブプロットを持つ詳細な可視化
- **実験的なコード**: `process_data_2()`フラグなど、試験的な機能を含む
- **完全な自己完結型**: 単一ファイルで全機能を実装

#### srcフォルダの特徴
- **モジュール化**: 機能を基底クラスと個別クラスに分割
- **Matplotlibサポート**: Plotlyに加えてMatplotlibでのプロット機能
- **改良されたエラー処理**: より堅牢なエラーハンドリング
- **恒星固有の最適化**: 各恒星の特性に応じたパラメータ設定

### 4. 具体的な実装の違い

#### データ読み込み
```python
# hirotoフォルダ
match = re.match(r"(.+)-s_lc\.fits$", fname_base)

# srcフォルダ
match = re.match(r"(.+)-\d+-\d+-s_lc\.fits$", fname_base)
match = re.match(r"[a-z]+\d+-s00(.+)-\d+-\d+-s_lc\.fits$", fname_base)
```

#### TESS応答関数ファイルパス
```python
# hirotoフォルダ
wave, resp = np.loadtxt(".\\tess-energy.csv", delimiter=",").T

# srcフォルダ  
wave, resp = np.loadtxt("data/tess-response-function-v1.0.csv", delimiter=",").T
```

#### ギャップ検出の閾値
```python
# hirotoフォルダ
gap_indices = np.where(diff_bjd >= 0.1)[0]

# srcフォルダ
gap_indices = np.where(diff_bjd >= 0.05)[0]
```

### 5. クラス変数の違い

#### hirotoフォルダ
```python
array_flare_ratio = np.array([])
array_energy_ratio = np.array([])
array_amplitude = np.array([])
average_flare_ratio = 0.0
array_observation_time = np.array([])
```

#### srcフォルダ
```python
array_flare_ratio = np.array([])
array_observation_time = np.array([])
array_energy_ratio = np.array([])
array_amplitude = np.array([])
array_starspot = np.array([])
array_starspot_ratio = np.array([])
array_data_name = np.array([])
array_per = np.array([])
array_per_err = np.array([])
```

### 6. 新機能の追加（srcフォルダのみ）

- **恒星黒点計算**: `starspot`, `starspot_ratio`
- **自転周期計算**: `rotation_period()`メソッド
- **Lomb-Scargle周期分析**: `astropy.timeseries.LombScargle`使用
- **Matplotlibプロット**: `plot_flare_matplotlib()`, `plot_energy_matplotlib()`
- **エネルギー閾値処理**: `flare_energy()`メソッドの改良

### 7. V889 Her固有の実装

srcフォルダにはV889 Her専用のカスタムデトレンドメソッドが実装されています：
- `difference_at_lag()`メソッド
- 複数のラグでの差分検出
- フレア候補の開始・終点検出アルゴリズム

## 開発段階の違い

### hirotoフォルダ
- **開発・実験段階**: Jupyter Notebookでの試行錯誤の痕跡
- **単一ファイル設計**: プロトタイプとしての性格が強い
- **Windowsパス**: ハードコードされたWindowsパス（".\\tess-energy.csv"）

### srcフォルダ  
- **製品版設計**: モジュール化されたクリーンなアーキテクチャ
- **クロスプラットフォーム**: Unixベースのパス設計
- **保守性**: コードの分離と再利用性を重視

## まとめ

| 項目 | hirotoフォルダ | srcフォルダ |
|------|----------------|-------------|
| 目的 | 開発・実験 | 製品・保守 |
| 設計 | 単一クラス | 継承ベース |
| 可読性 | 中 | 高 |
| 保守性 | 低 | 高 |
| 機能性 | 基本機能 + 実験的機能 | 拡張機能 + 安定版 |
| サイズ | 大（Notebook含む） | 小（モジュール化） |

**結論**: `hiroto`フォルダは開発初期段階の実験的な実装であり、`src`フォルダはそれを元に改良された製品版のコードベースと言えます。srcフォルダはオブジェクト指向の原則に従い、モジュール性、保守性、拡張性を大幅に向上させています。
