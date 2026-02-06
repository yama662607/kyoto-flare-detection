# daijiro / hiroto フォルダ統合詳細計画

## 0. 目的と前提

daijiro および hiroto フォルダの Python コードを、`src` 配下の継承設計と `notebooks` のワークフローに統合する。以下を前提とする。

* `src/base_flare_detector.py` の最新実装を中心に機能を統合する @src/base_flare_detector.py#1-490
* 既存の派生クラス (`FlareDetector_DS_Tuc_A`, `FlareDetector_EK_Dra`, `FlareDetector_V889_Her`) は縮約版のため、daijiro/hiroto 由来の詳細ロジックを段階的に移植する @src/flarepy_DS_Tuc_A.py.py#1-49 @src/flarepy_EK_Dra.py#1-28 @src/flarepy_V889_Her.py#1-108
* 実装変更はまだ実施しない。コード差分と計画のみを整理する。

## 1. 現状アーキテクチャの比較

| 領域 | src | daijiro | hiroto |
| --- | --- | --- | --- |
| クラス構造 | 基底クラス + 派生クラス | 単一クラス | 単一クラス |
| プロット | Plotly + Matplotlib（簡易） | Plotly + Matplotlib（詳細スタイル） | Plotly（6面マルチプロット） |
| フレア検出パイプライン | 継承前提の段階処理 | 各ファイルに完全実装 | 完全実装 + process_data_2 |
| データ取得 | `data/tess-response-function-v1.0.csv` | `./tess-energy.csv` | `./tess-energy.csv` |
| 固有パラメータ | 派生クラスで一部指定 | 固定値多数 | コンストラクタ引数で可変 |

### 1.1 src/base_flare_detector の主な仕様

* TESS 読み込み・ギャップ補正・デトレンド・誤差再推定・検出・可視化・統計を一貫提供 @src/base_flare_detector.py#27-413
* Matplotlib 用スタイル・Plotly 可視化を両対応済み @src/base_flare_detector.py#414-486
* エネルギー計算は応答関数 CSV を `data/` から読み込む @src/base_flare_detector.py#324-341

### 1.2 daijiro / hiroto クラスの特徴

* **daijiro**: 恒星ごとのチューニング（固定 `flux_mean`、`err_constant` 配列、連星補正など）@daijiro/flarepy_DS_Tuc_A.py#256-385
* **hiroto**: 詳細 Plotly プロット、`process_data_2`、`diff`/`remove` 等のユーティリティが豊富 @hiroto/flarepy_improved.py#676-1086

## 2. ギャップ分析（機能別）

| 項目 | src 現状 | daijiro / hiroto | ギャップと対応方針 |
| --- | --- | --- | --- |
| FITS 正規化 | 固定 `flux_mean` をコンストラクタで受け取り正規化 @src/base_flare_detector.py#124-134 | 実測平均 or 固定値を内部算出（恒星ごと）@daijiro/flarepy_DS_Tuc_A.py#256-266 @hiroto/flarepy_improved.py#223-231 | `BaseFlareDetector` に「未指定時は平均で正規化」「派生で事前設定可」を整理する |
| データ番号分岐 | `data_number > 74` で SAP/PDCSAP 切替（固定閾値）@src/base_flare_detector.py#109-118 | V889 は 90 超で切替など細分化 @daijiro/flarepy_V889_Her 2.py（確認済み） | 派生クラスで条件上書きできるよう hook 追加 |
| ギャップ閾値 | 0.05 日固定 @src/base_flare_detector.py#143-158 | V889: 0.004、hiroto: 0.1 など | `gap_threshold` パラメータ導入と派生クラスで設定 |
| デトレンド | FFT ローパス + スプライン（共通）@src/base_flare_detector.py#171-191 | V889 は差分ベースのフレア候補除去あり @src/flarepy_V889_Her.py#38-108 | V889 派生でメソッド上書き、共通処理の再利用方針明記 |
| 誤差再推定 | `err_constant_mean` を使用 @src/base_flare_detector.py#193-203 | daijiro は実測値配列から平均化、hiroto は 0.5 日窓 STD | 派生で配列渡し・共通で重み計算を許容する API に拡張 |
| フレア検出 | 5σ→1σ 拡張（共通）@src/base_flare_detector.py#205-256 | 仕様ほぼ同じ | 共通化済み、派生では閾値調整オプション化検討 |
| フレア後処理 | `flaredetect_check` で線形補正と E-decay 計算 @src/base_flare_detector.py#259-319 | hiroto/daijiro と概ね一致 | 追加補正不要 |
| エネルギー | 応答関数ファイル固定パス、単星の面積計算 @src/base_flare_detector.py#324-341 | DS Tuc A は連星加算、daijiro/hiroto は `./tess-energy.csv` を参照 | ファイルパスを設定化、DS Tuc A override で連星実装 |
| プロット | Plotly 2 面 + Matplotlib 2 面 @src/base_flare_detector.py#414-476 | daijiro: Matplotlib 詳細スタイル、hiroto: Plotly 6 面構成 | 詳細 Plotly/Matplotlib をオプション化 (`plot_flare_mode`) |
| 集計指標 | 星黒点・自転周期などを集計 @src/base_flare_detector.py#365-399 | hiroto/daijiro は同様 or 自作 | `BaseFlareDetector` に揃っているため派生は利用する |
| `process_data_2` | 実装なし | hiroto 版に「トランジット除去しない」分岐 @hiroto/flarepy_improved.py#975-1027 | `process_data` にフラグ追加 or サブクラスで実装提供 |

## 3. 改修計画

### 3.1 BaseFlareDetector の拡張

1. **正規化戦略の柔軟化**
   * `flux_mean` が指定されない場合は正の値平均を採用する分岐を追加
   * 派生クラスから計算済み値を渡す場合は現状通り利用

2. **FITS カラム選択ロジックのフック化**
   * `_select_flux_columns(data_number)` のような保護メソッドを新設し、デフォルトでは `>74` 判定を維持
   * 派生クラスで閾値やカラム名を上書き可能にする

3. **ギャップ閾値のパラメータ化**
   * `__init__` 引数に `gap_threshold` を追加し、既定値 0.05 をデフォルトにする
   * `apply_gap_correction` で固定値ではなくプロパティを使用

4. **誤差再推定のカスタマイズ支援**
   * 現状の `err_constant_mean` スケーリングに加え、派生クラスから配列や係数を注入できるよう設計（例: `self.err_scaling_series`）
   * EK Dra / V889 用に、計測値リストから平均を取るユーティリティを提供

5. **応答関数ファイルパスの設定化**
   * `tess_response_path` を `__init__` で受け取れるようにし、デフォルトを現行パスに設定
   * ファイル不存在時のエラーメッセージにヒントを追加

6. **詳細プロットモード追加**
   * Plotly: hiroto版 6 面サブプロットを `plot_flare(mode="detailed")` で呼び出せるよう整理
   * Matplotlib: daijiro スタイルを継承しつつ、保存ファイル名や DPI を可変化

7. **プロセス制御フラグ**
   * `process_data` に `skip_remove=False` や `custom_pipeline=None` を追加し、`process_data_2` 相当の動線を提供

### 3.2 派生クラス別タスク

| クラス | 現状 | 必要タスク |
| --- | --- | --- |
| `FlareDetector_DS_Tuc_A` | 固定パラメータのみ設定 @src/flarepy_DS_Tuc_A.py.py#8-27 | 1) `tess_band_energy` を連星計算に差し替え 2) `remove` を daijiro 実装と整合 3) `process_data` 呼び出し時に連星用 `tess_response_path` を渡す |
| `FlareDetector_EK_Dra` | 固定パラメータのみ @src/flarepy_EK_Dra.py#4-27 | 1) `err_constant` の実測配列を属性で保持し再推定に活用 2) `flux_mean` の妥当性検証（平均算出か固定値か） |
| `FlareDetector_V889_Her` | detrend を部分的に移植済み @src/flarepy_V889_Her.py#31-108 | 1) `apply_gap_correction` を 0.004 閾値で再実装 2) 低周波カットの再パラメータ化（30/40） 3) 事前スプライン用 `difference_at_lag` フローの完全移植確認 |

### 3.3 hiroto 特有機能取り込み

* `remove` / `no_remove` / `diff` / `calculate_precise_obs_time` 等のメソッドはすでに `BaseFlareDetector` に近い形で存在する。差分は以下を調整。
  * `diff` の実装差：hiroto は上下 2% 切り出しで振幅算出→ `Base` の `flux_diff` と整合確認 @hiroto/flarepy_improved.py#877-904 @src/base_flare_detector.py#365-371
  * `process_data_2` を `Base` にフラグ追加で吸収

### 3.4 ノートブック整備

1. 各ノートブックでのインポートとクラス生成を新しい派生クラスに変更
2. `process_data` 呼び出し時に追加パラメータ（例: `gap_threshold`, `use_matplotlib_plots`）を設定
3. Plotly ベースの可視化セルを詳細モードに合わせて更新（必要であれば新しい API 呼び出しに変更）
4. 旧 `FlareDetector` クラスを参照するセルは全て最新インターフェースに書き換える

### 3.5 テスト戦略

* ユニットテスト：派生クラスごとに最低限以下を検証
  1. FITS 読み込み後の `flux_mean` 正規化が期待通りか（サンプル FITS を用意）
  2. `apply_gap_correction` でギャップ閾値が反映されるか
  3. `detrend_flux` の出力が既存スナップショットと一致するか
  4. `tess_band_energy` の出力が daijiro 実装と一致するか（モックデータで比較）
* 統合テスト：ノートブックを `papermill` などでバッチ実行し、主要な中間結果（フレア件数やエネルギー積算）が旧成果物と一致するか確認

## 4. 実装ロードマップ（推奨順）

1. **基底クラス機能拡張**（3.1 のタスク）
2. **派生クラスの機能差分取り込み**（3.2）
3. **hiroto 特有機能の吸収 & API 整理**（3.3）
4. **ノートブック更新**（3.4）
5. **自動テスト/検証環境の整備**（3.5）

各ステップ後に `pnpm check` / `pnpm test` の実行、および必要に応じて TESS データを用いたスモークテストを推奨する。

## 5. リスクと検討事項

* `flux_mean` を固定値にするか動的計算するかで、検出感度が変わる可能性
* 応答関数ファイルのパス統一が必要（相対パス差異による FileNotFound エラー）
* V889 Her 特有の高周波パラメータは他恒星に影響しないよう派生クラス内で閉じる
* `process_data` の API 拡張は後方互換を維持するよう追加引数はキーワード専用にする

## 6. まとめ

本計画では、daijiro / hiroto の詳細ロジックを再利用しつつ、`BaseFlareDetector` を拡張して汎用的なフレア検出フレームワークを確立する。派生クラスは恒星ごとの差分のみに集中させ、ノートブックとテストを通じて動作を保証する。以降の実装フェーズでは、ここで列挙したタスクを順次実施し、各段階で検証結果を記録する。
