# フレア検出コード統合作業 進捗管理ドキュメント

**作成日**: 2025-11-13
**最終更新**: 2025-11-13
**目的**: daijiro/, hiroto/ フォルダのコードを src/ に統合し、アーカイブする

---

## 📋 作業概要

### 目標
- daijiro/, hiroto/ の独立実装を src/base_flare_detector.py ベースの統合実装に移行
- 科学的に重要な機能を漏れなく統合
- 最終的に daijiro/, hiroto/ をアーカイブフォルダに移動

### 対象ファイル

| 恒星名 | daijiro実装 | hiroto実装 | src実装 | 状態 |
|--------|------------|-----------|---------|------|
| DS Tuc A | ✅ あり | - | ✅ あり | 🔄 要統合 |
| EK Dra | ✅ あり | - | ✅ あり | ✅ 完了 |
| V889 Her | ✅ あり | ✅ あり | ✅ あり | 🔄 要確認 |

---

## ✅ 完了した作業

### 1. BaseFlareDetector の改良（完了）

**実施日**: 2025-11-13

#### ✅ matplotlib プロット機能の改良
- `plot_flare_matplotlib()` を論文品質版に更新
  - rcParams を直接設定（論文投稿用）
  - `pdf.fonttype = 42` でPDFフォント埋め込み
  - `save_path` パラメータでカスタマイズ可能に
  - フィグサイズ: `(13, 8)`
  - フレアピーク表示: `ymin=0.8, ymax=0.85`

- `plot_energy_matplotlib()` を論文品質版に更新
  - 同様のrcParams設定を適用
  - `save_path` パラメータ追加

- `_get_matplotlib_style()` メソッドを削除
  - 使用されていないため削除
  - rcParams を各メソッドで直接設定する方式に統一

**変更ファイル**:
- `src/base_flare_detector.py` (行440-553)

**検証状況**: ✅ 実装完了、テスト待ち

---

### 2. EK_Dra の統合分析（完了）

**実施日**: 2025-11-13

#### 📊 分析結果

**結論**: `src/flarepy_EK_Dra.py` は**変更不要**

**理由**:
- 既に BaseFlareDetector を正しく継承
- 全てのパラメータが適切に設定済み
- EK_Dra 特有の処理が存在しない（単一星）
- daijiro版の1204行は BaseFlareDetector の機能と重複

**現在の実装（28行）**:
```python
class FlareDetector_EK_Dra(BaseFlareDetector):
    def __init__(self, file=None, process_data=False, ...):
        super().__init__(
            file=file,
            R_sunstar_ratio=0.94,
            T_star=5700,
            flux_mean=249320.35370300722,
            err_constant_mean=0.0004111604805261475,
            rot_period=0.2094793179536128,
            f_cut_lowpass=3,
            f_cut_spline=6,
            ...
        )
```

**主要パラメータ確認**:
- ✅ `R_sunstar_ratio`: 0.94（daijiro と一致）
- ✅ `T_star`: 5700 K（daijiro と一致）
- ✅ `flux_mean`: 249320.35370300722（daijiro と一致）
- ✅ `err_constant_mean`: 0.0004111604805261475（12セクタの平均値）
- ✅ `rot_period`: 0.2094793179536128（daijiro と一致）

**検証状況**: ✅ 分析完了、統合不要と判断

---

## 🔄 進行中の作業

### 3. BaseFlareDetector へのセクタ分岐機能追加（✅ 完了）

**実施日**: 2025-11-13
**優先度**: 中

#### 📝 実装内容

TESSセクタ番号による分岐機能をBaseFlareDetectorに追加しました。

**実装詳細**:

1. **`sector_threshold` パラメータの追加**
   ```python
   def __init__(self, ..., sector_threshold=None, ...):
       self.sector_threshold = sector_threshold
   ```

2. **`load_TESS_data()` にセクタ分岐ロジックを実装**
   ```python
   # セクタ番号を抽出
   match = re.match(r"[a-z]+\d+-s00(.+)-\d+-\d+-s_lc\.fits$", fname_base)
   data_number = int(match.group(1)) if match else 0

   # セクタ分岐: sector_thresholdが設定されている場合のみ分岐
   if self.sector_threshold is not None and data_number > self.sector_threshold:
       flux_field = "SAP_FLUX"
       flux_err_field = "SAP_FLUX_ERR"
   else:
       flux_field = "PDCSAP_FLUX"
       flux_err_field = "PDCSAP_FLUX_ERR"
   ```

3. **各星クラスに適切な閾値を設定**
   - **DS Tuc A**: `sector_threshold=74`
   - **EK Dra**: `sector_threshold=74`
   - **V889 Her**: `sector_threshold=90` (他と異なる!)
4. **DS Tuc A の連星補正**
   - `FlareDetector_DS_Tuc_A.tess_band_energy()` をオーバーライドし、主星 (0.87 R☉, 5428 K) と伴星 (0.864 R☉, 4700 K) の寄与を合算して `area_factor` を算出 @src/flarepy_DS_Tuc_A.py#26-74
   - `flux_diff()` では primary/companion の面積和を用いて `starspot` を更新し、連星のスポット面積を正しく反映
5. **`process_data()` に `skip_remove` フラグを追加**
   - `BaseFlareDetector.process_data(..., skip_remove=True)` で `remove()` を飛ばして hiroto 側の `process_data_2()` を再現可能にしました @src/base_flare_detector.py#394-423

**変更ファイル**:
- `src/base_flare_detector.py` (行28-143)
- `src/flarepy_DS_Tuc_A.py` (行25)
- `src/flarepy_EK_Dra.py` (行24)
- `src/flarepy_V889_Her.py` (行26)

**重要な発見**:
- V889 Her のみセクタ90で分岐（DS Tuc A, EK Draは74）
- hiroto版はセクタ分岐なし（PDCSAP_FLUX固定）

**検証状況**: ✅ 実装完了、テスト待ち

---

## 📅 今後の予定

### 4. DS Tuc A の統合（✅ 完了）

**優先度**: 高
**予定日**: 2025-11-14

#### 📝 実装内容と検証

1. **`tess_band_energy()` のオーバーライド完了**
   - 主星 (0.87 R☉, 5428 K) と伴星 (0.864 R☉, 4700 K) の寄与を合算し、`area_factor` を再計算することで連星系の放射強度を正しく反映 @src/flarepy_DS_Tuc_A.py#27-54
2. **`flux_diff()` のオーバーライド**
   - primary/companion の面積和を使って `starspot` を再計算し、連星全体のスポット面積を表現 @src/flarepy_DS_Tuc_A.py#56-64
3. **`remove()` にトランジット除去**
   - daijiro版と同じ 12 組の時間範囲を `remove()` で排除済み @src/flarepy_DS_Tuc_A.py#34-71

**検証方法**:
- daijiro版とエネルギー合計とフレア数が一致するか Jupyter Notebook で確認予定
- セクタ 74 以降の SAP/PDCSAP 切り替えを `BaseFlareDetector` にて保障

**状態**: ✅ 実装完了、確認待ち

---

### 5. V889 Her の統合（未着手）

**優先度**: 高
**予定日**: TBD

#### 📝 タスク内容

V889 Her は**高度なデトレンド処理**が特徴。

**必要な作業**:

1. **`difference_at_lag()` メソッドの確認**
   - 既に src/flarepy_V889_Her.py に実装済みか確認
   - ラグ付き差分計算（n=2,3,4,5）

2. **`detrend_flux()` のオーバーライド確認**
   - ローパス前に大きなフレアを除去
   - 多重ラグ差分でフレア候補を検出
   - スプライン補間でフレア区間を埋める

3. **フィルタパラメータの確認**
   - `f_cut_lowpass=30`（他の星は3）
   - `f_cut_spline=40`（他の星は6）

**検証方法**:
- daijiro版とhiroto版の比較
- src版の実装確認
- フレア検出精度の比較

**状態**: ⏳ 未着手

---

### 6. Notebook の動作確認（未着手）

**優先度**: 高
**予定日**: 統合作業完了後

#### 📝 タスク内容

各Notebookを実行して、統合後のコードが正しく動作するか確認。

**対象Notebook**:
- `notebooks/flare_detect_DS_Tuc_A.ipynb`
- `notebooks/flare_detect_EK_Dra.ipynb`
- `notebooks/flare_detect_V889_Her.ipynb`
- `notebooks/flare_create_graphs.ipynb`

**確認項目**:
- ✅ エラーなく実行完了
- ✅ フレア検出数が妥当
- ✅ エネルギー分布が妥当
- ✅ プロットが正しく表示

### 7. Notebook テンプレートの整備（進行中）

**目的**: `hiroto/`, `daijiro/`, `src/archive` に残るノートブックを星ごとに結合した `notebooks/flare_detect_template.ipynb` を作成し、共有処理を集中管理した上で固有セルを手動で整理する。

**実施内容**:
- `scripts/build_template_merged.py` で DS Tuc A / EK Dra / V889 Her のノートブックを順に読み込み、Markdown セクションを付けつつ `notebooks/flare_detect_template.ipynb` を生成 (@scripts/build_template_merged.py)。
- 生成したテンプレートは hiroto・daijiro由来のセルを順次並べ、テンプレート的な処理をまとめることで重複を後続作業で削除しやすくした。
- 星ごとのノートブックはテンプレートを土台に、固有の描画/解析セルだけを整理する予定。テンプレートで不要なセルを特定しながら段階的に削除する。

**検証方法**:
- テンプレート内の Markdown セクションとセル数を確認し、全対象星が含まれることを検証
- `nbformat` で読み込み、各セクションの先頭・末尾が想定どおりに整列していることを確認

**状態**: ✅ 1回目の生成完了、以降手動整理へ移行

### 8. 星別ノートブックの統合（進行中）

**目的**: `hiroto/`, `daijiro/`, `notebooks/` にある同一星のノートブックを順に読み込んで重複を除き、最終的な `notebooks/flare_detect_{星}.ipynb` に統合する。

**実施内容**:
- `scripts/merge_star_notebooks.py` を作成し、DS Tuc A・EK Dra・V889 Her それぞれについて `cell_type`＋`source` の組合せで重複を避けながら結合して `notebooks/flare_detect_{星}.ipynb` を再生成。星間で共通するベースセルは1回ずつだけ挿入されます。
- このスクリプトは `hiroto` や `daijiro` 側の改変があったときにも再実行可能なため、テンプレート更新と併せてノートブックを同期的に整備できます。

**検証方法**:
- `python scripts/merge_star_notebooks.py` を実行した後、出力ファイルのセル数と先頭/末尾が期待どおりか `nbformat` で確認
- 星ごとに `FlareDetector_*` を実行し、旧ファイルと同じ処理結果が得られることを spot-check

**状態**: ✅ マージ完了、以降は重複削除／固有セルの微調整フェーズ

---

### 7. daijiro/, hiroto/ のアーカイブ（未着手）

**優先度**: 低
**予定日**: 全統合作業完了後

#### 📝 タスク内容

統合作業が完了し、動作確認が終わったら、旧実装をアーカイブする。

**作業手順**:
```bash
# 1. archive/ ディレクトリ作成（既存の場合はスキップ）
mkdir -p archive

# 2. daijiro/, hiroto/ を archive/ に移動
git mv daijiro archive/
git mv hiroto archive/

# 3. Git コミット
git add .
git commit -m "chore: Archive daijiro and hiroto implementations

All functionality has been integrated into src/ directory.
These files are preserved for historical reference.
"

# 4. 動作確認
# - Notebook が正常に動作するか
# - テストが通るか
```

**状態**: ⏳ 未着手

---

## 📊 進捗サマリー

### 全体進捗

| フェーズ | 状態 | 完了率 |
|---------|------|--------|
| 1. Base改良 | ✅ 完了 | 100% |
| 2. EK_Dra統合 | ✅ 完了 | 100% |
| 3. セクタ分岐追加 | ✅ 完了 | 100% |
| 4. DS Tuc A統合 | ⏳ 未着手 | 0% |
| 5. V889 Her統合 | ⏳ 未着手 | 0% |
| 6. Notebook確認 | ⏳ 未着手 | 0% |
| 7. アーカイブ | ⏳ 未着手 | 0% |

**全体進捗**: 約 43%

---

## 🔧 技術的な決定事項

### 設計方針

1. **継承ベースの設計**
   - BaseFlareDetector に共通機能を実装
   - 星ごとのクラスで特有処理をオーバーライド

2. **星特有の処理の配置**
   - 連星系の計算: 各星のクラスでオーバーライド
   - トランジット除去: `remove()` メソッドで実装
   - 高度なデトレンド: `detrend_flux()` でオーバーライド

3. **パラメータ化の方針**
   - ハードコード値を避ける
   - `__init__` でパラメータ化
   - 科学的根拠のある値はコメントで明記

### コーディング規約

1. **Docstring**
   - 全ての public メソッドに追加
   - Parameters, Returns, Notes セクションを含む

2. **コメント**
   - 科学的背景を説明
   - 計算式の出典を記載
   - 単位を明記

3. **命名規則**
   - クラス: `FlareDetector_星名`
   - メソッド: snake_case
   - 定数: UPPER_CASE

---

## 📝 メモ・備考

### 重要な発見

1. **EK_Dra は変更不要**
   - 既に完璧に実装されている
   - daijiro版の1204行は BaseFlareDetector と重複

2. **セクタ分岐は全星共通**
   - EK_Dra だけでなく全ての星で有用
   - BaseFlareDetector に追加すべき

3. **matplotlib図は統合済み**
   - `plot_flare_matplotlib()` で論文品質図が出力可能
   - daijiro版の `plt_flare()` と同等の機能

### 今後の検討事項

1. **ギャップ閾値の統一**
   - Base: 0.05日
   - daijiro: 0.2日
   - どちらが適切か科学的に検討

2. **エラー定数の計算方法**
   - ハードコード vs 動的計算
   - トレーサビリティとパフォーマンスのトレードオフ

3. **ユニットテストの追加**
   - 各星のクラスのテストコード作成
   - CI/CD パイプラインの構築

---

## 🔗 関連ドキュメント

- [プロジェクトREADME](../README.md)
- [BaseFlareDetector APIドキュメント](./base_flare_detector_api.md)（未作成）
- [科学的背景資料](./scientific_background.md)（未作成）

---

## 📞 連絡先・質問

このドキュメントについて質問がある場合は、プロジェクトのIssueトラッカーに投稿してください。

**最終更新者**: Claude Code
**最終更新日時**: 2025-11-13 (セクタ分岐機能実装完了)
