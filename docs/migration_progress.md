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

### 4. DS Tuc A の統合（未着手）

**優先度**: 高
**予定日**: TBD

#### 📝 タスク内容

DS Tuc A は**連星系**のため、主星と副星の両方を考慮した特殊処理が必要。

**必要な作業**:

1. **`tess_band_energy()` のオーバーライド**
   ```python
   # 主星の寄与
   Rstar = Rsun_cm * 0.87
   star_intensity = ...

   # 副星の寄与（DS Tuc A専用）
   Rcomp = Rsun_cm * 0.864
   comp_intensity = np.sum(dw * self.planck(wave * 1e-9, 4700) * resp)

   # 合計
   area_factor = (主星) + (副星)
   ```

2. **`flux_diff()` のオーバーライド**
   ```python
   # 主星 + 副星の面積
   total_area = (0.87 * 695510e3)**2 + (0.864 * 695510e3)**2
   self.starspot = 2 * np.pi * total_area * ...
   ```

3. **`remove()` メソッドの実装**
   - トランジット除去処理（12組の時間範囲）
   - 既に src/flarepy_DS_Tuc_A.py に実装済みか確認

**検証方法**:
- daijiro版とエネルギー計算結果を比較
- フレア検出数が一致するか確認
- Notebookで可視化して確認

**状態**: ⏳ 未着手

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

**状態**: ⏳ 未着手

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
