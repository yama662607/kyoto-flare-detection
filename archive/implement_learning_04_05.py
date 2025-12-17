#!/usr/bin/env python3
"""notebooks/learning の 04 / 05 学習ノートを本実装に差し替えるスクリプト"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
NB04_PATH = PROJECT_ROOT / "notebooks" / "learning" / "flare_learning_04_star_specifics_DS_Tuc_A_V889_Her.ipynb"
NB05_PATH = PROJECT_ROOT / "notebooks" / "learning" / "flare_learning_05_statistics_comparison.ipynb"


def build_nb04_cells() -> list[dict]:
    """04: 恒星ごとの特殊処理（DS Tuc A / V889 Her）"""
    cells: list[dict] = []

    # タイトルと導入
    cells.append({
        "cell_type": "markdown",
        "id": "title-04",
        "metadata": {},
        "source": [
            "# フレア検出 学習ノート 04: 恒星ごとの特殊処理（DS Tuc A / V889 Her）\n",
            "\n",
            "このノートでは、`FlareDetector_DS_Tuc_A` と `FlareDetector_V889_Her` を題材に、\n",
            "Base クラス `BaseFlareDetector` をどのように拡張しているかを学びます。\n",
            "\n",
            "- DS Tuc A: 連星系＋トランジットを持つ若い星\n",
            "- V889 Her: 強い活動性を持つ回転変動の大きな星\n",
            "\n",
            "それぞれの恒星に合わせて「どの段階を」「どのように」カスタマイズしているかを、\n",
            "コードと図を使って確認していきます。\n",
        ],
    })

    cells.append({
        "cell_type": "markdown",
        "id": "outline-04",
        "metadata": {},
        "source": [
            "## このノートで学ぶこと\n",
            "\n",
            "- なぜ恒星ごとに `remove` / `detrend_flux` / `tess_band_energy` / `flux_diff` を\n",
            "  カスタマイズする必要があるのか\n",
            "- DS Tuc A のトランジット除去ロジックと連星補正の考え方\n",
            "- V889 Her の「フレアだけを埋めてから」デトレンドする高度な手法\n",
            "- Base クラスとサブクラスの責務分担の整理\n",
            "\n",
            "前提として、01〜03 のノートで基本フローを一通り体験していることを想定しています。\n",
        ],
    })

    # パス設定セル
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "id": "path-setup-04",
        "metadata": {},
        "outputs": [],
        "source": [
            "import sys\n",
            "from pathlib import Path\n",
            "\n",
            "NOTEBOOK_DIR = Path().resolve()\n",
            "PROJECT_ROOT = NOTEBOOK_DIR.parent.parent\n",
            "if str(PROJECT_ROOT) not in sys.path:\n",
            "    sys.path.insert(0, str(PROJECT_ROOT))\n",
            "\n",
            'print("NOTEBOOK_DIR:", NOTEBOOK_DIR)\n',
            'print("PROJECT_ROOT:", PROJECT_ROOT)',
        ],
    })

    # 1. なぜ恒星ごとの実装が必要か
    cells.append({
        "cell_type": "markdown",
        "id": "why-star-specific",
        "metadata": {},
        "source": [
            "## 1. なぜ恒星ごとの実装が必要か\n",
            "\n",
            "`BaseFlareDetector` は「単一星」を前提にした共通ロジックを提供しますが、\n",
            "実際には恒星ごとに次のような違いがあります。\n",
            "\n",
            "- 連星系かどうか（DS Tuc A は主星 + 伴星）\n",
            "- トランジットや強い系外惑星信号があるかどうか\n",
            "- 長期変動やフレアの典型的な形がどれくらい複雑か（V889 Her は強い活動星）\n",
            "\n",
            "そのため、\n",
            "\n",
            "- DS Tuc A では `remove()` と `tess_band_energy()` / `flux_diff()` をオーバーライドして\n",
            "  連星補正とトランジット除去を行う\n",
            "- V889 Her では `detrend_flux()` をオーバーライドして\n",
            "  高度なフレアキャンセル付きデトレンドを行う\n",
            "\n",
            "といった「星ごとの拡張」が入っています。\n",
            "\n",
            "このノートでは、実際のクラスを使って\n",
            "\n",
            "- DS Tuc A のトランジット除去と連星補正\n",
            "- V889 Her の高度デトレンド\n",
            "\n",
            "を、図と数値の両方から確認していきます。\n",
        ],
    })

    # import とデータパス
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "id": "imports-04",
        "metadata": {},
        "outputs": [],
        "source": [
            "import numpy as np\n",
            "import plotly.graph_objects as go\n",
            "from plotly.subplots import make_subplots\n",
            "\n",
            "from src.flarepy_DS_Tuc_A import FlareDetector_DS_Tuc_A\n",
            "from src.flarepy_V889_Her import FlareDetector_V889_Her\n",
            "\n",
            'DATA_DIR_DS_TUC_A = PROJECT_ROOT / "data" / "TESS" / "DS_Tuc_A"\n',
            'DATA_DIR_V889_HER = PROJECT_ROOT / "data" / "TESS" / "V889_Her"\n',
            "\n",
            'print("DATA_DIR_DS_TUC_A:", DATA_DIR_DS_TUC_A)\n',
            'print("DATA_DIR_V889_HER:", DATA_DIR_V889_HER)\n',
        ],
    })

    # 2. DS Tuc A セクション（説明）
    cells.append({
        "cell_type": "markdown",
        "id": "ds-tuc-a-section",
        "metadata": {},
        "source": [
            "## 2. DS Tuc A: トランジット除去と連星補正\n",
            "\n",
            "DS Tuc A は主星＋伴星からなる連星系で、かつトランジットを持つ系外惑星が知られています。\n",
            "そのため、TESS の光度曲線には\n",
            "\n",
            "- 想定外のディッピング（トランジット）\n",
            "- 連星伴星による「余分な光」\n",
            "\n",
            "が含まれています。そのままフレア検出やエネルギー計算を行うと、\n",
            "\n",
            "- トランジットの谷をフレアと誤検出してしまう\n",
            "- 伴星の光まで含めてエネルギーを見積もってしまう\n",
            "\n",
            "といった問題が起きます。\n",
            "\n",
            "この節では、\n",
            "\n",
            "1. `remove()` によるトランジット区間のマスク\n",
            "2. `tess_band_energy()` / `flux_diff()` による「主星＋伴星」を考慮したスケーリング\n",
            "\n",
            "を、実際の光度曲線と数値で確認します。\n",
        ],
    })

    # DS Tuc A 図: remove 前後
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "id": "ds-tuc-a-plot",
        "metadata": {},
        "outputs": [],
        "source": [
            "if not DATA_DIR_DS_TUC_A.exists():\n",
            '    print("DS Tuc A のディレクトリが見つかりません。data/TESS/DS_Tuc_A を確認してください。")\n',
            "else:\n",
            '    fits_files_ds = sorted(DATA_DIR_DS_TUC_A.glob("*.fits"))\n',
            "    if not fits_files_ds:\n",
            '        print("DS Tuc A の FITS ファイルが見つかりません。")\n',
            "    else:\n",
            "        ds_file = fits_files_ds[0]\n",
            '        print("使用するファイル:", ds_file.name)\n',
            "\n",
            "        # トランジット除去前の光度曲線\n",
            "        ds_raw = FlareDetector_DS_Tuc_A(file=str(ds_file), process_data=False)\n",
            "        time_raw = ds_raw.atessBJD\n",
            "        flux_raw = ds_raw.amPDCSAPflux\n",
            "\n",
            "        # remove() 適用後の光度曲線\n",
            "        ds_removed = FlareDetector_DS_Tuc_A(file=str(ds_file), process_data=False)\n",
            "        ds_removed.remove()\n",
            "        time_removed = ds_removed.tessBJD\n",
            "        flux_removed = ds_removed.mPDCSAPflux\n",
            "\n",
            "        fig = make_subplots(\n",
            "            rows=2,\n",
            "            cols=1,\n",
            "            shared_xaxes=True,\n",
            "            vertical_spacing=0.05,\n",
            '            subplot_titles=("トランジット除去前", "トランジット除去後"),\n',
            "        )\n",
            "\n",
            "        fig.add_trace(\n",
            '            go.Scatter(x=time_raw, y=flux_raw, mode="lines", name="raw"),\n',
            "            row=1,\n",
            "            col=1,\n",
            "        )\n",
            "        fig.add_trace(\n",
            '            go.Scatter(x=time_removed, y=flux_removed, mode="lines", name="removed", line=dict(color="orange")),\n',
            "            row=2,\n",
            "            col=1,\n",
            "        )\n",
            "\n",
            '        fig.update_xaxes(title_text="Time [BJD]", row=2, col=1)\n',
            '        fig.update_yaxes(title_text="Normalized Flux", row=1, col=1)\n',
            '        fig.update_yaxes(title_text="Normalized Flux", row=2, col=1)\n',
            "        fig.update_layout(height=600)\n",
            "        fig.show()\n",
        ],
    })

    cells.append({
        "cell_type": "markdown",
        "id": "ds-tuc-a-plot-explain",
        "metadata": {},
        "source": [
            "### 図の読み方\n",
            "\n",
            "- 上段: 元の正規化光度。トランジットによる深い谷が周期的に現れています。\n",
            "- 下段: `remove()` 実行後。トランジットに対応する時間帯がマスクされ、\n",
            "  連続した回転変動＋フレアの上に乗った曲線になります。\n",
            "\n",
            "この処理により、フレア検出アルゴリズムは\n",
            "「トランジットの谷」をフレアと誤認識しにくくなります。\n",
        ],
    })

    # DS Tuc A: エネルギー・スポット指標
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "id": "ds-tuc-a-summary",
        "metadata": {},
        "outputs": [],
        "source": [
            "if 'ds_file' not in globals():\n",
            '    print("先に DS Tuc A のトランジット除去セルを実行してください。")\n',
            "else:\n",
            "    # process_data=True でフルパイプラインを実行\n",
            "    ds_proc = FlareDetector_DS_Tuc_A(file=str(ds_file), process_data=True)\n",
            "\n",
            '    print("--- DS Tuc A summary ---")\n',
            '    print("flare_number:", ds_proc.flare_number)\n',
            '    print("sum_flare_energy [erg]:", ds_proc.sum_flare_energy)\n',
            '    print("precise_obs_time [day]:", ds_proc.precise_obs_time)\n',
            "    if ds_proc.precise_obs_time > 0:\n",
            '        print("flare_ratio [1/day]:", ds_proc.flare_number / ds_proc.precise_obs_time)\n',
            '        print("energy_ratio [erg/day]:", ds_proc.sum_flare_energy / ds_proc.precise_obs_time)\n',
            "    else:\n",
            '        print("flare_ratio / energy_ratio は obs time が 0 のため計算できません。")\n',
            '    print("starspot [cm^2]:", ds_proc.starspot)\n',
            '    print("starspot_ratio:", ds_proc.starspot_ratio)\n',
            "\n",
            "    if ds_proc.energy is not None and len(ds_proc.energy) > 0:\n",
            "        sample_energy = float(ds_proc.energy[0])\n",
            '        print("sample flare energy [erg]:", sample_energy)\n',
            "    else:\n",
            '        print("エネルギーが計算されていません。フレア検出結果を確認してください。")\n',
        ],
    })

    cells.append({
        "cell_type": "markdown",
        "id": "ds-tuc-a-energy-explain",
        "metadata": {},
        "source": [
            "DS Tuc A では、`tess_band_energy()` 内で\n",
            "\n",
            "- 主星（R = 0.87 R☉, T ≈ 5428 K）\n",
            "- 伴星（R ≈ 0.864 R☉, T ≈ 4700 K）\n",
            "\n",
            "の両方の寄与を足し合わせて TESS 帯での光度を評価しています。\n",
            "これにより、同じフラックス増分 `count` でも、\n",
            "「連星系としての実効的な放射面積」に応じたエネルギーが見積もられます。\n",
            "\n",
            "また、`flux_diff()` でも主星＋伴星の面積を使ってスポット面積を評価しており、\n",
            "単一星の近似よりも現実的なスポットサイズが得られます。\n",
        ],
    })

    # 3. V889 Her セクション
    cells.append({
        "cell_type": "markdown",
        "id": "v889-her-section",
        "metadata": {},
        "source": [
            "## 3. V889 Her: フレアキャンセル付き高度デトレンド\n",
            "\n",
            "V889 Her は強い活動性を持つ恒星で、\n",
            "\n",
            "- 回転に伴う大きな長期変動\n",
            "- 頻繁に起こるフレア\n",
            "\n",
            "が同時に存在します。そのため「単純なローパスフィルタ」だけでは\n",
            "\n",
            "- フレア成分まで一緒に平滑化してしまう\n",
            "- 基本的な回転変動が十分に再現できない\n",
            "\n",
            "といった問題が起こります。\n",
            "\n",
            "`FlareDetector_V889_Her.detrend_flux()` では、まず差分列を使って\n",
            "フレア候補区間を推定し、その部分をいったん補間で埋めてから\n",
            "スプライン＋ローパスでベースラインを推定しています。\n",
            "この手順により、\n",
            "\n",
            "- フレアのピークはなるべく壊さずに\n",
            "- 回転変動だけをきれいに取り除いたデトレンド光度\n",
            "\n",
            "を得ることができます。\n",
        ],
    })

    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "id": "v889-her-plot",
        "metadata": {},
        "outputs": [],
        "source": [
            "if not DATA_DIR_V889_HER.exists():\n",
            '    print("V889 Her のディレクトリが見つかりません。data/TESS/V889_Her を確認してください。")\n',
            "else:\n",
            '    fits_files_v889 = sorted(DATA_DIR_V889_HER.glob("*.fits"))\n',
            "    if not fits_files_v889:\n",
            '        print("V889 Her の FITS ファイルが見つかりません。")\n',
            "    else:\n",
            "        v_file = fits_files_v889[0]\n",
            '        print("使用するファイル:", v_file.name)\n',
            "\n",
            "        v_det = FlareDetector_V889_Her(file=str(v_file), process_data=True)\n",
            "\n",
            "        time_ext = v_det.gtessBJD\n",
            "        flux_ext = v_det.gmPDCSAPflux\n",
            "        detrended = v_det.s2mPDCSAPflux\n",
            "\n",
            "        fig = make_subplots(\n",
            "            rows=2,\n",
            "            cols=1,\n",
            "            shared_xaxes=True,\n",
            "            vertical_spacing=0.05,\n",
            '            subplot_titles=("gap 補正後の光度曲線", "高度デトレンド後 (s2mPDCSAPflux)"),\n',
            "        )\n",
            "\n",
            "        fig.add_trace(\n",
            '            go.Scatter(x=time_ext, y=flux_ext, mode="lines", name="gap corrected", line=dict(width=1)),\n',
            "            row=1,\n",
            "            col=1,\n",
            "        )\n",
            "\n",
            "        fig.add_trace(\n",
            '            go.Scatter(x=v_det.tessBJD, y=detrended, mode="lines", name="detrended", line=dict(width=1, color="orange")),\n',
            "            row=2,\n",
            "            col=1,\n",
            "        )\n",
            "\n",
            '        fig.update_xaxes(title_text="Time [BJD]", row=2, col=1)\n',
            '        fig.update_yaxes(title_text="Normalized Flux", row=1, col=1)\n',
            '        fig.update_yaxes(title_text="Detrended Flux", row=2, col=1)\n',
            "        fig.update_layout(height=600)\n",
            "        fig.show()\n",
        ],
    })

    cells.append({
        "cell_type": "markdown",
        "id": "v889-her-explain",
        "metadata": {},
        "source": [
            "### difference_at_lag によるフレア候補抽出のイメージ\n",
            "\n",
            "実装では、\n",
            "\n",
            "- `diff_flux = np.diff(flux_ext)`\n",
            "- `difference_at_lag(flux_ext, n=2..5)`\n",
            "\n",
            "といった複数のラグ差分を取り、\n",
            "\n",
            "- どれかの差分が 0.01 以上\n",
            "- 時間差分 `diff_time` が極端に大きくない\n",
            "\n",
            "という条件を満たす点を「フレアが始まり得る場所」として候補に挙げています。\n",
            "その後、フレア前後で元の値に戻るまでの区間を探し、\n",
            "その区間を補間で埋めてからスプライン・ローパスを適用することで、\n",
            "長期変動の推定を安定させています。\n",
        ],
    })

    # 4. 責務分担の整理
    cells.append({
        "cell_type": "markdown",
        "id": "responsibility-04",
        "metadata": {},
        "source": [
            "## 4. Base クラスとサブクラスの責務分担の整理\n",
            "\n",
            "ここまでの内容を踏まえて、設計上の役割分担を整理すると次のようになります。\n",
            "\n",
            "- BaseFlareDetector\n",
            "  - TESS LC の読み込み (`load_TESS_data`)\n",
            "  - ギャップ補正 (`apply_gap_correction`)\n",
            "  - 一般的なローパス＆スプラインによるデトレンド (`detrend_flux`)\n",
            "  - 誤差の再推定 (`reestimate_errors`)\n",
            "  - フレア検出・エネルギー・回転周期・スポット指標の計算\n",
            "- FlareDetector_DS_Tuc_A\n",
            "  - `remove()` でトランジット区間を明示的にマスク\n",
            "  - `tess_band_energy()` / `flux_diff()` で主星＋伴星を考慮したスケーリング\n",
            "- FlareDetector_V889_Her\n",
            "  - `detrend_flux()` を上書きし、フレアを一時的に埋めてからデトレンド\n",
            "\n",
            "このように「共通ロジックは Base クラス」「恒星ごとの差分はサブクラス」で分離しておくと、\n",
            "\n",
            "- 新しい恒星を追加するときに、必要な部分だけをオーバーライドすればよい\n",
            "- 既存のアルゴリズム改善を Base クラスに集約できる\n",
            "\n",
            "といったメリットがあります。\n",
        ],
    })

    return cells


def build_nb05_cells() -> list[dict]:
    """05: 複数セクタ・複数恒星の統計と比較"""
    cells: list[dict] = []

    # タイトル・導入
    cells.append({
        "cell_type": "markdown",
        "id": "title-05",
        "metadata": {},
        "source": [
            "# フレア検出 学習ノート 05: 複数セクタ・複数恒星の統計と比較\n",
            "\n",
            "このノートでは、`BaseFlareDetector` で得られるフレア検出結果をまとめて扱い、\n",
            "複数セクタ・複数恒星のフレア活動を統計的に比較する方法を学びます。\n",
            "\n",
            "ここでは、以下の 3 つの観点に注目します。\n",
            "\n",
            "1. クラス変数に蓄積されるサマリ値の意味\n",
            "2. 単一恒星内でのセクタ間比較\n",
            "3. 複数恒星の間でのフレア発生率・エネルギー率の比較\n",
        ],
    })

    cells.append({
        "cell_type": "markdown",
        "id": "outline-05",
        "metadata": {},
        "source": [
            "## このノートで学ぶこと\n",
            "\n",
            "- `array_flare_ratio` / `array_energy_ratio` / `array_observation_time` などの役割\n",
            "- 単一恒星内でのセクタごとの FFD 的な図の作り方（簡易版）\n",
            "- 複数恒星のフレア発生率やエネルギー率の比較プロットの作り方\n",
        ],
    })

    # パス設定セル
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "id": "path-setup-05",
        "metadata": {},
        "outputs": [],
        "source": [
            "import sys\n",
            "from pathlib import Path\n",
            "\n",
            "NOTEBOOK_DIR = Path().resolve()\n",
            "PROJECT_ROOT = NOTEBOOK_DIR.parent.parent\n",
            "if str(PROJECT_ROOT) not in sys.path:\n",
            "    sys.path.insert(0, str(PROJECT_ROOT))\n",
            "\n",
            'print("NOTEBOOK_DIR:", NOTEBOOK_DIR)\n',
            'print("PROJECT_ROOT:", PROJECT_ROOT)',
        ],
    })

    # 1. クラス変数の整理
    cells.append({
        "cell_type": "markdown",
        "id": "class-vars-05",
        "metadata": {},
        "source": [
            "## 1. クラス変数に蓄積される情報の整理\n",
            "\n",
            "`BaseFlareDetector.process_data()` を呼ぶたびに、\n",
            "\n",
            "- `array_flare_ratio` : フレア発生率 [1/day]\n",
            "- `array_energy_ratio` : フレアエネルギーの総和 / 観測時間 [erg/day]\n",
            "- `array_observation_time` : 観測の代表時刻（たとえば BJD の中央値）\n",
            "- `array_starspot` : 推定されたスポット面積 [cm^2]\n",
            "- `array_starspot_ratio` : 光度変動から見積もるスポット被覆率\n",
            "- `array_data_name` : 各データの ID（FITS ファイル名から抽出）\n",
            "- `array_per`, `array_per_err` : 回転周期とその不確かさ\n",
            "\n",
            "といった値がクラス変数として蓄積されていきます。\n",
            "複数の FITS ファイル（複数セクタ・複数恒星）に対して `process_data()` を\n",
            "順に呼び出すことで、これらの配列を「簡易なカタログ」として扱うことができます。\n",
        ],
    })

    # import と helper 関数定義
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "id": "imports-05",
        "metadata": {},
        "outputs": [],
        "source": [
            "import re\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "import plotly.express as px\n",
            "\n",
            "from src.base_flare_detector import BaseFlareDetector\n",
            "from src.flarepy_EK_Dra import FlareDetector_EK_Dra\n",
            "from src.flarepy_DS_Tuc_A import FlareDetector_DS_Tuc_A\n",
            "from src.flarepy_V889_Her import FlareDetector_V889_Her\n",
            "\n",
            'DATA_DIR_EK_DRA = PROJECT_ROOT / "data" / "TESS" / "EK_Dra"\n',
            'DATA_DIR_DS_TUC_A = PROJECT_ROOT / "data" / "TESS" / "DS_Tuc_A"\n',
            'DATA_DIR_V889_HER = PROJECT_ROOT / "data" / "TESS" / "V889_Her"\n',
            "\n",
            'print("DATA_DIR_EK_DRA:", DATA_DIR_EK_DRA)\n',
            'print("DATA_DIR_DS_TUC_A:", DATA_DIR_DS_TUC_A)\n',
            'print("DATA_DIR_V889_HER:", DATA_DIR_V889_HER)\n',
        ],
    })

    # summary_df 構築
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "id": "build-summary-df",
        "metadata": {},
        "outputs": [],
        "source": [
            "def extract_sector(data_name: str) -> int:\n",
            '    """データ名から TESS セクタ番号 (sXXXX) を抽出する簡易関数。"""\n',
            '    m = re.search(r"-s(\\d{4})-", data_name)\n',
            "    return int(m.group(1)) if m else -1\n",
            "\n",
            "\n",
            "def run_detectors_for_all_stars(max_files_per_star: int = 3) -> pd.DataFrame:\n",
            '    """EK Dra / DS Tuc A / V889 Her について、いくつかの FITS を処理してサマリ表を作る。"""\n',
            "    rows: list[dict] = []\n",
            "\n",
            "    star_configs = [\n",
            '        ("EK Dra", DATA_DIR_EK_DRA, FlareDetector_EK_Dra),\n',
            '        ("DS Tuc A", DATA_DIR_DS_TUC_A, FlareDetector_DS_Tuc_A),\n',
            '        ("V889 Her", DATA_DIR_V889_HER, FlareDetector_V889_Her),\n',
            "    ]\n",
            "\n",
            "    for star_name, data_dir, Detector in star_configs:\n",
            "        if not data_dir.exists():\n",
            '            print(f"[warning] {star_name} のディレクトリ {data_dir} が見つかりません。")\n',
            "            continue\n",
            "\n",
            '        fits_files = sorted(data_dir.glob("*.fits"))[:max_files_per_star]\n',
            "        if not fits_files:\n",
            '            print(f"[warning] {star_name} の FITS ファイルが見つかりません。")\n',
            "            continue\n",
            "\n",
            "        for fpath in fits_files:\n",
            '            print(f"Processing {star_name}: {fpath.name}")\n',
            "            det = Detector(file=str(fpath), process_data=True)\n",
            "\n",
            "            if det.precise_obs_time <= 0:\n",
            "                flare_ratio = np.nan\n",
            "                energy_ratio = np.nan\n",
            "            else:\n",
            "                flare_ratio = det.flare_number / det.precise_obs_time\n",
            "                energy_ratio = det.sum_flare_energy / det.precise_obs_time\n",
            "\n",
            "            rows.append(\n",
            "                {\n",
            '                    "star": star_name,\n',
            '                    "data_name": det.data_name,\n',
            '                    "sector": extract_sector(det.data_name or ""),\n',
            '                    "flare_ratio": flare_ratio,\n',
            '                    "energy_ratio": energy_ratio,\n',
            '                    "starspot": det.starspot,\n',
            '                    "starspot_ratio": det.starspot_ratio,\n',
            '                    "rotation_period": det.per,\n',
            '                    "rotation_period_err": det.per_err,\n',
            "                }\n",
            "            )\n",
            "\n",
            "    df = pd.DataFrame(rows)\n",
            "    return df\n",
            "\n",
            "\n",
            "summary_df = run_detectors_for_all_stars(max_files_per_star=2)\n",
            "summary_df.head()\n",
        ],
    })

    cells.append({
        "cell_type": "markdown",
        "id": "summary-df-explain",
        "metadata": {},
        "source": [
            "上のセルでは、各恒星について最大 2 ファイルずつ処理し、\n",
            "\n",
            "- `flare_ratio` : フレア発生率 [1/day]\n",
            "- `energy_ratio` : フレアエネルギー率 [erg/day]\n",
            "- `starspot` / `starspot_ratio` : スポット面積と被覆率\n",
            "- `rotation_period` : 回転周期\n",
            "- `sector` : TESS セクタ番号\n",
            "\n",
            "を 1 行ずつのレコードにまとめています。\n",
            "この `summary_df` をもとに、セクタ間・恒星間の比較プロットを作っていきます。\n",
        ],
    })

    # 2. 単一恒星内のセクタ間比較
    cells.append({
        "cell_type": "markdown",
        "id": "single-star-compare",
        "metadata": {},
        "source": [
            "## 2. 単一恒星内のセクタ間比較（例：EK Dra）\n",
            "\n",
            "まずは EK Dra だけを取り出し、セクタごとのフレア発生率とエネルギー率を\n",
            "簡単な散布図で眺めてみます。\n",
        ],
    })

    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "id": "ekdra-sector-plot",
        "metadata": {},
        "outputs": [],
        "source": [
            'ek_df = summary_df[summary_df["star"] == "EK Dra"].copy()\n',
            "\n",
            "if ek_df.empty:\n",
            '    print("EK Dra のデータがありません。上のセルで summary_df を作成できているか確認してください。")\n',
            "else:\n",
            '    ek_df = ek_df.sort_values("sector")\n',
            "\n",
            "    fig = px.scatter(\n",
            "        ek_df,\n",
            '        x="sector",\n',
            '        y="flare_ratio",\n',
            '        size="energy_ratio",\n',
            '        hover_data=["data_name"],\n',
            '        labels={"sector": "TESS Sector", "flare_ratio": "Flare Rate [1/day]"},\n',
            '        title="EK Dra: セクタごとのフレア発生率とエネルギー率 (バブルサイズ)",\n',
            "    )\n",
            '    fig.update_yaxes(type="log")\n',
            "    fig.show()\n",
        ],
    })

    cells.append({
        "cell_type": "markdown",
        "id": "ekdra-plot-explain",
        "metadata": {},
        "source": [
            "図では、\n",
            "\n",
            "- x 軸: セクタ番号\n",
            "- y 軸: フレア発生率（対数軸）\n",
            "- バブルの大きさ: エネルギー率\n",
            "\n",
            "として表示しています。セクタごとの観測時間や検出閾値の違いによって、\n",
            "どの程度フレア活動が変動して見えるかを直感的に把握できます。\n",
        ],
    })

    # 3. 複数恒星比較
    cells.append({
        "cell_type": "markdown",
        "id": "multi-star-compare",
        "metadata": {},
        "source": [
            "## 3. 複数恒星の比較プロット\n",
            "\n",
            "次に、EK Dra / DS Tuc A / V889 Her を 1 枚の図にまとめ、\n",
            "\n",
            "- 回転周期\n",
            "- フレア発生率\n",
            "- スポット被覆率\n",
            "\n",
            "の関係をざっくり眺めてみます。\n",
        ],
    })

    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "id": "multi-star-plot",
        "metadata": {},
        "outputs": [],
        "source": [
            "if summary_df.empty:\n",
            '    print("summary_df が空です。上のセルで run_detectors_for_all_stars を実行してください。")\n',
            "else:\n",
            "    fig = px.scatter(\n",
            "        summary_df,\n",
            '        x="rotation_period",\n',
            '        y="flare_ratio",\n',
            '        color="star",\n',
            '        size="starspot_ratio",\n',
            '        hover_data=["sector", "data_name"],\n',
            "        labels={\n",
            '            "rotation_period": "Rotation Period [day]",\n',
            '            "flare_ratio": "Flare Rate [1/day]",\n',
            '            "starspot_ratio": "Starspot Filling Factor (relative)",\n',
            "        },\n",
            '        title="複数恒星における回転周期・フレア発生率・スポット比の関係",\n',
            "    )\n",
            '    fig.update_xaxes(type="log")\n',
            '    fig.update_yaxes(type="log")\n',
            "    fig.show()\n",
        ],
    })

    cells.append({
        "cell_type": "markdown",
        "id": "multi-star-plot-explain",
        "metadata": {},
        "source": [
            "この図から、\n",
            "\n",
            "- 回転が速い星ほどフレア発生率が高いか？\n",
            "- スポット被覆率が大きい星ほどフレアが多いか？\n",
            "\n",
            "といった物理的な傾向を、ざっくりと可視化できます。\n",
            "より厳密な解析を行う場合は、観測時間や閾値設定の違いをそろえたうえで\n",
            "FFD フィッティングなどを行う必要がありますが、このノートの目的は\n",
            "「クラス変数に蓄積された情報から、素早く比較プロットを作る」ことにあります。\n",
        ],
    })

    return cells


def update_notebook(path: Path, build_cells) -> None:
    if not path.exists():
        print(f"Skip {path} (not found)")
        return

    with path.open(encoding="utf-8") as f:
        nb = json.load(f)

    nb["cells"] = build_cells()

    with path.open("w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=2)

    print(f"Updated {path}")


def main() -> None:
    update_notebook(NB04_PATH, build_nb04_cells)
    update_notebook(NB05_PATH, build_nb05_cells)


if __name__ == "__main__":
    main()
