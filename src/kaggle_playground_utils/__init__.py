"""kaggle-playground-utils — reusable toolkit for Kaggle Playground Series.

Two generations of tooling live here:

  v0.1 (S6E4 toolkit):
    - features, train, registry, blend, signal_factory, greedy_selection, etc.
    - Built around a Registry-based model lifecycle

  v0.2 (S6E5 additions):
    - harvesting: bulk-scan Kaggle public notebooks via CLI
    - curated_blend: top-K LB-filtered public mix (L19 — the winning lever on S6E5)
    - drive: Colab Drive restore/sync helpers
    - environment: Colab vs local detection, Kaggle CLI auth setup

See README.md for usage examples.
See ~/knowledge-graph/kaggle/2026-19_s6e5-postmortem.md for empirical lessons.
"""

__version__ = "0.2.0"


# v0.2 exports (S6E5 additions) — explicit imports for IDE/REPL convenience
from kaggle_playground_utils.curated_blend import (  # noqa: F401
    PublicSubmission,
    RatioSweepResult,
    curated_mix,
    load_public_subset,
    predicted_lb_score,
    rank_norm,
    ratio_sweep,
    top_k_curated_mean,
)
from kaggle_playground_utils.drive import (  # noqa: F401
    restore_from_drive,
    sync_file_to_drive,
    sync_to_drive,
)
from kaggle_playground_utils.environment import (  # noqa: F401
    detect_environment,
    get_drive_path,
    setup_kaggle_auth,
)
from kaggle_playground_utils.harvesting import (  # noqa: F401
    bulk_harvest,
    categorize_entry,
    extract_claimed_lb,
    extract_claimed_oof_auc,
    list_top_kernels,
    load_manifest,
    slug_to_tag,
    validate_oof_on_pool,
    validate_submission,
)
