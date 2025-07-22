import re
import bitrecs
import datetime
from pathlib import Path
from datetime import datetime, timezone

"""
Constants:
    ROOT_DIR (Path): Root directory of the project.
    SCHEMA_UPDATE_CUTOFF (datetime): Cutoff date for schema updates.
    EPOCH_TEMPO (int): Number of blocks in an epoch.
    TEMPO_SYNC_INTERVAL (int): Length of seconds between tempo syncs.
    MAX_DENDRITE_TIMEOUT (int): Maximum timeout for dendrite requests.
    MIN_QUERY_LENGTH (int): Minimum length of a query.
    MAX_QUERY_LENGTH (int): Maximum length of a query.
    MIN_RECS_PER_REQUEST (int): Minimum recommendations per request.
    MAX_RECS_PER_REQUEST (int): Maximum recommendations per request.
    MAX_CONTEXT_TEXT_LENGTH (int): Maximum length of context text.
    MAX_CONTEXT_TOKEN_COUNT (int): Maximum number of tokens in context.
    MIN_CATALOG_SIZE (int): Minimum size of the catalog.
    MAX_CATALOG_SIZE (int): Maximum size of the catalog.
    ACTION_SYNC_INTERVAL (int): Interval for action syncs.
    VERSION_CHECK_INTERVAL (int): Interval for version checks.
    COOLDOWN_SYNC_INTERVAL (int): Interval for cooldown syncs.
    CATALOG_DUPE_THRESHOLD (float): Threshold for duplicate catalog entries.
    R2_SYNC_INTERVAL (int): Interval for R2 syncs.
    RE_PRODUCT_NAME (Pattern): Regular expression for validating product names.
    RE_REASON (Pattern): Regular expression for validating reasons.
    RE_MODEL_NAME (Pattern): Regular expression for validating model names.
    CONVERSION_SCORING_ENABLED (bool): Flag to enable conversion scoring.
    QUERY_BATCH_SIZE (int): Size of query batches.
    MIN_QUERY_BATCH_SIZE (int): Minimum size of query batches.
    BATCH_FAILURE_THRESHOLD (float): Threshold for batch failure.
    SCORE_DISPLAY_ENABLED (bool): Flag to enable score display.
    SCORE_DISPLAY_INTERVAL (int): Interval for score display updates.
    REWARD_ORPHANS (bool): Flag to enable rewarding orphaned miners.

"""

ROOT_DIR = Path(bitrecs.__file__).parent.parent
SCHEMA_UPDATE_CUTOFF = datetime(2025, 7, 28, tzinfo=timezone.utc)
EPOCH_TEMPO = 360
TEMPO_SYNC_INTERVAL = 120
MAX_DENDRITE_TIMEOUT = 5
MIN_QUERY_LENGTH = 3
MAX_QUERY_LENGTH = 40
MIN_RECS_PER_REQUEST = 1
MAX_RECS_PER_REQUEST = 20
MAX_CONTEXT_TEXT_LENGTH = 1_000_000
MAX_CONTEXT_TOKEN_COUNT = 600_000
MIN_CATALOG_SIZE = 6
MAX_CATALOG_SIZE = 100_000
ACTION_SYNC_INTERVAL = 14400
VERSION_CHECK_INTERVAL = 1200
COOLDOWN_SYNC_INTERVAL = 360
R2_SYNC_INTERVAL = 3600
CATALOG_DUPE_THRESHOLD = 0.05
RE_PRODUCT_NAME = re.compile(r"[^A-Za-z0-9 |-]")
RE_REASON = re.compile(r"[^A-Za-z0-9 ]")
RE_MODEL_NAME = re.compile(r"[^A-Za-z0-9-._/-:]")
CONVERSION_SCORING_ENABLED = False
QUERY_BATCH_SIZE = 14
MIN_QUERY_BATCH_SIZE = 3
BATCH_FAILURE_THRESHOLD = 0.90
SCORE_DISPLAY_ENABLED = True
SCORE_DISPLAY_INTERVAL = 180
REWARD_ORPHANS = True