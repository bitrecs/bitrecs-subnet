import os
import json
import logging
import sqlite3
import bittensor as bt
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from typing_extensions import List
from logging.handlers import RotatingFileHandler
from bitrecs.llms.prompt_factory import PromptFactory
from bitrecs.protocol import BitrecsRequest, SignedResponse
from bitrecs.utils import constants as CONST
from bitrecs.utils.constants import SCHEMA_UPDATE_CUTOFF, TRUNCATE_LOGS_DB_DAYS, TRUNCATE_LOGS_ENABLED

EVENTS_LEVEL_NUM = 38
DEFAULT_LOG_BACKUP_COUNT = 10
TIMESTAMP_FILE = 'timestamp.txt'
NODE_INFO_FILE = 'node_info.json'

def setup_events_logger(full_path, events_retention_size):
    logging.addLevelName(EVENTS_LEVEL_NUM, "EVENT")

    logger = logging.getLogger("event")
    logger.setLevel(EVENTS_LEVEL_NUM)

    def event(self, message, *args, **kws):
        if self.isEnabledFor(EVENTS_LEVEL_NUM):
            self._log(EVENTS_LEVEL_NUM, message, args, **kws)

    logging.Logger.event = event

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = RotatingFileHandler(
        os.path.join(full_path, "events.log"),
        maxBytes=events_retention_size,
        backupCount=DEFAULT_LOG_BACKUP_COUNT,
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(EVENTS_LEVEL_NUM)
    logger.addHandler(file_handler)

    return logger


def get_db_log_path() -> Path:
    """Get the path to the miner responses database file."""
    db_path_env = os.environ.get("MINER_RESPONSES_DB_PATH")
    if db_path_env:
        data_file = Path(db_path_env)
        bt.logging.trace(f"Using custom DB path: {data_file}")
    else:
        data_file = Path(CONST.ROOT_DIR) / "miner_responses.db"
    return data_file


def write_node_info(network, uid, hotkey, neuron_type, sample_size, v_limit, epoch_length) -> None:
    """Write node information for the auto-updater"""    
    node_info = {
        "network": network,
        "uid": uid,
        "hotkey": hotkey,
        "neuron_type": neuron_type,
        "sample_size": sample_size,
        "epoch_length": epoch_length,
        "v_limit": v_limit,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    }    
    tmp_file = NODE_INFO_FILE + '.tmp'    
    try:
        with open(tmp_file, 'w', encoding='utf-8') as f:
            json.dump(node_info, f, indent=2, ensure_ascii=False)
        os.replace(tmp_file, NODE_INFO_FILE)
    except Exception as e:
        bt.logging.error(f"Error writing node info: {e}")
        # Clean up temp file if it exists
        if os.path.exists(tmp_file):
            os.remove(tmp_file)
        raise


def read_node_info() -> dict:
    """Read node information"""
    try:
        with open(NODE_INFO_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        bt.logging.warning(f"Node info file not found: {NODE_INFO_FILE}")
        return {}
    except json.JSONDecodeError as e:
        bt.logging.error(f"Error parsing node info JSON: {e}")
        return {}
    except Exception as e:
        bt.logging.error(f"Error reading node info: {e}")
        return {}



def write_timestamp(current_time):    
    tmp_file = TIMESTAMP_FILE + '.tmp'
    with open(tmp_file, 'w') as f:
        f.write(str(current_time))
    os.replace(tmp_file, TIMESTAMP_FILE)  # Atomic operation to replace the file


def read_timestamp():
    try:
        with open(TIMESTAMP_FILE, 'r') as f:
            timestamp_str = f.read()
            return float(timestamp_str)
    except (FileNotFoundError, ValueError):
        return None


def update_table_schema(conn: sqlite3.Connection, required_columns: list) -> None:
    """Update table schema to include any missing columns before cutoff date."""
    if datetime.now(timezone.utc) > SCHEMA_UPDATE_CUTOFF:
        return
    cursor = conn.cursor()    
    cursor.execute("PRAGMA table_info(miner_responses)")
    existing_columns = {row[1] for row in cursor.fetchall()}
    for col in required_columns:
        if col not in existing_columns:
            bt.logging.info(f"Adding missing column: {col}")
            alter_sql = f'ALTER TABLE miner_responses ADD COLUMN "{col}" TEXT'
            cursor.execute(alter_sql)
    conn.commit()


def truncate_miner_log_db(since_date: datetime) -> int:
    """Truncate miner log database to remove entries older than since_date."""
    data_file = get_db_log_path()
    if not os.path.exists(data_file):
        bt.logging.error("No miner_responses.db found to truncate")
        return 0
    try:
        conn = sqlite3.connect(data_file)
        cursor = conn.cursor()
        cutoff_str = since_date.strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("DELETE FROM miner_responses WHERE created_at < ?", (cutoff_str,))
        deleted_rows = cursor.rowcount
        conn.commit()
        if deleted_rows > 1000:
            bt.logging.trace("Running VACUUM on miner_responses.db to reclaim space")
            cursor.execute("VACUUM")
            conn.commit()        
        conn.close()
        bt.logging.trace(f"Truncated {deleted_rows} rows older than {cutoff_str} from miner_responses.db")
        return deleted_rows
    except sqlite3.Error as e:
        bt.logging.error(f"SQLite error during truncation: {e}")
        return 0



def log_miner_responses_to_sql(step: int, 
                                responses: List[BitrecsRequest], 
                                rewards: np.ndarray = None, 
                                reward_notes: List[str] = None, 
                                elected: BitrecsRequest = None) -> None:
    try:        
        if TRUNCATE_LOGS_ENABLED:
            deleted = truncate_miner_log_db(datetime.now(timezone.utc) - pd.Timedelta(days=TRUNCATE_LOGS_DB_DAYS))
            bt.logging.trace(f"Truncated {deleted} old rows from miner log database.")
        frames = []
        for response in responses:
            if not isinstance(response, BitrecsRequest):
                bt.logging.warning(f"Skipping invalid response type: {type(response)}")
                continue
            response.context = ""
            response.user = ""
            data = {
                **response.to_headers(),
                **response.to_dict()
            }
            df = pd.json_normalize(data)
            if rewards is not None and len(rewards) > 0:
                df['reward'] = rewards[responses.index(response)] if responses.index(response) < len(rewards) else 0.0
            else:
                df['reward'] = 0.0
            if reward_notes is not None and len(reward_notes) > 0:
                df['reward_note'] = reward_notes[responses.index(response)] if responses.index(response) < len(reward_notes) else ""
            else:
                df['reward_note'] = ""
            if response.verified_proof and "proof" in response.verified_proof:
                signed_response = SignedResponse(**response.verified_proof)
                model = PromptFactory.extract_model_from_proof(signed_response)
                df['models_used'] = json.dumps([model])
            frames.append(df)
        final = pd.concat(frames)

        batch_elected_uid = elected.miner_uid if elected and elected.miner_uid else ""
        batch_elected_hotkey = elected.axon.hotkey if elected and elected.axon else ""
        batch_elected_process_time = elected.axon.process_time if elected and elected.axon else 0

        if len(final) > 0:
            utc_now = datetime.now(timezone.utc)
            created_at = utc_now.strftime("%Y-%m-%d %H:%M:%S")
            data_file = get_db_log_path()
            conn = sqlite3.connect(data_file)
            try:
                final['step'] = step
                final['created_at'] = created_at
                final['batch_elected_uid'] = batch_elected_uid
                final['batch_elected_hotkey'] = batch_elected_hotkey
                final['batch_elected_process_time'] = batch_elected_process_time

                dtype_dict = {col: 'TEXT' for col in final.columns}
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='miner_responses';")
                table_exists = cursor.fetchone() is not None                
                if not table_exists:
                    final.to_sql('miner_responses', conn, index=False, dtype=dtype_dict)
                else:                    
                    update_table_schema(conn, list(final.columns))
                    final.to_sql('miner_responses', conn, index=False, if_exists='append', dtype=dtype_dict)
                conn.commit()
                bt.logging.trace(f"DB Updated at step {step}")
            except sqlite3.Error as e:
                bt.logging.error(f"SQLite error: {e}")
                conn.rollback()
            finally:
                conn.close()

        bt.logging.info(f"Miner responses logged {len(final)}")
    except Exception as e:
        bt.logging.error(f"Error in logging miner responses: {str(e)}")
        bt.logging.error(f"Columns in dataframe: {list(final.columns)}")