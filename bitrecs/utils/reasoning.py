import os
import requests
import bittensor as bt
from typing import List
from dataclasses import dataclass, field    
from bitrecs import __version__ as this_version


@dataclass
class ReasoningReport:
    created_at: str = field(default_factory=str, description="Timestamp when the report was created")
    scored_on: str = field(default_factory=str, description="Date when the score was evaluated")
    miner_hotkey: str = field(default_factory=str, description="Miner hotkey identifier")
    f_score: float = field(default=0.0, description="F score for reasoning tasks")
    evaluated: int = field(default=0, description="Number of rec queries performed")
    rank: int = field(default=0, description="Global Rank based on reasoning score")

    
    @staticmethod
    def get_reports() -> List["ReasoningReport"]:
        """
        Load latest reasoning scores
        """
        reports = []
        try:            
            proxy_url = os.environ.get("BITRECS_PROXY_URL").removesuffix("/")
            reason_url = f"{proxy_url}/reasoning"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.environ.get('BITRECS_API_KEY')}",
                "User-Agent": f"Bitrecs-Node/{this_version}"
            }        
            report_json = requests.get(reason_url, headers=headers).json()
            data = report_json.get("data", [])
            if not data or len(data) == 0:
                bt.logging.error("No data found in reasoning report")
                return []
            
            for item in data:
                report = ReasoningReport(
                    created_at=item.get("created_at", ""),
                    scored_on=item.get("scored_on", ""),
                    miner_hotkey=item.get("miner_hotkey", ""),
                    evaluated=item.get("evaluated", 0),
                    f_score=item.get("f_score", 0.0),
                    rank=item.get("rank", 0)
                )
                reports.append(report)
            sorted_reports = sorted(reports, key=lambda x: x.rank, reverse=False)
            return sorted_reports
        except Exception as e:
            bt.logging.error(f"get_reports Exception: {e}")
        

    @staticmethod
    def get_delta_uids(reports: List["ReasoningReport"], metagraph: "bt.metagraph.Metagraph" ) -> List[int]:
        """Find the delta of reports vs metagraph UIDs."""
        delta_uids = []
        report_hotkeys = [r.miner_hotkey.strip().lower() for r in reports]
        for uid in range(metagraph.n.item()):
            if uid == 0:
                continue
            hk = metagraph.axons[uid].hotkey.strip().lower()
            if hk not in report_hotkeys:
                delta_uids.append(uid)
        return delta_uids