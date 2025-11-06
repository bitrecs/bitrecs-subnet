import requests
import bittensor as bt
import bitrecs.utils.constants as CONST
from typing import List
from dataclasses import dataclass, field    
from bitrecs import __version__ as this_version


@dataclass
class RarityReport:
    created_at: str = field(default_factory=str)
    model: str = field(default_factory=str)
    count: int = field(default=0)
    rarity: str = field(default_factory=str)
    tier: str = field(default_factory=str)
    bonus: float = field(default=0.0)
    
    @staticmethod
    def get_reports() -> List["RarityReport"]:
        """
        Load model rarity from the verified inference server.
        
        Returns:
            List[RarityReport]: A list of RarityReport instances.
        """
        reports = []
        try:            
            rarity_url = f"{CONST.VERIFIED_INFERENCE_URL}/rarity"
            headers = {
                "Content-Type": "application/json",
                "User-Agent": f"Bitrecs-Node/{this_version}"
            }        
            report_json = requests.get(rarity_url, headers=headers).json()
            rarity_report = report_json.get("rarity_report", {})
            models = rarity_report.get("models", [])            
            if not models:
                bt.logging.error("No models found in rarity report")
                return []
            
            for item in models:
                report = RarityReport(
                    created_at=item.get("created_at", ""),
                    model=item.get("model", ""),
                    count=item.get("count", 0),
                    rarity=item.get("rarity", ""),
                    tier=item.get("tier", ""),
                    bonus=item.get("bonus", 0.0)
                )
                reports.append(report)
            
            sorted_reports = sorted(reports, key=lambda x: x.count, reverse=False)
            return sorted_reports
        except Exception as e:
            bt.logging.error(f"get_reports Exception: {e}")
            return []


