import subprocess
import bitrecs.utils.constants as CONST

from shlex import split
from dataclasses import dataclass
from importlib.metadata import version
from bitrecs import __spec_version__ as spec_version
from bitrecs import __version__ as this_version
    
@dataclass
class LocalMetadata:
    """Metadata associated with the local neuron instance"""

    head: str
    remote_head: str
    btversion: str
    uid: int = 0
    coldkey: str = ""
    hotkey: str = ""
    version: str = ""
    spec: str = ""

    def to_dict(self):
        return {
            'head': self.head,
            'remote_head': self.remote_head,
            'btversion': self.btversion,
            'uid': self.uid,
            'coldkey': self.coldkey,
            'hotkey': self.hotkey,
            'version': self.version,
            'spec': self.spec
        }
    

    @staticmethod
    def local_metadata() -> "LocalMetadata":
        """Extract the version as current git commit hash"""
        commit_hash = "head error"
        remote_commit_hash = "remote head error"
        bittensor_version = "metadata exception"        
        try:
            bittensor_version = version("bittensor")
            result = subprocess.run(
                split("git rev-parse HEAD"),
                check=True,
                capture_output=True,
                cwd=CONST.ROOT_DIR,
            )
            commit = result.stdout.decode().strip()
            assert len(commit) == 40, f"Invalid commit hash: {commit}"
            commit_hash = commit[:16]

            # Get remote commit hash
            result = subprocess.run(
                split("git ls-remote origin -h refs/heads/main"),
                check=True,
                capture_output=True,
                cwd=CONST.ROOT_DIR,
            )
            remote_commit = result.stdout.decode().strip().split()[0]
            assert len(remote_commit) == 40, f"Invalid remote commit hash: {remote_commit}"
            remote_commit_hash = remote_commit[:16]

        except Exception:
            commit_hash = "exception unknown"

        return LocalMetadata(
            head=commit_hash,
            remote_head=remote_commit_hash,
            btversion=bittensor_version,
            version=this_version,
            spec=spec_version
        )
    

    @staticmethod
    def version_match() -> bool:
        meta = LocalMetadata.local_metadata()
        if not meta.head or not meta.remote_head:
            raise ValueError("Could not get local or remote head")
        return meta.head == meta.remote_head
    
    
    @staticmethod
    def version() -> str:
        return str(this_version)
    
    @staticmethod
    def spec() -> str:
        return str(spec_version)
