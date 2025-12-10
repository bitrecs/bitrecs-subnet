from enum import Enum


class ColorScheme(Enum):
    VIRIDIS = "viridis"
    ROCKET = "rocket"
    MAKOTO = "makoto"
    SPECTRAL = "spectral"

class ColorPalette:
    """Color schemes for matrix visualization"""
    SCHEMES = {
        ColorScheme.VIRIDIS: {
            "strong": "\033[38;5;46m",   # Strong Green
            "medium": "\033[38;5;37m",     # Teal
            "weak": "\033[38;5;31m",   # Deep Blue
            "minimal": "\033[38;5;55m",   # Dark Purple 
            "highlight": "\033[38;5;227m" # Bright Yellow
        },
        ColorScheme.ROCKET: {
            "strong": "\033[38;5;89m",    # Deep Plum
            "medium": "\033[38;5;161m",   # Reddish Purple
            "weak": "\033[38;5;196m",     # Warm Red
            "minimal": "\033[38;5;209m",   # Coral
            "highlight": "\033[38;5;223m"  # Light Peach
        },
        ColorScheme.MAKOTO: {
            "strong": "\033[38;5;232m",   # Near Black
            "medium": "\033[38;5;24m",    # Dark Blue
            "weak": "\033[38;5;67m",      # Steel Blue
            "minimal": "\033[38;5;117m",  # Light Sky Blue
            "highlight": "\033[38;5;195m" # Pale Blue
        },
        ColorScheme.SPECTRAL: {
            "strong": "\033[38;5;160m",   # Red
            "medium": "\033[38;5;215m",   # Orange
            "weak": "\033[38;5;229m",     # Soft Yellow
            "minimal": "\033[38;5;151m",  # Mint Green
            "highlight": "\033[38;5;32m"  # Cool Blue
        }
    }


class RarityTier(Enum):
    COMMON = "Common"
    MAGIC = "Magic"
    RARE = "Rare"      
    LEGENDARY = "Legendary"
    UNIQUE = "Unique"    
  
    @staticmethod
    def get_tier_icon(tier: str) -> str:
        """Return a colored icon for the given tier string (case-insensitive)."""
        tier_lower = tier.lower()
        icons = {
            "common": "\033[90m● Common\033[0m",             # Gray circle
            "magic": "\033[34m● Magic\033[0m",               # Blue circle
            "rare": "\033[35m● Rare\033[0m",                 # Purple circle
            "legendary": "\033[38;5;208m♦ Legendary\033[0m", # Orange diamond
            "unique": "\033[1;33m★ Unique\033[0m"            # Yellow star
        }
        return icons.get(tier_lower, "\033[91m?\033[0m")  # Default to "?" for invalid


