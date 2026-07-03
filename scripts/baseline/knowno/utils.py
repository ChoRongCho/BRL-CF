from __future__ import annotations


GREEN = "\033[32m"
YELLOW = "\033[33m"
RESET = "\033[0m"


def usage_total(usage) -> int:
    if not usage:
        return 0
    return (
        usage.get("total_tokens")
        or usage.get("total_token_count")
        or sum(value for key, value in usage.items() if key.endswith("tokens") and isinstance(value, int))
    )
