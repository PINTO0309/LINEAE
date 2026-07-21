"""Shared deployment configuration helpers."""


def resolve_num_select(configured: int, num_queries: int, override: int | None = None) -> int:
    """Resolve and validate the number of predictions retained for deployment."""
    num_queries = int(num_queries)
    num_select = int(configured if override is None else override)
    if not 0 < num_select <= num_queries:
        raise ValueError(
            f"num_select must be in [1, num_queries={num_queries}], got {num_select}"
        )
    return num_select
