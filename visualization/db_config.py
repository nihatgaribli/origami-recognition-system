"""Backward-compatible alias for older imports.

Some modules may still import `visualization.db_config`.
The active config lives in `visualization._db_config`.
"""

from visualization._db_config import *  # noqa: F401,F403
