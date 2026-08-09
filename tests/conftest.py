"""Test configuration: force the non-interactive matplotlib backend
before any test imports nav_map, so the suite runs identically on a
developer Mac and a headless CI runner."""
import matplotlib

matplotlib.use("Agg", force=True)
