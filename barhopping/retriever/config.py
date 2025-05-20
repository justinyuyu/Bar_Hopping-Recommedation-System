import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BARS_DB = os.path.join(PROJECT_ROOT, "data", "bars_tpe.db")