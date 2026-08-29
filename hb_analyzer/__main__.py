"""Allow running as: python -m hb_analyzer ..."""
from .cli import main
import sys
sys.exit(main())
