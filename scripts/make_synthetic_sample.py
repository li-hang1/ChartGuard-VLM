import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chartguard.synthetic import generate_revenue_bar_chart


def main() -> None:
    image_path, meta_path = generate_revenue_bar_chart("examples")
    print(f"image: {image_path}")
    print(f"metadata: {meta_path}")


if __name__ == "__main__":
    main()
