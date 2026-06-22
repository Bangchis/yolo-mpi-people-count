from __future__ import annotations

import socket
import os


def main() -> int:
    rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", "0"))
    world = int(os.environ.get("OMPI_COMM_WORLD_SIZE", "1"))
    host = socket.gethostname()

    try:
        import torch  # type: ignore
    except ImportError as exc:
        print(
            f"rank={rank}/{world} host={host} ERROR missing torch: {exc}. "
            "Install optional MPS helper deps with: pip install -e '.[mps]'",
            flush=True,
        )
        return 1

    built = torch.backends.mps.is_built()
    available = torch.backends.mps.is_available()
    print(
        f"rank={rank}/{world} host={host} "
        f"torch={torch.__version__} mps_built={built} mps_available={available}",
        flush=True,
    )
    return 0 if built and available else 2


if __name__ == "__main__":
    raise SystemExit(main())
