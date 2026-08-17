"""Run one bounded input through a frozen extensible-versus-legacy parity task.

Thin CLI wrapper around parity_support.run_parity for a single-input,
single-epoch smoke check against a frozen baseline (cvcv12 for rPCC;
trace212 or raonaturalimages5 for sPCC). Invoke from the pypredcoding root:
``python tests/test_smoke_1input.py --model sPCC --task trace212``.
"""

from __future__ import annotations

import argparse

from parity_support import run_parity


def main() -> int:
    """Parse CLI arguments and run the one-input smoke parity task.

    Returns:
        Process exit code: 0 on PASS, 1 on FAIL.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=("rPCC", "sPCC"))
    parser.add_argument("--task", required=True, choices=("cvcv12", "trace212", "raonaturalimages5"))
    parser.add_argument("--seed-init", type=int, default=1)
    parser.add_argument("--seed-shuffle", type=int, default=1)
    args = parser.parse_args()
    try:
        return run_parity(args.model, args.task, 1, args.seed_init, args.seed_shuffle, one_input=True)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())