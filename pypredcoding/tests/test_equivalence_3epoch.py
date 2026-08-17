"""Run bounded frozen extensible-versus-legacy parity over multiple epochs.

Thin CLI wrapper around parity_support.run_parity for a full-dataset,
multi-epoch (default 3) equivalence check against a frozen baseline
(cvcv12 for rPCC; trace212 or raonaturalimages5 for sPCC). Invoke from the
pypredcoding root:
``python tests/test_equivalence_3epoch.py --model rPCC --task cvcv12``.
"""

from __future__ import annotations

import argparse

from parity_support import run_parity


def main() -> int:
    """Parse CLI arguments and run the multi-epoch equivalence parity task.

    Returns:
        Process exit code: 0 on PASS, 1 on FAIL.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=("rPCC", "sPCC"))
    parser.add_argument("--task", required=True, choices=("cvcv12", "trace212", "raonaturalimages5"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed-init", type=int, default=1)
    parser.add_argument("--seed-shuffle", type=int, default=1)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    try:
        return run_parity(args.model, args.task, args.epochs, args.seed_init, args.seed_shuffle, one_input=False)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())