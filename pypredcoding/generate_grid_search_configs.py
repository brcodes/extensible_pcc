"""Generate timestamped grid-search config files (and folders) from a base config.

Define PARAMS_TO_SEARCH below as a plain dict mapping config keys to either
a fixed value, or a Sweep(...) of values to search over. A Sweep can also be
placed anywhere *inside* a larger value (nested in a dict/list/tuple), so you
can sweep a single field inside a bigger structure like rate_schedule without
rewriting the whole thing for every combination.

Examples
--------
Sweep a couple of top-level params (cartesian product across all Sweeps):

    PARAMS_TO_SEARCH = {
        'kU': {1: 0.02, 2: 0.94},
        'kr': {1: 0.05, 2: 0.125},
        'top_lyr_size': 48,
        'epoch_n': Sweep(2000, 4000),
    }

Sweep one number nested inside a larger structure (e.g. one point of a
piecewise-linear rate schedule), while everything else stays fixed:

    PARAMS_TO_SEARCH = {
        'rate_schedule': {
            'type': 'piecewise_linear',
            'param': 'kU',
            'layer': 2,
            'points': [(0, 0.94), (3500, 0.94), (4000, Sweep(1.05, 1.15, 1.30, 1.50))],
        },
    }

Both patterns can be combined and mixed freely; every Sweep found anywhere
in PARAMS_TO_SEARCH contributes one axis to the cartesian product of
generated configs.
"""

from __future__ import annotations

from datetime import datetime
from itertools import product
from os import makedirs
from os.path import basename, join, splitext
import argparse
import re
from typing import Any, Iterator

# Choose the config file you want as your base for the grid search. 
# User-defined Search/Sweep params will be applied to copies of this base config and put in the DIR defined below
# Search/Sweep params will appear in the generated config filenames according to sweep order and up to a num chars limit.
BASE_CONFIG_PATH = join('config', 'config_active.txt')
GRID_SEARCH_DIR_PREFIX = 'configs_grid_search'
GRID_SEARCH_FILENAME_MAX_LEN = 80


class Sweep:
    """Marks a value (or a value nested inside a dict/list/tuple) to search over."""

    _counter = 0

    def __init__(self, *values: Any, name: str | None = None) -> None:
        """Store the sweep values and assign an auto-numbered name if none is given.

        Raises:
            ValueError: If no values are provided.
        """
        if not values:
            raise ValueError('Sweep requires at least one value.')
        self.values = values
        if name is None:
            name = f'sweep{Sweep._counter}-'
            Sweep._counter += 1
        self.name = name


def piecewise_schedule(param: str, key: int, points: list[tuple[int, Any]]) -> dict[str, Any]:
    """Convenience builder for a piecewise-linear rate_schedule entry."""
    return {
        'type': 'piecewise_linear',
        'param': param,
        'key': key,
        'points': points,
    }

# ---------------------------------------------------------------------------
# Define your parameter search here.
# ---------------------------------------------------------------------------

PARAMS_TO_SEARCH = {
    'kU': {1: 0.02, 2: 0.94},
    'kr': {1: 0.05, 2: 0.125},
    'top_lyr_size': 48,
    'epoch_n': Sweep(2000, 4000),
    'rate_schedule': piecewise_schedule(
        param='kU',
        key=2,
        points=[(0, 0.94), (3500, 0.94), (4000, Sweep(1.05, 1.15, 1.30, 1.50))],
    ),
}


def _iter_sweeps(value: Any) -> Iterator[Sweep]:
    """Yield every Sweep instance nested within value, at any depth."""
    if isinstance(value, Sweep):
        yield value
    elif isinstance(value, dict):
        for sub_value in value.values():
            yield from _iter_sweeps(sub_value)
    elif isinstance(value, (list, tuple)):
        for sub_value in value:
            yield from _iter_sweeps(sub_value)


def _substitute_sweeps(value: Any, chosen: dict[Sweep, Any]) -> Any:
    """Return a copy of value with every Sweep replaced by chosen[sweep]."""
    if isinstance(value, Sweep):
        return chosen[value]
    if isinstance(value, dict):
        return {key: _substitute_sweeps(sub_value, chosen) for key, sub_value in value.items()}
    if isinstance(value, list):
        return [_substitute_sweeps(sub_value, chosen) for sub_value in value]
    if isinstance(value, tuple):
        return tuple(_substitute_sweeps(sub_value, chosen) for sub_value in value)
    return value


def _expand_params_to_search(params_to_search: dict[str, Any]) -> list[tuple[dict[str, str], list[tuple[str, Any]]]]:
    """Expand PARAMS_TO_SEARCH into (replacements, label_parts) per combination.

    Returns:
        One (replacements, label_parts) pair per point in the cartesian
        product of all Sweeps; replacements map config keys to repr strings,
        label_parts pair each sweep's name with its chosen value.
    """
    sweeps = []
    seen_ids = set()
    for value in params_to_search.values():
        for sweep in _iter_sweeps(value):
            if id(sweep) not in seen_ids:
                seen_ids.add(id(sweep))
                sweeps.append(sweep)

    if not sweeps:
        replacements = {key: repr(value) for key, value in params_to_search.items()}
        return [(replacements, [])]

    combinations = []
    for combo_values in product(*(sweep.values for sweep in sweeps)):
        chosen = dict(zip(sweeps, combo_values))
        replacements = {
            key: repr(_substitute_sweeps(value, chosen))
            for key, value in params_to_search.items()
        }
        label_parts = [(sweep.name, chosen[sweep]) for sweep in sweeps]
        combinations.append((replacements, label_parts))
    return combinations


def _remove_whitespace(value: str) -> str:
    """Strip all whitespace characters from a string."""
    return re.sub(r'\s+', '', value)


def _filename_fragment(label_parts: list[tuple[str, Any]]) -> str:
    """Build a filesystem-safe, length-capped filename fragment from sweep labels."""
    if not label_parts:
        return 'base'

    fragment = '_'.join(
        f'{name}{_remove_whitespace(repr(value))}' for name, value in label_parts
    )
    fragment = fragment.replace('/', '-')
    if len(fragment) > GRID_SEARCH_FILENAME_MAX_LEN:
        fragment = fragment[:GRID_SEARCH_FILENAME_MAX_LEN] + '..'
    return fragment


def _render_config(base_config_path: str, replacements: dict[str, str]) -> str:
    """Copy the base config text with target key=value lines replaced.

    Raises:
        ValueError: If the base config lacks a line for any replacement key.
    """
    with open(base_config_path, 'r') as config_file:
        lines = config_file.readlines()

    remaining_keys = set(replacements)
    rendered_lines = []

    for line in lines:
        stripped_line = line.strip()
        if stripped_line and not stripped_line.startswith('#') and '=' in stripped_line:
            key = stripped_line.split('=', 1)[0].strip()
            if key in replacements:
                rendered_lines.append(f'{key}={replacements[key]}\n')
                remaining_keys.remove(key)
                continue
        rendered_lines.append(line)

    if remaining_keys:
        missing = ', '.join(sorted(remaining_keys))
        raise ValueError(f'Base config is missing target parameter line(s): {missing}')

    return ''.join(rendered_lines)


def generate_grid_search_configs(base_config_path: str, params_to_search: dict[str, Any] | None = None, dry_run: bool = False) -> tuple[str, list[str]]:
    """Write one config file per sweep combination into a new timestamped directory.

    Returns:
        Tuple of the output directory and the generated config paths (paths
        are computed but nothing is written when dry_run is True).
    """
    if params_to_search is None:
        params_to_search = PARAMS_TO_SEARCH

    timestamp = datetime.now().strftime('%y%m%d_%H%M%S')
    output_dir = join('config', f'{GRID_SEARCH_DIR_PREFIX}_{timestamp}')

    if not dry_run:
        makedirs(output_dir, exist_ok=False)

    base_config_name = splitext(basename(base_config_path))[0]

    generated_paths = []
    for replacements, label_parts in _expand_params_to_search(params_to_search):
        config_text = _render_config(base_config_path, replacements)
        config_name = f'{base_config_name}_{_filename_fragment(label_parts)}.txt'
        config_path = join(output_dir, config_name)
        generated_paths.append(config_path)

        if not dry_run:
            with open(config_path, 'w') as config_file:
                config_file.write(config_text)

    return output_dir, generated_paths


def main() -> None:
    """Parse CLI args and generate (or dry-run) the grid-search configs."""
    parser = argparse.ArgumentParser(
        description='Generate timestamped grid-search config files from PARAMS_TO_SEARCH.'
    )
    parser.add_argument(
        '--base-config',
        default=BASE_CONFIG_PATH,
        help='Base config path to copy and edit. Default: config/config_active.txt',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate and print generated paths without writing files.',
    )
    args = parser.parse_args()

    output_dir, generated_paths = generate_grid_search_configs(
        args.base_config,
        dry_run=args.dry_run,
    )

    action = 'Would write' if args.dry_run else 'Wrote'
    print(f'{action} {len(generated_paths)} grid-search configs to: {output_dir}')
    for generated_path in generated_paths:
        print(generated_path)


if __name__ == '__main__':
    main()
