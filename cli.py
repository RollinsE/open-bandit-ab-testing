#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Command-line interface for the Open Bandit A/B testing pipeline."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from pipeline import create_parser, main as run_pipeline


def main():
    parser = create_parser()
    args = parser.parse_args()
    return run_pipeline(args)


if __name__ == '__main__':
    main()
