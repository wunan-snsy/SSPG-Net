import argparse

import yaml

from tests.smoke_test import run_smoke_test


def main():
    parser = argparse.ArgumentParser(description="SSPG-Net reference entry point")
    parser.add_argument("--config", default="configs/rtx3060_6gb.yaml")
    parser.add_argument("--mode", default="smoke_test", choices=("smoke_test",))
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    run_smoke_test(config)


if __name__ == "__main__":
    main()
