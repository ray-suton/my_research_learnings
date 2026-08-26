import argparse

from config_utils import format_config, load_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    config = load_config(args.config)
    print(format_config(config))

if __name__ == "__main__":
    main()
