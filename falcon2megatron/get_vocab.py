import json
from pathlib import Path
from argparse import ArgumentParser

import requests


def main(url: str, out: Path = Path("."), prefix: str = ""):
    print("Fetching file")
    req = requests.get(url)
    info = json.loads(req.content.decode("utf8"))

    print("Dumping output")
    out = out.absolute()
    with open(out/f"{prefix}vocab.json", "w+") as f:
        json.dump(info["model"]["vocab"], f)
    with open(out/f"{prefix}merges.txt", "w+") as f:
        for line in info["model"]["merges"]:
            f.write(f"{line}\n")

if __name__ == "__main__":
    parser = ArgumentParser(description="Download Falcon vocab and merge files")
    parser.add_argument("--url", default="https://huggingface.co/tiiuae/falcon-7b/raw/main/tokenizer.json",
                        help="Where to look for the tokenizier files")
    parser.add_argument("--out", type=Path, default=Path("."),
                        help="Directory to store the vocab and merge files")
    parser.add_argument("--prefix", default="",
                        help="Prefix of the output files")
    args = parser.parse_args()
    main(args.url, args.out, args.prefix)
