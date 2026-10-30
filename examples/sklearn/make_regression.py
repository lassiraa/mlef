"""Reproduce the synthetic regression fixture (Apache-2.0); no external downloads."""

import csv
import random
from pathlib import Path


def main() -> None:
    rng = random.Random(1729)
    with Path("regression.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "amount", "age", "tenure", "region", "target"])
        for index in range(240):
            amount, age, tenure = rng.uniform(0, 100), rng.randint(18, 80), rng.uniform(0, 20)
            region = rng.choice(["north", "south", "east"])
            target = (
                3 * amount
                + 0.1 * age
                + tenure**2
                + (30 if region == "north" else 0)
                + rng.gauss(0, 10)
            )
            writer.writerow([index, "" if index % 17 == 0 else amount, age, tenure, region, target])


if __name__ == "__main__":
    main()
