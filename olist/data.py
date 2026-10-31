"""Load Olist datasets."""

from pathlib import Path

import pandas as pd


class Olist:
    """Provide access to Olist CSV datasets."""

    def get_data(self):
        """Return the CSV tables as a dictionary of DataFrames."""
        csv_path = Path.home() / ".workintech" / "olist" / "data" / "csv"

        file_paths = sorted(
            path for path in csv_path.iterdir()
            if path.is_file() and path.suffix == ".csv"
        )

        data = {}

        for path in file_paths:
            key = (
                path.name
                .removeprefix("olist_")
                .removesuffix(".csv")
                .removesuffix("_dataset")
            )
            data[key] = pd.read_csv(path)

        return data

    def ping(self):
        """Confirm that the package can be imported."""
        print("pong")