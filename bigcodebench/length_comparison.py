# TODO: This license is not consistent with the license used in the project.
#       Delete the inconsistent license and above line and rerun pre-commit to insert a good license.
# # SPDX-FileCopyrightText: (c) {year} UIUC Security and Privacy Lab
# #
# # SPDX-License-Identifier: Apache-2.0

import os
from multiprocessing import Pool

from rich.console import Console
from rich.table import Table
from tqdm import tqdm
from transformers import AutoTokenizer


def tokenize_texts(texts: list[str], num_workers: int = None) -> list[list[int]]:
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")

    if num_workers is None:
        num_workers = os.cpu_count()

    with Pool(num_workers) as pool:
        token_lists = list(
            tqdm(
                pool.imap(tokenizer.encode, texts), total=len(texts), desc="Tokenizing"
            )
        )

    return token_lists


def tokenize_string(text: str) -> list[int]:
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
    return tokenizer.encode(text)


def plot_token_distribution(token_lists: list[list[int]]) -> None:
    import matplotlib.pyplot as plt

    token_counts = [len(tokens) for tokens in token_lists]

    plt.figure(figsize=(10, 6))
    plt.hist(token_counts, bins=50, edgecolor="black", alpha=0.7)
    plt.xlabel("Number of Tokens")
    plt.ylabel("Frequency")
    plt.title("Distribution of Token Counts")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    plt.savefig("results/cot_length_distribution.png")

    console = Console()

    stats_table = Table(
        title="Token Count Statistics", show_header=True, header_style="bold blue"
    )
    stats_table.add_column("Metric", style="cyan", width=20)
    stats_table.add_column("Value", justify="right", style="green")

    stats_table.add_row("Total Texts", f"{len(token_counts)}")
    stats_table.add_row("Min Tokens", f"{min(token_counts)}")
    stats_table.add_row("Max Tokens", f"{max(token_counts)}")
    stats_table.add_row("Mean Tokens", f"{sum(token_counts)/len(token_counts):.2f}")
    stats_table.add_row(
        "Median Tokens", f"{sorted(token_counts)[len(token_counts)//2]}"
    )

    console.print(stats_table)
