from __future__ import annotations

import numpy as np
import pandas as pd

from esm2_ton_generator.data import add_split_column, extract_sequence, load_ton_dataset


def test_extract_sequence_handles_multiple_formats() -> None:
    assert extract_sequence("1-I,F,Y,G,K,V,A,N,M,D") == "IFYGKVANMD"
    assert extract_sequence("22-A,P,M,F,K,D,V,H,W,P-1") == "APMFKDVHWP"
    assert extract_sequence("第二轮迭代") is None
    assert extract_sequence(np.nan) is None


def test_load_ton_dataset_aggregates_duplicates(tmp_path) -> None:
    toy_df = pd.DataFrame(
        {
            "exp": [
                "第一轮随机",
                "1-A,C,D,E,F,G,H,I,K,L",
                "2-A,C,D,E,F,G,H,I,K,L",
                "3-W,Y,V,T,S,R,Q,P,N,M-2",
            ],
            "ton": [np.nan, 10.0, 12.0, 20.0],
        }
    )
    excel_path = tmp_path / "toy.xlsx"
    toy_df.to_excel(excel_path, index=False)

    parsed = load_ton_dataset(excel_path, aggregate="mean")
    assert set(parsed["sequence"]) == {"ACDEFGHIKL", "WYVTSRQPNM"}

    ac_mean = parsed.loc[parsed["sequence"] == "ACDEFGHIKL", "ton"].iloc[0]
    assert ac_mean == 11.0


def test_add_split_column_covers_all_rows() -> None:
    sequences = [f"AAAAAAA{i:03d}"[-10:] for i in range(60)]
    ton = np.linspace(1.0, 30.0, num=60)
    df = pd.DataFrame({"sequence": sequences, "ton": ton})

    split_df = add_split_column(df, train_fraction=0.7, val_fraction=0.15, test_fraction=0.15, random_state=7)
    assert len(split_df) == len(df)
    assert set(split_df["split"].unique()) == {"train", "val", "test"}
