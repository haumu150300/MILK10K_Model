"""Create MILK10k submission CSV template.

Output columns:
lesion_id,AKIEC,BCC,BEN_OTH,BKL,DF,INF,MAL_OTH,MEL,NV,SCCKA,VASC
"""

from __future__ import annotations

import pandas as pd


LABEL_COLUMNS = [
	"AKIEC",
	"BCC",
	"BEN_OTH",
	"BKL",
	"DF",
	"INF",
	"MAL_OTH",
	"MEL",
	"NV",
	"SCCKA",
	"VASC",
]


def build_submit_template(
	test_metadata_path: str,
	output_path: str,
) -> None:
	test_df = pd.read_csv(test_metadata_path, usecols=["lesion_id"])
	lesion_ids = test_df["lesion_id"].drop_duplicates().sort_values()

	submit_df = pd.DataFrame({"lesion_id": lesion_ids})
	for col in LABEL_COLUMNS:
		submit_df[col] = 0.0

	submit_df.to_csv(output_path, index=False)


if __name__ == "__main__":
	build_submit_template(
		"MILK10k_Test_Metadata.csv",
		"MILK10k_Submit.csv",
	)
