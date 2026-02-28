import argparse
import os
import subprocess
import pandas as pd

def run(cmd: list[str]) -> None:
    print(f"Running: {' '.join(cmd)}")
    subprocess.check_call(cmd)

def upload_partitioned(parquet_path: str, s3_prefix: str, tmpdir: str) -> None:
    df = pd.read_parquet(parquet_path)
    if "dt" not in df.columns:
        raise ValueError(f"{parquet_path} missing required partition column: dt")
    
    os.makedirs(tmpdir, exist_ok=True)
    for dt, part in df.groupby("dt"):
        out_dir = os.path.join(tmpdir, f"dt={dt}")
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, "part-00000.parquet")
        part_no_dt = part.drop(columns=["dt"])
        part_no_dt.to_parquet(out_file, index=False)
        run(["aws", "s3", "cp", out_file, f"{s3_prefix}/dt={dt}/part-00000.parquet"])

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--transactions", required = True)
    p.add_argument("--chargebacks", required = True)
    p.add_argument("--s3-bucket", required=True, help="s3://fraud-ml")
    p.add_argument("--tmpdir", default=".tmp_upload")
    args = p.parse_args()

    tx_prefix = f"{args.s3_bucket}/lake/transactions"
    cb_prefix = f"{args.s3_bucket}/lake/chargebacks"

    upload_partitioned(args.transactions, tx_prefix, os.path.join(args.tmpdir, "transactions"))
    upload_partitioned(args.chargebacks, cb_prefix, os.path.join(args.tmpdir, "chargebacks"))

    print("Upload complete.")

if __name__ == "__main__":
    main()