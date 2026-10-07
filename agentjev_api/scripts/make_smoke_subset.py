"""从 prepared/ 按案例抽一个冒烟测试用的小子集，输出到 prepared_smoke/。

抽样单位是案例（case_id），同一案例的全部问题一起进出。案例 id 先排序
再用固定种子抽，结果与文件行序无关。选中案例的 jsonl 行原样复制，
文件名和字段格式与 prepared/ 完全一致。

用法：
  python scripts/make_smoke_subset.py \
      [--train 40 --dev 10 --calibration 10 --test 20 --seed 20260921]
"""
import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "dev", "calibration", "test")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=str(ROOT / "prepared"))
    parser.add_argument("--dest", default=str(ROOT / "prepared_smoke"))
    parser.add_argument("--seed", type=int, default=20260921)
    for name, n in zip(SPLITS, (40, 10, 10, 20)):
        parser.add_argument(f"--{name}", type=int, default=n, help=f"{name} 案例数")
    args = parser.parse_args()

    src, dest = Path(args.src), Path(args.dest)
    dest.mkdir(exist_ok=True)
    summary = {}
    for name in SPLITS:
        req_lines = (src / f"{name}_requests.jsonl").read_text(encoding="utf-8").splitlines()
        q_lines = (src / f"{name}_questions.jsonl").read_text(encoding="utf-8").splitlines()
        case_ids = sorted(json.loads(line)["id"] for line in req_lines)
        picked = set(random.Random(f"{args.seed}:{name}").sample(case_ids, getattr(args, name)))

        reqs = [line for line in req_lines if json.loads(line)["id"] in picked]
        qs = [line for line in q_lines if json.loads(line)["case_id"] in picked]
        assert len(reqs) == len(picked)
        assert {json.loads(line)["case_id"] for line in qs} == picked
        for suffix, lines in (("requests", reqs), ("questions", qs)):
            (dest / f"{name}_{suffix}.jsonl").write_text(
                "".join(line + "\n" for line in lines), encoding="utf-8"
            )
        summary[name] = {"cases": len(reqs), "questions": len(qs), "case_ids": sorted(picked)}

    (dest / "smoke_manifest.json").write_text(
        json.dumps({"seed": args.seed, "source": str(src), "splits": summary}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({k: {x: v[x] for x in ("cases", "questions")} for k, v in summary.items()}))


if __name__ == "__main__":
    main()
