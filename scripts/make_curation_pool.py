#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, random, argparse, datetime
from collections import defaultdict

def read_jsonl(p):
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line:
                yield json.loads(line)

def write_jsonl(p, rows):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def norm(x, fallback="unknown"):
    x = (x or "").strip().lower()
    return x if x else fallback

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/4_instruction/train_multiturn.jsonl")
    ap.add_argument("--out", default="data/6_curation/curation_pool_1000.jsonl")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260225)

    ap.add_argument("--langs", default="en,si,ta", help="langs to include")
    ap.add_argument("--styles", default="plain,singlish,tamilish,mixed", help="styles to include")

    ap.add_argument("--balance_by_lang", action="store_true")
    ap.add_argument("--balance_by_style", action="store_true")
    ap.add_argument("--balance_by_task", action="store_true",
                    help="optional: balance tasks within each (lang,style) bucket")

    args = ap.parse_args()
    rng = random.Random(args.seed)

    allowed_langs = {norm(x) for x in args.langs.split(",") if x.strip()}
    allowed_styles = {norm(x) for x in args.styles.split(",") if x.strip()}

    # Load + filter
    rows = []
    for r in read_jsonl(args.src):
        lang = norm(r.get("lang"))
        style = norm(r.get("style"))
        if lang in allowed_langs and style in allowed_styles:
            rows.append(r)

    if not rows:
        raise SystemExit("No rows found after lang/style filter.")

    target_n = min(args.n, len(rows))
    today = datetime.date.today().isoformat()

    # Bucket by (lang, style)
    buckets = defaultdict(list)
    for r in rows:
        key = (norm(r.get("lang")), norm(r.get("style")))
        buckets[key].append(r)

    langs_present = sorted({k[0] for k in buckets.keys()})
    styles_present = sorted({k[1] for k in buckets.keys()})

    # Decide which buckets are eligible
    eligible_keys = [k for k,v in buckets.items() if v]

    # Helper: sample from a bucket, optionally balanced by task
    def sample_from_bucket(bucket_rows, k):
        if not args.balance_by_task:
            tmp = bucket_rows[:]
            rng.shuffle(tmp)
            return tmp[:min(k, len(tmp))]

        by_task = defaultdict(list)
        for r in bucket_rows:
            t = norm(r.get("task"))
            by_task[t].append(r)

        tasks = list(by_task.keys())
        rng.shuffle(tasks)
        per_task_lists = {t: by_task[t][:] for t in tasks}
        for t in tasks:
            rng.shuffle(per_task_lists[t])

        picked = []
        while len(picked) < k:
            progressed = False
            for t in tasks:
                if per_task_lists[t]:
                    picked.append(per_task_lists[t].pop())
                    progressed = True
                    if len(picked) >= k:
                        break
            if not progressed:
                break
        return picked

    # Compute target per bucket
    # If balancing by lang/style, we try to spread evenly across buckets.
    final_sample = []

    if args.balance_by_lang or args.balance_by_style:
        # Start with uniform allocation across eligible buckets
        base = target_n // len(eligible_keys)
        rem = target_n % len(eligible_keys)

        rng.shuffle(eligible_keys)
        per_bucket = {k: base for k in eligible_keys}
        for i in range(rem):
            per_bucket[eligible_keys[i]] += 1

        # Sample per bucket
        for k, quota in per_bucket.items():
            final_sample.extend(sample_from_bucket(buckets[k], quota))

        # If underfilled due to small buckets, top up from leftovers
        if len(final_sample) < target_n:
            picked_ids = {r.get("id") for r in final_sample}
            leftovers = [r for r in rows if r.get("id") not in picked_ids]
            rng.shuffle(leftovers)
            final_sample.extend(leftovers[:(target_n - len(final_sample))])

        rng.shuffle(final_sample)

    else:
        # Plain random overall
        tmp = rows[:]
        rng.shuffle(tmp)
        final_sample = tmp[:target_n]

    # Add curation fields
    out_rows = []
    for r in final_sample[:target_n]:
        r = dict(r)
        rid = str(r.get("id","")).strip()
        r["id"] = f"cur_{rid}" if rid else None
        r["curation"] = {
            "status": "needs_review",
            "curated_by": "Charunya",
            "curated_at": today,
        }
        r["evidence"] = []
        r["notes"] = ""
        out_rows.append(r)

    write_jsonl(args.out, out_rows)

    # Print summary stats
    c_lang = defaultdict(int)
    c_style = defaultdict(int)
    c_task = defaultdict(int)
    for r in out_rows:
        c_lang[norm(r.get("lang"))] += 1
        c_style[norm(r.get("style"))] += 1
        c_task[norm(r.get("task"))] += 1

    print(f"Wrote {len(out_rows)} to {args.out}")
    print("Lang counts:", dict(sorted(c_lang.items())))
    print("Style counts:", dict(sorted(c_style.items())))
    print("Task counts (top 10):", dict(sorted(c_task.items(), key=lambda x: -x[1])[:10]))

if __name__ == "__main__":
    main()