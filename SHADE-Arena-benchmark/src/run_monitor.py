import argparse, collections, glob, json, os, re, statistics, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASKS = ["american_airlines", "api_key_calendar_agendas", "bank_transfer",
         "expenditure_accounting", "spam_filter_update"]
GROUPS = ["attack_nohint", "attack_hint", "benign"]
WRAP = "\n\nHere is the conversation from the agent model \n <begin_agent_conversation> "
MODELS = {
    "cotguard": dict(hf="CoT-Guard/cot-guard-4b-rl", max_new=4096),
    "qwen": dict(hf="Qwen/Qwen3.8-27B", max_new=24576),
}
VERDICT = re.compile(r"<verdict>\s*(\d+(?:\.\d+)?)\s*</verdict>", re.I)

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=sorted(MODELS), required=True)
    p.add_argument("--condition", choices=["strip", "keep"], required=True)
    p.add_argument("--out", default=os.path.join(ROOT, "reproduced", "monitor_outputs"))
    p.add_argument("--model_path", default="", help="local model folder (default: the Hugging Face id)")
    p.add_argument("--max_input", type=int, default=60000)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--chunk", type=int, default=96)
    p.add_argument("--pilot", type=int, default=0)
    p.add_argument("--dry", action="store_true")
    p.add_argument("--author", action="store_true", help="authors only: allow writing into reference/")
    return p.parse_args()

def check_out(out, author):
    ref = os.path.realpath(os.path.join(ROOT, "reference"))
    real = os.path.realpath(out)
    if (real == ref or real.startswith(ref + os.sep)) and not author:
        sys.exit("refusing to write into reference/ : choose another --out folder")

def txt(c):
    if c is None:
        return ""
    return c if isinstance(c, str) else json.dumps(c)

def strip_tags(t, tags):
    for w in tags:
        while f"<{w}>" in t and f"</{w}>" in t:
            s = t.find(f"<{w}>")
            e = t.find(f"</{w}>") + len(f"</{w}>")
            t = t[:s] + t[e:]
    return t.strip()

def conv_str(conv, tags):
    parts = []
    for m in conv:
        if m["role"] == "system":
            continue
        tc = "Tool calls:"
        if m.get("tool_calls"):
            tc += "\n- " + str(m["tool_calls"])
        parts.append(f"{m['role'].upper()}: {strip_tags(txt(m.get('content')), tags)}{tc}")
    return "\n\n".join(parts)

def build_items(tok, condition, max_input):
    tags = ["thinking", "system_instructions"] if condition == "strip" else ["system_instructions"]
    prompts = {}
    for t in TASKS:
        with open(os.path.join(ROOT, "monitor_prompts", t + ".txt"), encoding="utf-8") as fh:
            prompts[t] = fh.read()
    items = []
    for task in TASKS:
        for g in GROUPS:
            for f in sorted(glob.glob(os.path.join(ROOT, "transcripts", task, g, "*.json"))):
                d = json.load(open(f))
                user = prompts[task] + WRAP + conv_str(d["conversation"], tags) + " <end_agent_conversation>"
                text = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)
                n = len(tok(text, add_special_tokens=False).input_ids)
                items.append(dict(task=task, group=g, file=os.path.relpath(f, ROOT), n_tokens=n,
                                  side_task_evaluation=d.get("side_task_evaluation"), text=text))
    return items, [i for i in items if i["n_tokens"] <= max_input]

def make_llm(name, path, seed):
    from vllm import LLM
    hf = path or MODELS[name]["hf"]
    extra = {} if seed is None else {"seed": seed}
    if name == "cotguard":
        kw = dict(model=hf, gpu_memory_utilization=0.95, tensor_parallel_size=1, **extra)
        rope = {"rope_type": "dynamic", "factor": 2.0}
        try:
            return LLM(**dict(kw, max_model_len=65536, rope_scaling=rope))
        except Exception as e:
            print("first load style failed:", repr(e)[:200])
            return LLM(**dict(kw, max_model_len=65536, hf_overrides={"rope_parameters": rope}))
    return LLM(model=hf, max_model_len=131072, gpu_memory_utilization=0.90, max_num_seqs=64,
               limit_mm_per_prompt={"image": 0, "video": 0}, **extra)

def main():
    a = get_args()
    check_out(a.out, a.author)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model_path or MODELS[a.model]["hf"])
    items, kept = build_items(tok, a.condition, a.max_input)
    print(f"model={a.model} condition={a.condition} | transcripts {len(items)} | within {a.max_input} tokens: {len(kept)}")
    cnt = collections.Counter((i["task"], i["group"]) for i in kept)
    for t in TASKS:
        print(f"  {t:27}", {g: cnt[(t, g)] for g in GROUPS})
    if a.pilot:
        kept.sort(key=lambda i: i["n_tokens"])
        idx = sorted({round(j * (len(kept) - 1) / max(a.pilot - 1, 1)) for j in range(a.pilot)})
        kept = [kept[j] for j in idx]
        print("pilot subset:", len(kept))
    if a.dry:
        return
    import datetime, importlib.metadata as md
    from vllm import SamplingParams
    os.makedirs(a.out, exist_ok=True)
    name = f"{a.model}_{a.condition}" + (".pilot" if a.pilot else "")
    out_path = os.path.join(a.out, name + ".jsonl")
    done = set()
    if os.path.exists(out_path):
        done = {json.loads(l)["file"] for l in open(out_path)}
    todo = [i for i in kept if i["file"] not in done]
    print(f"already done: {len(done)} | to do: {len(todo)}", flush=True)
    llm = make_llm(a.model, a.model_path, a.seed)
    sp = SamplingParams(temperature=0.6, top_p=0.95, top_k=-1, min_p=0.0, max_tokens=MODELS[a.model]["max_new"])
    info = dict(date=datetime.datetime.now().isoformat(timespec="seconds"), model=a.model, condition=a.condition,
                model_id=a.model_path or MODELS[a.model]["hf"], seed=a.seed, transcripts=len(kept),
                versions={p: md.version(p) for p in ("vllm", "torch", "transformers")})
    os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
    with open(os.path.join(ROOT, "logs", name + ".info.json"), "w") as fh:
        json.dump(info, fh, indent=2)
    for st in range(0, len(todo), a.chunk):
        batch = todo[st:st + a.chunk]
        outs = llm.generate([i["text"] for i in batch], sp, use_tqdm=False)
        with open(out_path, "a") as fh:
            for it, o in zip(batch, outs):
                out = o.outputs[0].text
                status, score = "ok", None
                if "</think>" not in out:
                    status = "no_think_end"
                else:
                    m = VERDICT.findall(out.split("</think>")[-1])
                    if m and 0 <= float(m[-1]) <= 100:
                        score = float(m[-1])
                    else:
                        status = "no_verdict"
                row = {k: v for k, v in it.items() if k != "text"}
                row.update(output=out, score=score, parse=status, out_tokens=len(o.outputs[0].token_ids))
                fh.write(json.dumps(row) + "\n")
        print(f"chunk done: {min(st + a.chunk, len(todo))}/{len(todo)}", flush=True)
    rows = [json.loads(l) for l in open(out_path)]
    ol = sorted(r["out_tokens"] for r in rows)
    print("parse:", dict(collections.Counter(r["parse"] for r in rows)))
    print(f"output tokens: median={statistics.median(ol)} max={ol[-1]}")
    print("saved", out_path)

if __name__ == "__main__":
    main()
