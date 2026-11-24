from pathlib import Path
import csv
import json
import re
import os
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "reports"
RAW = OUT / "runs"
RAW.mkdir(parents=True, exist_ok=True)

def rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))

DOCS = {r["id"]: r["text"] for r in rows(os.getenv("KG_PASSAGES", DATA / "passages.tsv"))}
LEX = rows(DATA / "dictionary.tsv")
NAMES = {r["name"]: r["type"] for r in LEX}
ALIAS = {a: r["name"] for r in LEX for a in [r["name"], *(r["aliases"] or "").split("|")] if a}
TYPES = list(dict.fromkeys(NAMES.values()))
# Symmetric edges sort endpoints lexicographically; directed labels read from subject to object.
SYMMETRIC = {"朋友", "青梅竹马", "恋爱", "配偶", "关系恶化", "同学"}
PRED = ["就读", "毕业于", "身份", "爱慕", "同学", "关系恶化", "恋爱", "朋友", "帮助", "父母是", "兄姐是", "青梅竹马", "配偶", "到访", "领导", "抵制", "推行", "召开", "就业机会", "任职于", "经营", "参与", "发电报给", "患病", "表白", "拒绝", "救护"]
def canonical(s):
    return ALIAS.get(s, s)
def edge(s, p, o):
    s, o = canonical(s), canonical(o)
    if p in SYMMETRIC and s > o:
        s, o = o, s
    return [s, p, o]
def norm_edges(triples):
    return [list(t) for t in sorted({tuple(edge(*v)) for v in triples})]
def find_spans(text, terms):
    candidates = []
    for surface, typ in terms:
        if not surface or typ not in TYPES:
            continue
        for m in re.finditer(re.escape(surface), text):
            candidates.append((m.start(), m.end(), typ))
    result = []
    occupied = set()
    for a,b,t in sorted(set(candidates), key=lambda v:(v[0]-v[1],v[0],v[2])):
        if not occupied.intersection(range(a,b)):
            occupied.update(range(a,b))
            result.append([a,b,t])
    return sorted(result)
def mentions(text):
    return find_spans(text, [(a,NAMES[n]) for a,n in ALIAS.items()])
def save(path, data):
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding="utf-8")
def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
def result(method, task, predictions, seconds, **meta):
    save(RAW / (method+"_"+task+".json"), dict(method=method,task=task,seconds=seconds,predictions=predictions,**meta))
