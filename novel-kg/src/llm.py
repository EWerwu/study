"""Three separate zero-shot runs. Store the original local model response and per-call timers."""
import os
import json
import argparse
from urllib.request import Request, urlopen
from time import perf_counter
from common import *

MODEL = os.getenv("KG_LLM", "qwen2.5:7b")
HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
if "://" not in HOST:
    HOST = "http://" + HOST
def prompt_for(task, part):
    doc = "\n".join(i+" "+s for i,s in DOCS.items() if i.startswith(part+"."))
    intro = "你是中文小说信息抽取器。只用以下部内原文，不使用原著知识。输出一个合法JSON对象，不要解释。"
    if task == "ner":
        rule = ("抽取具名人物、文中用称谓唯一指代的人、具体机构、具体地点、身份职业、命名事件、制度、疾病、显式时间。"
                "不要一般代词、年龄和泛指地点。按句号编号，以原文连续词面列出，每词在每句只列一次，重叠优先最长词面。"
                "格式 {句子编号:[[原文词面,类型],...]}，类型只取"+"/".join(TYPES)+"。")
    else:
        vocab = "实体规范名及简称："+ "、".join(r["name"]+("="+r["aliases"] if r["aliases"] else "") for r in LEX)+"。"
        spec = ("关系限于："+ "、".join(PRED)+"。父母是、兄姐是均由孩子或弟妹指向长辈或兄姐。"
                "单方爱慕和双向恋爱区分，就业机会与已任职区分。每条须有对应的句子证据，不能把情节常识当证据。"
                "宾语救护对象若为未具名职业可用身份节点。可用段内指代，不能给无名人物擅补本名。")
        if task == "re":
            rule = vocab+spec+"按句子编号抽取已明确陈述的关系。输出 {句子编号:[[主语规范名,关系,宾语规范名],...]}。"
        else:
            rule = vocab+spec+("直接从本部原文设计一张属性图。输出 {\"nodes\":[规范名,...],"
                               "\"edges\":[[主节点,关系,客节点,句子编号],...]}。只列有证据的边和这些边的端点。")
    return intro+"\n"+rule+"\n"+doc
def ask(task, part):
    prompt = prompt_for(task,part)
    body = dict(model=MODEL,prompt=prompt,stream=False,format="json",
                options=dict(temperature=0,seed=42,num_ctx=6144,num_predict=3200))
    req = Request(HOST+"/api/generate",data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
    start = perf_counter()
    with urlopen(req,timeout=1500) as r:
        response = json.load(r)
    elapsed = perf_counter()-start
    record = dict(model=MODEL,task=task,part=part,prompt=prompt,response=response["response"],seconds=elapsed,
                  done_reason=response.get("done_reason"),eval_count=response.get("eval_count"),
                  load_duration_ns=response.get("load_duration"),eval_duration_ns=response.get("eval_duration"))
    return record
def run(task, fresh):
    pred = {i:[] for i in DOCS}
    rejected, declared, times = [], [], []
    for part in ("1","2","3"):
        trace = RAW / f"qwen-{task}-{part}.json"
        if not fresh and trace.exists():
            record=load(trace)
            if record["prompt"]!=prompt_for(task,part) or record["model"]!=MODEL:
                raise ValueError("模型或提示已变，请使用 --fresh")
        else:
            record=ask(task,part)
            save(trace,record)
        # Sentence IDs such as 1.01 may arrive as JSON numbers. Preserve their literal spelling.
        obj=json.loads(record["response"],parse_float=str)
        times.append(record["seconds"])
        def add(i, v):
            if i not in DOCS or not i.startswith(part+".") or not isinstance(v,list):
                rejected.append([i,v]);return
            if task=="ner":
                if len(v)==2 and all(isinstance(a,str) for a in v) and v[0] in DOCS[i] and v[1] in TYPES:
                    pred[i].append(v)
                else: rejected.append([i,v])
            elif len(v)==3 and all(isinstance(a,str) for a in v) and v[1] in PRED and canonical(v[0]) in NAMES and canonical(v[2]) in NAMES:
                pred[i].append(edge(*v))
            else: rejected.append([i,v])
        if task=="graph":
            if not isinstance(obj.get("nodes"),list) or not isinstance(obj.get("edges"),list):
                raise ValueError("LLM图结构不完整："+str(trace))
            declared.extend([canonical(n) for n in obj["nodes"] if isinstance(n,str) and canonical(n) in NAMES])
            for row in obj["edges"]:
                if isinstance(row,list) and len(row)==4 and isinstance(row[-1],str):
                    add(row[-1],row[:3])
                else: rejected.append([None,row])
        else:
            for i, vv in obj.items():
                if isinstance(vv,list):
                    for v in vv: add(i,v)
                else: rejected.append([i,vv])
        print(task,part,round(times[-1],2),flush=True)
    for i,values in pred.items():
        pred[i]=find_spans(DOCS[i],values) if task=="ner" else norm_edges(values)
    result("qwen",task,pred,sum(times),model=MODEL,rejected=rejected,declared_nodes=sorted(set(declared)),part_seconds=times,validation_version=2)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--task",choices=["all","ner","re","graph"],default="all")
    p.add_argument("--fresh",action="store_true")
    args=p.parse_args()
    for task in (["ner","re","graph"] if args.task=="all" else [args.task]):
        run(task,args.fresh)
