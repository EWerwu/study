"""Create three scoped property graphs; benchmark actual Bolt transactions and read them back."""
import argparse
import hashlib
import os
import statistics
from neo4j import GraphDatabase
from common import *
from reference import reference

def records(pred):
    return [dict(h=h,p=p,t=t,sid=i) for i,edges in pred.items() for h,p,t in edges]
def source_graph(method):
    if method=="reference":return reference()[1], []
    if method=="spacy":return load(RAW/"spacy_re.json")["predictions"], []
    run=load(RAW/"qwen_graph.json")
    if run.get("validation_version",1)<2:
        from llm import run as parse_traces
        parse_traces("graph",False)
        run=load(RAW/"qwen_graph.json")
    return run["predictions"],run["declared_nodes"]
def build(driver,method,db):
    pred,declared=source_graph(method)
    edges=records(pred)
    names=sorted(set(declared)|{n for v in edges for n in (v["h"],v["t"])})
    nodes=[dict(name=n,kind=NAMES[n]) for n in names]
    # A new dataset version gets its own scope. Repeated identical imports are idempotent.
    digest=hashlib.sha256(json.dumps([nodes,edges],sort_keys=True).encode()).hexdigest()[:16]
    scope="novel-course-"+method+"-"+digest
    with driver.session(database=db) as ses:
        ses.run("CREATE CONSTRAINT novel_course_entity IF NOT EXISTS FOR (n:NovelCourseEntity) REQUIRE (n.scope,n.name) IS UNIQUE").consume()
        existing=ses.run("MATCH (n:NovelCourseEntity {scope:$s}) RETURN count(n) AS c",s=scope).single()["c"]
        def write(tx):
            tx.run("UNWIND $rows AS v MERGE (n:NovelCourseEntity {scope:$s,name:v.name}) SET n.kind=v.kind",s=scope,rows=nodes).consume()
            for p in sorted({v['p'] for v in edges}):
                assert p in PRED
                cypher=("UNWIND $rows AS v MATCH (a:NovelCourseEntity {scope:$s,name:v.h}),"
                        "(b:NovelCourseEntity {scope:$s,name:v.t}) MERGE (a)-[r:`"+p+"` {sid:v.sid}]->(b) SET r.evidence=v.evidence")
                tx.run(cypher,s=scope,rows=[{**v,"evidence":DOCS[v["sid"]]} for v in edges if v["p"]==p]).consume()
        ts=perf_counter();ses.execute_write(write);write_s=perf_counter()-ts
        q=("MATCH (a:NovelCourseEntity {scope:$s})-[r]->(b:NovelCourseEntity {scope:$s}) "
           "RETURN a.name AS h,type(r) AS p,b.name AS t,r.sid AS sid ORDER BY sid,h,p,t")
        ts=perf_counter();back=[r.data() for r in ses.run(q,s=scope)];read_s=perf_counter()-ts
        actual=sorted(r["n"] for r in ses.run("MATCH (a:NovelCourseEntity {scope:$s}) RETURN a.name AS n",s=scope))
        key=lambda v:(v["sid"],v["h"],v["p"],v["t"])
        assert sorted(edges,key=key)==sorted(back,key=key) and actual==names
        times=[]
        lookup=("MATCH (a:NovelCourseEntity {scope:$s,name:$name})-[r]-(b:NovelCourseEntity {scope:$s}) "
                "RETURN DISTINCT b.name AS name ORDER BY name")
        for k in range(21):
            ts=perf_counter();neighbours=[r['name'] for r in ses.run(lookup,s=scope,name="孙少平")];elapsed=perf_counter()-ts
            if k: times.append(elapsed)
        info=ses.run("CALL dbms.components() YIELD versions RETURN versions[0] AS v").single()["v"]
        data=dict(scope=scope,version=info,existing_nodes=existing,nodes=actual,edges=back,write_seconds=write_s,
                  read_seconds=read_s,lookup_median_ms=statistics.median(times)*1000,lookup_count=20,neighbours=neighbours)
        save(RAW/f"neo4j_{method}.json",data)
        print(method,len(nodes),len(edges),"verified",flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--source",choices=["all","reference","spacy","qwen"],default="all");args=p.parse_args()
    secret=os.environ.get("NEO4J_PASSWORD")
    auth=(os.getenv("NEO4J_USER","neo4j"),secret) if secret else None
    with GraphDatabase.driver(os.getenv("NEO4J_URI","bolt://127.0.0.1:7687"),auth=auth) as driver:
        driver.verify_connectivity()
        for m in (["reference","spacy","qwen"] if args.source=="all" else [args.source]):
            build(driver,m,os.getenv("NEO4J_DATABASE","neo4j"))
