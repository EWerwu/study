from common import *

def reference():
    ner, triples = {}, {}
    for r in rows(DATA/"reference.tsv"):
        i=r["id"];s=DOCS[i]
        surfaces=(r["surfaces"] or "").split("|")
        for v in filter(None,surfaces):
            assert v in ALIAS and v in s,(i,v)
        ner[i]=find_spans(s,[(v,NAMES[canonical(v)]) for v in surfaces if v])
        facts=[]
        for row in filter(None,(r["triples"] or "").split(";")):
            h,p,t=row.split("/")
            assert h in NAMES and t in NAMES and p in PRED,(i,row)
            facts.append(edge(h,p,t))
        triples[i]=norm_edges(facts)
    assert list(ner)==list(DOCS)
    return ner,triples
