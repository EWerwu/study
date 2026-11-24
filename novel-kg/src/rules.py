"""spaCy Chinese character tokenizer, PhraseMatcher entities and Matcher relation triggers."""
import spacy
from spacy.matcher import Matcher, PhraseMatcher
from spacy.util import filter_spans
from common import *

# (predicate, trigger direction relative to surface order, left types, right types, phrases).
PAT = [
 ("就读","LR","人物","机构","到|就读于"),("身份","LR","人物","身份","成了|当了|成为|做了|是|从"),
 ("就业机会","LR","人物","身份","获得了当|有机会当"),("毕业于","LR","人物","机构","从|毕业于"),
 ("爱慕","LR","人物","人物","爱慕|对"),("同学","LR","人物","人物","同班同学"),
 ("朋友","LR","人物","人物","朋友|与"),("恋爱","LR","人物","人物","女友|恋爱"),
 ("父母是","RL","人物","人物","女儿|儿子"),("兄姐是","LR","人物","人物","哥哥|姐姐"),
 ("兄姐是","RL","人物","人物","弟弟|妹妹"),("配偶","LR","人物","人物","妻子|丈夫|与|和"),
 ("到访","LR","人物","地点","到|到了|回到了|从"),("任职于","LR","人物","机构,地点","到|到了|在"),
 ("帮助","LR","人物,身份","人物","帮助了|帮助"),("领导","LR","人物","机构","领导|带领"),
 ("召开","LR","人物","事件","召开|主持"),("参与","LR","人物","事件","在|参加|参与"),
 ("抵制","LR","人物","制度","抵制|反对"),("推行","LR","人物,机构","制度","推广|实行"),
 ("经营","LR","人物","地点","经营|的"),("发电报给","LR","人物","人物","给"),
 ("患病","LR","人物","疾病","确诊|患上"),("表白","LR","人物","人物","向"),
 ("拒绝","LR","人物","人物","拒绝"),("救护","LR","人物","身份","抢救|救护"),
 ("青梅竹马","LR","人物","人物","和|与"),("关系恶化","LR","人物","人物","与|和"),
]
GUARDS={"朋友":"友情|朋友","恋爱":"恋爱|女友|男友","青梅竹马":"青梅竹马","关系恶化":"关系.{0,4}(恶劣|恶化)",
        "配偶":"结.{0,2}婚|夫妻|妻子|丈夫","爱慕":"爱慕|情愫|暗恋","就读":"就读|读书|念书",
        "毕业于":"毕业","任职于":"工作|上班|尽力干活|当了记者","发电报给":"电报","表白":"表白",
        "经营":"经营|扩建|发展","参与":"采访|参与|参加"}
def run():
    start=perf_counter();nlp=spacy.blank("zh")
    lex=PhraseMatcher(nlp.vocab,attr="ORTH")
    for typ in TYPES:
        lex.add(typ,[nlp.make_doc(a) for a,n in ALIAS.items() if NAMES[n]==typ])
    rules=Matcher(nlp.vocab)
    for n,(_,_,_,_,phrases) in enumerate(PAT):
        rules.add(str(n),[[{"ORTH":x.text} for x in nlp.make_doc(p)] for p in phrases.split("|")])
    setup=perf_counter()-start;start=perf_counter();pred={}
    for i,s in DOCS.items():
        doc=nlp(s);entities=[]
        for match,a,b in lex(doc):
            entities.append(doc[a:b])
        entities=filter_spans(entities)
        triples=[]
        for idx,a,b in rules(doc):
            p,d,l,r,_=PAT[int(nlp.vocab.strings[idx])]
            x,y=doc[a:b].start_char,doc[a:b].end_char
            before=[e for e in entities if e.end_char<=x and NAMES[canonical(e.text)] in l.split(",") and not re.search(r"[，；。！？]",s[e.end_char:x])]
            after=[e for e in entities if e.start_char>=y and NAMES[canonical(e.text)] in r.split(",") and not re.search(r"[，；。！？]",s[y:e.start_char])]
            if not before or not after:continue
            left,right=before[-1],after[0]
            lo=max(s.rfind(c,0,x) for c in "，；。！？")+1
            stops=[s.find(c,y) for c in "，；。！？" if s.find(c,y)>=0]
            clause=s[lo:min(stops,default=len(s))]
            if p in GUARDS and not re.search(GUARDS[p],clause):continue
            h,t=(left.text,right.text) if d=="LR" else (right.text,left.text)
            if canonical(h)!=canonical(t):triples.append(edge(h,p,t))
        pred[i]=norm_edges(triples)
    result("spacy","re",pred,perf_counter()-start,setup_seconds=setup,version=spacy.__version__)
if __name__=="__main__":run()
