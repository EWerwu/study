"""Character-level BERT tagger, BiLSTM with a trainable linear-chain CRF, BERT relation classifier."""
import argparse
import os
import random
from time import perf_counter
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from common import *
from training import synth, PATTERNS

TAGS=["O"]+[p+"-"+t for t in TYPES for p in ("B","I")]
R=["无关系"]+PRED
def bio(size,spans):
    tags=[0]*size
    for a,b,t,*_ in spans:
        tags[a]=TAGS.index("B-"+t)
        tags[a+1:b]=[TAGS.index("I-"+t)]*(b-a-1)
    return tags
def decode(ids):
    out=[]
    for i,k in enumerate(ids):
        tag=TAGS[k]
        if tag=="O": continue
        if tag.startswith("I-") and out and out[-1][1]==i and out[-1][2]==tag[2:]:
            out[-1][1]+=1
        else: out.append([i,i+1,tag[2:]])
    return out
class CRF(nn.Module):
    def __init__(self,k):
        super().__init__()
        self.trans=nn.Parameter(torch.zeros(k,k));self.start=nn.Parameter(torch.zeros(k));self.end=nn.Parameter(torch.zeros(k))
    def loss(self,e,y,mask):
        logits=self.start+e[:,0]
        score=self.start[y[:,0]]+e[:,0].gather(1,y[:,:1]).flatten()
        for i in range(1,e.size(1)):
            scores=logits.unsqueeze(2)+self.trans+e[:,i].unsqueeze(1)
            logits=torch.where(mask[:,i,None],scores.logsumexp(1),logits)
            gold=self.trans[y[:,i-1],y[:,i]]+e[:,i].gather(1,y[:,i,None]).flatten()
            score+=gold*mask[:,i]
        last=y.gather(1,mask.sum(1,keepdim=True)-1).flatten()
        return ((logits+self.end).logsumexp(1)-score-self.end[last]).mean()
    def best(self,e):
        d=self.start+e[0];back=[]
        for step in e[1:]:
            d,idx=(d[:,None]+self.trans).max(0);back.append(idx);d=d+step
        path=[int((d+self.end).argmax())]
        for idx in reversed(back): path.append(int(idx[path[-1]]))
        return path[::-1]
class BiTag(nn.Module):
    def __init__(self,size):
        super().__init__()
        self.emb=nn.Embedding(size,64,padding_idx=0)
        self.enc=nn.LSTM(64,72,bidirectional=True,batch_first=True)
        self.head=nn.Linear(144,len(TAGS));self.crf=CRF(len(TAGS))
    def forward(self,x,lens):
        packed=pack_padded_sequence(self.emb(x),lens.cpu(),batch_first=True,enforce_sorted=False)
        out,_=self.enc(packed);out,_=pad_packed_sequence(out,batch_first=True)
        return self.head(out)
def tensor_rows(rows,pad=0):
    return nn.utils.rnn.pad_sequence([torch.tensor(x) for x in rows],batch_first=True,padding_value=pad).to(DEVICE)
def lengths(rows):return torch.tensor([len(x) for x in rows],device=DEVICE)
def train_bi(train):
    vocab={c:i+2 for i,c in enumerate(sorted(set("".join(x[0] for x in train))))}
    encode=lambda text:[vocab.get(c,1) for c in text]
    model=BiTag(len(vocab)+2).to(DEVICE)
    opt=torch.optim.Adam(model.parameters(),lr=.006)
    ts=perf_counter()
    for _ in range(12):
        random.shuffle(train)
        for j in range(0,len(train),48):
            batch=train[j:j+48]; enc=[encode(x[0]) for x in batch]
            arr=tensor_rows(enc); y=tensor_rows([bio(len(s),v) for s,v,_ in batch])
            logits=model(arr,lengths(enc));loss=model.crf.loss(logits,y,arr!=0)
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
    fitted=perf_counter()-ts;model.eval();ts=perf_counter();out={}
    with torch.no_grad():
        for i,s in DOCS.items():
            x=[encode(s)];out[i]=decode(model.crf.best(model(tensor_rows(x),lengths(x))[0]))
    result("bilstm","ner",out,perf_counter()-ts,training_seconds=fitted,epochs=12,examples=len(train),device=str(DEVICE))
def train_bert(train,checkpoint):
    from transformers import BertTokenizer, BertModel
    ts=perf_counter()
    tokenizer=BertTokenizer.from_pretrained(checkpoint)
    enc=BertModel.from_pretrained(checkpoint,add_pooling_layer=False).to(DEVICE)
    # One token per original character guarantees alignment, including digits and punctuation.
    def ids(s):return [tokenizer.cls_token_id]+tokenizer.convert_tokens_to_ids(list(s))+[tokenizer.sep_token_id]
    size=enc.config.hidden_size
    tag=nn.Linear(size,len(TAGS)).to(DEVICE)
    rel=nn.Sequential(nn.Linear(3*size,192),nn.GELU(),nn.Linear(192,len(R))).to(DEVICE)
    opt=torch.optim.AdamW([{"params":enc.parameters(),"lr":3e-5},{"params":tag.parameters(),"lr":.001},{"params":rel.parameters(),"lr":.001}])
    def features(h,a,b,c,d):
        return torch.cat([h[0],h[1+a:1+b].mean(0),h[1+c:1+d].mean(0)])
    load_s=perf_counter()-ts;ts=perf_counter()
    for epoch in range(4):
        random.shuffle(train)
        for j in range(0,len(train),24):
            batch=train[j:j+24];x=tensor_rows([ids(s) for s,_,_ in batch])
            h=enc(x,attention_mask=x.ne(0)).last_hidden_state
            y=tensor_rows([[-100]+bio(len(s),v)+[-100] for s,v,_ in batch],-100)
            pairs=[];labels=[]
            for h0,(_,v,p) in zip(h,batch):
                a,b,_,_=next(k for k in v if k[3]=="s");c,d,_,_=next(k for k in v if k[3]=="o")
                pairs.extend([features(h0,a,b,c,d),features(h0,c,d,a,b)])
                labels.extend([R.index(p),R.index(p) if p in SYMMETRIC else 0])
            loss=nn.functional.cross_entropy(tag(h).flatten(0,1),y.flatten())+nn.functional.cross_entropy(rel(torch.stack(pairs)),torch.tensor(labels,device=DEVICE))
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(enc.parameters(),1.0);opt.step()
        print("BERT epoch",epoch+1,flush=True)
    fit=perf_counter()-ts;enc.eval();tag.eval();rel.eval()
    meta=dict(checkpoint="hfl/rbt3",epochs=4,examples=len(train),device=str(DEVICE),training_seconds=fit,load_seconds=load_s)
    with torch.no_grad():
        ts=perf_counter();ner={}
        for i,s in DOCS.items():
            h=enc(tensor_rows([ids(s)])).last_hidden_state[0]
            ner[i]=decode(tag(h[1:-1]).argmax(-1).tolist())
        result("bert","ner",ner,perf_counter()-ts,**meta)
        ts=perf_counter();rels={}
        for i,s in DOCS.items():
            triples=[]
            for m in re.finditer(r"[^，；。！？]+",s):
                clause=m.group();v=mentions(clause)
                if len(v)<2:continue
                h=enc(tensor_rows([ids(clause)])).last_hidden_state[0]
                for a,b,t in v:
                    for c,d,u in v:
                        if a==c or canonical(clause[a:b])==canonical(clause[c:d]):continue
                        scores=rel(features(h,a,b,c,d)).softmax(0)
                        pred=R[int(scores.argmax())]
                        if pred=="无关系" or float(scores.max())<.50:continue
                        ok=(t,u)==PATTERNS[pred][:2] or (pred,t,u) in {("帮助","身份","人物"),("任职于","人物","地点"),("推行","机构","制度")}
                        if ok:triples.append(edge(clause[a:b],pred,clause[c:d]))
            rels[i]=norm_edges(triples)
        result("bert","re",rels,perf_counter()-ts,**meta)
if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--method",choices=["bert","bilstm","all"],default="all")
    args=parser.parse_args()
    DEVICE=torch.device(os.getenv("KG_DEVICE","cpu"));torch.set_num_threads(int(os.getenv("KG_THREADS","4")))
    for name in (["bilstm","bert"] if args.method=="all" else [args.method]):
        torch.manual_seed(42);random.seed(42)
        train=synth()
        if name=="bilstm":train_bi(train)
        else:train_bert(train,os.getenv("KG_BERT","hfl/rbt3"))
