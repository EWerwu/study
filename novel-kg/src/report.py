"""Generate Markdown tables and the offline graph exclusively from recorded predictions."""
from collections import Counter
from common import *
from reference import reference

N_LABEL={"bert":"中文 BERT（RBT3）","bilstm":"BiLSTM-CRF","qwen":"本地 Qwen2.5 7B","spacy":"spaCy 中文规则"}
def keys(pred):return {(i,*row) for i,vs in pred.items() for row in vs}
def metrics(actual,target):
    a,b=len(actual),len(target); hit=len(actual&target)
    pr=hit/a if a else 0;rec=hit/b if b else 0
    return (a,b,hit,pr,rec,2*hit/(a+b) if a+b else 0)
def mt(m):
    a,b,h,p,r,f=m
    return [a,b,h,f"{p:.1%}",f"{r:.1%}",f"{f:.1%}"]
def table(head,rows):
    def line(r):return "| "+" | ".join(str(x).replace("|","\\|").replace("\n"," ") for x in r)+" |"
    return "\n".join([line(head),line(["---"]*len(head)),*[line(r) for r in rows]])
def md(name,lines):
    (OUT/name).write_text("\n\n".join(lines)+"\n",encoding="utf-8")
def display(v,task):
    if task=="ner":
        i,a,b,t=v
        return f"{i}「{DOCS[i][a:b]}」{t} [{a},{b})"
    i,h,p,t=v
    return f"{i}（{h}，{p}，{t}）"
def projected(e):
    return {(v["sid"],v["h"],v["p"],v["t"]) for v in e}
def compare(task,methods,gold,heading):
    goal=keys(gold);runs={m:load(RAW/f"{m}_{task}.json") for m in methods}
    stats={m:metrics(keys(v["predictions"]),goal) for m,v in runs.items()}
    content=[heading,table(["方法","预测","参考","命中","精确率","召回率","F1","三部推理耗时/s","训练耗时/s"],
             [[N_LABEL[m],*mt(stats[m]),f'{v["seconds"]:.3f}',f'{v["training_seconds"]:.2f}' if "training_seconds" in v else "—"] for m,v in runs.items()])]
    content.append(table(["方法","第一部 F1","第二部 F1","第三部 F1"],[
        [N_LABEL[m],*[f'{metrics({k for k in keys(v["predictions"]) if k[0].startswith(p)}, {k for k in goal if k[0].startswith(p)})[-1]:.1%}' for p in ("1.","2.","3.")]] for m,v in runs.items()]))
    for m,v in runs.items():
        got=keys(v["predictions"]);fp=sorted(got-goal);fn=sorted(goal-got)
        err=lambda vals:"；".join(display(x,task) for x in vals[:3]) or "无"
        content+=[f"**{N_LABEL[m]}**　多报 {len(fp)} 条，漏报 {len(fn)} 条。多报示例：{err(fp)}。漏报示例：{err(fn)}。"]
    return content, runs, stats
def generate():
    refner,refre=reference();gold=keys(refre)
    ent,runs1,sm1=compare("ner",["bert","qwen","bilstm"],refner,"## 二、实体抽取")
    resec,runs2,sm2=compare("re",["spacy","bert","qwen"],refre,"## 三、关系抽取")
    entcount=sum(map(len,refner.values()))
    md("01_实体词典.md",["# 《平凡的世界》三部梗概 · 实体词典",
        f"收录 {len(NAMES)} 个规范实体，使用 {len(TYPES)} 个类型。括号、破折号与前后修饰语不计入词面。各简称独立参与实体识别，归并后共用一个图节点。",
        table(["类型","规范名","其他词面","覆盖原句"],[[r["type"],r["name"],r["aliases"] or "—","、".join(i for i,v in refner.items() if any(canonical(DOCS[i][a:b])==r["name"] for a,b,t in v)) or "仅嵌于称谓"] for r in LEX]),
        "“金波之妹”作为原文能辨认、姓名未给出的人物整体标注；“金波”作为亲属关系的端点单独保留。“秀莲”“润生”沿用截图写法。场所节点“煤矿”把第三部末句的“矿山”视为同一工作场所。“省报”按文中所指机构保留原称。"])
    corpus=["# 三张图片的文字转录","换行按句末句号重排，句内分号保留。编号首位是图片序号，小数点后是所在部的句序。"]
    ocr_stats=[]
    for part in (1,2,3):
        d=load(RAW/f"ocr-{part}.json")
        diff="；".join(f"「{v['raw']}」改为「{v['checked']}」" for v in d["changes"]) or "与校对稿一致"
        ocr_stats.append([f"[图 {part}](../data/images/part{part}.png)",f"{d['seconds']:.2f}",diff])
    corpus.extend(["RapidOCR 在本机逐图识别。去除标题、空格和换行后，与按原图核对的转录稿逐字对齐。",table(["图片","识别耗时/s","OCR 与校对稿的差异"],ocr_stats)])
    for p,title in enumerate(["第一部","第二部概要","第三部"],1):
        corpus.append("## "+title)
        corpus.extend(f'<a id="s{i.replace(".","-")}"></a> **{i}**　{s}' for i,s in DOCS.items() if i.startswith(str(p)+"."))
    md("00_原文与句号.md",corpus)
    detail=["# 三种方法的关系三元组","主语、关系、宾语按顺序排列。每个句号都链接到原文。对称关系用规范名排序，箭头只表示存储顺序。参考列允许结合前文确定指代。"]
    for i,s in DOCS.items():
        detail.append(f'## {i}　[查看原句](00_原文与句号.md#s{i.replace(".","-")})')
        detail.append(s)
        methodpred=[("参考标注",refre[i])]+[(N_LABEL[m],runs2[m]["predictions"][i]) for m in runs2]
        detail.append(table(["来源","（主语，关系，宾语）"],[[m,"<br>".join("（"+"，".join(v)+"）" for v in vs) or "∅"] for m,vs in methodpred]))
    md("03_关系三元组.md",detail)
    details=["# 实体识别逐句对照","方括号为从 0 开始、包含起点且不含终点的字符区间。相同词形在同句多次出现时按位置分别计分。"]
    for i,s in DOCS.items():
        details += [f"## {i}",s,table(["来源","词面 / 类型 / 字符区间"],[
          [m,"；".join(f"{s[a:b]} / {t} / [{a},{b})" for a,b,t in vs) or "∅"]
          for m,vs in [("参考标注",refner[i])]+[(N_LABEL[m],runs1[m]["predictions"][i]) for m in runs1]])]
    md("02_实体识别逐句对照.md",details)
    kg=load(RAW/"qwen_graph.json");kt=keys(kg["predictions"])
    graphs={m:load(RAW/f"neo4j_{m}.json") for m in ["reference","spacy","qwen"]}
    graph_names={"reference":"参考图 · Neo4j","spacy":"spaCy 抽取图 · Neo4j","qwen":"Qwen 直接生成图 · Neo4j 回读"}
    gt_nodes=set(graphs["reference"]["nodes"]);kg_nodes=set(kg["declared_nodes"])
    endpoints={x for _,h,p,t in kt for x in (h,t)}
    for m,g in graphs.items():
        expected={"reference":gold,"spacy":keys(runs2["spacy"]["predictions"]),"qwen":kt}[m]
        assert projected(g["edges"])==expected
    # The data originate in the checked vocabulary. Still quote all strings for Cypher re-import.
    quote=lambda value:json.dumps(value,ensure_ascii=False)
    commands=["// 在 cypher-shell 交互窗口执行 :source reports/06_图谱导入.cypher",
              "CREATE CONSTRAINT novel_course_entity IF NOT EXISTS FOR (n:NovelCourseEntity) REQUIRE (n.scope,n.name) IS UNIQUE;"]
    for g in graphs.values():
        scope=quote(g["scope"])
        for n in g["nodes"]:
            commands.append(f"MERGE (n:NovelCourseEntity {{scope:{scope},name:{quote(n)}}}) SET n.kind={quote(NAMES[n])};")
        for v in g["edges"]:
            assert v["p"] in PRED
            commands.append(f"MATCH (a:NovelCourseEntity {{scope:{scope},name:{quote(v['h'])}}}), (b:NovelCourseEntity {{scope:{scope},name:{quote(v['t'])}}}) MERGE (a)-[r:`{v['p']}` {{sid:{quote(v['sid'])}}}]->(b) SET r.evidence={quote(DOCS[v['sid']])};")
    (OUT/"06_图谱导入.cypher").write_text("\n".join(commands)+"\n",encoding="utf-8")
    gr=["## 四、图谱构建与性能",
        "路径 A 把 spaCy 产生的带句号三元组交给 Neo4j，按规范名合并节点、按关系类型和证据句建边。路径 B 对每一部另发一次本地模型请求，从原文直接输出 nodes 和 edges。另将参考图和模型图也写入同一台 Neo4j 并回读，用于核对入库与查询。",
        table(["库中图谱","节点","证据边","去掉句号后的不同三元组","写入事务/s","整图回读/s","一跳查询中位数/ms","写前已有节点"],[
          [graph_names[m],len(g["nodes"]),len(g["edges"]),len({(v['h'],v['p'],v['t']) for v in g['edges']}),f'{g["write_seconds"]:.3f}',f'{g["read_seconds"]:.3f}',f'{g["lookup_median_ms"]:.3f}',g["existing_nodes"]] for m,g in graphs.items()]),
        table(["构建路径","证据边预测","参考","命中","P","R","F1","生成或抽取/s","再写入/s"],
          [[label,*mt(metrics(v,gold)),f"{elapsed:.3f}",f'{graphs[m]["write_seconds"]:.3f}'] for m,label,v,elapsed in [
            ("spacy","规则抽取 → Neo4j",keys(runs2["spacy"]["predictions"]),runs2["spacy"]["seconds"]),
            ("qwen","LLM 直接生成图 → Neo4j",kt,kg["seconds"])]]),
        f"模型直接列出的有效节点为 {len(kg_nodes)} 个，对参考图节点的精确率、召回率分别为 {metrics(kg_nodes,gt_nodes)[3]:.1%}、{metrics(kg_nodes,gt_nodes)[4]:.1%}。边引用但模型未显式声明的节点有 {len(endpoints-kg_nodes)} 个，在入库时补齐。模型声明而没有有效边的孤立节点有 {len(kg_nodes-endpoints)} 个。因谓词、实体或证据编号不合要求而剔除的边记录有 {len(kg['rejected'])} 条。",
        "Neo4j 列使用真实 Bolt 事务耗时，包含本机通信、提交和按关系类型分批写入，排除建索引、启动服务和模型推理。每张图对“孙少平”查询去重的一跳邻居，预热 1 次后计 20 次的中位数。写前已有节点数用于识别重复导入。小图上的写入时间受缓存和首次执行计划影响。",
        "三张图均已逐条从 Neo4j 读回，核对节点名、关系方向、谓词和证据句号。回读一致只能验证存储。事实质量由与参考证据边的匹配来衡量。"]
    sample_query="MATCH (n:NovelCourseEntity {scope: '"+graphs["reference"]["scope"]+"', name: '孙少平'})-[r]-(m) RETURN n,r,m"
    false_edges=sorted(kt-gold)
    if false_edges:
        gr.append("直接建图后仍需核对可执行的边。以下记录通过了实体、谓词和句子编号校验，但与参考事实不符："+ "；".join(display(x,"re") for x in false_edges[:3])+"。")
    gr.append("Neo4j Browser 中查看本次参考图的一跳关系：\n\n```cypher\n"+sample_query+"\n```")
    selection={"父母是","兄姐是","配偶","恋爱","朋友","关系恶化","青梅竹马"}
    people=[(i,h,p,t) for i,h,p,t in sorted(gold) if p in selection and NAMES[h]==NAMES[t]=="人物"]
    names={n:f"n{j}" for j,n in enumerate(sorted({v for _,h,p,t in people for v in (h,t)}))}
    diagram=["flowchart LR",*[f'    {ident}["{name}"]' for name,ident in names.items()]]
    for h,p,t in sorted({(h,p,t) for _,h,p,t in people}):
        connector=" ---" if p in SYMMETRIC else " -->"
        diagram.append(f'    {names[h]}{connector}|{p}| {names[t]}')
    gr.append("**参考图中的人物关系节选**　无向线表示对称关系，箭头表示关系从谁指向谁。不同部别出现的关系并列展示。\n\n```mermaid\n"+"\n".join(diagram)+"\n```")
    graphpage={"docs":DOCS,"kinds":NAMES,"graphs":{m:dict(title=graph_names[m],nodes=g["nodes"],edges=g["edges"]) for m,g in graphs.items()}}
    template=(ROOT/"src"/"view.html").read_text(encoding="utf-8")
    (OUT/"05_知识图谱.html").write_text(template.replace("__GRAPH_DATA__",json.dumps(graphpage,ensure_ascii=False).replace("<","\\u003c")),encoding="utf-8")
    bestner=max(sm1,key=lambda m:sm1[m][-1]);bestre=max(sm2,key=lambda m:sm2[m][-1])
    scores=lambda pred,ident,vs: "、".join(N_LABEL[m] for m,v in pred.items() if tuple(edge(*vs)) in {tuple(x) for x in v["predictions"][ident]}) or "无"
    case_rows=[]
    for ident,vs,why in [("1.03",("孙少平","朋友","田晓霞"),"主语写作“他”，需回到前句识别。"),("2.03",("孙少平","就业机会","煤矿工人"),"“获得……好机遇”保留机会语气。"),("2.03",("田晓霞","任职于","省报"),"女友、学校、职业和工作单位出现在同一长句。"),("2.05",("润生","兄姐是","田润叶"),"需要把“她”回指到上句的润叶。"),("3.02",("田福军","发电报给","孙少平"),"“给……发了封电报”确定了方向。"),("3.04",("秀莲","患病","肺癌"),"确诊者与疾病之间隔着对庆典的叙述。")]:
        case_rows.append([ident," → ".join(vs),scores(runs2,ident,vs),why])
    mrules=["## 一、语料与评价约定",
        f"三张图逐句转录为 {len(DOCS)} 句，共 {sum(map(len,DOCS.values()))} 个正文字符。另按图文逐句编制参考标注，含 {entcount} 个实体提及、{len(gold)} 条带证据关系；没有独立第二标注者复核。21 句全部用于本次对照，模型的监督训练另用脚本生成的 896 条短句（27 种关系及无关系各 32 条）。词典与规则据此题材设计，因此属于小语料、已知领域的课堂比较。",
        "实体采用平面 BIO 标注，按〔句号、字符起点、终点、类型〕完全一致计命中。人物涵盖具名人物及“金波之妹”一类可唯一指认的称谓；机构、地点、身份、事件、制度、疾病和时间限于词典所述范围。年龄、金额、泛称亲属与未命名地点暂不计入目标。身份保留中心词，例如“优秀工人”标作“工人”。",
        "关系采用 27 类限定谓词，按〔句号、规范主语、谓词、规范宾语〕计命中。同一句重复陈述只计一次，不同句可为同一边提供不同证据。配偶、恋爱、同学、朋友、青梅竹马、关系恶化按无向语义去重；单方爱慕、帮助和亲属方向单独计分。父母是、兄姐是由晚辈或弟妹指向父母或兄姐。不同时间部别并存，不推出永久有效。",
        "P＝命中/预测；R＝命中/参考；微平均 F1＝2×命中/(预测+参考)。表中的推理耗时为三部合计；本地模型计入请求及加载，神经模型的训练耗时另列。BERT 两项任务共享一次联合训练，两个表中的训练时间是同一次计时。不同程序可能同期占用电脑，耗时只反映这一轮运行。",
        "本次运行使用 Windows、本地 Ollama（qwen2.5:7b）、PyTorch 2.5.1、Transformers 4.44.2、spaCy 3.7.5 和 Neo4j Community "+graphs["reference"]["version"]+"。BERT 和 BiLSTM-CRF 使用 CPU；本机显卡为 NVIDIA GeForce RTX 4060 Laptop GPU。"]
    analysis=["## 五、从原句解释差异",table(["句号","参考关系","抽对的方法","语言线索"],case_rows),
        f"按本次严格匹配的 F1，实体识别最高的是 {N_LABEL[bestner]}（{sm1[bestner][-1]:.1%}），关系抽取最高的是 {N_LABEL[bestre]}（{sm2[bestre][-1]:.1%}）。样本仅有三段相连的剧情，词典又已覆盖人物，排序不能推广到另一部小说。",
        "BERT 使用 hfl/rbt3 三层中文预训练编码器，联合微调字级实体分类头和有向实体对分类头。关系阶段用词典定位同一分句内的候选，按两端类型和 0.50 概率阈值筛选。BiLSTM-CRF 从合成短句学习字嵌入和标签转移，用 Viterbi 解码；未在训练字符表中出现的字共用未知字向量。长句、简称和训练文本没有出现过的人名对它尤其困难。",
        "spaCy 使用中文逐字分词、PhraseMatcher 和 Matcher。其局部规则要求触发词两侧有可匹配端点。Qwen 实体任务独立于词典预测词面与类型，再对齐到原句；相同词面按所有完整出现的位置展开，重叠优先长词。Qwen 关系与直接建图各自独立，输入带规范词表和谓词表，同一部内允许指代。参考标注还可调用前部明示事实，因此跨部省略对算法更难。",
        f"同一模型的三个任务分别保留原始回复和提示。实体和关系清洗分别记录 {len(runs1['qwen']['rejected'])}、{len(runs2['qwen']['rejected'])} 项不合规范输出。成绩对应完成校验的系统结果；可打开 runs 目录审计被剔除的值和句号。",
        "若扩充到新章节，可先请两位标注者独立标注，再按章节拆分开发与测试，并用多次运行报告中位数与波动。这样可以检验本次方法差异是否稳定。"]
    md("04_实验比较.md",["# 《平凡的世界》实体、关系与知识图谱实验",*mrules,*ent,*resec,*gr,*analysis])
if __name__=="__main__":generate()
