"""Recognise the supplied three numbered pages and align OCR to the visually proofread transcript."""
import csv
import hashlib
from difflib import SequenceMatcher
from rapidocr_onnxruntime import RapidOCR
from common import DATA, RAW, rows, save, perf_counter
def main():
    engine=RapidOCR(intra_op_num_threads=2,inter_op_num_threads=1)
    proof=rows(DATA/"passages.tsv");verified=[]
    for page in rows(DATA/"images.tsv"):
        part=page["part"];path=DATA/"images"/page["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest()==page["sha256"],"图片已更换，须重新校对。"
        ts=perf_counter();lines,_=engine(str(path),use_cls=False);seconds=perf_counter()-ts
        if not lines:raise ValueError("未识别出文本 "+page["file"])
        lines.sort(key=lambda x:sum(p[1] for p in x[0])/4)
        texts=[v[1] for v in lines]
        if not texts[0].startswith("第"):raise ValueError("页标题识别异常。")
        raw="".join(texts[1:]).replace(" ","")
        refs=[r for r in proof if r["id"].startswith(part+".")]
        target="".join(r["text"] for r in refs)
        align=SequenceMatcher(None,raw,target,autojunk=False)
        if align.ratio()<.85:raise ValueError("OCR 与校对稿差异过大，请复核。")
        fixed=[];changes=[]
        for op,a,b,c,d in align.get_opcodes():
            if op=="equal":fixed.append(raw[a:b])
            else:
                changes.append(dict(offset=a,raw=raw[a:b],checked=target[c:d]))
                fixed.append(target[c:d])
        corrected="".join(fixed)
        assert corrected==target
        start=0
        for row in refs:
            stop=start+len(row["text"]);verified.append((row["id"],corrected[start:stop]));start=stop
        save(RAW/f"ocr-{part}.json",dict(image=page["file"],seconds=seconds,lines=texts,changes=changes,alignment_ratio=align.ratio()))
    with (RAW/"verified_passages.tsv").open("w",encoding="utf-8",newline="") as f:
        w=csv.writer(f,delimiter="\t",lineterminator="\n");w.writerow(["id","text"]);w.writerows(verified)
    print("三张原图识别、校对与分句完成。",flush=True)
if __name__=="__main__":main()
