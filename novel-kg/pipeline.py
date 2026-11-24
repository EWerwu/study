"""Course project entry point. 'report' replays all saved measurements without dependencies."""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
def run(file,*args):
    subprocess.run([sys.executable,str(ROOT/"src"/file),*args],cwd=ROOT,check=True)
def main():
    parser=argparse.ArgumentParser(description="三张中文梗概的实体、关系与知识图谱流水线")
    parser.add_argument("stage",choices=["all","ocr","neural","rules","llm","graph","report"],nargs="?",default="report")
    parser.add_argument("--fresh",action="store_true",help="重新调用本地模型，替换原始响应")
    cfg=parser.parse_args()
    stages=["ocr","neural","rules","llm","graph","report"] if cfg.stage=="all" else [cfg.stage]
    for stage in stages:
        print("运行 "+stage,flush=True)
        if stage=="ocr":
            run("ocr.py")
            os.environ["KG_PASSAGES"]=str(ROOT/"reports"/"runs"/"verified_passages.tsv")
        elif stage=="llm":run("llm.py",*(["--fresh"] if cfg.fresh else []))
        else:run(dict(neural="neural.py",rules="rules.py",graph="graph.py",report="report.py")[stage])
if __name__=="__main__":main()
