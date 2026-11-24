"""Standalone synthetic supervision; no names, sentences or labels read from the test reference."""
import random
import re
from common import TYPES, PRED

POOLS = {
    "人物": "林星 许明 江远 宋兰 谢淮 陆宁 宁川 叶青 林小雅 姚志新 胡晓楠 周启文 张文安 刘红英 韩书平 夏雨 陈飞 梁心怡 唐子安 卢思敏".split(),
    "机构": "长岭中学 镇政府 红星小学 晨报 艺术学院 物资局 车队 合作社 教育局 县教育委员会 北河中学 青年报 锦城师范".split(),
    "地点": "四川 江南 海南 河北 福建 银矿 林场 诊所 牧场 陶窑 采石场 海港 门诊部".split(),
    "身份": "医生 工程师 船员 店主 干部 总编辑 农户 站长 电工 学徒 护士 校长 维修工 通讯员 经理 司机".split(),
    "事件": "防汛演练 代表大会 工会会议 抗震采访 春季运动会 暴雨灾害 洪灾 扶贫行动".split(),
    "制度": "轮岗制 分红制 计件制 轮班制度 绩效制度".split(),
    "疾病": "肝炎 胃癌 肺炎 贫血 哮喘".split(),
    "时间": "1960年冬 1994年夏 2003年初 1998年 三年后 半年后 今年春天".split(),
}
# Each entry specifies subject/object types and paraphrases with explicit argument roles.
PATTERNS = {
 "就读":("人物","机构",["{s}到{o}读书","{s}就读于{o}","在{o}念书的{s}来了"]),
 "毕业于":("人物","机构",["{s}从{o}毕业","{s}毕业于{o}","从{o}毕业的{s}回乡"]),
 "身份":("人物","身份",["{s}成了一名{o}","{o}{s}走过来了","{s}以前是{o}","{s}当了{o}"]),
 "爱慕":("人物","人物",["{s}爱慕{o}","{s}对{o}产生了情愫","{o}是{s}暗恋的人"]),
 "同学":("人物","人物",["{s}和同班同学{o}到场","{s}与{o}是同学"]),
 "关系恶化":("人物","人物",["{s}与{o}渐渐交恶","{s}和{o}的关系变坏了"]),
 "恋爱":("人物","人物",["{s}和{o}恋爱了","{s}的女友{o}来了","{s}和{o}成为恋人"]),
 "朋友":("人物","人物",["{s}同{o}建立了友情","{s}是{o}的朋友"]),
 "帮助":("人物","人物",["{s}帮助了{o}","{o}得到了{s}的帮助","在{s}帮助下，{o}度过难关"]),
 "父母是":("人物","人物",["{s}的父亲是{o}","{o}的女儿{s}来了","{s}是{o}的儿子"]),
 "兄姐是":("人物","人物",["{s}的哥哥是{o}","{o}的弟弟{s}长大了","{s}是{o}的妹妹"]),
 "青梅竹马":("人物","人物",["{s}与{o}青梅竹马","{s}和{o}自幼一起长大"]),
 "配偶":("人物","人物",["{s}与{o}结了婚","{s}和{o}成为夫妻","{s}的妻子{o}回来了"]),
 "到访":("人物","地点",["{s}来到了{o}","{s}从{o}出来","{s}又回到了{o}"]),
 "领导":("人物","机构",["{s}领导{o}行动","{s}带领{o}前进"]),
 "抵制":("人物","制度",["{s}决定抵制{o}","{s}反对{o}"]),
 "推行":("人物","制度",["{s}率先实行{o}","{s}决定推广{o}"]),
 "召开":("人物","事件",["{s}连夜召开{o}","{s}主持{o}"]),
 "就业机会":("人物","身份",["{s}获得当{o}的机会","{s}有望成为{o}"]),
 "任职于":("人物","机构",["{s}到{o}工作","{s}在{o}上班","{s}受聘于{o}"]),
 "经营":("人物","地点",["{s}经营着{o}","{s}自己开办了{o}"]),
 "参与":("人物","事件",["{s}参与{o}","{s}在{o}中工作"]),
 "发电报给":("人物","人物",["{s}给{o}发去电报","{o}接到{s}拍来的电报"]),
 "患病":("人物","疾病",["{s}确诊{o}","{s}患上了{o}"]),
 "表白":("人物","人物",["{s}向{o}表白","{o}收到了{s}的告白"]),
 "拒绝":("人物","人物",["{s}拒绝了{o}","{o}遭到了{s}的拒绝"]),
 "救护":("人物","身份",["{s}抢救了{o}","{s}为了救护{o}受了伤"]),
}
def synth(seed=42, per=32):
    r=random.Random(seed)
    samples=[]
    for pred in [*PRED,"无关系"]:
        for k in range(per):
            if pred=="无关系":
                st,ot=r.choice(list(POOLS)),r.choice(list(POOLS))
                template=r.choice(["{s}出现在前页，文章随后提起{o}","谈到{s}，墙上还画着{o}","有人提及{o}，远处传来{s}的消息"])
            else:
                st,ot,variants=PATTERNS[pred]
                if pred=="推行" and k%4==0: st="机构"
                if pred=="帮助" and k%5==0: st="身份"
                if pred=="任职于" and k%4==0: ot="地点"
                template=variants[k%len(variants)]
            s=r.choice(POOLS[st]);o=r.choice([x for x in POOLS[ot] if x!=s and s not in x and x not in s])
            parts=[]; spans=[]; offset=0
            for piece in re.split(r"(\{s\}|\{o\})",template):
                if piece in ("{s}","{o}"):
                    word,typ=(s,st) if piece=="{s}" else (o,ot)
                    spans.append((offset,offset+len(word),typ,piece[1]))
                else: word=piece
                parts.append(word);offset+=len(word)
            samples.append(("".join(parts)+"。",spans,pred))
    r.shuffle(samples)
    return samples
