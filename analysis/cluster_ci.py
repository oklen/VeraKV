import json,random,collections
random.seed(0)
def load(*t):
    r=[]
    for x in t: r+=json.load(open("/home/tiger/mu_merged_%s.json"%x))
    return r
def qa_ci(rs,B=2000):
    n=len(rs); accs=[]
    sc=[1.0 if x["score"]==1.0 else 0.0 for x in rs]
    for _ in range(B):
        accs.append(sum(sc[random.randrange(n)] for _ in range(n))/n)
    accs.sort(); return accs[int(B*.025)],accs[int(B*.975)]
def ep_ci(rs,B=2000):
    by=collections.defaultdict(list)
    for x in rs: by[x["episode_id"]].append(1.0 if x["score"]==1.0 else 0.0)
    eps=list(by.values()); E=len(eps); accs=[]
    for _ in range(B):
        pick=[eps[random.randrange(E)] for _ in range(E)]
        flat=[s for ep in pick for s in ep]
        accs.append(sum(flat)/len(flat))
    accs.sort(); return accs[int(B*.025)],accs[int(B*.975)],E
for name,tags in [("R0 lex/plain",("R0",)),("R1 lex/struct",("R1",)),("R2 hyb/struct",("R2",)),
                  ("flagship/plain",("FPA","FPB")),("flagship/struct",("FLSA","FLSB"))]:
    rs=load(*tags)
    if not rs: print(name,"MISSING"); continue
    acc=sum(1 for x in rs if x["score"]==1.0)/len(rs)
    q=qa_ci(rs); e=ep_ci(rs)
    print("%-16s acc=%.4f  QA-CI[%.4f,%.4f]  EPISODE-CI[%.4f,%.4f] (E=%d)"%(name,acc,q[0],q[1],e[0],e[1],e[2]))
