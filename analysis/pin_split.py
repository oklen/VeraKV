import json,re
def load(*t):
    r=[]
    for x in t: r+=json.load(open("/home/tiger/mu_merged_%s.json"%x))
    return r
def split_acc(rs):
    cite=[]; non=[]
    for x in rs:
        q=str(x.get("question",""))
        (cite if re.search(r"(?:step|turn)s?\s*#?\s*\d+",q,re.I) else non).append(1 if x["score"]==1.0 else 0)
    return sum(cite)/len(cite), len(cite), sum(non)/len(non), len(non)
R2=load("R2"); FL=load("FLSA","FLSB")
c2,nc2,n2,nn2=split_acc(R2); cf,ncf,nf,nnf=split_acc(FL)
print("step-citing frac = %.1f%% (%d/%d)"%(100*ncf/(ncf+nnf),ncf,ncf+nnf))
print("hybrid(R2):      cite=%.4f  noncite=%.4f"%(c2,n2))
print("causal(FLSA/B):  cite=%.4f  noncite=%.4f"%(cf,nf))
print("pin effect:      cite=%+.4f  noncite=%+.4f  overall=%+.4f"%(cf-c2,nf-n2,(sum(1 for x in FL if x['score']==1.0)/len(FL))-(sum(1 for x in R2 if x['score']==1.0)/len(R2))))
