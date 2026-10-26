def summarize(out: dict):
    keep = ["iters","r","s","eps_pri","eps_dual","comm_rounds"]
    return {k: out.get(k) for k in keep if k in out}