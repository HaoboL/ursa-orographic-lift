#!/usr/bin/env python3
"""Frozen URSA S+E V2 validation on the full local 932-case BO04 corpus."""
from __future__ import annotations
import argparse, hashlib, json, math, os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd

for n in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[n]="1"
ROOT=Path(os.environ.get("URSA_PROJECT_ROOT", Path(__file__).resolve().parents[1])); sys.path[:0]=[str(ROOT/"src"),str(ROOT/"scripts")]
from orographic_cfd.evve_terrain_shelter_v7 import build_ridge_segment_inventory,evaluate_wemod_incident_ratios
from orographic_cfd.evve_terrain_shelter_v7 import build_flow_aligned_grid
from orographic_cfd.ursa_turbulent_interaction_support_v3t import extract_turbulent_interaction_inventory
import score_ursa_e_v2_perdigao_rhi_reuse_v1 as e2

THRESHOLDS=(.2,.5,1.0); HEIGHT=5.0

def terrain_hash(a):
    x=np.ascontiguousarray(a,dtype=np.float32); h=hashlib.sha256(); h.update(str(x.shape).encode()); h.update(b"\0"); h.update(memoryview(x).cast("B")); return h.hexdigest()

def metrics(pred,truth,valid):
    p=pred[valid]; t=truth[valid]; out={"cell_count":int(p.size),"mae":float(np.mean(abs(p-t))),"positive_overprediction_mae":float(np.mean(np.maximum(p-t,0))),"thresholds":{}}
    for q in THRESHOLDS:
        pp=p>=q; aa=t>=q; tp=np.count_nonzero(pp&aa); fn=np.count_nonzero(~pp&aa)
        out["thresholds"][str(q)]={"false_lift":float(np.mean(pp&~aa)),"accuracy":float(np.mean(pp==aa)),"recall":float(tp/(tp+fn)) if tp+fn else None}
    return out

def component_anchors(inv,shape,angle):
    starts=np.asarray(inv.component_segment_starts,int); ends=np.r_[starts[1:],inv.segment_count]; c=math.cos(math.radians(angle)); s=math.sin(math.radians(angle)); out=[]
    for begin,end in zip(starts,ends,strict=True):
        j=begin+int(np.argmax(inv.crest_elevation_m[begin:end])); x=inv.segment_s_m[j]*c-inv.segment_n_m[j]*s; y=inv.segment_s_m[j]*s+inv.segment_n_m[j]*c
        row=int(np.clip(round(y/30),0,shape[0]-1)); col=int(np.clip(round(x/30),0,shape[1]-1)); out.append((row,col))
    return out

def one(path_s):
    path=Path(path_s); case=path.parent.name
    with np.load(path,allow_pickle=False) as z: dem=np.asarray(z["dem"],float); rough=np.asarray(z["roughness"],float); ua=np.asarray(z["u_100m"],float); va=np.asarray(z["v_100m"],float)
    with np.load(path.with_name("outputs.npz"),allow_pickle=False) as z: truth=np.asarray(z["w"][0],float)
    u=float(np.mean(ua[3:6,3:6])); v=float(np.mean(va[3:6,3:6])); east=v; north=u; speed=math.hypot(east,north); angle=math.degrees(math.atan2(north,east))%360
    gy,gx=np.gradient(dem,30,30,edge_order=2); base_full=(u*gy+v*gx)/np.sqrt(1+gx*gx+gy*gy)
    # One input-only deterministic draw from every 3x3 spatial block.  This
    # covers all 100x100 blocks (10,000 equal-area cells) without consulting
    # FuXi w, BO04 error, E, or any validation metric.
    seed=int.from_bytes(hashlib.sha256(case.encode("ascii")).digest()[:8],"little")
    rng=np.random.default_rng(seed); br,bc=np.indices((100,100))
    sr=(3*br+rng.integers(0,3,size=br.shape)).ravel(); sc=(3*bc+rng.integers(0,3,size=bc.shape)).ravel()
    base=base_full[sr,sc]; truth_sample=truth[sr,sc]
    ridge=build_ridge_segment_inventory(dem,rough,flow_to_math_deg=angle,source_resolution_m=30,spacing_m=30)
    qx=sc.astype(float)*30.; qy=sr.astype(float)*30.; qground=dem[sr,sc]
    old=evaluate_wemod_incident_ratios(ridge,query_x_m=qx,query_y_m=qy,query_ground_elevation_m=qground,height_agl_m=[HEIGHT],effective_moment_coefficient=.8,base_variant="downwind_valley",shear_variant="one_seventh",combination="linear_sum",query_chunk_size=512).physical_far_ratio_hq[0].astype(float)
    retention=old.copy(); pair_count=0; flow=np.array([east/speed,north/speed]); cross=np.array([-flow[1],flow[0]])
    starts=np.asarray(ridge.component_segment_starts,int); ends=np.r_[starts[1:],ridge.segment_count]; components=[]
    for ci,(begin,end) in enumerate(zip(starts,ends,strict=True)):
        j=begin+int(np.argmax(ridge.crest_elevation_m[begin:end])); components.append({"s":float(ridge.segment_s_m[j]),"n":float(ridge.segment_n_m[j]),"h":float(ridge.crest_elevation_m[j]-ridge.downwind_base_elevation_m[j]),"n0":float(ridge.component_min_n_m[ci]),"n1":float(ridge.component_max_n_m[ci])})
    for target in components:
        upstream=[source for source in components if source["s"]<target["s"] and min(source["n1"],target["n1"])>=max(source["n0"],target["n0"])]
        if not upstream: continue
        source=max(upstream,key=lambda x:x["s"]); ht=target["h"]
        if source["h"]<=0 or ht<=0: continue
        pair_count+=1; factor=1.30*np.clip(source["h"]/ht,.5,2.)**.25
        tx=target["s"]*flow[0]-target["n"]*flow[1]; ty=target["s"]*flow[1]+target["n"]*flow[0]; ds=(qx-tx)*flow[0]+(qy-ty)*flow[1]; dn=(qx-tx)*cross[0]+(qy-ty)*cross[1]
        span=max(target["n1"]-target["n0"],30.); region=(ds>=0)&(ds<=19.2*max(ht,1.))&(abs(dn)<=.5*span)
        retention[region]=np.minimum(retention[region],np.clip(1-(1-old[region])*factor,0,1))
    aligned=build_flow_aligned_grid(dem,rough,flow_to_math_deg=angle,source_resolution_m=30,spacing_m=30); tint=extract_turbulent_interaction_inventory(aligned); pf=e2.pressure_factors(aligned,tint)
    frame=pd.DataFrame({"x_epsg3763_m":qx,"y_epsg3763_m":qy,"z_msl_m":qground+HEIGHT,"height_agl_m":np.full(qground.size,HEIGHT)})
    z0=np.asarray(ridge.roughness_length_m,float) if ridge.segment_count==tint.segment_count else float(np.median(rough))
    exposure=e2.exposure(frame,tint,0.,0.,z0,pf)["total"]; valid=exposure<.05
    corrected=np.minimum(base,0)+retention*np.maximum(base,0)
    return {"case_id":case,"terrain_group_sha256":terrain_hash(dem),"sample_seed":seed,"sample_cell_count":int(qground.size),"pair_count":pair_count,"masked_fraction":float(np.mean(~valid)),"raw":metrics(base,truth_sample,valid),"v2":metrics(corrected,truth_sample,valid)}

def finite_mean(x):
    a=np.asarray(x,float); return float(np.mean(a[np.isfinite(a)]))
def macro(rows,variant):
    groups={}
    for r in rows: groups.setdefault(r["terrain_group_sha256"],[]).append(r[variant])
    def avg(vals):
        out={"cell_count":finite_mean([x["cell_count"] for x in vals]),"mae":finite_mean([x["mae"] for x in vals]),"positive_overprediction_mae":finite_mean([x["positive_overprediction_mae"] for x in vals]),"thresholds":{}}
        for t in THRESHOLDS: out["thresholds"][str(t)]={m:finite_mean([x["thresholds"][str(t)][m] for x in vals if x["thresholds"][str(t)][m] is not None]) for m in ("false_lift","accuracy","recall")}
        return out
    return avg([avg(v) for v in groups.values()])

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--data-root",type=Path,required=True); ap.add_argument("--output",type=Path,required=True); ap.add_argument("--workers",type=int,default=1); ap.add_argument("--max-cases",type=int); a=ap.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    paths=sorted(a.data_root.glob("cases/case_*/inputs.npz")); paths=paths[:a.max_cases] if a.max_cases else paths; started=time.monotonic(); rows=[]
    if a.workers==1:
        iterator=(one(str(p)) for p in paths)
        for i,row in enumerate(iterator,1): rows.append(row); elapsed=time.monotonic()-started; print(json.dumps({"completed":i,"total":len(paths),"percent":100*i/len(paths),"elapsed_s":elapsed,"eta_s":elapsed/i*(len(paths)-i),"case_id":row["case_id"]}),flush=True)
    else:
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            fs={pool.submit(one,str(p)):p for p in paths}
            for i,f in enumerate(as_completed(fs),1): row=f.result(); rows.append(row); elapsed=time.monotonic()-started; print(json.dumps({"completed":i,"total":len(paths),"percent":100*i/len(paths),"elapsed_s":elapsed,"eta_s":elapsed/i*(len(paths)-i),"case_id":row["case_id"]}),flush=True)
    rows.sort(key=lambda r:r["case_id"]); out={"schema":"ursa.v2-bo04-full-corpus-spatial-stratified-validation.v2","status":"complete" if len(rows)==932 else "dry_run","model":"frozen S+E V2 generalized to automatically detected successive-ridge targets","input_contract":"DEM + single global wind vector","sampling":{"case_coverage":"all available cases","rule":"one deterministic input-only random cell per nonoverlapping 3x3 spatial block","cells_per_case":10000,"full_cells_per_case":90000,"selection_uses_reference_w":False},"case_count":len(rows),"terrain_group_count":len({r["terrain_group_sha256"] for r in rows}),"cases_with_detected_pair":sum(r["pair_count"]>0 for r in rows),"detected_pair_count":sum(r["pair_count"] for r in rows),"masked_fraction":float(np.mean([r["masked_fraction"] for r in rows])),"aggregate":{"raw":macro(rows,"raw"),"v2":macro(rows,"v2")},"case_rows":rows,"runtime_s":time.monotonic()-started}
    a.output.parent.mkdir(parents=True,exist_ok=False); a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n"); print(json.dumps({"status":"complete","output":str(a.output)}),flush=True)
if __name__=="__main__": main()
