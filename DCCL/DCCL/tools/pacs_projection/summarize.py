"""Comparison of the fixed PACS pairs, with existing A8 results as the only baseline."""
import csv
import json
from pathlib import Path
from statistics import mean
import subprocess
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tools.pacs_causal.metrics import coverage
from tools.pacs_projection.analyze import class_domain_drift, protection
from tools.pacs_projection.queue import A8, DOMAINS, GROUPS, ROOT, atomic_json, completed, folds, historical_fold, say


SETUPS=("A8","B8","A16")
CHECKPOINTS=("iid_best","swad")
PLANES=("z","z_ref","E_beta","S_removed","Pz","Pz_ref","E_orth","E_full_res")
CL_BY_KIND=("category","domain")
LABELS=("Art Painting","Cartoon","Photo","Sketch")


def csv_rows(path, rows):
    keys=list(dict.fromkeys(k for item in rows for k in item))
    with path.open("w",newline="",encoding="utf-8") as handle:
        w=csv.DictWriter(handle,fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def performance_rows():
    result=[]
    for i,d in enumerate(DOMAINS):
        for group in SETUPS:
            folder=historical_fold(i) if group=="A8" else folds(group,i)[0]
            if group=="A8":
                fold,=json.loads((folder/".github-archive"/"evaluation.json").read_text())
            else:
                commit=json.loads((ROOT/"logs"/f"{group}.complete.json").read_text())["commit"]
                fold=completed(folder,group,i,commit)
            selection,swad=fold["iid_checkpoint"],fold["swad_evaluation"]
            result.append(dict(held_out=d,group=group,iid=selection["test_in"],
                iid_test_out=selection["test_out"],source_best=selection["train_out"],
                swad=swad["metrics"]["test_in"],swad_test_out=swad["metrics"]["test_out"],
                swad_source_val=swad["metrics"]["train_out"],iid_step=selection["step"],
                stop_update=swad["stop_update"],swad_converged=swad["converged"],
                swad_fallback=not swad["converged"],dead_valley=swad["dead_valley"],
                first_swad_step=swad["start"],last_swad_step=swad["end"],
                checkpoint_dir=str(folder/"checkpoints")))
    return result


def a8_features(domain,variant,previous):
    name="IID" if variant=="iid_best" else "SWAD"
    record=next(r for r in previous["results"] if r["held_out"]==domain and r["checkpoint"]==name)
    with np.load(A8/"features"/domain/"features.npz",allow_pickle=False) as original:
        d,y,train=(original[k] for k in ("domain","y","is_source_train"))
        ids=original["sample_id"]
    path=(A8/"features"/domain/"features.npz" if variant=="iid_best" else
          A8/"diagnosis_20261120"/"cache"/(domain+"_swad.npz"))
    with np.load(path,allow_pickle=False) as bank:
        np.testing.assert_array_equal(ids,bank["sample_id"])
        z,q=bank["z"],bank["q"]
        perturb={k:bank[k] for k in bank.files if k.startswith(("delta_r_","delta_z_"))}
    q_env,ranks=protection(z,q,y,d,train)
    probes={}
    for rep in PLANES:
        query="S_raw" if rep=="S_removed" else rep
        probes[rep]={kind:next(a for a in record["probes"] if a["representation"]==query
                               and a["scale"]=="original" and a["probe"]==kind) for kind in CL_BY_KIND}
    raw=[dict(style=r["style"],response=r["response"],space="Q",
                  **{k:r[k] for k in ("mean","median","p10","p90","valid")})
         for r in record["coverage"] if r["direction"]=="Learned Q"]
    assert len(raw)==4
    for name,value in perturb.items():
        raw.append(dict(style=name[8:],response=name[:7],space="Q_env_after_class_protection",**coverage(value,q_env)))
    return dict(group="A8",held_out=domain,checkpoint=variant,protection=ranks,probes=probes,
                geometry={r["representation"]:r for r in record["geometry"]},
                within_class_domain_drift_fraction_Q_env=class_domain_drift(z,q,q_env,y,d,~train),
                removed_energy_over_z=next(r["removed_energy_over_z"] for r in record["directions"]
                                            if r["direction"]=="Learned Q"),coverage=raw)


def features_rows():
    prev=json.loads((A8/"diagnosis_20261120"/"tables"/"diagnostic_summary.json").read_text())
    all_rows=[]
    for group in SETUPS:
        for d in DOMAINS:
            for ck in CHECKPOINTS:
                old=(A8/"diagnosis_20261120"/"cache"/(d+".json"))
                if group=="A8":
                    result=a8_features(d,ck,prev)
                else:
                    result=json.loads((ROOT/"probes"/group/d/(ck+".json")).read_text())
                    assert result["split_sha256"]==json.loads(old.read_text())["data_split_sha256"]
                all_rows.append(result)
            if group=="B8":
                assert all_rows[-1]["fixed_mean_sha256"]==all_rows[-2]["fixed_mean_sha256"]
    return all_rows


PC=lambda x:f"{100*x:.3f}%"
PP=lambda x:f"{100*x:+.3f} pp"


def header(columns):
    return ["| "+" | ".join(columns)+" |","|"+"|".join("---" for _ in columns)+"|"]


def table_row(values):
    return "| "+" | ".join(map(str,values))+" |"


def mean_probe(rows,rep,kind):
    return mean(r["probes"][rep][kind]["acc"] for r in rows)


def coverage_mean(rows,response,style,space="Q"):
    return mean(next(x["mean"] for x in r["coverage"] if
        (x["response"],x["style"],x["space"])==(response,style,space)) for r in rows)


def comparison_figure(scores):
    fig,axes=plt.subplots(1,2,figsize=(11.8,4.4))
    fig.subplots_adjust(left=.085,right=.98,bottom=.22,top=.76,wspace=.22)
    ink,gray,colors="#1d3542","#48606d",("#667788","#147e75","#6959a2")
    fig.suptitle("PACS  |  Two isolated DICM changes",weight="bold",fontsize=17,x=.085,y=.97,ha="left",color=ink)
    fig.text(.085,.875,"Paired target folds. The dashed lines mark each configuration's four-fold mean.",
             color=gray,fontsize=9)
    for ax,key,title in zip(axes,("swad","swad_test_out"),("SWAD test_in","SWAD test_out")):
        for group,color in zip(SETUPS,colors):
            rows=[r for r in scores if r["group"]==group]
            valid=[100*r[key] for r in rows]
            ax.plot(range(4),valid,"o-",color=color,linewidth=1.6,markersize=5,label=group)
            ax.axhline(mean(valid),color=color,linestyle="--",linewidth=.8,alpha=.58)
            for i,r in enumerate(rows):
                if r["swad_fallback"]:
                    ax.scatter(i,valid[i],color=color,marker="x",s=75,zorder=5)
        ax.set_title(title,loc="left",weight="bold",color=ink)
        ax.set_xticks(range(4),("Art","Cartoon","Photo","Sketch"))
        ax.set_ylabel("Accuracy (%)")
        ax.spines[["right","top"]].set_visible(False)
        ax.grid(axis="y",color="#dbe5e9")
    fig.legend(*axes[0].get_legend_handles_labels(),loc="lower center",ncol=3,frameon=False)
    for extension in ("png","pdf"):
        fig.savefig(ROOT/"reports"/f"swad_comparison.{extension}",dpi=210,bbox_inches="tight",facecolor="white")
    plt.close(fig)


def write_report(scores, analysis):
    scores_mean={g:{k:mean(r[k] for r in scores if r["group"]==g)
                 for k in ("iid","iid_test_out","swad","swad_test_out","source_best","swad_source_val")} for g in SETUPS}
    feature={g:{ck:[r for r in analysis if r["group"]==g and r["checkpoint"]==ck]
                for ck in CHECKPOINTS} for g in SETUPS}
    summary=[
        "# PACS DICM 固定参考与 Rank16 对照",
        "",
        "A8 为历史逐样本冻结参考结果。B8 只把稳定投影改为三个源域训练均值的等权平均，PIRE、CDRM、ISR 仍使用逐样本冻结参考。"
        "A16 只把 rank 从 8 改为 16，基从头初始化。每折数据划分、初始 Backbone 和分类器均与 A8 核对一致。"
        "四域、单种子、CE+ISR，保持原 SWAD 协议。目标域测试准确率只用于最终对照。",
        "",
        *header(("目标域","方案","IID test_in","IID test_out","SWAD test_in","SWAD test_out","源域最佳","停止更新","SWAD 状态")),
    ]
    for r in scores:
        status="正常早停" if r["dead_valley"] and r["swad_converged"] else (
               "收敛" if r["swad_converged"] else "未收敛回退")
        summary.append(table_row((r["held_out"],r["group"],*(PC(r[k]) for k in (
                     "iid","iid_test_out","swad","swad_test_out","source_best")),r["stop_update"],status)))
    summary += [""]
    for g in SETUPS:
        a=scores_mean[g]
        summary += [f"{g} 的四域 IID test_in/test_out 为 **{PC(a['iid'])} / {PC(a['iid_test_out'])}**，"
                    f"SWAD test_in/test_out 为 **{PC(a['swad'])} / {PC(a['swad_test_out'])}**。"
                    f"源域 IID 最佳与最终模型验证准确率为 {PC(a['source_best'])} / {PC(a['swad_source_val'])}。"]
    if any(r["swad_fallback"] for r in scores):
        summary += ["","上列含回退模型的方案均值是四域最终输出的描述，不能解释为四个正常收敛 SWAD 的均值。"]
    summary += ["","![PACS 目标域配对结果](swad_comparison.png)","",
                "## 固定参考和秩变化的配对影响",""]
    for g in ("B8","A16"):
        a=scores_mean["A8"]; b=scores_mean[g]
        r=feature[g]["swad"]; ref=feature["A8"]["swad"]
        orig=mean_probe(r,"z","domain"); new=mean_probe(r,"E_beta","domain")
        old=mean_probe(ref,"E_beta","domain")
        summary += [f"**A8 与 {g}。** SWAD test_in 变化 {PP(b['swad']-a['swad'])}，"
                    f"test_out 变化 {PP(b['swad_test_out']-a['swad_test_out'])}。"
                    f"保留表示的源域可辨识性为 {PC(new)}，A8 为 {PC(old)}，该组原始 z 为 {PC(orig)}。"]
        for d in ("cartoon","sketch"):
            index={(t["held_out"],t["group"]):t for t in scores}
            x,y=index[d,"A8"],index[d,g]
            summary.append(f"- {d} 的 SWAD test_in/test_out 变化为 {PP(y['swad']-x['swad'])} / {PP(y['swad_test_out']-x['swad_test_out'])}。")
        u=mean(t["geometry"]["z"]["same_class_cross_domain_ms_unit_l2"] for t in r)
        v=mean(t["geometry"]["E_beta"]["same_class_cross_domain_ms_unit_l2"] for t in r)
        summary += ["", f"{g} 的类别探针由 {PC(mean_probe(r,'z','category'))} 到 "
                    f"{PC(mean_probe(r,'E_beta','category'))}。同类跨域单位化平方距离由 {u:.4f} 到 {v:.4f}。"]
    summary += ["","## 类别线索、扰动覆盖与删除能量","",
                "域探针为源域类别×域平衡的线性读出。Q 内保护子空间由按源域等权的类均值差异构成，是否有额外可利用的域方向，用同一批留出扰动和剩余空间覆盖共同判断。"
                "方向覆盖随 rank 和删除能量变化，因此报告原覆盖率、剩余覆盖、每维覆盖和实际删除量。","",
                *header(("方案","权重","z 类/域","E 类/域","S 类/域","Pz 类/域","删除能量","同类跨域距离 z/E","类间/类内 z/E","保护后 rank","Δr 模糊 Q/剩余","Δr 色阶 Q/剩余"))]
    for g in SETUPS:
        for ck in CHECKPOINTS:
            r=feature[g][ck]
            labels=["/".join(PC(mean_probe(r,n,k)) for k in ("category","domain")) for n in ("z","E_beta","S_removed","Pz")]
            def pair_g(key):
                return "/".join(f"{mean(t['geometry'][m][key] for t in r):.4f}" for m in ("z","E_beta"))
            ranks="、".join(str(x) for x in sorted({t["protection"]["remaining_rank"] for t in r}))
            def cv(style):
                return "/".join(PC(coverage_mean(r,"delta_r",style,sp)) for sp in ("Q","Q_env_after_class_protection"))
            summary.append(table_row((g,ck,*labels,PC(mean(t["removed_energy_over_z"] for t in r)),
                           pair_g("same_class_cross_domain_ms_unit_l2"),pair_g("between_over_within_class"),ranks,
                           cv("gaussian_blur_1p6"),cv("posterize_4bit"))))
    a,b=(feature[g]["swad"] for g in ("A8","A16"))
    summary += ["",f"Rank16 的移除分量类别探针相对 A8 变化 {PP(mean_probe(b,'S_removed','category')-mean_probe(a,'S_removed','category'))}，"
                f"域探针变化 {PP(mean_probe(b,'S_removed','domain')-mean_probe(a,'S_removed','domain'))}。"
                "两项读数衡量固定探针可恢复的线索。秩上升和能量变化均可能影响它们，跨来源的独立域信息需要结合类保护后覆盖来判断。",""]
    styles=("gaussian_blur_1p6","posterize_4bit")
    for style,label in zip(styles,("模糊","色阶")):
        full=coverage_mean(b,"delta_r",style)-coverage_mean(a,"delta_r",style)
        protected=coverage_mean(b,"delta_r",style,"Q_env_after_class_protection")-coverage_mean(
            a,"delta_r",style,"Q_env_after_class_protection")
        summary.append(f"Rank16 的{label}残差响应覆盖相对 A8 变化 {PP(full)}，类保护后变化 {PP(protected)}。")
    for g in SETUPS:
        r=feature[g]["swad"]
        drift=mean(t["within_class_domain_drift_fraction_Q_env"] for t in r)
        summary.append(f"{g} 的类保护剩余空间捕获 Q 内同类跨源域中心偏移能量的 {PC(drift)}。")
    summary.append("")
    summary += [
        "## 后续保留判断",
        "",
        "B8 在同一权重下执行 $E=(I-0.75P)z+0.75P\\bar z_{\\mathrm{ref}}$。"
        "$I-0.75P$ 在 Q 内和 Q 外的特征值分别为 0.25 和 1，因此映射可逆。"
        "线性探针的变化反映当前特征几何与有限样本读出。端到端固定参考也会经 CE 改变 Backbone，"
        "需要结合不同模型的 z 和 E 区分训练作用与固定投影作用。",
        "",
    ]
    reference=feature["A8"]["swad"]
    reference_source=scores_mean["A8"]["swad_source_val"]
    converged={g for g in SETUPS if all(r["swad_converged"] for r in scores if r["group"]==g)}
    supported=[]
    for g in ("B8","A16"):
        current=feature[g]["swad"]
        delta_src=scores_mean[g]["swad_source_val"]-reference_source
        delta_domain=mean_probe(current,"E_beta","domain")-mean_probe(reference,"E_beta","domain")
        delta_class=mean_probe(current,"E_beta","category")-mean_probe(reference,"E_beta","category")
        relative_to_z=mean_probe(current,"E_beta","domain")-mean_probe(current,"z","domain")
        summary.append(f"{g} 相对 A8 的源域验证准确率、保留表示域探针、保留表示类别探针变化"
                       f"分别为 {PP(delta_src)}、{PP(delta_domain)}、{PP(delta_class)}。")
        if {"A8",g}<=converged and delta_src>=0 and delta_domain<0 and delta_class>=0 and relative_to_z<0:
            supported.append(g)
    summary.append("")
    if supported:
        best=max(supported,key=lambda g:scores_mean[g]["swad_source_val"])
        summary.append(f"按源域验证，{best} 同时满足验证分类和类别探针保持、域探针下降，优先做多种子验证。")
    else:
        summary.append("源域诊断尚未发现同时满足验证分类和类别探针保持、域探针下降的方案。"
                       "结合类别保护后覆盖和删除能量检查 Q 的学习目标，再决定是否更改分解机制。")
    summary.append("Cartoon、Sketch 与其余目标域的成绩仅作为最终配对结果。")
    summary += ["","全部探针和类保护只用源域。诊断反映已测扰动、线性可辨识性和该轮单种子泛化。",
                "各模型各折的 t-SNE 由同一批源域验证图像的 z、E、S 等表示联合嵌入，比较范围是同一模型与折。",
                "两种权重、逐折样本量、原始坐标及覆盖明细见归档。","",
                "[目标域明细](performance.csv) · [特征明细](features.csv) · [扰动覆盖](coverage.csv)",""]
    (ROOT/"reports"/"PACS-DICM-B8-A16-对照报告.md").write_text("\n".join(summary),encoding="utf-8")
    comparison_figure(scores)


def run():
    while not all((ROOT/"logs"/f"{g}.analysis_complete.json").is_file() for g in GROUPS):
        for group in GROUPS:
            marker=ROOT/"logs"/f"{group}.analysis_complete.json"
            if marker.is_file():
                continue
            pane=subprocess.run(["tmux","list-panes","-t",f"=dicm_pacs_proj_{group}_analysis",
                                  "-F","#{pane_dead}"],capture_output=True,text=True)
            if (pane.returncode or "0" not in pane.stdout.splitlines()) and not marker.is_file():
                raise RuntimeError(f"Missing {group} analysis completion; inspect logs/{group}.analysis.log")
        time.sleep(60)
    scores=performance_rows()
    features=features_rows()
    out=ROOT/"reports"
    csv_rows(out/"performance.csv",scores)
    probes=[]
    responses=[]
    for item in features:
        keys={k:item[k] for k in ("group","held_out","checkpoint")}
        probes.append(dict(keys,**item["protection"],removed_energy_over_z=item["removed_energy_over_z"],
                           within_class_domain_drift_fraction_Q_env=item["within_class_domain_drift_fraction_Q_env"],
                           **{f"{r}_{k}":item["probes"][r][k]["acc"] for r in PLANES for k in CL_BY_KIND},
                           **{f"{r}_{k}":v for r,n in item["geometry"].items() for k,v in n.items() if isinstance(v,(float,int))}))
        for r in item["coverage"]:
            dimension=item["protection"]["Q_rank" if r["space"]=="Q" else "remaining_rank"]
            responses.append(dict(keys,**r,dimension=dimension,coverage_per_basis_dimension=r["mean"]/dimension))
    csv_rows(out/"features.csv",probes)
    csv_rows(out/"coverage.csv",responses)
    write_report(scores,features)
    atomic_json(ROOT/"logs"/"report.complete.json",dict(performance=len(scores),feature_configurations=len(features)))
    say("Completed PACS Chinese report and feature comparison",out)


if __name__=="__main__":
    run()
