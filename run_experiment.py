#!/usr/bin/env python3
"""Reproduce the SDP mmWave beamforming results and all report figures.

Run at repository root: python run_experiment.py
Figures are deliberately monochrome with distinct line styles for B/W print.
"""
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

from src.array_model import Experiment
from src.solvers import primal_admm, dual_admm, alm_dual, check_kkt

ROOT=Path(__file__).resolve().parent
FIG=ROOT/'figures'
RES=ROOT/'results'
FIG.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.family':'DejaVu Sans',
    'font.size':10,
    'axes.titlesize':12,
    'axes.labelsize':10,
    'legend.fontsize':9,
    'axes.edgecolor':'0.1',
    'savefig.facecolor':'white',
    'figure.facecolor':'white',
})

def save(fig, name):
    fig.savefig(FIG/name, dpi=300,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def graph_array(exp):
    fig,ax=plt.subplots(figsize=(8.5,3.2))
    xx=np.arange(exp.n_antennas)
    ax.plot([-0.5,7.5],[0,0],lw=1.2,color='0.2')
    ax.scatter(xx,np.zeros_like(xx),s=95,c='white',edgecolors='black',linewidths=1.5,marker='s',zorder=3)
    for i in xx:
        ax.text(i,-0.2,str(i+1),ha='center',fontsize=9)
    ax.annotate('',xy=(4,1.25),xytext=(4,.08),arrowprops=dict(arrowstyle='-|>',color='black',lw=1.4))
    ax.text(4.2,1.15,'Broadside  0 deg',ha='left',va='center')
    ax.annotate('',xy=(4.95,-.48),xytext=(4.05,-.48),arrowprops=dict(arrowstyle='<->',lw=1.1))
    ax.text(4.5,-.68,'d = lambda/2',ha='center',fontsize=10)
    ax.set_xlim(-.7,7.7);ax.set_ylim(-.9,1.55)
    ax.set_xticks([]);ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_title('Eight-element uniform linear array')
    save(fig,'Fig01_Array_geometry.png')


def graph_workflow():
    fig,ax=plt.subplots(figsize=(8.3,5.4)); ax.set_xlim(0,10);ax.set_ylim(0,9.5);ax.axis('off')
    nodes=[(2,8.2,6,0.85,'ULA steering vector and real I/Q covariance'),
       (2,6.75,6,.85,'Directional power:  tr[B(theta) Q]'),
       (2,5.3,6,.85,'Primal SDP, dual SDP, KKT conditions'),
       (2,3.85,6,.85,'Affine constraints + PSD cone'),
       (.65,2.25,3.95,.85,'Dual ADMM (c-i)'),
       (5.4,2.25,3.95,.85,'Reduced ALM (c-ii)'),
       (2.0,.6,6.,.85,'Numerical validation and beam patterns')]
    for x,y,w,h,s in nodes:
        ax.add_patch(Rectangle((x,y),w,h,fill=False,edgecolor='black',lw=1.2))
        ax.text(x+w/2,y+h/2,s,ha='center',va='center',fontsize=9.2)
    for y in (8.2,6.75,5.3):
        ax.annotate('',xy=(5,y-.57),xytext=(5,y),arrowprops=dict(arrowstyle='->',color='black',lw=1.2))
    ax.annotate('',xy=(2.6,3.1),xytext=(4.6,3.85),arrowprops=dict(arrowstyle='->',color='black',lw=1.1))
    ax.annotate('',xy=(7.4,3.1),xytext=(5.4,3.85),arrowprops=dict(arrowstyle='->',color='black',lw=1.1))
    ax.annotate('',xy=(3.2,1.45),xytext=(2.6,2.25),arrowprops=dict(arrowstyle='->',color='black',lw=1.1))
    ax.annotate('',xy=(6.8,1.45),xytext=(7.4,2.25),arrowprops=dict(arrowstyle='->',color='black',lw=1.1))
    ax.set_title('Numerical optimization workflow')
    save(fig,'Fig02_Model_workflow.png')


def graph_beam(theta,p_ref,p_opt,limit,side):
    fig,ax=plt.subplots(figsize=(8.3,4.0))
    y_ref=10*np.log10(np.maximum(p_ref,1e-7))
    y_opt=10*np.log10(np.maximum(p_opt,1e-7))
    ax.plot(theta,y_ref,'k--',lw=1.4,label='Uniform baseline')
    ax.plot(theta,y_opt,'k-',lw=1.8,label='SDP covariance')
    # sidelobe limit is only imposed on selected angles, never through the main lobe
    threshold=10*np.log10(limit)
    ax.hlines(threshold,-90,-20,linestyles=':',colors='0.45',lw=1.5,label='Sidelobe cap')
    ax.hlines(threshold,20,90,linestyles=':',colors='0.45',lw=1.5)
    ax.axvspan(-20,20,color='0.92',alpha=.55,zorder=-5)
    ax.axvline(0,color='0.65',lw=.8,ls='-.')
    ax.set(xlabel='Angle (degree)',ylabel='Power (dB, desired gain = 0 dB)',
           title='Optimized beam pattern versus uniform beamformer',xlim=(-90,90),ylim=(-45,2))
    ax.grid(True,alpha=.18,linestyle=':');ax.legend(loc='lower right')
    save(fig,'Fig03_Beam_pattern_comparison.png')


def graph_sample_constraint(ang,sample,limit):
    fig,ax=plt.subplots(figsize=(8.2,3.9))
    angle=np.asarray(ang,dtype=float)
    values=np.asarray(sample,dtype=float)
    for k,(subangle,subpower) in enumerate(((angle[angle<0], values[angle<0]),
                                             (angle[angle>0], values[angle>0]))):
        ax.plot(subangle,subpower,'k-',lw=1.25,marker='.',markersize=4,
                label='SDP sampled power' if k==0 else None)
    ax.axhline(limit,color='0.4',ls='--',lw=1.3,label='Constraint (0.025)')
    ax.axvspan(-20,20,color='0.92',alpha=.55,zorder=-5)
    ax.set(xlabel='Constrained sidelobe angle (degree)',ylabel='Directional power',
           title='Power at the 40 constrained sidelobe angles',ylim=(-.002,.031),xlim=(-95,95))
    ax.set_xticks([-90,-60,-40,-20,0,20,40,60,90])
    ax.grid(True,ls=':',alpha=.25);ax.legend(loc='upper left')
    save(fig,'Fig04_Sidelobe_constraints.png')


def graph_admm(hist):
    fig,ax=plt.subplots(figsize=(8.2,4.0))
    it=np.array([p['iteration'] for p in hist]);pr=np.array([p['primal_residual'] for p in hist]);dr=np.array([p['dual_residual'] for p in hist]);eq=np.array([p['equality_residual'] for p in hist]);
    ax.semilogy(it,np.maximum(pr,1e-14),'k-',lw=1.35,label='Primal consistency')
    ax.semilogy(it,np.maximum(dr,1e-14),'k--',lw=1.35,label='Dual residual')
    ax.semilogy(it,np.maximum(eq,1e-14),'k:',lw=1.35,label='Affine feasibility')
    ax.axhline(1e-7,color='0.5',ls='-.',lw=1.0,label='Stopping tolerance')
    ax.set(xlabel='ADMM iteration',ylabel='Residual (log scale)',title='Dual ADMM residual convergence')
    ax.grid(True,which='major',ls=':',alpha=.25);ax.legend()
    save(fig,'Fig05_ADMM_residuals.png')


def graph_alm_obj(hist):
    fig,ax=plt.subplots(figsize=(7.0,3.7))
    it=[x['outer_iteration'] for x in hist]
    obj=[x['objective'] for x in hist]
    ax.plot(it,obj,'ko-',lw=1.4,markersize=6,label='Dual objective at each outer step')
    ax.set_xticks(it)
    ax.set(xlabel='ALM outer iteration',ylabel='Objective value',
           title='Reduced ALM: outer-iteration objective')
    ax.grid(True,ls=':',alpha=.26);ax.legend(fontsize=8,loc='best')
    save(fig,'Fig06_ALM_objective.png')


def graph_alm_residuals(hist):
    fig,ax=plt.subplots(figsize=(7.0,3.7))
    it=np.array([x['outer_iteration'] for x in hist])
    eq=np.array([x['equality_residual'] for x in hist]);pg=np.array([x['projected_gradient_residual'] for x in hist]);
    ax.semilogy(it,np.maximum(eq,1e-14),'ko-',lw=1.3,label='Equality residual')
    ax.semilogy(it,np.maximum(pg,1e-14),'ks--',lw=1.3,ms=5,label='Projected gradient')
    ax.set_xticks(it);ax.set(xlabel='ALM outer iteration',ylabel='Residual (log scale)',
                             title='Reduced ALM: feasibility and stationarity')
    ax.grid(True,which='major',ls=':',alpha=.25);ax.legend(fontsize=9)
    save(fig,'Fig07_ALM_residuals.png')


def graph_eigs(Q,S):
    q=np.maximum(0,np.linalg.eigvalsh((Q+Q.T)/2)[::-1]);s=np.maximum(0,np.linalg.eigvalsh((S+S.T)/2)[::-1]);
    q=q/max(q.max(),1e-15);s=s/max(s.max(),1e-15);
    fig,ax=plt.subplots(figsize=(8,3.8))
    i=np.arange(len(q))
    ax.plot(i+1,q,'ko-',lw=1.2,ms=4,label='Primal covariance Q')
    ax.plot(i+1,s,'ks--',lw=1.2,ms=4,label='Dual slack S')
    ax.set(xlabel='Eigenvalue index (descending)',ylabel='Eigenvalue / largest eigenvalue',
        title='Normalized PSD eigenvalue spectra')
    ax.set_xticks(np.arange(1,17,2));ax.grid(True,ls=':',alpha=.24);ax.legend()
    save(fig,'Fig08_PSD_eigen_spectra.png')


def graph_power_tradeoff(metrics):
    fig,ax=plt.subplots(figsize=(7,3.7))
    vals=[0.125,metrics['primal_objective']]
    bars=ax.bar([0,1],vals,width=0.52,color=['white','0.65'],edgecolor='black',hatch=['//','\\\\'])
    ax.set_xticks([0,1]);ax.set_xticklabels(['Uniform','SDP optimized'])
    ax.set(ylabel='Total transmit power',title='Power cost of sidelobe suppression')
    ax.set_ylim(0,0.15)
    for b,v in zip(bars,vals):ax.text(b.get_x()+b.get_width()/2,v+.002,f'{v:.6f}',ha='center',fontsize=9)
    ax.grid(axis='y',ls=':',alpha=.3)
    save(fig,'Fig09_Power_comparison.png')


def write_csv(p,rows):
    if not rows: return
    with open(p,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()));w.writeheader();w.writerows(rows)


def main():
    exp=Experiment()
    print('Solving primal SDP using hand-coded ADMM...')
    primal=primal_admm(exp)
    print('Solving dual SDP using hand-coded ADMM...')
    dual=dual_admm(exp)
    print('Solving dual SDP using reduced augmented Lagrangian...')
    alm=alm_dual(exp)
    Q=primal.solution['Q'];y=dual.solution['y']
    metrics=check_kkt(exp,Q,y)
    theta=np.linspace(-90,90,1801)
    p_opt=exp.powers(Q,theta)
    p_ref=exp.uniform_power(theta)
    mask=(np.abs(theta)>=20)
    p0max=float(p_ref[mask].max())
    p1max=float(p_opt[mask].max())
    metrics.update({'n_antennas':exp.n_antennas,'carrier_hz':exp.carrier_hz,
         'gamma':exp.sidelobe_limit,'number_sidelobe_samples':len(exp.sidelobe_angles_deg),
         'dual_admm_iterations':dual.iterations,'primal_admm_iterations':primal.iterations,
         'alm_outer_iterations':alm.iterations, 'alm_objective':alm.objective,
         'alm_eq_residual':alm.residual,
         'dense_sidelobe_max_uniform':p0max,
         'dense_sidelobe_max_optimized':p1max,
         'dense_sidelobe_reduction_percent':100*(1-p1max/p0max),
         'dense_sidelobe_reduction_dB':10*np.log10(p1max/p0max),
         'power_increase_percent':100*(metrics['primal_objective']/0.125-1)})
    # Independent quality gates, matching the precision claimed in the paper.
    assert abs(metrics['primal_objective']-metrics['dual_objective'])<3e-6,metrics
    assert metrics['max_sidelobe_violation']<2e-6,metrics
    assert metrics['max_target_violation']<2e-6,metrics
    assert metrics['min_eig_Q']>-2e-7,metrics
    assert metrics['min_eig_dual_slack']>-2e-6,metrics
    assert abs(alm.objective-dual.objective)<1e-5,metrics
    assert abs(metrics['primal_objective']-0.12633145)<3e-5,metrics
    print('Numerical checks passed:',json.dumps({k:v for k,v in metrics.items() if k not in ('active_sidelobe_multiplier_angles',)},indent=2))
    (RES/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    with open(RES/'beam_pattern.csv','w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['angle_deg','optimized_power','uniform_power'])
        for a,o,b in zip(theta,p_opt,p_ref):w.writerow([f'{a:.2f}',f'{o:.12g}',f'{b:.12g}'])
    with open(RES/'sidelobe_samples.csv','w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['angle_deg','optimized_power','limit','dual_multiplier'])
        for i,a in enumerate(exp.sidelobe_angles_deg):
            w.writerow([float(a),exp.power(Q,float(a)),exp.sidelobe_limit,float(y[i+1])])
    write_csv(RES/'primal_admm_trace.csv',primal.history)
    write_csv(RES/'dual_admm_trace.csv',dual.history)
    write_csv(RES/'alm_trace.csv',alm.history)
    np.savez_compressed(RES/'matrices.npz',Q=Q,dual_multipliers=y,dual_slack=dual.solution['S'])

    graph_array(exp);graph_workflow()
    graph_beam(theta,p_ref,p_opt,exp.sidelobe_limit,exp.sidelobe_angles_deg)
    graph_sample_constraint(exp.sidelobe_angles_deg,exp.powers(Q,exp.sidelobe_angles_deg),exp.sidelobe_limit)
    graph_admm(dual.history);graph_alm_obj(alm.history);graph_alm_residuals(alm.history)
    graph_eigs(Q,dual.solution['S']);graph_power_tradeoff(metrics)
    print('Saved 9 monochrome figures and numerical result files.')

if __name__=='__main__':main()
