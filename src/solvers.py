"""Hand-coded primal/dual ADMM and reduced augmented-Lagrangian solvers.

The primal SDP is:
  min_Q,s tr(Q)
  s.t. <B0,Q>-s0=g,  <Bk,Q>+sk=gamma,
       Q>=0 (PSD), s>=0.
The dual is:
  max_y c.T@y
  s.t. D(y)+S=I, y>=0, S>=0 (PSD),
  D(y)=sum_i y_i D_i, D0=B0, Dk=-Bk.

All matrix inner products are Frobenius products.  The implementations
use eigendecomposition for the PSD cone and exact affine projections.
"""
from dataclasses import dataclass
from typing import List, Dict
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve
from .array_model import Experiment


def sym(A):
    return 0.5 * (A + A.T)


def psd_project(A: np.ndarray) -> np.ndarray:
    eigval, eigvec = np.linalg.eigh(sym(A))
    return sym((eigvec * np.maximum(eigval, 0)) @ eigvec.T)


def negative_part(A: np.ndarray) -> np.ndarray:
    eigval, eigvec = np.linalg.eigh(sym(A))
    return sym((eigvec * np.minimum(eigval, 0)) @ eigvec.T)


@dataclass
class SolverResult:
    objective: float
    iterations: int
    solution: object
    history: List[Dict[str, float]]
    residual: float


def primal_admm(exp: Experiment, rho=1.0, tol=1e-7, max_iter=25000,
                log_every=20) -> SolverResult:
    B0, Bs, _, _ = exp.dual_data()
    B = np.concatenate([B0[None], Bs], axis=0)
    m, n, _ = B.shape
    F = B.copy()
    F[0] = -B0 # not used for A: sign only on slack vector
    slack_sign = np.r_[-1., np.ones(m-1)]
    target = np.r_[exp.desired_gain, np.repeat(exp.sidelobe_limit, m-1)]
    gram = np.einsum('ijk,ljk->il', B, B) + np.eye(m)
    gram_chol = cho_factor(gram, check_finite=False)
    # Gram matrix is SPD, thanks to the slack-variable identity.

    def A(Q, slack):
        return np.einsum('ijk,jk->i', B, Q) + slack_sign * slack

    def AT(y):
        return np.einsum('i,ijk->jk', y, B), slack_sign * y

    ZQ = np.eye(n) * 0.2
    Zs = np.ones(m) * 0.1
    UQ = np.zeros((n, n)); Us = np.zeros(m)
    trace_C = np.eye(n)
    hist = []
    for it in range(1, max_iter+1):
        VQ = ZQ-UQ-trace_C/rho
        Vs = Zs-Us
        lam = cho_solve(gram_chol, target-A(VQ,Vs),check_finite=False)
        Aq, As = AT(lam)
        XQ, Xs = VQ+Aq, Vs+As
        prevQ, prevs = ZQ, Zs
        ZQ = psd_project(XQ+UQ)
        Zs = np.maximum(Xs+Us,0.0)
        UQ += XQ-ZQ
        Us += Xs-Zs
        rp = float(np.hypot(np.linalg.norm(XQ-ZQ), np.linalg.norm(Xs-Zs)))
        rd = float(rho * np.hypot(np.linalg.norm(ZQ-prevQ), np.linalg.norm(Zs-prevs)))
        eq = float(np.linalg.norm(A(ZQ,Zs)-target))
        if it==1 or it % log_every==0 or (rp<tol and rd<tol and eq<tol):
            hist.append({'iteration':int(it),'objective':float(np.trace(ZQ)),
                         'primal_residual':rp,'dual_residual':rd,'equality_residual':eq})
        if rp<tol and rd<tol and eq<tol:
            break
    else:
        raise RuntimeError('Primal ADMM did not converge: increase max_iter')
    return SolverResult(float(np.trace(ZQ)), it, {'Q':ZQ,'slacks':Zs},hist,eq)


def dual_admm(exp: Experiment, rho=1.0, tol=1e-7,
              max_iter=25000,log_every=20) -> SolverResult:
    _, _, D, c = exp.dual_data()
    m,n,_ = D.shape
    I = np.eye(n)
    gram = np.einsum('ijk,ljk->il', D,D)
    # Affine projector solves (I+gram)*t=r.
    H = np.eye(m)+gram
    H_chol=cho_factor(H, check_finite=False)
    z = np.zeros(m); T=0.1*I
    uy = np.zeros(m); US=np.zeros((n,n))
    hist = []

    def A(y,S):
        return np.einsum('i,ijk->jk',y,D)+S

    for it in range(1,max_iter+1):
        Vy=z-uy+c/rho
        VS=T-US
        R=I-A(Vy,VS)
        t=cho_solve(H_chol,np.einsum('ijk,jk->i',D,R),check_finite=False)
        Y=R-np.einsum('i,ijk->jk',t,D)
        y_aff=Vy+t; S_aff=VS+Y
        prev_z,prev_T=z,T
        z=np.maximum(y_aff+uy,0.0)
        T=psd_project(S_aff+US)
        uy += y_aff-z
        US += S_aff-T
        rp=float(np.hypot(np.linalg.norm(y_aff-z),np.linalg.norm(S_aff-T)))
        rd=float(rho*np.hypot(np.linalg.norm(z-prev_z),np.linalg.norm(T-prev_T)))
        eq=float(np.linalg.norm(A(z,T)-I))
        if it==1 or it%log_every==0 or (rp<tol and rd<tol and eq<tol):
            hist.append({'iteration':int(it),'objective':float(c@z),
                         'primal_residual':rp,'dual_residual':rd,'equality_residual':eq})
        if rp<tol and rd<tol and eq<tol:
            break
    else:
        raise RuntimeError('Dual ADMM did not converge: increase max_iter')
    return SolverResult(float(c@z),it, {'y':z,'S':T},hist,eq)


def alm_dual(exp:Experiment,rho=1.,tol=1e-8,max_outer=30) -> SolverResult:
    _,_,D,c=exp.dual_data()
    m,n,_=D.shape
    I=np.eye(n)
    Lam=np.zeros((n,n)); y=np.zeros(m)
    hist=[]
    for k in range(max_outer):
        def smooth_obj_grad(v):
            M=I-np.einsum('i,ijk->jk',v,D)-Lam/rho
            Mneg=negative_part(M)
            value=-float(c@v)+(rho/2)*float(np.sum(Mneg*Mneg))-np.sum(Lam*Lam)/(2*rho)
            grad=-c-rho*np.einsum('ijk,jk->i',D,Mneg)
            return value,grad
        result=minimize(smooth_obj_grad,y,method='L-BFGS-B',jac=True,
                        bounds=[(0,None)]*m,
                        options={'maxiter':12000,'ftol':1e-14,'gtol':1e-10,'maxls':60})
        y=result.x.copy()
        P=I-np.einsum('i,ijk->jk',y,D)
        S=psd_project(P-Lam/rho)
        eqmat=S-P
        Lam += rho*eqmat
        _,grad=smooth_obj_grad(y)
        pg=float(np.linalg.norm(y-np.maximum(y-grad,0)))
        residual=float(np.linalg.norm(eqmat))
        hist.append({'outer_iteration':k, 'objective':float(c@y),
                     'equality_residual':residual, 'projected_gradient_residual':pg,
                     'inner_iterations':int(result.nit)})
        if residual<tol and pg<1e-7:
            break
    else:
        raise RuntimeError('ALM did not converge: increase max_outer')
    return SolverResult(float(c@y),len(hist),{'y':y,'S':S,'Lambda':Lam},hist,residual)


def check_kkt(exp:Experiment, Q:np.ndarray, y:np.ndarray) -> dict:
    B0,Bs,D,c=exp.dual_data()
    S=np.eye(exp.signal_dim)-np.einsum('i,ijk->jk',y,D)
    target_power=float(np.sum(B0*Q))
    side_powers=np.einsum('ijk,jk->i',Bs,Q)
    qtrace=float(np.trace(Q))
    dual_obj=float(c@y)
    return {'primal_objective':qtrace,'dual_objective':dual_obj,
        'primal_dual_gap':qtrace-dual_obj,'target_gain':target_power,
        'maximum_sampled_sidelobe':float(max(side_powers)),
        'max_target_violation':float(max(0,exp.desired_gain-target_power)),
        'max_sidelobe_violation':float(max(0,max(side_powers)-exp.sidelobe_limit)),
        'min_eig_Q':float(min(np.linalg.eigvalsh(sym(Q)))),
        'min_eig_dual_slack':float(min(np.linalg.eigvalsh(sym(S)))),
        'complementary_inner_product':float(np.sum(Q*S)),
        'dual_multiplier_min':float(min(y)),
        'target_dual_multiplier':float(y[0]),
        'active_sidelobe_multiplier_angles':[(float(angle),float(y[k+1]))
            for k,angle in enumerate(exp.sidelobe_angles_deg) if y[k+1]>1e-6]}
