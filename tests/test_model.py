import unittest
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.array_model import Experiment
from src.solvers import psd_project,negative_part

class TestMath(unittest.TestCase):
    def setUp(self):
        self.exp=Experiment()

    def test_ula_dimension_and_power(self):
        e=self.exp
        self.assertEqual(len(e.sidelobe_angles_deg),40)
        self.assertEqual(e.directional_matrix(0).shape,(16,16))
        self.assertAlmostEqual(e.power(np.eye(16)/16,0.0),1.0,places=12)
        for ang in [0.,15.,30.]:
            B=e.directional_matrix(ang)
            self.assertGreaterEqual(np.linalg.eigvalsh(B).min(),-1e-12)

    def test_psd_projection(self):
        M=np.array([[3.,2.],[2.,-2.]])
        pos=psd_project(M)
        neg=negative_part(M)
        self.assertTrue(np.allclose(pos+neg,M,rtol=1e-12,atol=1e-12))
        self.assertGreaterEqual(np.linalg.eigvalsh(pos).min(),-1e-12)
        self.assertLessEqual(np.linalg.eigvalsh(neg).max(),1e-12)
        self.assertAlmostEqual(np.sum(pos*neg),0,places=12)

    def test_reduced_alm_gradient(self):
        e=self.exp
        _,_,D,c=e.dual_data()
        rho=1.0
        rng=np.random.default_rng(2026)
        y=rng.uniform(0.02,0.04,size=len(c))
        Lam=rng.normal(scale=.03,size=(16,16));Lam=(Lam+Lam.T)/2
        def f(v):
            M=np.eye(16)-np.einsum('i,ijk->jk',v,D)-Lam/rho
            Mn=negative_part(M)
            val=-c@v+rho/2*np.sum(Mn*Mn)-np.sum(Lam*Lam)/(2*rho)
            grad=-c-rho*np.einsum('ijk,jk->i',D,Mn)
            return val,grad
        base,grad=f(y)
        eps=1e-6
        for j in [0,1,3,17,40]:
            v1=y.copy();v1[j]+=eps
            v2=y.copy();v2[j]-=eps
            fd=(f(v1)[0]-f(v2)[0])/(2*eps)
            self.assertAlmostEqual(fd,grad[j],delta=2e-6)

if __name__=='__main__':unittest.main()
