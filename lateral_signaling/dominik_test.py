import scipy.stats as ss
import scipy.integrate as si
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

pdf_gen = lambda x, A0, x0, s0, A1, x1, s1: A0*np.exp(-(x-x0)**2/(2*s0**2))+A1*np.exp(-(x-x1)**2/(2*s1**2))
pdf1 = lambda x: pdf_gen(x, 1., 0, 1., 0., 1, 1.)
pdf2 = lambda x: pdf_gen(x, 0.5, 0, 1., 0.5, 1, 1.)
norm1, _ = si.quad(pdf1, -np.inf, np.inf)
norm2, _ = si.quad(pdf2, -np.inf, np.inf)

class dist1_class(ss.rv_continuous):
    def _pdf(self, x):
        return pdf1(x)/norm1
class dist2_class(ss.rv_continuous):
    def _pdf(self, x):
        return pdf2(x)/norm2

X = np.linspace(-3,3, 50)
dist1 = dist1_class()
dist2 = dist2_class()
Y1 = dist1.pdf(x=X)
Y2 = dist2.pdf(x=X)

#sample_size_space = np.geomspace(10, 1e4, 5)
sample_size_space = np.geomspace(10, 1e4, 4)

plt.figure(figsize = (4,len(sample_size_space)*3))
for i, sample_size in enumerate(tqdm(sample_size_space)):
    plt.subplot(len(sample_size_space),2,1+2*i)
    sample1 = dist1.rvs(size=int(sample_size))
    sample2 = dist2.rvs(size=int(sample_size))
    _, p = ss.ks_2samp(sample1, sample2)
    plt.ylabel(f'N={int(sample_size)}, p={p:.3e}')
    plt.plot(X, Y1, 'C0', label = 'PDF 1')
    plt.hist(sample1, bins=30, density=True, alpha=.5, color='C0')
    plt.subplot(len(sample_size_space),2,2+2*i)
    plt.plot(X, Y2, 'C1', label = 'PDF 2')
    plt.hist(sample2, bins=30, density=True, alpha=.5, color='C1')
plt.tight_layout()
plt.savefig('big_difference.png')
