# Linearized Wasserstein dictionary learning


This repository contains preliminary algorithms for Wasserstein dictionary learning (in the sense of  <a href="https://epubs.siam.org/doi/10.1137/17M1140431">[Schmitz et al.]</a>) in the Linearized Optimal Transport (LOT) framework (in the sense of <a href="https://www.imagedatascience.com/wang_ijcv_13.pdf">[Wang et al.]</a>).

Let $\rho$ be a reference absolutely continuous probability measure, supported on a compact convex set for simplicity (say $\rho$ is the Lebesgue measure on $[0, 1]^d$).

For a dataset of probability measures $(\mu_i)_{1 \leq i \leq N} \in P_2(R^d)$, compute their LOT embeddings as the optimal transport map from $\rho$ to themselves:
$$\forall i \in \{1, \dots, N\}, \quad T_{\mu_i} := \arg \min_{T \vert T_\# \rho = \mu_i} \int_{[0,1]^d} ||T(x) - x||^2 d \rho(x).$$

Vecotrize each transport maps $T_{\mu_i}$ as follows: