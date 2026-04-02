---
layout: distill
title: TOPOS \\ Topological Optimal-transport Partitioned Operator Solver
description: Neural operators have emerged as a powerful paradigm in scientific machine learning, enabling resolution-invariant mappings across infinite-dimensional function spaces for applications including weather forecasting, fluid dynamics simulation and structural analysis. Despite their success on structured grids, these architectures struggle with arbitrary 3D geometries requiring retraining for each new shape and violating the need for zero-shot generalisation in real-world tasks. We introduce TOPOS (Topological Optimal-transport Partitioned Operational System), a unified framework that standardises irregular physical domains into topology-aware latent workbenches using instance-dependent optimal transport mappings and genus-based routing. For a given input mesh with density $\mu$, TOPOS computes a diffeomorphic transport $T$ to a uniform reference $\nu$ on a sphere $(g=0)$ or torus $(g=1)$ workbench, applies a spectral neural operator as a solver and decodes solutions back via inverse transport. This four-stage pipeline ensures topological integrity, discretisation invariance and computational speedups through three-dimensional to two-dimensional manifold reduction. TOPOS thus provides a universal physics engine, learning geometry-agnostic physical laws once and deploying them zero-shot across diverse topologies.
date: 2026-04-02
future: true
htmlwidgets: true
hidden: true

# Mermaid diagrams
mermaid:
  enabled: true
  zoomable: true

# Anonymize when submitting
authors:
  - name: Anonymous

# authors:
#   - name: Mamta Saini
#     url: "https://en.wikipedia.org/wiki/Albert_Einstein"
#     affiliations:
#       name: IAS, Princeton

# must be the exact same name as your blogpost
bibliography: topos.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: Equations
  - name: Images and Figures
    subsections:
      - name: Interactive Figures
  - name: Citations
  - name: Footnotes
  - name: Code Blocks
  - name: Diagrams
  - name: Tweets
  - name: Layouts
  - name: Other Typography?

# Below is an example of injecting additional post-specific styles.
# This is used in the 'Layouts' section of this post.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }
---

## Introduction

Solving partial differential equations (PDEs) on intricate 3D geometric domains continues to pose a core difficulty in computational science. Traditional finite element and spectral solvers perform well on regular geometries but falter on complex shapes, where costly meshing procedures demand repeated refinement cycles. This geometric complexity hinders progress in fields like fluid dynamics, structural engineering, and geophysics. The issue is stark in automotive aerodynamics, where simulating a single vehicle design can consume over 300 CPU-hours \citep{elrefaie2024drivaernetparametriccardataset} or 10 GPU-hours \citep{GINO}.

Deep learning offers compelling alternatives for tackling PDEs on irregular domains, delivering substantial speedups over classical methods \citep{bhatnagar2019prediction, pfaff2020learning, thuerey2020deep, hennigh2021nvidia}. These techniques thrive at reduced resolutions, slashing runtime demands. Yet, many remain tied to fixed mesh resolutions, curbing their versatility. Neural operators overcome this by providing discretization-agnostic PDE solvers, marking a pivotal advance in the field.

Neural operators for irregular geometries. These models learn PDE solution operators directly from data in a mesh-agnostic fashion \citep{FNO, kovachki2023neural, lu2021learning}. Discretization invariance sets them apart from standard neural networks, ideal for PDE tasks. Cutting-edge developments target complex geometries by projecting them into canonical latent spaces amenable to spectral tools like the Fast Fourier Transform (FFT) \citep{Geo-FNO,yin2024dimon,ahmad2024diffeomorphiclatentneuraloperators}. The Geometry-Aware Fourier Neural Operator (Geo-FNO) \citep{Geo-FNO} pioneered diffeomorphic mappings from physical domains to uniform grids, unlocking FFT acceleration in latent coordinates. Still, Geo-FNO relies on class-shared deformations that overlook instance-specific traits and uses matrix-heavy Fourier operations, impeding 3D scalability. The Geometry-Informed Neural Operator (GINO) \citep{Geo-FNO} fused Graph Neural Operators (GNO) \citep{GINO} with FNO \citep{li2020neural}, blending graph locality for meshes with FFT globalism to conquer high-Reynolds 3D flows. Its graph-centric design, however, inherits locality biases and 3D memory overheads, amplified at scale. Parallel efforts employ transformer tokens \citep{hao2023gnot, wu2024transolver, alkin2024universal} or implicit representations \citep{yin2022continuous, serrano2023operator, chen2023implicit, chen2022crom} for geometry encoding, versatile yet prone to losing metric fidelity and lacking invertibility for inverse design tasks. Prevailing neural operators thus wrestle with 3D costs and cross-geometry transfer. We counter this by recasting geometry embedding as per-instance optimal transport, enabling tailored manifold operators that redefine complex shape handling. OTNO \citep{li2025geometric} embeds meshes as densities using per-instance optimal transport,
\begin{align}
    T = \arg\min_{T_\sharp \mu = \nu} \int \|x - T(x)\|^2 d\mu(x),
\end{align} yielding 5x speedups \citep{li2025geometric}. Yet, shared maps ignore instance topology (genus $g$), graphs suffer locality, and 3D latents burden scaling.

% Optimal transport for geometry encoding. Optimal transport offers a precise lens for minimal-cost density transformations. We recast surface meshes as density measures capturing curvature and complexity, solving for transport maps to uniform latent densities. This sidesteps interpolation artifacts like clustering, delivering smooth, metric-preserving diffeomorphisms akin to r-adaptive meshing \citep{budd2015geometry} but topology-agnostic. Sinkhorn approximations \citep{cuturi2013sinkhorn} render it computationally viable. Our framework embeds mesh submanifolds into latent spaces with intact geometry.

\begin{figure*}[t]
    \centering
    \includegraphics[width=\textwidth]{topos_image.png}
    \caption{The TOPOS Architecture. A four-stage pipeline that utilizes an Optimal Transport (OT) encoder to standardize irregular 3D meshes, a topological router to compute the Euler characteristic ($\chi$), and a Spectral Neural Operator (FNO) to solve PDEs in a structured latent domain before decoding the solution back to the physical mesh.}
    \label{fig:topos_architecture}
\end{figure*}

TOPOS bridges this by understanding \textit{messy} geometries to \textit{clean} spectral workbenches using a four-stage pipeline. 

\textbf{Stage 1 (OT-Encoder)} computes instance-dependent $T$ pushing mesh density $\mu$ on $\Omega$ to uniform $\nu$ on latent $\hat{\Omega}$, preserving metrics diffeomorphically-unlike Geo-FNO's shared maps. 

\textbf{Stage 2 (Topological Router)} selects workbench via Euler characteristic i.e., $\chi = V-E+F = 2-2g$ (sphere for $g=0$, torus for $g=1$), averting tears. 
% \begin{align*}
    % \chi = V-E+F = 2-2g
% \end{align*} (sphere for $g=0$, torus for $g=1$), averting tears. 

\textbf{Stage 3 (Latent Solver)} deploys FNO on the grid for global physics i.e., $G(u) = \mathcal{F}^{-1}(R_\phi \cdot \mathcal{F}(Wu+b))$
% \begin{align*}
    % G(u) = \mathcal{F}^{-1}(R_\phi \cdot \mathcal{F}(Wu+b))
% \end{align*}   

\textbf{Stage 4 (Decoder)} pulls back via $T^{-1}$ with soft rasterization. This ensures zero-shot super-resolution, topological integrity, and $O(N\log N)$ 2D efficiency vs. $O(N^3)$ 3D (from OTNO), learning physics once for unseen shapes of matching genus.

## Methodology

We formalize TOPOS as a neural operator $\mathcal{G}_\theta: f \mapsto u$ that approximates solutions to parametric PDEs \begin{align}
    \mathcal{N}[u] = f
\end{align} defined on irregular domains $\Omega \subset \mathbb{R}^3$, represented by meshes $\mathcal{M} = (V, E, F)$ with associated density measures $\mu$. The architecture decomposes this operator learning problem into a principled four-stage pipeline that systematically transforms unstructured physical geometries into structured latent representations amenable to efficient spectral methods, then faithfully reconstructs solutions on the original mesh (Fig.~\ref{fig:topos_architecture}).

In the first stage, the OT-Encoder addresses the fundamental mismatch between arbitrary mesh topologies and the uniform grid requirements of spectral neural operators. Treating the input mesh as an empirical measure 
\begin{align*}
    \mu = \frac{1}{|V|} \sum_{v_i \in V} \delta_{v_i},
\end{align*}
we solve the classical Monge optimal transport problem:
\begin{equation}
\label{eq:ot}
T = \arg\min_{T_\sharp \mu = \nu} \int_\Omega c(x, T(x)) \, d\mu(x), \quad c(x,y) = \|x - y\|^2_2
\end{equation}
where $\nu$ denotes a uniform reference density on a candidate latent workbench $\hat{\Omega}$ (such as a unit sphere or periodic grid). This formulation, solved via entropic regularization with Sinkhorn iterations~\citep{cuturi2013sinkhorn}, yields a diffeomorphic transport map $T: \Omega \to \hat{\Omega}$. Input and solution fields are subsequently pushed forward as $\tilde{f} = f \circ T^{-1}$ and $\tilde{u} = u \circ T^{-1}$. This stage is essential to standardize vertex counts and spacing across diverse geometries; intuitively, it rearranges an irregular ``sandpile'' of mesh points into a perfectly uniform ``sandbox'' while uniquely preserving the shape's geometric and physical structure-unlike Geo-FNO's class-shared deformations \citep{li2023fourier}.

The second stage, the Topological Router, ensures that the chosen workbench respects the input geometry's intrinsic topology, preventing non-diffeomorphic mappings that cause gradient pathologies. We compute the Euler characteristic
\begin{equation}
\label{eq:euler}
\chi(\mathcal{M}) = |V| - |E| + |F| = 2 - 2g
\end{equation}
to determine the genus $g$, then route to topology-compatible domains: a sphere ($S^2$ with spherical harmonics) for $g=0$ (closed surfaces), a torus or periodic grid (enabling FFT) for $g=1$ (objects with holes), or a cubic lattice for volumetric cases. A lightweight learnable selector $r_\psi(\chi, \mathcal{M})$ adaptively weights these candidates during training. This routing is critical for maintaining continuous gradients and topological fidelity; by matching ``connectivity holes,'' it avoids the tearing artifacts absent in prior work like OTNO~\citep{li2025geometric}, particularly for high-genus structures such as pipes or cavities.

Once standardized and topologically aligned, Stage 3-the Latent Solver-deploys a resolution-invariant spectral operator on the selected $\hat{\Omega}_g$. We instantiate this with the Fourier Neural Operator over $K$ layers:
\begin{align}
\label{eq:fno}
G_k(v_{k-1}) &= \sigma \left( \mathcal{F}^{-1} \big( R_{\phi_k} \cdot \mathcal{F}(v_{k-1}) \big) + W_k v_{k-1} \right), \\
v_K &= \Pi \big( G_K(\ell_0(\tilde{f})) \big)
\end{align}
where $R_{\phi_k}$ are learnable spectral kernels, $\sigma$ is GELU, and $\Pi$ projects to solution space~\citep{FNO}. An optional physics-informed loss \begin{align}
    \mathcal{L}_\text{PDE} = \|\mathcal{N}[\hat{u}] - \tilde{f}\|^2_\Omega
\end{align} 
enforces consistency. This design scalably captures nonlocal physics via $O(N \log N)$ FFT operations; conceptually, it distills ``pure physical laws'' (e.g., Navier-Stokes dynamics) into frequency patterns on an idealized grid, enabling shape-independent generalization.

Finally, the Decoder in Stage 4 reprojects the latent solution $\hat{u} \in \hat{\Omega}_g$ back to the physical domain via the inverse transport: $u_\text{phys} = \hat{u} \circ T$. In practice, we implement this pullback through differentiable soft rasterization:
\begin{equation}
\begin{aligned}
\label{eq:decoder}
u_\text{phys}(x_i) & = \sum_{j \in \hat{\Omega}_g} \hat{u}(T(x_i)) \cdot w_{ij} \\ w_{ij} & = \operatorname{\texttt{softmax}} \big( \|T(x_i) - y_j\|^2 \big)
\end{aligned}
\end{equation}
or bipartite graph interpolation, yielding the full operator \begin{align}
    \mathcal{G}_\theta(f) = \text{Dec}_\theta(\text{Solve}_g(\text{Enc}_\theta(f))).
\end{align}
This step delivers interpretable fields on the original mesh; akin to snapping a stretched rubber sheet back to its natural form, it seamlessly transfers physics-informed predictions without distortion.

The full model trains end-to-end by minimizing 
\begin{align}
    \mathcal{L} = \|u - \mathcal{G}_\theta(f)\|^2 + \lambda \mathcal{L}_\text{PDE},
\end{align}
achieving zero-shot generalization: physics learned on one low-resolution shape deploys instantly to unseen high-resolution instances of matching genus.

\begin{table}[!h]
\centering
\scriptsize
\begin{tabular}{p{0.6cm} p{7cm}}
\toprule
\textbf{Symbol} & \textbf{Meaning} \\
\midrule

$f$ & Input field or source term (e.g., forcing, coefficients, boundary/initial conditions) \\
$u$ & Ground-truth PDE solution field corresponding to $f$ \\
$\tilde{u}$ & Predicted solution field in latent domain (pulled back to physical space via $T$) \\

$\mathcal{N}$ & Differential operator defining the governing PDE (e.g., Navier–Stokes, diffusion) \\
$\mathcal{G}_\theta$ & Neural operator mapping input field $f$ to solution field $u$ \\
$\mathcal{L}$ & Total training loss (data misfit plus optional physics-informed term) \\
$\mathcal{L}_\text{PDE}$ & Physics-informed loss enforcing PDE residual consistency \\
$T$ & Diffeomorphic optimal-transport map from physical domain $\Omega$ to latent workbench $\hat{\Omega}$ \\
$T^{-1}$ & Inverse transport map used to decode latent solutions back to the physical mesh \\
$\mu$ & Input mesh density measure on physical domain $\Omega$ \\
$\nu$ & Uniform reference density on latent workbench $\hat{\Omega}$ (sphere/torus/grid) \\

$\Omega$ & Irregular physical domain in $\mathbb{R}^3$ \\
$\hat{\Omega}$ & Latent workbench domain (e.g., sphere or torus) after optimal transport \\
$\mathcal{M} = (V,E,F)$ & Mesh representation: vertices $V$, edges $E$, faces $F$ \\
$V,E,F$ & Vertex, edge, and face sets of the mesh, respectively \\
$g$ & Genus (number of handles/holes) of the input geometry \\
$\chi$ & Euler characteristic used to infer genus and route to topology-compatible workbench \\
$\chi(\mathcal{M})$ & Euler characteristic of mesh $\mathcal{M}$, given by $|V|-|E|+|F|$ \\

$\tilde{f}$ & Input field $f$ pushed forward to latent domain via $T^{-1}$ \\
$\hat{u}$ & Latent-domain solution field before decoding to physical mesh \\
$u_\text{phys}$ & Decoded solution field on the original physical mesh \\
$c(x,y)$ & Optimal-transport cost function, here squared Euclidean distance $\|x-y\|_2^2$ \\
$w_{ij}$ & Soft rasterization weights used for differentiable decoding/interpolation \\
$r_\psi$ & Learnable topological router selecting appropriate latent workbench \\
$\theta$ & Trainable parameters of the overall neural operator (encoder, solver, decoder) \\

$G$ & Generic neural operator or layer mapping latent fields (e.g., FNO layer) \\
$G_k$ & $k$-th FNO layer in the latent solver stack \\
$v_{k-1}, v_K$ & Latent feature fields before layer $k$ and after the final layer $K$ \\
$R_{\phi_k}$ & Learnable spectral kernel parameters in Fourier space for layer $k$ \\
$W, W_k$ & Linear (pointwise) operators applied in physical or latent space \\
$b$ & Bias term in the affine transformation within the spectral operator \\
$\phi_k$ & Parameters associated with spectral kernel $R_{\phi_k}$ in layer $k$ \\
$\Pi$ & Projection head mapping final latent features to solution space \\
$\sigma$ & Pointwise nonlinearity (e.g., GELU activation) in the FNO layers \\

$\mathcal{F}, \mathcal{F}^{-1}$ & Fourier transform and inverse Fourier transform operators \\
$N$ & Number of degrees of freedom (e.g., grid points) in the discretization \\
\bottomrule
\end{tabular}
\end{table}
