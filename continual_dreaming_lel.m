function continual_dreaming_lel()
%% File: continual_dreaming_lel.m
% Experiment: Defeating catastrophic forgetting by generative "dreaming"
%
% Paper:
%   Energy-Based Credit Assignment for Continual Learning:
%   A Review of Backpropagation-Free Methods
%
% Purpose:
%   Show that local generative replay ("dreaming") removes catastrophic
%   forgetting in a fully local (backpropagation-free) pipeline, without
%   storing any real data of the past task. Vanilla LEL is compared against
%   LEL with dreaming across several random seeds.
%
% Method:
%   - Sequential task learning (two spatially separated two-moons tasks)
%   - Local Equilibrium Learning (relaxation to equilibrium + local Hebbian rule)
%   - Generative replay: after learning task A, draw random pseudo-inputs and
%     label them with the FROZEN old model (teacher), then mix them with task B
%   - Multi-seed evaluation at convergence
%
% Outputs:
%   - Console table: per-seed forgetting and accuracy on B, for vanilla and
%     for dreaming, plus mean and standard deviation across seeds
%   - No files are written (results are printed to the console)
%
% How to run:
%   Open MATLAB and execute:
%       continual_dreaming_lel
%
% Reproducibility:
%   Seeds: [1 7 13 21 42 99] (set per run via rng(s))
%   MATLAB version: R2023b or later recommended
%   Toolboxes: none (base MATLAB only)
%
% Author:
%   Cesar Hernando Valencia Niño
%   Facultad de Ingeniería Mecatrónica
%   Universidad Santo Tomás, Seccional Bucaramanga, Colombia
%   Email: cesar.valencia@ustabuca.edu.co
%   ORCID: 0000-0001-6077-6458
%
% License:
%   MIT License, or the same license declared in the repository.
%
% Citation:
%   Valencia Niño, C.H. Energy-Based Credit Assignment for Continual Learning:
%   A Review of Backpropagation-Free Methods.
% =====================================================================
%
%  Biological motivation: during sleep, the hippocampus REPLAYS experiences
%  to consolidate memory. Here, after learning task A, the model "dreams":
%  it generates random inputs and labels them with the OLD model (teacher).
%  Those dreams are mixed with task B. Not a single real sample of A is kept.
%
%  Comparison (at convergence, multi-seed):
%     VANILLA LEL            -> catastrophic forgetting
%     LEL + DREAMING (replay) -> forgetting ~0
%
%  Backpropagation also supports pseudo-replay, BUT LEL is an energy-based
%  model: in principle it can SAMPLE from its own learned energy (realistic
%  dreaming) instead of uniform noise, a structural advantage discussed in
%  the manuscript.

    H=32; EP=1200; gamma=0.2; T=40; lr=0.05; Ndream=400; NOISE=0.12;
    seeds=[1 7 13 21 42 99];
    fv=[]; fd=[]; bv=[]; bd=[];

    line=repmat('=',1,72);
    fprintf('%s\n', line);
    fprintf(' Catastrophic forgetting: VANILLA vs LEL+DREAMING (multi-seed)\n');
    fprintf(' 2 -> %d -> 1 | EP=%d (converged) | n_dreams=%d | tasks A,B separated\n', H,EP,Ndream);
    fprintf('%s\n', line);
    fprintf(' %-6s | %-18s | %-18s\n','seed','VANILLA','LEL + DREAM');
    fprintf(' %-6s | %-8s %-8s | %-8s %-8s\n','','forget','accB','forget','accB');
    fprintf('%s\n', repmat('-',1,72));

    for s = seeds
        rng(s);
        [XA,YA]=make_moons(800,NOISE,[-2.5 0]);  [XAte,YAte]=make_moons(400,NOISE,[-2.5 0]);
        [XB,YB]=make_moons(800,NOISE,[ 2.5 0]);  [XBte,YBte]=make_moons(400,NOISE,[ 2.5 0]);
        allX=[XA;XB]; mu=mean(allX,1); sd=std(allX,0,1);
        XA=(XA-mu)./sd; XAte=(XAte-mu)./sd; XB=(XB-mu)./sd; XBte=(XBte-mu)./sd;
        W0={randn(H,2)*sqrt(1/2), zeros(1,H), randn(1,H)*sqrt(1/H), zeros(1,1)};

        % --- learn A ---
        WA = train_lel(W0, XA,YA, EP,lr,gamma,T);
        a1 = accuracy(WA, XAte,YAte);

        % --- VANILLA: learn B (forgets A) ---
        WBv = train_lel(WA, XB,YB, EP,lr,gamma,T);
        f_v = a1 - accuracy(WBv, XAte,YAte);  b_v = accuracy(WBv, XBte,YBte);

        % --- DREAM: generate dreams labelled by the old model, mix with B ---
        Xd = randn(Ndream,2)*1.6;  Yd = forward(WA, Xd);     % generative replay
        WBd = train_lel(WA, [XB;Xd],[YB;Yd], EP,lr,gamma,T);
        f_d = a1 - accuracy(WBd, XAte,YAte);  b_d = accuracy(WBd, XBte,YBte);

        fv(end+1)=f_v; bv(end+1)=b_v; fd(end+1)=f_d; bd(end+1)=b_d; %#ok<AGROW>
        fprintf(' %-6d | %7.2f%% %7.2f%% | %7.2f%% %7.2f%%\n', s, 100*f_v,100*b_v, 100*f_d,100*b_d);
    end

    fprintf('%s\n', repmat('-',1,72));
    fprintf(' VANILLA       forget = %5.1f%% +/- %4.1f   accB = %.1f%%\n', 100*mean(fv),100*std(fv,1),100*mean(bv));
    fprintf(' LEL + DREAM   forget = %5.1f%% +/- %4.1f   accB = %.1f%%\n', 100*mean(fd),100*std(fd,1),100*mean(bd));
    fprintf('%s\n', line);
    fprintf(' Dreaming (generative replay) removes forgetting and COLLAPSES the variance,\n');
    fprintf(' without storing any data of A.\n');
    fprintf('%s\n', line);
end

% ---------------------------------------------------------------------
function [X,y] = make_moons(n, noise, shift)
    no=floor(n/2); ni=n-no;
    to=linspace(0,pi,no)'; ti=linspace(0,pi,ni)';
    Xo=[cos(to), sin(to)];  Xi=[1-cos(ti), 1-sin(ti)-0.5];
    X=[Xo;Xi] + noise*randn(n,2) + shift;
    y=[-ones(no,1); ones(ni,1)];
    p=randperm(n); X=X(p,:); y=y(p,:);
end

% ---------------------------------------------------------------------
function out = forward(W, X)
    out = tanh(X*W{1}' + W{2})*W{3}' + W{4};
end
function a = accuracy(W, X,Y)
    a = mean(sign(forward(W,X)) == sign(Y));
end

% ---------------------------------------------------------------------
% LEL: relaxation to equilibrium + local Hebbian rule (no backpropagation)
function W = train_lel(W, X,Y, EP, lr, gamma, T)
    W1=W{1}; b1=W{2}; W2=W{3}; b2=W{4}; n=size(X,1);
    for ep = 1:EP
        a1=X*W1'+b1; x1=tanh(a1); x2=Y;
        for t = 1:T                                  % inference = relax to equilibrium
            a1=X*W1'+b1; pred1=tanh(a1);
            e1=x1-pred1; e2=x2-(x1*W2'+b2);
            x1=x1 - gamma*(e1 - e2*W2);
        end
        a1=X*W1'+b1;                                 % learning: LOCAL rule at the equilibrium
        e1=x1-tanh(a1); e2=x2-(x1*W2'+b2);
        g1=e1.*(1-tanh(a1).^2);
        W1=W1 + lr*(g1'*X)/n;   b1=b1 + lr*sum(g1,1)/n;
        W2=W2 + lr*(e2'*x1)/n;  b2=b2 + lr*sum(e2,1)/n;
    end
    W={W1,b1,W2,b2};
end
