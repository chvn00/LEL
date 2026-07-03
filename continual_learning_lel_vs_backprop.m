function continual_learning_lel_vs_backprop()
%% File: continual_learning_lel_vs_backprop.m
% Experiment: Continual learning baseline, LEL vs. backpropagation
%
% Paper:
%   Backpropagation-Free Continual Learning:
%   A Unified Energy-Based Framework, Reproducible Benchmark, and Open Challenges
%
% Purpose:
%   Evaluate catastrophic forgetting in a sequential learning setting and
%   compare unprotected Local Equilibrium Learning (LEL) against a matched
%   backpropagation baseline. Task A is learned first, then task B is learned
%   without revisiting A; the script reports how much each method forgets of A.
%
% Method:
%   - Sequential task learning (two spatially separated two-moons tasks:
%     task A centred at x = -2.5, task B centred at x = +2.5)
%   - Local Equilibrium Learning (relaxation to equilibrium + local Hebbian rule)
%   - Matched backpropagation baseline (same initial weights and data)
%   - Fairness control: results are reported both at few epochs and at
%     convergence, because low forgetting at few epochs is only undertraining
%     on B, not real retention
%
% Outputs:
%   - Console table with, for each method and regime:
%       accuracy on A after learning A, accuracy on A after learning B,
%       forgetting of A, and accuracy on B
%   - No files are written (results are printed to the console)
%
% How to run:
%   Open MATLAB and execute:
%       continual_learning_lel_vs_backprop
%
% Reproducibility:
%   Seed: 7 (set via rng(7))
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
%   Valencia Niño, C.H. Backpropagation-Free Continual Learning:
%   A Unified Energy-Based Framework, Reproducible Benchmark, and Open Challenges.
% =====================================================================
%
%  Question: when learning task A and then task B WITHOUT revisiting A,
%  how much does each method FORGET of A? (catastrophic forgetting)
%
%  Benchmark (domain-incremental, NOT adversarial):
%     Task A = two-moons shifted to the left  (centre x = -2.5)
%     Task B = two-moons shifted to the right (centre x = +2.5)
%  The two tasks occupy different regions, so a shared network COULD retain
%  both; forgetting here is therefore informative.
%
%  Fairness control (key): we measure both at "few epochs" and "converged".
%  Low forgetting at few epochs is only UNDERTRAINING on B, not real
%  retention. The honest question is the forgetting AT CONVERGENCE on B.
%
%  Finding (see the discussion in the manuscript): at convergence, BOTH
%  methods forget catastrophically (~37%). Vanilla LEL does NOT beat
%  backpropagation, which is consistent with Theorem 2 (equivalence at the
%  equilibrium).

    rng(7);
    H = 32;  NTR = 800;  NTE = 400;  NOISE = 0.12;

    % ---- Spatially separated tasks ----
    [XA,YA]     = make_moons(NTR,NOISE,[-2.5 0]);
    [XAte,YAte] = make_moons(NTE,NOISE,[-2.5 0]);
    [XB,YB]     = make_moons(NTR,NOISE,[ 2.5 0]);
    [XBte,YBte] = make_moons(NTE,NOISE,[ 2.5 0]);

    % ---- Standardise with pooled statistics (identical for both methods) ----
    allX = [XA; XB];  mu = mean(allX,1);  sd = std(allX,0,1);
    XA=(XA-mu)./sd; XAte=(XAte-mu)./sd; XB=(XB-mu)./sd; XBte=(XBte-mu)./sd;

    % ---- Shared initial weights ----
    W0 = {randn(H,2)*sqrt(1/2), zeros(1,H), randn(1,H)*sqrt(1/H), zeros(1,1)};

    line = repmat('=',1,76);
    fprintf('%s\n', line);
    fprintf(' Continual learning (A -> B without revisiting A) | separated tasks\n');
    fprintf(' architecture 2 -> %d -> 1 | same init and data for both methods\n', H);
    fprintf('%s\n', line);
    fprintf(' %-22s | %-9s | %-9s | %-9s | %-9s\n','Method / regime','acc A','acc A','FORGET','acc B');
    fprintf(' %-22s | %-9s | %-9s | %-9s | %-9s\n','','after A','after B','of A','after B');
    fprintf('%s\n', repmat('-',1,76));

    % --- Backprop: few epochs vs converged ---
    run_block('BACKPROP  (400 ep)',  @() chain_bp (W0, XA,YA,XB,YB, 400,  0.2), XAte,YAte,XBte,YBte);
    run_block('BACKPROP  (conv.)',   @() chain_bp (W0, XA,YA,XB,YB, 1500, 0.2), XAte,YAte,XBte,YBte);
    fprintf('%s\n', repmat('-',1,76));
    % --- LEL: few epochs vs converged ---
    run_block('LEL       (400 ep)',  @() chain_lel(W0, XA,YA,XB,YB, 400,  0.05,0.2,60), XAte,YAte,XBte,YBte);
    run_block('LEL       (conv.)',   @() chain_lel(W0, XA,YA,XB,YB, 1500, 0.05,0.2,60), XAte,YAte,XBte,YBte);

    fprintf('%s\n', repmat('-',1,76));
    fprintf(' Honest reading: the low FORGET at 400 ep is UNDERTRAINING on B.\n');
    fprintf(' At CONVERGENCE both forget ~catastrophically -> vanilla LEL does NOT beat\n');
    fprintf(' backpropagation (consistent with Theorem 2). Open challenge: local\n');
    fprintf(' consolidation that respects the contractive condition of the equilibrium.\n');
    fprintf('%s\n', line);
end

% ---------------------------------------------------------------------
function run_block(name, trainfun, XAte,YAte,XBte,YBte)
    W = trainfun();                 % chain_* returns {W1,b1,W2,b2, accA_afterA}
    accA1 = W{5};
    accA2 = accuracy(W{1},W{2},W{3},W{4}, XAte,YAte);
    accB  = accuracy(W{1},W{2},W{3},W{4}, XBte,YBte);
    fprintf(' %-22s | %7.2f%% | %7.2f%% | %7.2f%% | %7.2f%%\n', ...
        name, 100*accA1, 100*accA2, 100*(accA1-accA2), 100*accB);
end

% ---------------------------------------------------------------------
function out = chain_bp(W0, XA,YA,XB,YB, EP, lr)
    [W1,b1,W2,b2] = train_backprop(W0{1},W0{2},W0{3},W0{4}, XA,YA, EP, lr);
    accA_afterA   = accuracy(W1,b1,W2,b2, XA,YA);
    [W1,b1,W2,b2] = train_backprop(W1,b1,W2,b2, XB,YB, EP, lr);
    out = {W1,b1,W2,b2, accA_afterA};
end

function out = chain_lel(W0, XA,YA,XB,YB, EP, lr,gamma,T)
    [W1,b1,W2,b2] = train_lel(W0{1},W0{2},W0{3},W0{4}, XA,YA, EP, lr,gamma,T);
    accA_afterA   = accuracy(W1,b1,W2,b2, XA,YA);
    [W1,b1,W2,b2] = train_lel(W1,b1,W2,b2, XB,YB, EP, lr,gamma,T);
    out = {W1,b1,W2,b2, accA_afterA};
end

% ---------------------------------------------------------------------
function [X,y] = make_moons(n, noise, shift)
    no = floor(n/2);  ni = n - no;
    to = linspace(0,pi,no)';  ti = linspace(0,pi,ni)';
    Xo = [cos(to),   sin(to)];
    Xi = [1-cos(ti), 1-sin(ti)-0.5];
    X  = [Xo; Xi] + noise*randn(n,2) + shift;
    y  = [-ones(no,1); ones(ni,1)];
    p  = randperm(n);  X = X(p,:);  y = y(p,:);
end

% ---------------------------------------------------------------------
% BACKPROP: global backward gradient (the 1986 rule)
function [W1,b1,W2,b2] = train_backprop(W1,b1,W2,b2, X,Y, EP, lr)
    n = size(X,1);
    for ep = 1:EP
        a1 = X*W1' + b1;  h1 = tanh(a1);  out = h1*W2' + b2;
        d_out = (out - Y)/n;
        dW2 = d_out'*h1;  db2 = sum(d_out,1);
        d_a1 = (d_out*W2).*(1-h1.^2);
        dW1 = d_a1'*X;    db1 = sum(d_a1,1);
        W2 = W2 - lr*dW2;  b2 = b2 - lr*db2;
        W1 = W1 - lr*dW1;  b1 = b1 - lr*db1;
    end
end

% ---------------------------------------------------------------------
% LEL: relaxation to equilibrium + local Hebbian rule (no backpropagation)
function [W1,b1,W2,b2] = train_lel(W1,b1,W2,b2, X,Y, EP, lr, gamma, T)
    n = size(X,1);
    for ep = 1:EP
        a1 = X*W1' + b1;  x1 = tanh(a1);  x2 = Y;
        for t = 1:T
            a1 = X*W1' + b1;  pred1 = tanh(a1);
            e1 = x1 - pred1;  e2 = x2 - (x1*W2' + b2);
            x1 = x1 - gamma*(e1 - e2*W2);
        end
        a1 = X*W1' + b1;
        e1 = x1 - tanh(a1);  e2 = x2 - (x1*W2' + b2);
        g1 = e1 .* (1 - tanh(a1).^2);
        W1 = W1 + lr*(g1'*X)/n;   b1 = b1 + lr*sum(g1,1)/n;
        W2 = W2 + lr*(e2'*x1)/n;  b2 = b2 + lr*sum(e2,1)/n;
    end
end

% ---------------------------------------------------------------------
function a = accuracy(W1,b1,W2,b2, X,Y)
    out = tanh(X*W1' + b1)*W2' + b2;
    a = mean(sign(out) == sign(Y));
end
