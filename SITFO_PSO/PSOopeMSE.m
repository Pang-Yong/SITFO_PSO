function [Current_decs,Current_V] = PSOopeMSE(Current_decs,Pbest_decs,Gbest_decs,Current_V,Kriging_dmse)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2022 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    [c1,c2] = deal(2.05);
    Chi = 0.757;
    [N,D] = size(Current_decs);
    r1        = rand(N,D);
    r2        = rand(N,D);
    New_V    = Chi*(Current_V+c1*r1.*(Pbest_decs-Current_decs)+c2*r2.*(Gbest_decs-Current_decs)+Kriging_dmse);
    %
    Current_decs=Current_decs+New_V;
    Current_V = New_V;

end 