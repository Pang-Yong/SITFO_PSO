function EI = CalEI(A,Current_objs,Current_mse)
% Solution update in EGO, where a solution with the best expected
% improvement is re-evaluated

%------------------------------- Copyright --------------------------------
% Copyright (c) 2022 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------


    Gbest = min(A.objs);

    N = size(Current_objs,1);
    EI = zeros(N,1);
    for i = 1 : N
        y = Current_objs(i);
        s         = sqrt(Current_mse(i));
        EI(i)     = (Gbest-y)*normcdf((Gbest-y)/s)+s*normpdf((Gbest-y)/s);
    end
    
end