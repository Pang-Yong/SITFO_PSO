function loc = OptEI(A,KrigingModel,Problem,x0)
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
    lo = Problem.lower;
    up = Problem.upper;
    
    fun=@(x)objfunc(x, KrigingModel,Gbest);
    %loc = ga(fun,size(lo,2),[],[],[],[],lo,up);
    loc = fmincon(fun,x0,[],[],[],[],lo,up);

end



function value=objfunc(x, KrigingModel,Gbest)
    [y,~,Kriging_mse,~] = predictor(x,KrigingModel);
    Kriging_mse= max(Kriging_mse,0);   
    s = sqrt(Kriging_mse);
    EI     = (Gbest-y)*normcdf((Gbest-y)/s)+s*normpdf((Gbest-y)/s);
    value = -EI;
end