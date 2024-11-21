classdef SITFOPSO < ALGORITHM
% <single> <real> <large/none> <expensive>
% Surrogate information transfer and fusion optimization algorithm with PSO
% ThetaStar ---  1e-5 --- pre-specified threshold 
% maxiter --- 10 --- maximum number of iterations
% nMCpoints --- 10000 --- number of MC samples

%------------------------------- Reference --------------------------------
% Pang, Y., Zhang, S., Jin, Y., Wang, Y., Lai, X., & Song, X. (2024). 
% Surrogate information transfer and fusion in high-dimensional expensive 
% optimization problems. Swarm and Evolutionary Computation, 88, 101586.
%--------------------------------------------------------------------------

    methods
        function main(Algorithm,Problem)

            %% Parameter setting
            [ThetaStar,maxiter,  nMCpoints] = Algorithm.ParameterSet(1e-5,20,10000);
            
            %% Generate the random population
            N          = Problem.N;
            P          = UniformPoint(N,Problem.D,'Latin');
            Population = Problem.Evaluation(repmat(Problem.upper-Problem.lower,N,1).*P+repmat(Problem.lower,N,1));
            MaxNode    = 20;
            THETA=5;


          %% Initialize swarm
            Swarm = [Population.decs,Population.objs];

            Kriging_objs = Population.objs;
            Kriging_mse = ones(size(Kriging_objs));
            Kriging_dmse = zeros(N,Problem.D);
            weight=5;
            Positive_mse = Kriging_mse;

          %% Optimization
            
            A   = Population;
            conv = [];
            while Algorithm.NotTerminated(A)  
                conv = [conv,min(A.objs)];
                PbestSwarm = Swarm;
                [~,best]       = min(PbestSwarm(:,Problem.D+1));
                Gbest = PbestSwarm(best,:);
                % Paramaters of RBFNN
                spread  =  sqrt(sum((max(A.decs,[],1)-min(A.decs,[],1)).^2));
                train_X = A.decs;
                train_Y = A.objs;
                [~,distinct] = unique(round(train_X*1e6)/1e6,'rows');  
                train_X   = train_X(distinct,:);
                train_Y   = train_Y(distinct,:);
                 
                % information transfer
                if  std(Positive_mse)/mean(Positive_mse)<ThetaStar
                    rbfnet = srgtsnewrb(train_X',train_Y',0.1,spread,MaxNode,1,'off');
                    output = MCGlobalSensitivity(rbfnet,[Problem.lower;Problem.upper],nMCpoints);
                    sensitivity=output.individual.';
                    weight=sensitivity*Problem.D/sum(sensitivity);
                    KrigingModel   = dacefit_tr(train_X,train_Y,'regpoly0','corrgauss',THETA, weight, [1e-5], [100]);
                    THETA = KrigingModel.theta;
                else
                    rbfnet = srgtsnewrb(train_X',train_Y',0.1,spread,MaxNode,1,'off');
                    KrigingModel   = dacefit_tr(train_X,train_Y,'regpoly0','corrgauss',THETA, weight);
                    THETA = KrigingModel.theta;
                end

                Current_V = repmat(2*(Problem.upper-Problem.lower),N,1).*rand(N,Problem.D)+repmat(-1*(Problem.upper-Problem.lower),N,1);
                for j =1:maxiter
                    if  std(Positive_mse)/mean(Positive_mse)>ThetaStar
                        [Swarm(:,1:Problem.D),Current_V] = PSOope(Swarm(:,1:Problem.D),PbestSwarm(:,1:Problem.D),Gbest(:,1:Problem.D),Current_V);
                    else
                        [Swarm(:,1:Problem.D),Current_V] = PSOope(Swarm(:,1:Problem.D),PbestSwarm(:,1:Problem.D),Gbest(:,1:Problem.D),Current_V);
                    end 
                    
                    Swarm(:,1:Problem.D) = max(Swarm(:,1:Problem.D),repmat(Problem.lower,N,1));
                    Swarm(:,1:Problem.D) = min(Swarm(:,1:Problem.D),repmat(Problem.upper,N,1));
                    Swarm(:,Problem.D+1)= sim(rbfnet,Swarm(:,1:Problem.D)')';
                    
                    
                    replace = PbestSwarm(:,Problem.D+1) > Swarm(:,Problem.D+1);
                    PbestSwarm(replace,1:Problem.D) = Swarm(replace,1:Problem.D);
                    PbestSwarm(replace,Problem.D+1) = Swarm(replace,Problem.D+1);
                    [~,best] = min(PbestSwarm(:,Problem.D+1));
                    Gbest = PbestSwarm(best,:);                   
                    
                    range = mean(Problem.upper - Problem.lower);
                    for i = 1:N
                        [Kriging_objs(i),~,Kriging_mse(i),Kriging_dmse(i,:)] = predictor(Swarm(i,1:Problem.D),KrigingModel);
                        Kriging_mse(i) = max(Kriging_mse(i),0);
                        rho = (1/norm(Kriging_dmse(i,:))).*0.1.*range.*exp(-(Problem.FE-Problem.N)) ;
                        Kriging_dmse(i,:) = Kriging_dmse(i,:).* rho;
                    end                     

                    Positive_mse = Kriging_mse;
                    Positive_mse(find(Positive_mse<1e-5),:) = [];
                end
                             
                

                
              %%  Infill

                %add point without considering uncertainty
                    
                loc = OptEI(A,KrigingModel,Problem,PbestSwarm(randi(N),1:Problem.D));
                new2 = Problem.Evaluation(loc);  

                if ~isempty(new2)
                    A = [A,new2];
                end   
  
                P = UniformPoint(N,Problem.D,'Latin');
                Swarm(:,1:Problem.D) = repmat(Problem.upper-Problem.lower,N,1).*P+repmat(Problem.lower,N,1);
                Swarm(:,Problem.D+1) = sim(rbfnet,Swarm(:,1:Problem.D)')';
                if size(conv,2) == 80
                    conv = conv';
                    conv'  
                end
            end
        end
    end
end