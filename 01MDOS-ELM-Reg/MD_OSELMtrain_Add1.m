function [PY,OutputWeight,H,M,memory,Xps,Yps]=MD_OSELMtrain_Add1(X,Y,ActivationFunction,Similarity ,Elm_Type,a,beta,X1,InputWeight,BiasMatrix,k,M1,memory)

%% 计算隐含层输出矩阵H k565 
[inputn1,Xps]=mapminmax(X);
[P1]=mapminmax(X1);
[outputn1,Yps]=mapminmax(Y);
%[T1]=mapminmax(Y1);
%%%%%%%%%%% Macro definition
REGRESSION=0;
CLASSIFIER=1;
%%%%%%%%%%%%
T=outputn1;
P=inputn1;
Q=size(P,2);
BiasMatrix_1=repmat( BiasMatrix(:,1),1,Q);
tempH=InputWeight*P;
tempH=tempH+BiasMatrix_1;
%%%%%%%%%%% Calculate hidden neuron output matrix H
switch lower(ActivationFunction)
     case {'sig','sigmoid'}
        %%%%%%%% Sigmoid 
        H = 1 ./ (1 + exp(-tempH))';
     case {'sin','sine'}
        %%%%%%%% Sine
        H = sin(tempH)';    
     case {'hardlim'}
        %%%%%%%% Hard Limit
        H = hardlim(tempH)';            
        %%%%%%%% More activation functions can be added here                
end
%% 计算上一代模型对当前的误差
   error=nthroot(msereg(abs(T'-H*beta)),2);
%% 计算记忆因子
  memory(k)=Memory(P,P1,Similarity,error,a);
 %% 不遗忘数据
    %M是工作矩阵（过渡的）        
      fact=1./memory(k);
      M =fact.*(M1 - M1 * H' * (memory(k).*eye(size(H,1)) + H * M1 * H')^(-1) * H * M1); 
      OutputWeight= memory(k).*beta + (M * H' * (T' - memory(k).*H* beta));  
      PY_1=(H * OutputWeight)';  % predicitons of the training set
      PY =mapminmax('reverse',PY_1,Yps);
end

