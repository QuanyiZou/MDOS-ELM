function [PY,OutputWeight,H,M,memory,Xps,Yps]=MD_OSELMtrain_Add2(X,Y,ActivationFunction,Similarity ,Elm_Type,u,a,beta,X1,Y1,InputWeight,BiasMatrix,Hq,k,M1,memory)

%% 计算隐含层输出矩阵H k
[inputn1,Xps]=mapminmax(X);
[P1]=mapminmax(X1);
[outputn1,Yps]=mapminmax(Y);
[T1]=mapminmax(Y1);
%%%%%%%%%%% Macro definition
REGRESSION=0;
CLASSIFIER=1;
%%%%%%%%%%%%%
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
  %%
     %%  遗忘数据 k>=u
      W=prod(memory(k-u+1:k));%累积
      Ht=cat(1,- W.*Hq,H);
      Ht1=cat(1,Hq,H);
      [R]=size(Hq,1);
      [Q]=size(H,1);
       M =( M1 - M1*Ht'*(memory(k).*eye(R+Q) + Ht1 * M1 * Ht')^(-1) * Ht1* M1); 
      tmepmem=memory(k).^2;
      Tt=cat(1,T',T1');
      OutputWeight =tmepmem.*beta+ (M * Ht' * (Tt -tmepmem.* Ht1 * beta));  
 %% 
      PY_1=(H * OutputWeight)';  % predicitons of the training set
      PY =mapminmax('reverse',PY_1,Yps);
end
