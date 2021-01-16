function [PY,OutputWeight,H,M,memory,Xps,Yps]=MD_OSELMtrain_Add1(X,Y,ActivationFunction,Similarity ,Elm_Type,a,beta,X1,InputWeight,BiasMatrix,k,M1,memory)
%%%%%%%%%%%%% Macro definition
 REGRESSION=0;
 CLASSIFICATION=1;
%% 归一化
 if Elm_Type==REGRESSION
    [inputn1,Xps]=mapminmax(X);
    [outputn1,Yps]=mapminmax(Y);
     P1=mapminmax(X1);
    else
    [inputn1,Xps]=mapminmax(X);
    P1=mapminmax(X1);
    outputn1=Y;
    Yps=0;
 end
 T=outputn1;
 P=inputn1;
 Q=size(P,2);
BiasMatrix_1=repmat( BiasMatrix(:,1),1,Q);
%%   
 NumberofTrainingData=size(P,2);
if Elm_Type~=REGRESSION
    %%%%%%%%%%%% Preprocessing the data of classification
    sorted_target=sort(T,2);
    label=zeros(1,1);                               %   Find and save in 'label' class label from training and testing data sets
    label(1,1)=sorted_target(1,1);
    j=1;
    for i = 2:(NumberofTrainingData)
        if sorted_target(1,i) ~= label(1,j)
            j=j+1;
            label(1,j) = sorted_target(1,i);
        end
    end
    number_class=j;
    NumberofOutputNeurons=number_class;
   
    %%%%%%%%%% Processing the targets of training
    temp_T=zeros(NumberofOutputNeurons, NumberofTrainingData);
    for i = 1:NumberofTrainingData
        for j = 1:number_class
            if label(1,j) == T(1,i)
                break; 
            end
        end
        temp_T(j,i)=1;
    end
    T=temp_T*2-1;
end  
%% 计算隐含层输出矩阵H k
%%%%%%%%%%%%%
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
 
     %% 不遗忘数据
      %M是工作矩阵（过渡的） 
      fact=1./memory(k);
      M =fact.*(M1 - M1 * H' * (memory(k).*eye(size(H,1)) + H * M1 * H')^(-1) * H * M1); 
      OutputWeight= memory(k).*beta + (M * H' * (T' - memory(k).*H* beta));  
      PY_1=(H * OutputWeight)';  % predicitons of the training set
    %%
 if Elm_Type == REGRESSION
         PY =mapminmax('reverse',PY_1,Yps); 
    elseif Elm_Type == CLASSIFICATION
        for i = 1:NumberofTrainingData
        [x, label_index_expected(i,:)]=max(PY_1(:,i));  
        end        
          for i=1:NumberofTrainingData
              for j=1: number_class
                if label_index_expected(i,:)==j;
                 tY(i,:)=label(j);
                 break;
                end
               end
          end 
    PY=tY';     
  end
 end

