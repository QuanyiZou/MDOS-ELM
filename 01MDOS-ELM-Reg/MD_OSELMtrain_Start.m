function [ELMout,BiasofHiddenNeurons,BiasMatrix,InputWeight,OutputWeight,Xps,Yps H,M] = MD_OSELMtrain_Start(X,Y,Elm_Type,NumberofHiddenNeurons,ActivationFunction)
%% πÈ“ªªØ
[inputn1,Xps]=mapminmax(X);
[outputn1,Yps]=mapminmax(Y);
%% 
%%%%%%%%%%% Macro definition
REGRESSION=0;
CLASSIFIER=1;
%%%%%%%%%%%%%
T=outputn1;
P=inputn1;
NumberofTrainingData=size(P,2);
NumberofInputNeurons=size(P,1);

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
end                                                 %   end if of Elm_Type

%%%%%%%%%%% Calculate weights & biases
%%%%%%%%%%% Random generate input weights InputWeight (w_i) and biases BiasofHiddenNeurons (b_i) of hidden neurons
InputWeight=rand(NumberofHiddenNeurons,NumberofInputNeurons)*2-1;
BiasMatrix=rand(NumberofHiddenNeurons,1);
tempH=InputWeight*P;
clear P;                                            %   Release input of training data 
ind=ones(1,NumberofTrainingData);
BiasofHiddenNeurons=BiasMatrix(:,ind);              %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
tempH=tempH+BiasofHiddenNeurons;
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
clear tempH;                                        %   Release the temparary array for calculation of hidden neuron output matrix H

%%%%%%%%%%% Calculate output weights OutputWeight (beta_i)
OutputWeight=pinv(H) * T';
M=pinv(H'*H);
PY=(H * OutputWeight)';  % predicitons of the training set
ELMout =mapminmax('reverse',PY,Yps);
end