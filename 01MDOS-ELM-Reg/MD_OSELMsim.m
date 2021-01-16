function [TY] = MD_OSELMsim(X,BiasMatrix,InputWeight,OutputWeight,ActivationFunction,Xps,Yps)
P = mapminmax('apply',X,Xps);
tempH_test=InputWeight*P;
Q=size(P,2);
BiasMatrix_1=repmat( BiasMatrix(:,1),1,Q);
  tempH_test=tempH_test + BiasMatrix_1;
switch lower(ActivationFunction)
     case {'sig','sigmoid'}
        %%%%%%%% Sigmoid 
        H = 1 ./ (1 + exp(-tempH_test))';
     case {'sin','sine'}
        %%%%%%%% Sine
        H = sin(tempH_test)';    
     case {'hardlim'}
        %%%%%%%% Hard Limit
        H = hardlim(tempH_test)';            
        %%%%%%%% More activation functions can be added here                
end
PY=(H * OutputWeight)';                       %   TY: the actual output of the testing data
TY=mapminmax('reverse',PY,Yps);
end
