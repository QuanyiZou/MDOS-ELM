function [TY] = MD_OSELMsim(X,BiasMatrix,InputWeight,OutputWeight,ActivationFunction,Elm_Type,Xps,Yps,number_class,label)
REGRESSION=0;
CLASSIFICATION=1;
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
PY_1=(H * OutputWeight)';                       %   TY: the actual output of the testing data
 if Elm_Type == REGRESSION
        TY =mapminmax('reverse',PY_1,Yps); 
    elseif Elm_Type == CLASSIFICATION
        for i = 1:Q
        [x, label_index_expected(i,:)]=max(PY_1(:,i));  
        end        
          for i=1:Q
              for j=1: number_class
                if label_index_expected(i,:)==j;
                 tY(i,:)=label(j);
                 break;
                end
               end
          end 
    TY=tY';     
  end
end
