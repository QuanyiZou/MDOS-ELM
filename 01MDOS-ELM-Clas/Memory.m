function memory= Memory(x1,x2,Similarity,error,a)
x1=x1';
x2=x2';
X1=mean(x1);
X2=mean(x2);
   switch Similarity %lower(Similarity)
        case{'Ed'}
           sim=1./(1+norm(X1-X2));
        case{'cos'}
           cosine=(X1*X2')./(norm(X1).*norm(X2));
           sim=cosine./2+0.5;
         case{'PC'}
             ppr=corrcoef(x,y);
             sim=ppr./2+0.5;
        case{'RHSM'}            
           sim=1./(1+RHSM(X1,X2,a));
   end 
    e=norm(error);
    memory=(1-exp(-sim./e));
end

