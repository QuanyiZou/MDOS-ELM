function RHSM= RHSM(x1,x2,a)
r=abs(x1-x2);
n=length(r);
for i=1:n
    if r>a
        p(i)=a*r(i)-(a.^2)/2;
    else
        p(i)=(r(i).^2)/2;
    end  
end
RHSM=mean(p);
end
