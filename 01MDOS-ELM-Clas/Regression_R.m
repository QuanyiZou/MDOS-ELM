function [R] = Regression_R(PY,TY)
%% ��R �ع����ϵ����
 SST=sum((PY-TY).^2);
 TY_1=mean(TY);
 SSR=sum((TY-TY_1).^2);
 R=1-SST/SSR;
end

