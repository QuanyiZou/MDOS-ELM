function [R2] = Regression_R(PY,TY)
%% ��R �ع����ϵ����
N=length(TY)
R2 = (N * sum(PY .* TY) - sum(PY) * sum(TY))^2 / ((N * sum((PY).^2) - (sum(PY))^2) * (N * sum((TY).^2) - (sum(TY))^2)); 
end

