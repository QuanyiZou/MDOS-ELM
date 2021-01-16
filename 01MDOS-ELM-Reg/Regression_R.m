function [R2] = Regression_R(PY,TY)
%% 求R 回归决定系数；
N=length(TY)
R2 = (N * sum(PY .* TY) - sum(PY) * sum(TY))^2 / ((N * sum((PY).^2) - (sum(PY))^2) * (N * sum((TY).^2) - (sum(TY))^2)); 
end

