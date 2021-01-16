clc
clear
%% 加载数据
%%%%%%%%% Loading Data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
load('-mat','Online_Train_data');
load('-mat','Online_Test_data');
%% 加载预测输出
load('-mat','PY_trn');
load('-mat','PY_tst');
load('-mat','k')
%% 求均方根误差和偏差绝对值
for k2=0:k
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%训练偏差绝对值
    absbias_Single_trn(k2+1,:)=abs(trnY.net{k2+1}-PY_trn{k2+1}); 
    maxtrnabsbias(k2+1,:)=max(absbias_Single_trn(k2+1,:));
%测试偏差绝对值
    absbias_Single_tst(k2+1,:)=abs(tstY.net{k2+1}-PY_tst{k2+1}); 
    maxtstabsbias(k2+1,:)=max(absbias_Single_tst(k2+1,:));    
 %训练均方根误差
    trn_mse_Single(k2+1,:)=msereg(absbias_Single_trn(k2+1,:));
    trn_smse_Single(k2+1,:)=nthroot((msereg(absbias_Single_trn(k2+1,:))),2);
%测试均方根误差
    tst_mse_Single(k2+1,:)=msereg(absbias_Single_tst(k2+1,:));
    tst_smse_Single(k2+1,:)=nthroot((msereg(absbias_Single_tst(k2+1,:))),2);
 %求训练(R^2)回归决定系数
    trn_regression_R(k2+1,:)=Regression_R(trnY.net{k2+1},PY_trn{k2+1});
%求测试(R^2)回归决定系数
    tst_regression_R(k2+1,:)=Regression_R(tstY.net{k2+1},PY_tst{k2+1});
end
%% 保存
save('k2','k2');
save('Errors','trn_smse_Single','tst_smse_Single','maxtrnabsbias','maxtstabsbias', 'trn_regression_R', 'tst_regression_R')
%% 
Errors=[trn_smse_Single,tst_smse_Single,maxtrnabsbias maxtstabsbias,trn_regression_R,tst_regression_R]