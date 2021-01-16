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
  %训练的正确率
   n_ge_trn=size(trnY.net{k2+1},2);
   q=length(find(trnY.net{k2+1}==PY_trn{k2+1}));%预测正确的个数
   Acc_trn(k2+1,:)=q/n_ge_trn;
 %测试的正确率
   n_ge_tst=size(tstY.net{k2+1},2);
   p=length(find(tstY.net{k2+1}==PY_tst{k2+1}));%预测正确的个数
   Acc_tst(k2+1,:)=p/n_ge_tst;  
end
%% 保存
save('Accuracy','Acc_trn','Acc_tst');
save('k2','k2');
Accuracy=[Acc_trn,Acc_tst]
