clc
clear all
%% ��������
load('-mat','Online_Train_data');
load('-mat','Online_Test_data');
load('-mat','PY_trn');
load('-mat','PY_tst');
load('-mat','k')
for k3=0:k
  %% ѵ���Ļ�������
  confusion_matrix_trn{k3+1}=confusion_matrix_fun(trnY.net{k3+1},PY_trn{k3+1});
%% ���ԵĻ�������
  confusion_matrix_tst{k3+1}=confusion_matrix_fun(tstY.net{k3+1},PY_tst{k3+1});
end
%%
save('k3','k3')