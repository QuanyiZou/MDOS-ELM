%%%%% �����������
clc
clear all
%% ��������
load('-mat','Online_Train_data');
load('-mat','Online_Test_data');
load('-mat','PY_trn');
load('-mat','PY_tst');
load('-mat','k3');
k3=k3+1;% 
%% ѵ���������
  neg=size(trnY.net{k3+1},2);
  figure
  plot(1:neg, trnY.net{k3+1},'-', 1:neg,PY_trn{k3+1},'*-');
  legend('Expected Outputs','Predictions of Single Online Model')
  hold on 
  xlabel('Samples in the Training Set')
  hold on 
  ylabel('Y')
  tit=sprintf('Online Model on Training Set %d',k3);
  title(tit);
%% �����������
  neg_tst=size(tstY.net{k3+1},2);
  figure
  plot(1:neg_tst, tstY.net{k3+1},'-', 1:neg_tst,PY_tst{k3+1},'*-');
  legend('Expected Outputs','Predictions of Single Online Model')
  hold on 
  xlabel('Samples in the Testing Set')
  hold on 
  ylabel('Y')
  tit=sprintf('Online Model on Testing Set %d',k3);
  title(tit);
%% 
save('k3','k3')