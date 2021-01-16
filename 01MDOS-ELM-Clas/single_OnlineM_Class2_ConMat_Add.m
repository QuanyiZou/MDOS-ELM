clc
clear all
%% Êı¾İÏÂÔØ
load('-mat','Online_Train_data');
load('-mat','Online_Test_data');
load('-mat','PY_trn');
load('-mat','PY_tst');
load('-mat','k3');
k3=k3+1;% 
%% ÑµÁ·µÄ»ìÏı¾ØÕó
  confusion_matrix_trn{k3+1}=confusion_matrix_fun(trnY.net{k3+1},PY_trn{k3+1});
%% ²âÊÔµÄ»ìÏı¾ØÕó
  confusion_matrix_tst{k3+1}=confusion_matrix_fun(tstY.net{k3+1},PY_tst{k3+1});
%% 
save('k3','k3');