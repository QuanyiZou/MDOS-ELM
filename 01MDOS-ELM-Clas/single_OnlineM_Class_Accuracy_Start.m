clc
clear
%% ��������
%%%%%%%%% Load1ng Data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
load('-mat','Online_Train_data');
load('-mat','Online_Test_data');
%% ����Ԥ�����
load('-mat','PY_trn');
load('-mat','PY_tst');
k2=0; % k2=0 original sample block; k2=1,..the k2-th new accquired sample block
%% ����ȷ��
  %ѵ������ȷ��
   n_ge_trn=size(trnY.net{k2+1},2);
   q=length(find(trnY.net{k2+1}==PY_trn{k2+1}));%Ԥ����ȷ�ĸ���
   Acc_trn(k2+1,:)=q/n_ge_trn;
  %���Ե���ȷ��
   n_ge_tst=size(tstY.net{k2+1},2);
   p=length(find(tstY.net{k2+1}==PY_tst{k2+1}));%Ԥ����ȷ�ĸ���
   Acc_tst(k2+1,:)=p/n_ge_tst;
%% ����
save('Accuracy','Acc_trn','Acc_tst');
save('k2','k2');
Accuracy=[Acc_trn,Acc_tst]
%% 