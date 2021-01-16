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
%% �����������ƫ�����ֵ
%ѵ��ƫ�����ֵ
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%ѵ��ƫ�����ֵ
    absbias_Single_trn(k2+1,:)=abs(trnY.net{k2+1}-PY_trn{k2+1}); 
    maxtrnabsbias(k2+1,:)=max(absbias_Single_trn(k2+1,:));
%����ƫ�����ֵ
    absbias_Single_tst(k2+1,:)=abs(tstY.net{k2+1}-PY_tst{k2+1}); 
    maxtstabsbias(k2+1,:)=max(absbias_Single_tst(k2+1,:));    
 %ѵ�����������
    trn_mse_Single(k2+1,:)=msereg(absbias_Single_trn(k2+1,:));
    trn_smse_Single(k2+1,:)=nthroot((msereg(absbias_Single_trn(k2+1,:))),2);
%���Ծ��������
    tst_mse_Single(k2+1,:)=msereg(absbias_Single_tst(k2+1,:));
    tst_smse_Single(k2+1,:)=nthroot((msereg(absbias_Single_tst(k2+1,:))),2);
%��ѵ��(R^2)�ع����ϵ��
    trn_regression_R(k2+1,:)=Regression_R(trnY.net{k2+1},PY_trn{k2+1});
%�����(R^2)�ع����ϵ��
    tst_regression_R(k2+1,:)=Regression_R(tstY.net{k2+1},PY_tst{k2+1});
%% ����
save('Errors','trn_smse_Single','tst_smse_Single','maxtrnabsbias','maxtstabsbias','trn_regression_R', 'tst_regression_R');
save('k2','k2');
Errors=[trn_smse_Single,tst_smse_Single,maxtrnabsbias maxtstabsbias,trn_regression_R,tst_regression_R]
%Errors_smse=[trn_smse_Single,tst_smse_Single]
%Errors_max=[maxtrnabsbias maxtstabsbias]
%Errors_R=[trn_regression_R,tst_regression_R]
%% 