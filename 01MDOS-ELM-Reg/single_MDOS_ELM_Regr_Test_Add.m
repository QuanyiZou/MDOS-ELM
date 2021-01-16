clc 
clear all
%% ����ѵ������
load('-mat','Online_Test_data');
load('-mat','Test_time');
load('-mat','k1');
k1=k1+1;
%% ����ģ��
no_model = sprintf('Single_MDOS_ELM_%d',k1);
load('-mat',no_model);
load('-mat','PY_tst');
%% ����
to=clock;
PY_tst{k1+1}= MD_OSELMsim(tstX.net{k1+1},BiasMatrix,InputWeight,OutputWeight{k1+1},'sig',Xps,Yps);
Test_time(k1+1,:)=etime(clock,to)/neg_tst_online(:,k1+1)*10^3; %ms
%% ������Խ��
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
save('Test_time','Test_time');
save('PY_tst','PY_tst');
save('k1','k1');



