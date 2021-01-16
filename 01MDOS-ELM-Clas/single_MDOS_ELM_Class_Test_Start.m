clc
clear all
%% 读入训练数据
load('-mat','Online_Test_data');
load('-mat','LableInformation');
%% 加载初始模型
k1=0; % k1=0 original sample block; k1=1,..the k1-th new accquired sample block
no_model = sprintf('Single_MDOS_ELM_%d',k1);
load('-mat',no_model);
%%  测试
to=clock;
PY_tst{k1+1}= MD_OSELMsim(tstX.net{k1+1},BiasMatrix,InputWeight,OutputWeight{k1+1},'sig',Task,Xps,Yps,number_class,label);
Test_time(k1+1,:)=etime(clock,to)/neg_tst_online(:,k1+1)*10^3; %ms
%% 保存结果 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
save('Test_time','Test_time');
save('PY_tst','PY_tst');
save('k1','k1');