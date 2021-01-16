clc
clear all
 %% 读入训练数据
load('-mat','Online_Train_data');
%% 初始训练
k=0; %% k=0 original sample block; k=1,..the k-th new accquired sample block
NN=100 % NumberofHiddenNeurons
IN=10;
u=3;  % period of validity
Task=0;  %0=regression,1=classification
err_tp=inf;to=clock;
for i=1:IN
    [PY1,BiasofHiddenNeurons_1,BiasMatrix_1,InputWeight_1,OutputWeight_1,Xps_1,Yps_1,H_1,M_1]= MD_OSELMtrain_Start(trnX.net{k+1},trnY.net{k+1},Task,NN,'sig');
    err1=nthroot(msereg(abs(PY1-trnY.net{k+1})),2);   
    if err1<err_tp
       BiasofHiddenNeurons=BiasofHiddenNeurons_1; 
       BiasMatrix=BiasMatrix_1;
       InputWeight=InputWeight_1;
       OutputWeight_t=OutputWeight_1;
       Xps=Xps_1;
       Yps=Yps_1;
       err_tp=err1;
       PY_trn{k+1}=PY1;
       H1=H_1;
       M1=M_1;
    end
end
OutputWeight{k+1}=OutputWeight_t; H{k+1}=H1; M{k+1}=M1; memory=[0];
Train_time(k+1,:)=etime(clock,to);
%% 保存初始模型 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
no_model = sprintf('Single_MDOS_ELM_%d',k);
save(no_model,'BiasMatrix','InputWeight','OutputWeight','Xps','Yps','H','M','NN','u','IN','Task');
no_memory = sprintf('memory_%d',k);
save(no_memory,'memory');
save('PY_trn','PY_trn');
save('k','k');
save('Train_time','Train_time')
%% 
