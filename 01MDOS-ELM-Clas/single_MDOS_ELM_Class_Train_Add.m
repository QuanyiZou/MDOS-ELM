clc 
clear all
%% 加载数据
load('-mat','Online_Train_data');
load('-mat','k');
%% 加载上一代模型
no_model = sprintf('Single_MDOS_ELM_%d',k);
load('-mat',no_model);
no_memory = sprintf('memory_%d',k);
load('-mat',no_memory);
k=k+1;
%% 加载上一代模型的预测输出和训练时间
load('-mat','PY_trn');
load('-mat','Train_time');
%% 在线更新模型
err_tp=inf;
to=clock;
for i=1:IN
        if k<u
          [PY1,OutputWeight_1,H_1,M_1,memory,Xps_1,Yps_1]=MD_OSELMtrain_Add1(trnX.net{k+1},trnY.net{k+1},'sig','Ed',Task,0,OutputWeight{k},trnX.net{k},InputWeight,BiasMatrix,k,M{k},memory);
        else
          [PY1,OutputWeight_1,H_1,M_1,memory,Xps_1,Yps_1]=MD_OSELMtrain_Add2(trnX.net{k+1},trnY.net{k+1},'sig','Ed',Task,u,0,OutputWeight{k},trnX.net{k},trnY.net{k-u+2},InputWeight,BiasMatrix,H{k-u+2},k,M{k},memory);
       end
         err1=nthroot(msereg(abs(PY1-trnY.net{k+1})),2);   
         if err1<err_tp
            BiasMatrix=BiasMatrix;
            InputWeight=InputWeight;
            OutputWeight{k+1}=OutputWeight_1;
            M{k+1}=M_1;
            H{k+1}=H_1;
            Xps=Xps_1;
            Yps=Yps_1;
            PY_trn{k+1}=PY1;
         end
end
Train_time(k+1,:)=etime(clock,to);
%%  保存模型和预测输出
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
no_model = sprintf('Single_MDOS_ELM_%d',k);
save(no_model,'BiasMatrix','InputWeight','OutputWeight','Xps','Yps','H','M','NN','u','IN','Task');
no_memory = sprintf('memory_%d',k);
save(no_memory,'memory');
save('PY_trn','PY_trn');
save('k','k');
save('Train_time','Train_time')
%% 

