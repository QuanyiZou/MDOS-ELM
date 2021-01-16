clc
clear all 
%% 加载数据
load('-mat','Online_Train_data')
%% 建模
NN=100 % NumberofHiddenNeurons
IN=3;
new_accquired=2
Task=0;  %0=regression,1=classification
u=3;
err_tp=inf;
for k=0:new_accquired
   to=clock;
   for i=1:IN
       if (k+1)==1
            [PY1,BiasofHiddenNeurons_1,BiasMatrix_1,InputWeight_1,OutputWeight_1,Xps_1,Yps_1,H_1,M_1]= MD_OSELMtrain_Start(trnX.net{k+1},trnY.net{k+1},Task,NN,'sig');
            memory(1)=0;
       elseif k<u &&(k+1)>1
            [PY1,OutputWeight_1,H_1,M_1,memory,Xps_1,Yps_1]=MD_OSELMtrain_Add1(trnX.net{k+1},trnY.net{k+1},'sig','Ed',Task,0,OutputWeight{k},trnX.net{k},InputWeight,BiasMatrix,k,M{k},memory);
       elseif k>=u
            [PY1,OutputWeight_1,H_1,M_1,memory,Xps_1,Yps_1]=MD_OSELMtrain_Add2(trnX.net{k+1},trnY.net{k+1},'sig','Ed',Task,u,0,OutputWeight{k},trnX.net{k},trnY.net{k-u+2},InputWeight,BiasMatrix,H{k-u+2},k,M{k},memory);
       end
       err1=nthroot(msereg(abs(PY1-trnY.net{k+1})),2);   
       if err1<err_tp
           BiasMatrix=BiasMatrix_1;
           InputWeight=InputWeight_1;
           OutputWeight{k+1}=OutputWeight_1;
           M{k+1}=M_1;
           H{k+1}=H_1;
           Xps=Xps_1;
           Yps=Yps_1;
           PY_trn{k+1}=PY1;
         end       
   end
  Train_time(k+1,:)=etime(clock,to);
  no_model = sprintf('Single_MDOS_ELM_%d',k);
  save(no_model,'BiasMatrix','InputWeight','OutputWeight','Xps','Yps','H','M','NN','u','IN','Task');
  no_memory = sprintf('memory_%d',k);
  save(no_memory,'memory');
end
%%  保存模型 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
save('Train_time','Train_time');
save('PY_trn','PY_trn');
save('k','k');
%% 

