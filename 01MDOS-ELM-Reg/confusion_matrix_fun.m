function confusion_matrix=confusion_matrix_fun(x,y)
  result_1=[x' y'];
  re=sortrows(result_1,1);
  k1=length(find(re(:,1)==re(1,1)));%ʵ������
  TP=length(find(re(1:k1,1)==re(1:k1,2)));%(������)
  FN=k1-TP;%�ٷ���
  k2=length(re(:,1))-k1;%ʵ�ʷ���
  TN=length(find(re(k1+1:end,1)==re(k1+1:end,2)));%(������)
  FP=k2-TN;%������
  confusion_matrix=[TP,FN;FP,TN];
 end