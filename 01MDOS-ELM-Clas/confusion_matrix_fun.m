function confusion_matrix=confusion_matrix_fun(x,y)
  result_1=[x' y'];
  re=sortrows(result_1,1);
  k1=length(find(re(:,1)==re(1,1)));%实际正例
  TP=length(find(re(1:k1,1)==re(1:k1,2)));%(真正例)
  FN=k1-TP;%假反例
  k2=length(re(:,1))-k1;%实际反例
  TN=length(find(re(k1+1:end,1)==re(k1+1:end,2)));%(正反例)
  FP=k2-TN;%假正例
  confusion_matrix=[TP,FN;FP,TN];
 end