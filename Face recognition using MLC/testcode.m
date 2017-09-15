clear
Data = imageSet('orl_faces','recursive');
a=1;
b=1;
for j=1:40
    for i=1:6;
        X= read(Data(j),i);        
 for c=1:92
     for r=1:112
         y(b,a)=X(r,c);   %% train data%%
         a=a+1;
     end;
 end;
 b=b+1;
 a=1;
    end;
end;
y=y';
d=double(y);
a1=1;
b1=1;
for j=1:40
    for i=7:10;
        X= read(Data(j),i);        
 for c=1:92
     for r=1:112
         y1(b1,a1)=X(r,c);
         a1=a1+1;
     end;
 end;
 b1=b1+1;
 a1=1;
    end;
end;
y1=y1';
testD=double(y1);
[score_pca,score_train_pca]=princcomp(d,testD);
[score_lda,score_train_lda]=lind(d,testD);
[score_svm,score_train_svm]=supvecmac(d,testD);
[score_knn,score_train_knn]=knearne(d,testD);
[score_dec,score_train_dec]=treedec(d,testD);
score_sum=score_lda+score_pca+score_svm+score_knn+score_dec;
score_train_sum=score_train_lda+score_train_pca+score_train_svm+score_train_knn+score_train_dec;
temp=[zeros(1,4),ones(1,156)]';
for i=1:40
        target(:,i)=temp(:,:);
       temp=circshift(temp,4);    %% creating Target value%%
end;
ezroc3(score_sum,target,2,'sum rule',1);
temp1=[zeros(1,6),ones(1,234)]';
for i=1:40
        target1(:,i)=temp1(:,:);
       temp1=circshift(temp1,6);    %% creating Target value%%
end;
ezroc3(score_train_sum,target1,2,' train sum rule',1);
 for i=1:160
     for j=1:40
         tempmat=[score_lda(i,j),score_pca(i,j),score_svm(i,j),score_knn(i,j),score_dec(i,j)];
         score_max(i,j)=max(tempmat);
     end;
 end;
 for i=1:160
     for j=1:40
         tempmat=[score_lda(i,j),score_pca(i,j),score_svm(i,j),score_knn(i,j),score_dec(i,j)];
         score_min(i,j)=min(tempmat);
     end;
 end;
ezroc3(score_max,target,2,'max rule',1);
ezroc3(score_min,target,2,'min rule',1);
a2=0.1;
for a1=0.3:0.1:0.8;
     a3=0.8-a1;
     score_weights=a1*score_lda+a2*score_svm+a3*score_knn;
    ezroc3(score_weights,target,2,strcat(num2str(a1),num2str(a2),num2str(a3)),1);
end
score_product=score_lda.*score_pca.*score_svm.*score_knn;
ezroc3(score_product,target,2,'product rule',1);
 for i=1:160
     for j=1:40
         tempmat=[score_lda(i,j),score_pca(i,j),score_svm(i,j),score_knn(i,j),score_dec(i,j)];
         score_median(i,j)=median(tempmat);
     end;
 end;
 ezroc3(score_median,target,2,'median rule',1);
 
