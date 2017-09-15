function [score,score_train]=lind(d,testD)
%load('bsca.mat');
%load('wsca.mat');
%load('V.mat');
m=mean(d,2);
M=repmat(m,[1,240]);   %% Calculating Mean of data set%%
j=1;
for i=0:6:234
    mea(:,j)=mean(d(:,i+1:i+6),2);
    me(:,i+1:i+6)=repmat(mea(:,j),[1,6]);    %% Calculating Mean of Each Class%%
    j=j+1;
end;

  temp=zeros(10304,10304);
       wsca=zeros(10304,10304);
       for i =0:6:234
           temp=(d(:,i+1:i+6)-me(:,i+1:i+6))*((d(:,i+1:i+6)-me(:,i+1:i+6))');
           wsca=temp+wsca;                %%calculating with in scatter matrix%%
       end;
    temp1=zeros(10304,10304);
    bsca=zeros(10304,10304);
    for i=1:40
        temp1=(mea(:,i)-m)*((mea(:,i)-m)');
        bsca=temp1+bsca;            %%Calculating between scatter matrix%%
        
    end;
 Va=cov(((d-M)'));
 [V,D]=eig(Va,'vector');
 PCAVk=V(:,(10304-159:10304));    %% PCA Eigen Space selection%%
wscaproj=PCAVk'*wsca*PCAVk;
bscaproj=PCAVk'*bsca*PCAVk; %% within scatter and between projecting into PCA
[LDAV,LDAD]=eig(bscaproj,wscaproj,'vector');
Proj=PCAVk*LDAV;
Projk=Proj(:,1:39);        %% LDA Eigen Space selection%%    
trainpro=Projk'*(d-M);    %% Train Projection onto Eigen Space%%
testm=mean(testD,2);
testM=repmat(testm,[1,160]);
testpro=Projk'*(testD-testM);
diff=pdist2(trainpro',testpro'); %% Finding Eucledian Distances betweern Train and Test%%
norm=max(diff(:));
normmat=1/norm*(diff);
euc=pdist(trainpro');
euc=squareform(euc);
m=max(euc(:));
euc=1/m*euc;
normmat=normmat';
j=1;
for i=0:6:234
score(:,j)=mean(normmat(:,i+1:i+6),2);
j=j+1;
end;
euc=euc';
j=1;
for i=0:6:234
score_train(:,j)=mean(euc(:,i+1:i+6),2);
j=j+1;
end;
end



