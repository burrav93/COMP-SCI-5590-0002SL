function [score,score_train]=knearne(d,testD)
load('labels.mat')
%load('V.mat')
%load('D.mat')
m=mean(d,2);
M=repmat(m,[1,240]);
Va=cov(((d-M)'));        %% Finding Co-variance of PCA%%
[V,D]=eig(Va,'vector');
Vk=V(:,(10304-9:10304)); 
trainpro=Vk'*(d-M);
Kv=fitcknn(trainpro',labels,'NumNeighbors',10,'Standardize',1,'NSMethod','kdtree','Distance','euclidean');
testm=mean(testD,2);
testM=repmat(testm,[1,160]);
testpro=Vk'*(testD-testM);
[~,score]=predict(Kv,testpro');
[~,score_train]=predict(Kv,trainpro');
score=1-score;
score_train=1-score_train;
end