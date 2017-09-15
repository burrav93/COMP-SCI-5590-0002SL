function[score,score_train]=supvecmac(d,testD)
%load('V.mat')
 %load('D.mat')
 load('labels.mat')
m=mean(d,2);
M=repmat(m,[1,240]);
Va=cov(((d-M)'));        %% Finding Co-variance of PCA%%
[V,D]=eig(Va,'vector');
Vk=V(:,(10304-9:10304));   %% Selecting few Eigen Vectors   
trainpro=Vk'*(d-M);
t=templateSVM('Standardize',1,'KernelFunction','Gaussian','KernelScale',5);
Sv=fitcecoc(trainpro',labels,'learners',t);    %% Using fitcecoc to classify the data set%%
%t=templateSVM('Standardize',1,'KernelFunction','Polynomial','PolynomialOrder',1);
%Sv=fitcecoc(trainpro',labels,'learners',t);

testm=mean(testD,2);
testM=repmat(testm,[1,160]);
testpro=Vk'*(testD-testM);
[~,score]=predict(Sv,testpro');   %% Calculating scores using predict function%%
[~,score_train]=predict(Sv,trainpro');
max_svm=max(score(:));
min_svm=min(score(:));
score=(score-min_svm)/(max_svm-min_svm);
score=1-score;
max_train=max(score_train(:));
min_train=min(score_train(:));
score_train=(score_train-min_train)/(max_train-min_train);
score_train=1-score_train;
end