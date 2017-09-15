
%This code to apply LDA (Linear Discriminant Analysis) 


% This example deals with 2 classes
c1=[1 2;2 3;3 3;4 5;5 5]  % the first class 5 observations
c2=[1 0;2 1;3 1;3 2;5 3;6 5] % the second class 6 observations
% scatter(c1(:,1),c1(:,2),6,'r'),hold on;
% scatter(c2(:,1),c2(:,2),6,'b');


% Number of observations of each class
n1=size(c1,1)
n2=size(c2,1)

%Mean of each class
mu1=mean(c1)
mu2=mean(c2)

% Average of the mean of all classes
mu=(mu1+mu2)/2

% Center the data (data-mean)
d1=c1-repmat(mu1,size(c1,1),1)
d2=c2-repmat(mu2,size(c2,1),1)


% Calculate the within class scatter (SW)
s1=d1'*d1
s2=d2'*d2
sw=s1+s2
invsw=inv(sw)


% if more than 2 classes calculate between class variance (SB)
sb1=n1*(mu1-mu)'*(mu1-mu)
sb2=n2*(mu2-mu)'*(mu2-mu)
SB=sb1+sb2
v=invsw*SB

% find eigen values and eigen vectors of the (v)
[evec,eval]=eig(v)

% Sort eigen vectors according to eigen values (descending order) and
% select eigen vectors according to eigen values to generate fisher space
%% training and test data will be projected to the fisher space




