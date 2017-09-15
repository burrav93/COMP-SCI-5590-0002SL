function [score,score_train]=princcomp(d,testD)
%load('V.mat')
%load('D.mat')
m=mean(d,2);
M=repmat(m,[1,240]);
Va=cov(((d-M)'));
[V,D]=eig(Va,'vector');
Vk=V(:,(10304-9:10304));
trainpro=Vk'*(d-M);
testm=mean(testD,2);
testM=repmat(testm,[1,160]);
testpro=Vk'*(testD-testM);
diff=pdist2(trainpro',testpro');
% for i=1:160
%     for j=1:240
%         diff(i,j)=sum(abs(testpro(:,i)-trainpro(:,j)));
%     end;
% end;
 norm=max(diff(:));
 normmat=1/norm*(diff);
% for i=1:200
%     for j=1:200
%         if(normmat(i,j)>0.60)
%             normmat(i,j)=1;
%         else
%             normmat(i,j)=0;
%         end;
%     end;
% end;
euc=pdist(trainpro');
euc=squareform(euc);
m=max(euc(:));
euc=1/m*euc;
tar1=[zeros(1,6),ones(1,234)];
tar1=repmat(tar1,[6,1]);
target1=zeros(240,240);

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

