
function [Output]=BAFC_classifier(Input,Mode,DistanceType)
%% 
if strcmp(Mode,'OfflineTraining')==1
    fprintf('Ofline Training Started.\n');
    data_train=Input.TrainingData;
    label_train=Input.TrainingLabel;
    seq=unique(label_train);
    data_train1={};
    N=length(seq);
    %%
    if strcmp(DistanceType,'Cosine')==1 
        data_train=data_train./(repmat(sqrt(sum(data_train.^2,2)),1,size(data_train,2)));
        DistanceType='Euclidean';
    end
    if strcmp(DistanceType,'Euclidean')==1
        for i=1:1:N
            data_train1{i}=data_train(label_train==seq(i),:);   
            delta(i)=mean(sum(data_train1{i}.^2,2))-sum(mean(data_train1{i},1).^2);  
            [centre{i},Member{i},averdist{i}, Avgdist{i}, CenDen{i}, member2{i} , sum_of_support{i}, Age{i}]=offline_training_Euclidean(data_train1{i},delta(i));
            
        end
        L=zeros(1,N);
        mu={};
        XX=zeros(1,N);
        ratio=zeros(1,N);
        for i=1:1:N
            mu{i}=mean(data_train1{i},1);
            [L(i),W]=size(data_train1{i});
            XX(i)=0;                      
            for ii=1:1:L(i)
                XX(i)=XX(i)+sum(data_train1{i}(ii,:).^2);  
            end
            XX(i)=XX(i)./L(i); 
            ratio(i)=averdist{i}/(2*(XX(i)-sum(mu{i}.^2)));
        end
        TrainedClassifier.seq=seq;     
        TrainedClassifier.ratio=ratio; 
        TrainedClassifier.miu=mu;         
        TrainedClassifier.XX=XX;          
        TrainedClassifier.L=L;           
        TrainedClassifier.centre=centre;  
        TrainedClassifier.Member=Member;  
        TrainedClassifier.averdist=averdist;
        TrainedClassifier.NoC=N;
        TrainedClassifier.delta=delta;
        TrainedClassifier.Avgdist = Avgdist;
        TrainedClassifier.CenDen = CenDen;
        TrainedClassifier.Sum_of_support = sum_of_support;
        TrainedClassifier.Member2 = member2;
        TrainedClassifier.Age = Age;
    end  % end of if
    
    Output.TrainedClassifier=TrainedClassifier; 
end


%--------------------------------------------------------------------------------------------------
%% Start of Evolving classifier 
if strcmp(Mode,'EvolvingTraining')==1
    fprintf('Evolving Training Started.\n\n');
    data_train=Input.TrainingData;            
    label_train=Input.TrainingLabel;         
    TrainedClassifier=Input.TrainedClassifier;  
    if strcmp(DistanceType,'Cosine')==1    
        data_train=data_train./(repmat(sqrt(sum(data_train.^2,2)),1,size(data_train,2)));
        DistanceType='Euclidean';
    end
    if strcmp(DistanceType,'Euclidean')==1
        seq=TrainedClassifier.seq;            %  
        ratio=TrainedClassifier.ratio;        % 
        mu=TrainedClassifier.miu;             %
        XX=TrainedClassifier.XX;              %
        L=TrainedClassifier.L;                %
        centre=TrainedClassifier.centre;      %
        Member=TrainedClassifier.Member;      %
        averdist=TrainedClassifier.averdist;  %
        delta = TrainedClassifier.delta;      %
        N = TrainedClassifier.NoC;            %  
        
        Avgdist = TrainedClassifier.Avgdist;
        CenDen = TrainedClassifier.CenDen ;
        Sum_of_support = TrainedClassifier.Sum_of_support;
        Member2 = TrainedClassifier.Member2;
        Age = TrainedClassifier.Age;
        dying_cloud = zeros(0,1);  
        for i=1:1:N
            seq2=find(label_train==seq(i));
            data_train2{i}=data_train(seq2,:); 
            dying_cloud{1,i} = zeros(0,2);
            dead_cloud{1,i}  = zeros(0,1);
        end
        for i=1:1:N
            for j=1:1:size(data_train2{i},1) 
                L(i)=L(i)+1;           
                XX(i)=XX(i).*(L(i)-1)/(L(i))+sum(data_train2{i}(j,:).^2)./(L(i)); 
                mu{i}=mu{i}.*(L(i)-1)/(L(i))+data_train2{i}(j,:)./(L(i));         
                delta(i)=XX(i)-sum(mu{i}.^2);        
                threshold=2*delta(i)*ratio(i);    
                [centre{i},Member{i},Member2{i},Sum_of_support{i}, Age{i}, Avgdist{i}] = evolving_training_Euclidean(j,data_train2{i}(j,:),mu{i},centre{i},Member{i},delta(i),threshold ,Member2{i},Sum_of_support{i}, Age{i}, Avgdist{i} );
                threshold=[];
                [Age{i}] = age_calculation( Age{i}, j, centre{i} , Member2{i}, Sum_of_support{i}); 
                [ dying_cloud{i}, d_cloud ] = Cloud_storing(  Age{i}, dying_cloud{i},centre{i} , Member{i}  );
                dead_cloud{1,i} = d_cloud;
                [centre{i}] = Delete_cloud(centre{i}, dead_cloud{i});
                [centre{i}, Age{i}, Avgdist{i}, Member{i}, Member2{i} ] = Merge_cloud(centre{i}, Age{i}, Avgdist{i}, Member{i}, Member2{i}, j, Sum_of_support{i} );
            end
        end
        TrainedClassifier.ratio=ratio;
        TrainedClassifier.miu=mu;
        TrainedClassifier.XX=XX;
        TrainedClassifier.L=L;
        TrainedClassifier.centre=centre;
        TrainedClassifier.Member=Member;
        TrainedClassifier.averdist=averdist;
        TrainedClassifier.NoC=N;
        TrainedClassifier.delta=delta;
        TrainedClassifier.Avgdist = Avgdist;
        TrainedClassifier.CenDen = CenDen;
        TrainedClassifier.Sum_of_support = Sum_of_support;
        TrainedClassifier.Member2 = Member2;
        TrainedClassifier.Age = Age;
        TrainedClassifier.Dying_cloud = dying_cloud;
        TrainedClassifier.Dead_cloud = dead_cloud;
    end
    Output.TrainedClassifier=TrainedClassifier;
end
%----------------------------------------------------------------------------------------------
%%  Validation saction
if strcmp(Mode,'Validation')==1
    fprintf('Validation Started.\n');
    TrainedClassifier=Input.TrainedClassifier; 
    seq=TrainedClassifier.seq;                
    data_test=Input.TestingData;          
    label_test=Input.TestingLabel;        
    N=TrainedClassifier.NoC;             
    if strcmp(DistanceType,'Cosine')==1
        data_test=data_test./(repmat(sqrt(sum(data_test.^2,2)),1,size(data_test,2))); 
        DistanceType='Euclidean';
    end
    if strcmp(DistanceType,'Euclidean')==1
        centre=TrainedClassifier.centre;
        dist=zeros(size(data_test,1),N);     
        for i=1:1:N
            dist(:,i)=min(pdist2(data_test,centre{i},'euclidean').^2,[],2);    
        end
        [~,label_est]=min(dist,[],2);         
        label_est=seq(label_est);              
    end                                       
    
    Output.TrainedClassifier=Input.TrainedClassifier;         
    Output.ConfusionMatrix=confusionmat(label_test,label_est);
    Output.EstimatedLabel=label_est;                          
end
end




%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [centre,member,member2,sum_of_support, Age, avgdist]=evolving_training_Euclidean(NO,data,mu,centre,member,delta,threshold,member2,sum_of_support, Age, avgdist)
% fprintf('sajal, %d .\n', NO);
dist1=pdist2(centre,mu,'cosine').^2;   
dist2=pdist2(data,mu,'cosine').^2;     
if dist2>max(dist1)||dist2<min(dist1)    
    centre(end+1,:)=data;              
    member(end+1,1)=1;
    member2(end+1, 1) = 1;
    sum_of_support(end+1,1) = NO;
    Age(end+1, 1) = 0;
    avgdist(end+1,1 ) = 0;
else
    [dist3,pos3]=min(pdist2(data,centre,'cosine').^2 );
    threshold = mean(avgdist);
    if dist3>threshold                   
        centre(end+1,:)=data;
        member(end+1,1)=1;
        member2(end+1, 1) = 1;
        sum_of_support(end+1,1) = NO;
        Age(end+1, 1) = 0;
        avgdist(end+1, 1) = 0;
    else             
        centre(pos3,:)=member(pos3,1)/(member(pos3,1)+1)*centre(pos3,:) + 1/(member(pos3,1)+1)*data;
        member(pos3,1)=member(pos3,1)+1;
        
        if pos3 <= size(member2,1) 
            member2(pos3,1) = member2(pos3,1)+1;
            sum_of_support(pos3,1) = sum_of_support(pos3,1)+NO;
            avgdist(pos3,1) = (avgdist(pos3,1)*  (member(pos3,1)-1) + dist3 )/ member(pos3,1)  ;
        end
    end
    
end
end

%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [centre3,Mnumber,averdist,Avgdist,cenden,member2,sum_of_support,Age ]=offline_training_Euclidean(data,delta)

[L,W]=size(data);
dist00=pdist(data,'cosine').^2;  
dist0=squareform(dist00);
dist00=sort(dist00,'ascend');                                 
averdist=mean(dist00);     
[UD,J,K]=unique(data,'rows');                         
F = histc(K,1:numel(J));  
LU=length(UD(:,1));      

%%
density=sum(dist0,2)./sum(sum(dist0,2));
density=F./density(J); 
dist=dist0(J,J);
[~,pos]=max(density);
seq=1:1:LU;
seq=seq(seq~=pos);  
Rank=zeros(LU,1); 
Rank(1,:)=pos;   
for i=2:1:LU 
    pos = Rank(Rank>=1);
    [aa,pos0]=min(dist(pos,seq));
    [a, pos0] = min(aa);
    pos=seq(pos0);
    Rank(i,:)=pos;  
    seq=seq(seq~=pos);
end
data2=UD(Rank,:);
data2den=density(Rank);
Gradient=zeros(2,LU-2);
Gradient(1,:)=data2den(1:1:LU-2)-data2den(2:1:LU-1);
Gradient(2,:)=data2den(2:1:LU-1)-data2den(3:1:LU);
seq2=2:1:LU-1;
seq1=find(Gradient(1,:)<0&Gradient(2,:)>0); 
if Gradient(2,LU-2)<0  
    seq3=[1,seq2(seq1),LU];
else
    seq3=[1,seq2(seq1)];
end 
centre0=data2(seq3,:); 
nc=size(centre0,1);
dist3=pdist2(centre0,data,'cosine').^2;
[~,seq4]=min(dist3,[],1);
centre1=zeros(nc,W);      
Mnumber=zeros(nc,1);
Avgdist = zeros(nc, 1);
miu=mean(data,1);

for i=1:1:nc
    seq5=find(seq4==i);
    Mnumber(i)=length(seq5);
    x = dist3(i,seq5);
    Avgdist(i) = mean(x);
    centre1(i,:)=mean(data(seq5,:),1);  

end
dist4=pdist(centre1,'cosine').^2; 
dist5=squareform(dist4);             
seqme2=zeros(nc);  
Avgdist = repmat(Avgdist,1, nc);
seqme2(dist5<=(Avgdist/2))=1;            
 start = 1;
for i= 1:1:nc
    if i == 1
        seq = find(seqme2(i,:) == 1);
        remove_cloud = seq;
        centre2(start,:) = mean(centre1(seq,:),1);
        start = start+1;
    else 
        already_removed = find(remove_cloud == i);
        if isempty(already_removed)
            seq = find(seqme2(i,:) == 1);
            seq22 = zeros(0);
            for j=1:1:length(seq)
                if find(remove_cloud == seq(j))
                    remove_cloud = remove_cloud;
                else
                    remove_cloud(end+1) = seq(j);
                    seq22(end+1) = seq(j);
                end
            end
            centre2(start,:) = mean(centre1(seq22,:),1);
            start = start+1;
        end
    end
end                         
nc=size(centre2,1);

dist6=pdist2(centre2,data,'cosine').^2; 
[~,seq7]=min(dist6,[],1);
centre3=zeros(nc,W);
Avgdist = zeros(nc, 1);
sum_of_support = zeros(nc,1);
Mnumber=zeros(nc,1);
for i=1:1:nc
    seq8=find(seq7==i);
    Mnumber(i)=length(seq8);
    x = dist6(i,seq8);
    Avgdist(i) = mean(x);
    centre3(i,:)=mean(data(seq8,:),1);
end                                     
nc=size(centre3,1);
fprintf('Number of cloud created in offline: %d.\n\n', nc);%------------- 

dist7 = pdist2(centre3,data,'cosine').^2;
dist8 = pdist(data, 'cosine').^2;

dist80 = squareform(dist8);
distance = sum(dist80);
total_dist = sum(distance);

k = size(data,1);
cenden=zeros(1,nc);
for i=1:1:nc
   cenden(i) = total_dist/ (2*k* sum(dist7(i,:))) ;
end
cenden = cenden';  
member2 = zeros(nc,1);
sum_of_support = zeros(nc,1);
Age = zeros(nc,1);
end
%%
function [Age] =  age_calculation(Age, J,  centre, member2, sum_of_support)
    NC = size(centre,1);
    for i = 1:1:NC
        if ( (member2(i) >= 1) & (sum_of_support(i) >= 1) )
            Age(i) =  J - (sum_of_support(i)/ member2(i));
        end
    end
end

%%
function [dying_cloud, dead_cloud] =  Cloud_storing(Age, dying_cloud, centre, member)
    for i = 1:1:size(dying_cloud,1)
        if(dying_cloud(i,1) >= 1  &  dying_cloud(i,2) >= -0.1 )
            dying_cloud(i,2) = dying_cloud(i,2) - .001;
        end
    end
    seq = find(dying_cloud(:,1)>0 & dying_cloud(:,2)<=0);
    dead_cloud = dying_cloud(seq,1);
    NC = size(centre,1);
    for i = 1:1:NC
        if (Age(i) > mean(Age) + std(Age))  | (( Age(i) == 0 ) & member(i) == 0 )
            pos = find(dying_cloud(:,1) == i);
            if isempty(pos)  
                dying_cloud(end+1,1) = i;
                dying_cloud(end,  2) = .5;
            end
            
        elseif  ( (Age(i) <= mean(Age) + std(Age)) &  find(dying_cloud(:,1) == i) ) 
            dying_cloud(i,1) = 0;
            dying_cloud(i,2) = 0;
        end
    end
end
%%
function [centre] = Delete_cloud(centre, dead_cloud)
    NC = size(dead_cloud,1);
    for i= 1:1:NC
        centre(dead_cloud(i), 1) = 900;
    end
end
%%
function [centre, Age, Avgdist, Member, Member2] = Merge_cloud(centre, Age, Avgdist, Member, Member2, J, sum_of_support)
    NC = size(centre,1);
    for i = 1:1:NC
         if centre(i,1) ~= 900
             dist3=pdist2(centre(i,:),centre,'cosine').^2;
             dist3(i) = [];  
             [dist2, j] = min(dist3);
             if j>= i 
                 j = j+1; 
             end
             if dist2 < ( (Avgdist(i)/2)  + (Avgdist(j)/2) )
                 centre(i,:) = (centre(i,:)*Member(i) +  centre(j,:)*Member(j)) / (Member(i)+ Member(j));
                 centre(j,1) = 900;
                 if( ((Member2(i) >= 1) & (sum_of_support(i) >= 1) ) | ((Member2(j) >= 1) & (sum_of_support(j) >= 1) )  )
                     Age(i) = J - ( ( sum_of_support(i)+ sum_of_support(j) ) / ( Member2(i)+ Member2(j) ) );
                 end
                 Avgdist(i) = (Avgdist(i) + Avgdist(j) + dist2 )/2;
                 Member(i) = Member(i) + Member(j);
                 Member2(i) = Member2(i) + Member2(j);
             end
         end
    end

end
