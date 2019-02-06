function val = read_stats(dir,save)

val = 0;
printdocs = 1;

beta = [];
load(strcat(save,'/train_stats.mat'));
fileID = fopen(strcat(save,'/train_alphabet.txt'),'r');
tok = textscan(fileID, '%s');
fclose(fileID);
fileID = fopen(strcat(save,'/train_target_alphabet.txt'),'r');
label = textscan(fileID, '%s');
fclose(fileID);

if printdocs
    % fileID = fopen(strcat(dir,'/train_doc.txt'),'r');
    [doctxt] = textscan(fopen(strcat(dir,'/train_doc.txt')), '%s','Delimiter','\n');
    doctxt = doctxt{1};
    % [doctxt] = textread(strcat(dir,'/train_doc.txt'), '%s', -1, 'delimiter','\n','whitespace','','bufsize',1000000);
    % fclose(fileID);
end

rep = fopen(strcat(save,'/train_report.txt'),'w');
repW = fopen(strcat(save,'/topic_words.csv'),'w');
fprintf(repW, "topic-id,word,probability\n");
repL = fopen(strcat(save,'/topic_lift.csv'),'w');
fprintf(repL, "topic-id,word,lift\n");
repS = fopen(strcat(save,'/topic_stats.csv'),'w');
fprintf(repS, "topic-id,proportion,eff-no-words\n");

nW = 20;
K = size(beta,1);
L = size(lambda,1);

%  access tok like:  tok{1}(20) 

df = sum(topic_type,1);
df = df / sum(df);
% df = df.';
tf = sum(doc_topic,1);
[tc, tx] = sort(tf,'descend');
ptf = tf / sum(tf);

doc = sum(doc_topic,2);

fprintf(rep, 'Effective no. topics=%f/%d\n', ent(ptf), K);
for ik = 1:K
    k = tx(ik);
    nTheta = topic_type(k,:);
    thisBeta = beta(k,:);% 
    nT = sum(nTheta);
    nB = sum(thisBeta);
    thisTheta = (nTheta + thisBeta) / (nT + nB);
    [out,idx] = sort(thisTheta,'descend');
    % size(thisTheta)
    fprintf(rep, 'Topic %d probability=%f eff.no.words=%f\n  FREQ: ', k, ptf(k), ent(thisTheta));
    topicwords(k) = "";
    fprintf(repS,"%d,%f,%f\n", k, ptf(k), ent(thisTheta));
    
   for i = 1:nW
        nm = tok{1}(idx(i));
        fprintf(rep, ' %s (%f)', nm{1}, out(i));
       topicwords(k) = strcat(topicwords(k),strcat(" ",nm{1}));
       fprintf(repW, '%d,%s,%f\n', k, nm{1}, out(i));
   end
    %  compute surprises
    fprintf(rep, '\n  SURPRISE: ');
    nTheta = topic_type(k,:);
    nTheta = max(0,nTheta - 1.5 * sqrt(nTheta));
    nT = sum(nTheta);   
    thisTheta = (nTheta + thisBeta)/(nT + nB);
    weightTheta = thisTheta ./ df;
    [out,idx] = sort(weightTheta,'descend');
     for i = 1:nW
        nm = tok{1}(idx(i));
        fprintf(rep, ' %s', nm{1});
        fprintf(repL, "%d,%s,%f\n", k, nm{1}, out(i));
     end
    
    %  compute top docs
    dt = doc_topic(:,k) ./ doc;
    [out,idx] = sort(dt,'descend');
    fprintf(rep, '\n  Top docs: \n');
    for i = 1:10
        if printdocs
            tw = doctxt(idx(i));
            sl = size(tw{1},2);
            if sl>600 
                sl = 600;
            end
            fprintf(rep, ' (%d) %s\n', idx(i), tw{1}(1:sl));
        else
            fprintf(rep, ' %d', idx(i));
        end
    end
    fprintf(rep,"\n");
    if printdocs==0
       fprintf("\n");
    end
end

%  last "label" is the base case, so ignore
[~,lbidx] = sort(label{1});

for li = 1:L-1
    if li <= size(lbidx,1)
      l = lbidx(li);
      if l<L 
       lpr = lambda(l,:);
       lpr = lpr / sum(lpr);
       nm = label{1}(l);
       fprintf(rep,"label %d::%s  eff.no.topics=%f\n", l, nm{1}, ent(lpr) );
       [out,idx] = sort(lpr,'descend');
       fprintf(rep,"  top topics are\n");
       for i = 1:6
            fprintf(rep,"  %d (%f) :: %s\n", idx(i), lpr(idx(i)), topicwords(idx(i)));
       end
       fprintf(rep,"\n");
      end
    end
end
fclose(rep);
fclose(repW);
fclose(repS);
fclose(repL);

val = 1;
