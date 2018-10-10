function me = ent(prob)
up = sort(prob,'descend');
me = 0;
dim = 0;
if size(up,1)==1
    dim = 2;
else
    if size(up,2)==1
        dim =  1;
    end
end
for i = 1:size(up,dim)
    if up(i)>0
        me = me - up(i) * log(up(i));
    end
end
me = exp(me);
