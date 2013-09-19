
%load_random_data;
load Intel218Questions
feat_types_printable  = {'Raw','Gradiometers','Magnetometers','Euclidean Norm of Gradiometers','Windowed Mean',...
    'Windowed Slope','STFT Power','STFT Phase','Continuous Haar Wavelet'};
feat_types_printable_short  = {'Raw','Grads','Mags','Grad Norm','W Mean',...
    'W Slope','Power','Phase','Haar'};
nfeats = 9;
%% p values for POVE, all subject results

Intel218Questions_noperson = Intel218Questions(1:218 ~= 13);

for i = 1:length(Intel218Questions_noperson),
    cur_str = Intel218Questions_noperson{i};
    cur_str(2:end) = lower(cur_str(2:end));
    Intel218Questions_noperson{i}=cur_str;
end

subj_size = size(all_pove,2);
q_size = length(Intel218Questions_noperson );
topn_target = 5;
p_pass = zeros(nfeats,subj_size,217);

metas_to_keep = [1 2 3 7 8];
meta_results = zeros(nfeats,length(metas_to_keep));

all_null_mean =0;
total_above = zeros(nfeats,1);
for i = 1:nfeats,
    
    if i>10,
        
        rtmp = squeeze(mean(all_data(1).all_pove(:,:,1:218~=13),2));
    
    else 
        rtmp = squeeze(mean(all_data(i).all_pove(:,:,1:218~=13),2));
    
    end
    fprintf('Feat %s mean null %.4f\n',feat_types{i},mean(rtmp(:)));
    all_null_mean = all_null_mean+mean(rtmp(:));
    
    atmp = squeeze(mean(all_pove(i,:,:,1:218~=13),3));
    
    c = numel(atmp);
    c_m = sum(1./(1:c));
    cutoffs=(1:c)/c/c_m*0.05/nfeats/217;
    
    %     ps = 1-normcdf(atmp(:),mean(rtmp(:)),std(rtmp(:)));
    %     thresh=sum(ps' < cutoffs);
    %     fprintf('feat %s %i\n',feat_types{i},thresh);
    %     [y1,ind]=sort(ps);
    %     all_subjs = [];
    %     all_qs = [];
    %     for a = 1:thresh,
    %         cur_q = rem(ind(a),length(Intel218Questions_noperson))+1;
    %         cur_s = ceil(ind(a)/length(Intel218Questions_noperson));
    %         fprintf('\t\tQ %s S %i %f\n',Intel218Questions_noperson{cur_q},cur_s,y(a));
    %         all_subjs(end+1)=cur_s;
    %         all_qs(end+1)=cur_q;
    %     end
    %     unique(all_subjs)
    %     unique(all_qs)
    
    [f,x]=ecdf(rtmp(:));
    
    yval = interp1(x(2:end),f(2:end),atmp(:),'linear');
    %     pp=spline(x(2:end),f(2:end));
    %     yval=ppval(pp,atmp(:));
    yval(yval>1)=1;
    yval(isnan(yval(:)) & atmp(:)>0)=1;
    yval(isnan(yval(:)) & atmp(:)<0)=0;
    yval(yval<0)=0;
    ps=1-yval;
    psr_thresh = reshape(ps' < cutoffs,size(atmp));
    p_pass(i,:,:)=psr_thresh;
    
    %     for a = 1:length(atmp(:)),
    %         ecdf_cut = 1-f(find(x>atmp(a),1,'first'));
    %         if isempty(ecdf_cut),
    %             ps(a)=0;
    %         else
    %             ps(a) = ecdf_cut;
    %         end
    %     end
    [y2,ind]=sort(ps);
    thresh=sum(ps' < cutoffs);
    %fprintf('feat %s %i\n',feat_types{i},thresh);
    [y,ind]=sort(ps);
    all_subjs = [];
    all_qs = [];
    subj_qs = repmat({{}},q_size,1);
    
    for a = 1:thresh,
        cur_q=ceil(ind(a)/subj_size);
        cur_s=ind(a)-floor(ind(a)/subj_size)*subj_size;
        if cur_s == 0,
            cur_s = subj_size;
        end
        %         cur_q = rem(ind(a),length(Intel218Questions_noperson))+1;
        %         cur_s = ceil(ind(a)/length(Intel218Questions_noperson));
        all_subjs(end+1)=cur_s;
        all_qs(end+1)=cur_q;
        tmp=subj_qs{cur_q};
        tmp_struct = struct('subj',cur_s,'p',y(a),'p_adj',cutoffs(a));
        tmp{end+1} = tmp_struct;
        subj_qs{cur_q}=tmp;
        %fprintf('\t\tQ %s S %i %f\n',Intel218Questions_noperson{cur_q},cur_s,y(a));
    end
    
    make_table=0;
    if make_table==1,
        %        if length(subj_qs{q}) > 1,
        %            fprintf('\t\tQ %s # %i\n',Intel218Questions_noperson{q},length(subj_qs{q}));
        %        end
        fprintf('\\begin{table}[htdp]\n');
        fprintf('\\caption{The most decodable semantic features using the %s MEG feature.\n',feat_types_printable{i});
        fprintf('Questions are ordered by the number of subjects for which that question could be decoded, and by average p-value threshold.  ');
        fprintf('Only the top %i per number of subject group are shown.}\n', topn_target);
        fprintf('\\small{\n');
        fprintf('\\begin{center}\n');
        fprintf('\\begin{tabular}{|l|l|l|}\n');
        fprintf('\\hline\n');
        fprintf('Semantic Feature & Average threshold & Subjs above threshold\\\\ \\hline\n');
        for nq = length(subj_ids):-1:1,
            %         fprintf('NQ^^^^^^^^^^^^^^^%i\n',nq)
            cur_inds = [];
            cur_ps = [];
            for q = 1:length(subj_qs),
                if length(subj_qs{q})== nq,
                    cur_subjs = subj_qs{q};
                    cur_inds(end+1) = q;
                    cur_ps(end+1) =0;
                    for sq = 1:length(cur_subjs),
                        cur_ps(end) = cur_ps(end) + cur_subjs{sq}.p_adj;
                    end
                    cur_ps(end) = cur_ps(end)/nq;
                end
            end
            if ~isempty(cur_ps),
                fprintf('\\multicolumn{3}{|c|}{Semantic features with %i subjects above threshold (%i total)}\\\\\n \\hline\n',...
                    nq,length(cur_ps))
                
                [y,ind]=sort(cur_ps);
                topn = min(length(ind),topn_target);
                for q = ind(1:topn),
                    fprintf(' %s & $%.4f*10^{-5}$ & ',...
                        Intel218Questions_noperson{cur_inds(q)}, cur_ps(q)*10^5);
                    cur_subjs = subj_qs{cur_inds(q)};
                    subj_vec = [];
                    for sq = 1:length(cur_subjs),
                        subj_vec(end+1) = cur_subjs{sq}.subj;
                    end
                    subj_vec = sort(subj_vec);
                    for sq = 1:length(subj_vec),
                        if sq>1,
                            fprintf(', ');
                        end
                        fprintf('%i',subj_vec(sq));
                    end
                    fprintf(' \\\\\n');
                end
                
                fprintf(' \\hline\n')
            end
            
        end
        
        fprintf('\\end{tabular}\n');
        fprintf('\\end{center}\n');
        fprintf('}\n');
        fprintf('\\label{t.topn_%s}\n',feat_types{i});
        fprintf('\\end{table}\n\n\n\n');
    end
    
    
    c=217;
    c_m = sum(1./(1:c));
    cutoffs=(1:c)/c/c_m*0.05/nfeats;
    
    psr = reshape(ps',size(atmp));
    chi_p = 1-chi2cdf(-2*sum(log(psr)),size(atmp,1)*2);
    
    [chisort,chi_i] = sort(chi_p);
    
%     psr_threshed = reshape(ps',size(atmp));
%     
%     psr_threshed(reshape(ps' < cutoffs,size(atmp))) = NaN;
%     psr=psr_threshed; 
%     chi_p = chi2pdf(-2*sum(log(psr)),size(atmp,1)*2);
%     [chisort,chi_i]=sort(chi_p);
    
    chi_inds = (chisort < cutoffs);

    chi_top=min(sum(chi_inds),10);
    
    fprintf('\n\n\n***********\n%s total above threshold: %i\n\n',feat_types_printable{i},sum(chi_inds));
    total_above(i) = sum(chi_inds);
    
    for chii= 1:chi_top,
        fprintf('%s\t%f\n',Intel218Questions_noperson{chi_i(chii)},...
            chisort(chii));
    end
    
    load my_meta
    meta.alive = meta.alive(meta.alive ~=13);
    meta.alive = [meta.alive meta.manmade meta.electronics];
    meta = rmfield(meta,'manmade');
    meta = rmfield(meta,'electronics');
    fld = fields(meta);
    fld = fld(metas_to_keep);
    
    fprintf('\n\n\n')
    for fi = 1:length(fld),
        cur_inds = meta.(fld{fi});
        cur_inds(cur_inds >13) = cur_inds(cur_inds >13) -1;
        tot_above = 0;
        for tot_i = 1:length(cur_inds),
            tot_above = tot_above + chi_inds(chi_i==cur_inds(tot_i));
        end
        fprintf('Meta category %s percent significant: %.2f\n',fld{fi},...
        tot_above/numel(meta.(fld{fi}))*100);
        meta_results(i,fi) = tot_above/numel(meta.(fld{fi}))*100;
    end
    fprintf('\n\n\n')
    
    subplot(2,1,1)
    hist(atmp(:),30)
    xlim([-0.4 0.75])
    title(sprintf('Real data %s',feat_types{i}))
    subplot(2,1,2)
    hist(rtmp(:),30)
    xlim([-0.4 0.75])
    title(sprintf('Random data %s',feat_types{i}))
    %waitforbuttonpress
    
    subplot(1,1,1)
    hmin = min(min(atmp(:)),min(rtmp(:)))-0.01;
    hmax = max(max(atmp(:)),max(rtmp(:)))+0.01;
    hstep = (hmax-hmin)/40;
    hinds = hmin:hstep:hmax;
    nr=histc(rtmp(:),hinds);
    n=histc(atmp(:),hinds);
    h=bar(hinds,[nr./sum(nr) n./sum(n)]*100,'histc');
    xlim([hmin,hmax])
    legend({'Permuted','True'},'location','Best','fontsize',14);
    xlabel('Percent Variance Explained','fontsize',14)
    ylabel('Probability','fontsize',14)
    title(sprintf('Empirical PDF of POVE for %s MEG Feature',feat_types_printable{i}),'fontsize',14)
    %waitforbuttonpress
    save2pdf(sprintf('~/research/fmri/graphs/feat_proj/l2_zscore_prefold_hist_%s_vs_null.pdf',...
        feat_types{i}));
end
fprintf('\n')
fprintf('All null mean is %.4f\n',all_null_mean/nfeats);

%%
% perform stat analysis on distances


nfeats = length(feat_types);
all_null_mean =0;
total_above = zeros(nfeats,1);
for i = 1:nfeats,
    
    rtmp = squeeze(median(all_rank_rand(i,:,:,:),4));
    fprintf('Feat %s mean null %.4f\n',feat_types{i},mean(rtmp(:)));
    all_null_mean = all_null_mean+mean(rtmp(:));
    atmp = squeeze(median(rank_1000_all(i,:,4,:),4));
    [f,x]=ecdf(rtmp(:));
    yval = interp1(x(2:end),f(2:end),atmp(:),'linear');
    %     pp=spline(x(2:end),f(2:end));
    %     yval=ppval(pp,atmp(:));
    yval(yval>1)=1;
    yval(isnan(yval(:)))=f(2);
   
    ps=yval;
    psr = reshape(ps',size(atmp));
    chi_p = 1-chi2cdf(-2*sum(log(psr)),length(atmp)*2);
    fprintf('%f, %.4f, %.4f\n',10^13*chi_p, 100*(1-max(atmp/941)), 100*(1-min(rtmp(:)/941)));
    fprintf('%f, %.4f, %.4f\n',10^13*chi_p,100*( 1-min(atmp)/941), max(rtmp(:)));
end


%%

  load my_meta
    meta.alive = meta.alive(meta.alive ~=13);
    meta.alive = [meta.alive meta.manmade meta.electronics];
    meta = rmfield(meta,'manmade');
    meta = rmfield(meta,'electronics');
    fld = fields(meta);
    fld = fld(metas_to_keep);
    
    
fprintf('MEG feature & ');
fprintf('%s & ',fld{:});
fprintf('All \\\\ \n ');
for i = 1:size(meta_results,1),
    fprintf('%s & ',feat_types_printable_short{i});
    fprintf('%.1f \\%% & ',meta_results(i,:));
    fprintf('%.1f \\%% \\\\ \n ',100*total_above(i)/217);
end

%%
% for presenation

for f = 1:size(meta_results,2),
    fprintf('\n\n\n\\multicolumn{1}{|c|}{MEG feature} & ');
    fprintf('\\multicolumn{1}{|c|}{%s} \\\\\n\\hline\n',fld{f});
    for i = 1:size(meta_results,1),
        
        fprintf('%s & ',feat_types_printable_short{i});
        fprintf('%.1f \\%% \\\\\n',meta_results(i,f));
       % fprintf('%.1f \\%% \\\\ \n ',100*total_above(i)/217);
    end
end

fprintf('\n\n\n\\multicolumn{1}{|c|}{MEG feature} & ');
    fprintf('\\multicolumn{1}{|c|}{Mean} \\\\\n\\hline\n');
    for i = 1:size(meta_results,1),
        
        fprintf('%s & ',feat_types_printable_short{i});
        %fprintf('%.1f \\%% \\\\\n',meta_results(i,f));
       fprintf('%.1f \\%% \\\\ \n ',100*total_above(i)/217);
    end


%% p value of all subjects, 2 vs 2 results

for i = 1:nfeats,
    
    
    rtmp = all_twos_rand(i,:);
    rtmp = rtmp(~isnan(rtmp(:)));
    
    fprintf('Feat %s mean null %.4f\n',feat_types{i},mean(rtmp(:)));
    
    cur_vals = all_twos(i,:);
    cur_vals = cur_vals(~isnan(cur_vals));
    
    atmp = 1-chi2cdf(-2*sum(log(cur_vals)),length(cur_vals)*2);
    
    [f,x]=ecdf(rtmp(:));
    
    yval = interp1(x(2:end),f(2:end),atmp,'linear');
    yval(yval>1)=1;
    yval(isnan(yval(:)) & atmp(:)>0)=1;
    yval(isnan(yval(:)) & atmp(:)<0)=0;
    yval(yval<0)=0;
    ps=1-yval;
    
    if ps <= 0.05/nfeats,
       fprintf('%s passed\n',feat_types{i});
    else
       fprintf('***********%s failed\n',feat_types{i});
    end
    
    
end


%% mean of subjects
Intel218Questions_noperson = Intel218Questions(1:218 ~= 13);

for i =1:6,
    rtmp = squeeze(mean(all_data(i).all_pove(:,:,1:218~=13),2));
    atmp = squeeze(mean(mean(all_pove(i,:,:,1:218~=13),3),2));
    
    c = numel(atmp);
    c_m = sum(1./(1:c));
    cutoffs=(1:c)/c/c_m*0.05/7;
    
    %     ps = 1-normcdf(atmp(:),mean(rtmp(:)),std(rtmp(:)));
    %     thresh=sum(ps' < cutoffs);
    %     fprintf('feat %s %i\n',feat_types{i},thresh);
    %     [y1,ind]=sort(ps);
    %     for a = 1:thresh,
    %         cur_q = rem(ind(a),length(Intel218Questions_noperson))+1;
    %         fprintf('\t\tQ %s %f\n',Intel218Questions_noperson{cur_q},y(a));
    %     end
    
    %     ps(:)=0;
    %     [f,x]=ecdf(rtmp(:));
    %     for a = 1:length(atmp(:)),
    %         ecdf_cut = 1-f(find(x>atmp(a),1,'first'));
    %         if isempty(ecdf_cut),
    %             ps(a)=0;
    %         else
    %             ps(a) = 1-f(find(x>atmp(a),1,'first'));
    %         end
    %     end
    
    ps(:)=0;
    [f,x]=ecdf(rtmp(:));
    
    yval = interp1(x(2:end),f(2:end),atmp(:),'linear');
    
    %     pp=spline(x(2:end),f(2:end));
    %     yval=ppval(pp,atmp(:));
    yval(yval>1)=1;
    yval(yval<0)=0;
    ps=1-yval;
    
    [y2,ind]=sort(ps);
    thresh=sum(ps' < cutoffs);
    fprintf('feat %s %i\n',feat_types{i},thresh);
    [y,ind]=sort(ps);
    for a = 1:thresh,
        cur_q = rem(ind(a),length(Intel218Questions_noperson))+1;
        fprintf('\t\tQ %s %f\n',Intel218Questions_noperson{cur_q},y(a));
    end
    
    subplot(2,1,1)
    hist(atmp(:),30)
    xlim([-0.4 0.75])
    title(sprintf('Real data %s',feat_types{i}))
    subplot(2,1,2)
    hist(rtmp(:),30)
    xlim([-0.4 0.75])
    title(sprintf('Random data %s',feat_types{i}))
    %waitforbuttonpress
    
    subplot(1,1,1)
    hmin = min(min(atmp(:)),min(rtmp(:)))-0.01;
    hmax = max(max(atmp(:)),max(rtmp(:)))+0.01;
    hstep = (hmax-hmin)/40;
    hinds = hmin:hstep:hmax;
    nr=histc(rtmp(:),hinds);
    n=histc(atmp,hinds);
    bar(hinds,[nr./sum(nr) n./sum(n)],'histc')
    xlim([hmin,hmax])
    legend({'Permuted data','True data'},'location','NorthWest','fontsize',14);
    xlabel('Percent Variance Explained','fontsize',14)
    ylabel('Probability','fontsize',14)
    title(sprintf('Empirical PDF of POVE for %s MEG Feature',feat_types_printable{i}),'fontsize',14)
    %waitforbuttonpress
    save2pdf(sprintf('~/research/fmri/graphs/feat_proj/l2_zscore_prefold_hist_%s_vs_null.pdf',...
        feat_types{i}));
end
fprintf('\n')

%% for all subjects
hmin = min(min(atmp(:)),min(rtmp(:)))-0.01;
hmax = max(max(atmp(:)),max(rtmp(:)))+0.01;
hstep = (hmax-hmin)/30;
hinds = hmin:hstep:hmax;
nr=histc(rtmp(:),hinds);
n=histc(atmp,hinds);
bar(hinds,[nr./sum(nr) n./sum(n)],'histc')
xlim([hmin,hmax])
legend({'Permuted data','True data'},'location','best');

%%

e_correct = zeros(1,218);
for i =1:218,
    [b,~,j]=unique(sem_matrix(:,i));
    e_correct(i)=sum(([sum(j==1) sum(j==2) sum(j==3) sum(j==4) sum(j==5)]./60).^2);
end
ec_sub = e_correct(1:218~=13);

for i = 1:6,
    rtmp = squeeze(mean(all_data(i).all_pove(:,:,1:218~=13),2));
    atmp = squeeze(mean(all_pove(i,:,:,1:218~=13),3));
    
    num_perms = size(all_data(i).all_pove,1);
    am = mean(atmp);
    amrep = repmat(am,[num_perms,1]);
    ecrep = repmat(ec_sub,[num_perms,1]);
    
    subplot(2,2,1)
    hist(reshape(rtmp,[1,numel(rtmp)]),100)
    title('Histogram of POVE on permuted data')
    subplot(2,2,2)
    hist(reshape(rtmp-amrep,[1,numel(rtmp)]),100)
    title('Histogram adjusted for actual performance')
    fprintf('%s\t%.3f\t%.3f\t%.3f\n',feat_types_printable{i},mean(rtmp(:)),mean(reshape(rtmp-amrep,[1,numel(rtmp)])),mean(atmp(:)));
    
    subplot(2,2,3)
    am_sorted = sort(am);
    p=polyfit(am,mean(rtmp),2);
    y2=polyval(p,am_sorted);
    plot(amrep(:),rtmp(:),'ro','Color',[0.7 0.7 0.7])
    hold on
    plot(am,mean(rtmp),'rx','linewidth',2,'MarkerSize',8)
    plot(am_sorted,y2)
    hold off
    xlabel('actual performance')
    ylabel('mean random performance');
    axis tight;
    title(sprintf('%s -- Actual performance vs mean random performance',feat_types_printable{i}))
    %     title(sprintf('%s -- Actual performance vs mean random performance %.2ft^2 + %.2ft + %.2f',...
    %         feat_types_printable{i},p(1),p(2),p(3)))
    
    subplot(2,2,4)
    ec_sorted = sort(ec_sub);
    p=polyfit(ec_sub,mean(rtmp),2);
    y2=polyval(p,ec_sorted);
    plot(ecrep(:),rtmp(:),'ro','Color',[0.7 0.7 0.7])
    hold on
    plot(ec_sub,mean(rtmp),'rx','linewidth',2,'MarkerSize',8)
    plot(ec_sorted,y2)
    hold off
    xlabel('expected % with same label')
    ylabel('mean random performance');
    axis tight;
    title(sprintf('%s -- Expected same label vs mean random performance ',feat_types_printable{i}))
    %     title(sprintf('%s -- Expected same label vs mean random performance %.2ft^2 + %.2ft + %.2f',...
    %         feat_types_printable{i},p(1),p(2),p(3)))
    
    waitforbuttonpress;
    save2pdf(sprintf('~/research/fmri/graphs/feat_proj/l2_zscore_%s_actual_vs_permuted_pove.pdf',...
        feat_types{i}));
end

%%

e_correct = zeros(1,218);
for i =1:218,
    [b,~,j]=unique(sem_matrix(:,i));
    e_correct(i)=sum(([sum(j==1) sum(j==2) sum(j==3) sum(j==4) sum(j==5)]./60).^2);
end
ec_sub = e_correct(1:218~=13);
ec_sorted = sort(ec_sub);

for i = 1:1,
    rtmp = squeeze(mean(all_data(i).all_pove(:,:,1:218~=13),2));
    atmp = squeeze(mean(all_pove(i,:,:,1:218~=13),3));
    
    num_perms = size(all_data(i).all_pove,1);
    am = mean(atmp);
    amrep = repmat(am,[num_perms,1]);
    ecrep = repmat(ec_sub,[num_perms,1]);
    
    
    subplot(2,2,1)
    hist(reshape(rtmp,[1,numel(rtmp)]),100)
    axis tight;
    title('Histogram of POVE on permuted data')
    subplot(2,2,2)
    hist(reshape(rtmp-amrep,[1,numel(rtmp)]),100)
    axis tight;
    title('Histogram adjusted for actual performance')
    fprintf('%s\t%.3f\t%.3f\t%.3f\n',feat_types_printable{i},mean(rtmp(:)),mean(reshape(rtmp-amrep,[1,numel(rtmp)])),mean(atmp(:)));
    
    %     subplot(2,2,3:4)
    %     am_sorted = sort(am);
    %     p=polyfit(am,mean(rtmp),2);
    %     y2=polyval(p,am_sorted);
    %     plot(amrep(:),rtmp(:),'ro','color',[0.8 0.8 0.8])
    %     hold on
    %     plot(am,mean(rtmp),'rx','linewidth',2)
    %     plot(am_sorted,y2)
    %     hold off
    %     xlabel('Actual POVE')
    %     ylabel('POVE on permuted data');
    %     title(sprintf('Actual POVE vs POVE on permuted data (%s MEG Feature)',feat_types_printable{i}));%,...       p(1),p(2),p(3)))
    %     axis tight;
    %     legend({'POVE perm.','Mean POVE perm.','LS fit'},'location','northwest')
    %     waitforbuttonpress;
    %     save2pdf(sprintf('~/research/fmri/graphs/feat_proj/l2_zscore_%s_actual_vs_permuted_pove.pdf',...
    %         feat_types{i}));
    
    
    
    subplot(2,2,3:4)
    am_sorted = sort(am);
    p=polyfit(ec_sub,mean(rtmp-amrep),2);
    y2=polyval(p,ec_sorted);
    plot(ecrep(:),rtmp(:)-amrep(:),'ro','color',[0.8 0.8 0.8])
    hold on
    plot(ec_sub,mean(rtmp-amrep),'rx','linewidth',2)
    plot(ec_sorted,y2)
    hold off
    xlabel('estimated percent same label')
    ylabel('POVE on permuted data');
    title(sprintf('estimated percent right vs adjusted POVE on permuted data (%s MEG Feature)',feat_types_printable{i}));%,...       p(1),p(2),p(3)))
    axis tight;
    % legend({'POVE perm.','Mean POVE perm.','LS fit'},'location','northwest')
    waitforbuttonpress;
    save2pdf(sprintf('~/research/fmri/graphs/feat_proj/l2_zscore_%s_actual_vs_permuted_pove.pdf',...
        feat_types{i}));
end

%%

for cur_subj = 1:9
    %cur_subj = 1:9,
    %subplot(3,3,cur_subj);
    load my_meta
    meta.alive = meta.alive(meta.alive ~=13);
    meta.alive = [meta.alive meta.manmade meta.electronics];
    meta = rmfield(meta,'manmade');
    meta = rmfield(meta,'electronics');
    f = fields(meta);
    f = f([1 2 3 7 8]);
    
    cbar=zeros(length(f),9);
    cbar_err=zeros(length(f),9);
    for i = 1:length(f),
        cur_data = all_pove(:,cur_subj,:,meta.(f{i}));
        cbar(i,:) =mean(squeeze(mean(mean(cur_data,3),4)),2);
        cbar_err(i,:) =std(reshape(cur_data,1,[]))/sqrt(numel(cur_data(1,:)));
        
    end
    
    h = barweb(cbar, 1.96*cbar_err);
    
    set(h.bars,'lineWidth',1)
    set(h.errors,'lineWidth',1)
    set(h.ax,'lineWidth',1)
    yl = get(h.ax,'ylim');
    axis([0.5 length(f)+0.5 yl(1) yl(2)*1.1])
    
    set(gca,'xticklabel',f,'fontsize',14);
    if length(c)==1,
        
        title(sprintf('POVE for meta-categories for subject %i',cur_subj));
    else
        title(sprintf('POVE for meta-categories for all subjects'));
    end
    
    legend(feat_types_printable_short,'fontsize',14,'location','best')
    %axis tight;
    waitforbuttonpress
end



%%



