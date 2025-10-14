function [Rfilter_train]=gaussian_filter(filter_sigma,dt1,sx_train)
            %{
                Function:
                    gaussian filter for 1-dimentional time series
                Input:
                    filter_sigma: std of gaussial fiulter
                    
            %}
            bin=4; % gaussian filter bin :ms
            taus=-bin*filter_sigma:dt1:bin*filter_sigma;
            Gfilter=exp(-taus.^2./(2*filter_sigma^2));
            Gfilter=Gfilter./sum(Gfilter(:));
            Rfilter_train = zeros(size(sx_train));
            for i = 1:size(sx_train,2)
                Rfilter_train(:,i) = conv(sx_train(:,i),Gfilter,'same');
            end
        end