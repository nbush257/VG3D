function [label_out,last_tracked] = manual_remove_bad_contacts(X,label_in,varargin)
%% function label = manual_label(X,C,[start],[win])
% =========================================================================
% this function takes in either a boolean C vector or an integer label
% vector in order to manually clean up contact. It allows you to label as
% either: contact(1), noncontact(-1), or uncertain(0). We will try to use
% the uncertain data to look back at the video if needed.
% =========================================================================
% NEB 20170731
%% input handling and window init
if length(varargin)==2
    win = varargin{2};
    starts = varargin{1};
elseif length(varargin)==1
    starts = varargin{1};
    win=5000;
else
    starts = 1;
    win=5000;
end
stops = starts + win-1;
tracking = true;

last_tracked=starts;
if isempty(label_in)
    label_in = false(size(X,1),1);
end
label_out = label_in;
[~,ccstarts,ccends] = convertContact(label_in);

%% init ui key mappings
add_to_contact = uint8(['a']);
remove_from_contact = uint8(['s']);
uncertain = uint8(['d']);
skip = uint8(['q']); % Need to depricate
%% init figure
close all
f = figure('units','normalized','outerposition',[0 .5 1 .4]);
mTextBox = uicontrol('style','text');
set(mTextBox,'units','normalized','Position',[.01 .5 .1 .1]);
mTextBox.BackgroundColor = 'w';
mTextBox.HorizontalAlignment = 'left';
s_legend = sprintf('Space = advance with labelling\na = contact\ns = not contact\nd = label as unknown\nleft click to remove');
set(mTextBox,'String',s_legend);
%% Get lims
ftemp = figure();
plot(X)
title('click the upper and lower poinds of where to plot')
[~,yy] = ginput(2);
close(ftemp)
yy = sort(yy);


%% Start UI tracking
try
    while tracking==true % Loop over windows
        if stops>length(label_in)
            tracking=false;
            stops = length(label_in);
        end
        window_c = label_out(starts:stops)==1;
        cstarts = ccstarts-starts+1;
        cends = ccends-starts+1;
        
%         [~,cstarts,cends] = convertContact(window_c);
        
        last_tracked = starts-1;
        x = 0;
        % Loop UI over contact intervals
        while ~isempty(x)
            cla
            x = [];
            but_press = [];
            plot(1:win,X(starts:stops,:),'linewidth',1);     
%             ylim([min(nanmin(X)) max(nanmax(X))]);
            ylim([yy(1),yy(2)])
            xlim([0,win])
            % use temp var booleans for shading
            tempC = label_out==1;
            
            
            shadeVector(tempC(starts:stops),'k');
            
            
            title_string = sprintf('Frames: %i  to  %i',starts,stops);
            title(title_string)
            
            % get first UI
            [x,~,but_press] = ginput(1);
            x = round(x);
            if isempty(but_press) || isempty(x)
                break
            end
            
            
            % if UI is skip, then skip the labelling of this window
            if ismember(but_press,skip)
                break
            end
            
            % check for spacebar or enter to continue to next frame
            if but_press == 32 || isempty(x)
                break
            end

            
            % align to global vector position
            
            
            % spacebar or enter advances the frames
            if any(but_press == 32) || isempty(x)
                break
            end
            temp = x-cstarts;
            [~,contact_idx] = min(temp(temp>0));
            nearest_start = cstarts(contact_idx);
            nearest_end = cends(contact_idx);
            if x>nearest_end
                continue
            end
            
%             if ismember(but_press,add_to_contact) || but_press==3
%                 label_out(starts+nearest_start:starts+nearest_end)=1;
%             elseif ismember(but_press,remove_from_contact) || but_press==1
%                 label_out(starts+nearest_start:starts+nearest_end)=0;
%             end
%             
            if label_out(starts+x)
                label_out(starts-1+nearest_start:starts+nearest_end)=0;
            else
                label_out(starts-1+nearest_start:starts+nearest_end)=1;
            end
            
        end
        
        % advance window
        starts =stops+1;
        stops = starts+win-1;
        save('temp.mat','label_out','last_tracked')
    end
    last_tracked = stops;
catch
    warning('caught an error, returning...')
    close all
    return
end
close all
