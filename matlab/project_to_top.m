function project_to_top(to_E3D_file,tracked_3D_file,toMerge_file,out_file)
load(to_E3D_file,'CP','E3D_flag','C')
load(tracked_3D_file,'frame_size','calibInfo')
load(toMerge_file,'tws')
%%
CP_2D = nan(length(C),2);
BP = nan(length(C),2);
for ii=1:length(C)
    if ~E3D_flag(ii)
        continue
    end
    if length(tws(ii).x)==0
        E3D_flag(ii)=0;
        continue
    end
       
%     wIn.x = xw3d{ii};
%     wIn.y = yw3d{ii};
%     wIn.z = zw3d{ii};
%     
    cp_struct.x = CP(ii,1);
    cp_struct.y = CP(ii,2);
    cp_struct.z = CP(ii,3);
       
    
%     [~,top_proj]=BackProject3D(wIn,calibInfo(1:4),calibInfo(5:8),calibInfo(9:10));
    [~,CP_2D(ii,:)] = BackProject3D(cp_struct,calibInfo(1:4),calibInfo(5:8),calibInfo(9:10));
    
%     xw2d{ii} = top_proj(:,1);
%     yw2d{ii} = top_proj(:,2);
    BP(ii,:) = [tws(ii).x(1) tws(ii).y(1)];    
end
xw2d = {tws.x};
yw2d = {tws.y};

[~,CP] = CPonWhisker2D(CP_2D,xw2d,yw2d);
E2D_flag = E3D_flag;
if size(CP,1)~=length(xw2d)
    warning('CP is not the same size for file %i',to_E3D_file)
end

save(out_file,'*w2d','C','CP','E2D_flag','BP','calibInfo','frame_size')
    

%% 
