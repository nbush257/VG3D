function pix2m = get_pix2m(to_E2D_file,to_E3D_file)
load(to_E3D_file,'*w3d')
load(to_E2D_file,'calibInfo','*w2d','E2D_flag')
pix2m = nan(length(xw3d),1);
for ii =1:length(xw3d)

    if ~E2D_flag(ii)
        continue
    end
    
    l3d = arclength3d(xw3d{ii},yw3d{ii},zw3d{ii});
    l2d = arclength(xw2d{ii},yw2d{ii});
    pix2mm = l2d/l3d;
    pix2m(ii) = 1/(pix2mm*1000);
end
pix2m = nanmedian(pix2m);