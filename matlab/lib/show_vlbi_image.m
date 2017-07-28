function show_vlbi_image(x,Nx,Ny)
% function show_image(x,Nx,Ny)

clf
colormap(hot)
imagesc(flipud(reshape(x,Nx,Ny)'))
axis equal off

colorbar

end