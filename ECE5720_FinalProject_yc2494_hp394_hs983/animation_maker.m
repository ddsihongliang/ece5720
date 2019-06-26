clc
clear all

%%
loops = 1000;
G(loops) = struct('cdata',[],'colormap',[]);

f= figure('Position', [10 10 860 860]);
whitebg('k');
set(gcf,'Color','k');

a1 = load('0.csv');

for i = 0:loops
f = string(i);
a = load(f+'.csv');

p = scatter3(a(:,1),a(:,2),a(:,3),2,[a1(:,1)],'filled');
% hold on 
%scatter(a(1:3,1),a(1:3,2),20,'r');
% hold off

view(i/10,15)
axis manual
ylim([-1000 1000])
xlim([-1000 1000])
zlim([-1000 1000])
G(i+1) = getframe(gcf,[0 0 800 800]);              
drawnow
end

v = VideoWriter('sphere.avi');
v.Quality = 100;
v.FrameRate = 18;
open(v)
writeVideo(v,G);
close(v)

