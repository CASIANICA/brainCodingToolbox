% config
root_dir = '/home/huanglijie/workingdir/brainDecoding';
stim_dir = fullfile(root_dir, 'stimulus');
stim_file = fullfile(stim_dir, 'Stimuli.mat');

% stimuli processing
load(stim_file)
s = size(st);
saliences = zeros(s(1), s(2), s(4));

% compute the saliency maps for the sequence
% get default GBVS params, but compute only 'I' intensity and
% 'F' flicker channels, and reduce # of levels for speedup
param = makeGBVSParams;
param.channels = 'IF';
param.levels = 3;
% previous frame information, initialized to empty
motinfo = [];
for i = 1:s(4)
    [out motinfo] = gbvs(st(:, :, :, i), param, motinfo);
    saliences(:, :, i) = out.master_map_resized;
end
% save output
outfile = fullfile(stim_dir, 'train_stim_salience.mat');
save(outfile, 'saliences')
