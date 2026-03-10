function export_case(case_name, output_path, upstream_root)
% Export a curated ISETCam parity case from Octave into a MAT file.

if nargin < 3
    error('export_case requires case_name, output_path, and upstream_root');
end

addpath(genpath(upstream_root));
try
    ieInitSession;
catch
    % Some cases only need the path, not a full session.
end

payload = struct();
payload.case_name = case_name;

switch case_name
    case 'scene_macbeth_default'
        scene = sceneCreate();
        payload.wave = sceneGet(scene, 'wave');
        payload.photons = sceneGet(scene, 'photons');
        payload.mean_luminance = sceneGet(scene, 'mean luminance');

    case 'scene_checkerboard_small'
        scene = sceneCreate('checkerboard', 8, 4);
        payload.wave = sceneGet(scene, 'wave');
        payload.photons = sceneGet(scene, 'photons');
        payload.mean_luminance = sceneGet(scene, 'mean luminance');

    case 'scene_uniform_bb_small'
        scene = sceneCreate('uniform bb', 16, 4500);
        payload.wave = sceneGet(scene, 'wave');
        payload.photons = sceneGet(scene, 'photons');
        payload.mean_luminance = sceneGet(scene, 'mean luminance');

    case 'scene_frequency_orientation_small'
        params = struct();
        params.angles = linspace(0, pi / 2, 4);
        params.freqs = [1 2 4 8];
        params.blockSize = 16;
        params.contrast = 0.8;
        scene = sceneCreate('frequency orientation', params);
        payload.wave = sceneGet(scene, 'wave');
        payload.photons = sceneGet(scene, 'photons');
        payload.mean_luminance = sceneGet(scene, 'mean luminance');

    case 'scene_harmonic_small'
        params = struct();
        params.freq = [1 5];
        params.contrast = [0.2 0.6];
        params.ph = [0 pi/3];
        params.ang = [0 0];
        params.row = 64;
        params.col = 64;
        params.GaborFlag = 0.2;
        scene = sceneCreate('harmonic', params);
        payload.wave = sceneGet(scene, 'wave');
        payload.photons = sceneGet(scene, 'photons');
        payload.mean_luminance = sceneGet(scene, 'mean luminance');

    case 'scene_sweep_frequency_small'
        scene = sceneCreate('sweep frequency', 64, 12);
        payload.wave = sceneGet(scene, 'wave');
        payload.photons = sceneGet(scene, 'photons');
        payload.mean_luminance = sceneGet(scene, 'mean luminance');

    case 'utility_unit_frequency_list'
        payload.even = unitFrequencyList(50);
        payload.odd = unitFrequencyList(51);

    case 'utility_energy_quanta_1d'
        wave = (400:10:700)';
        energy = linspace(0.1, 3.1, numel(wave))';
        photons = Energy2Quanta(wave, energy);
        payload.wave = wave;
        payload.energy = energy;
        payload.photons = photons;
        payload.energy_roundtrip = Quanta2Energy(wave, photons);

    case 'utility_energy_quanta_matrix'
        wave = [400 500 600]';
        energy = [0.2 0.4; 0.5 0.7; 0.8 1.0];
        photons = Energy2Quanta(wave, energy);
        payload.wave = wave;
        payload.energy = energy;
        payload.photons = photons;
        payload.energy_roundtrip = Quanta2Energy(wave, photons')';

    case 'utility_blackbody_energy_small'
        wave = (400:10:700)';
        temperatures = [3000 5000];
        payload.wave = wave;
        payload.temperatures = temperatures;
        payload.energy = blackbody(wave, temperatures, 'energy');

    case 'utility_blackbody_quanta_small'
        wave = (400:10:700)';
        temperatures = [3000 5000];
        payload.wave = wave;
        payload.temperatures = temperatures;
        payload.photons = blackbody(wave, temperatures, 'photons');

    case 'utility_ie_param_format_string'
        original = 'Exposure Time';
        payload.original = original;
        payload.formatted = ieParamFormat(original);

    case 'metrics_xyz_from_energy_1d'
        wave = (400:10:700)';
        energy = linspace(0.05, 1.55, numel(wave));
        payload.wave = wave;
        payload.energy = energy;
        payload.xyz = ieXYZFromEnergy(energy, wave);

    case 'metrics_xyz_to_luv_1d'
        xyz = [20.0 30.0 15.0];
        white_point = [95.047 100.0 108.883];
        payload.xyz = xyz;
        payload.white_point = white_point;
        payload.luv = xyz2luv(xyz, white_point);

    case 'metrics_xyz_to_lab_1d'
        xyz = [20.0 30.0 15.0];
        white_point = [95.047 100.0 108.883];
        payload.xyz = xyz;
        payload.white_point = white_point;
        payload.lab = ieXYZ2LAB(xyz, white_point);

    case 'metrics_xyz_to_uv_1d'
        xyz = [20.0 30.0 15.0];
        payload.xyz = xyz;
        [u, v] = xyz2uv(xyz, 'uv');
        payload.uv = [u v];

    case 'metrics_cct_from_uv_1d'
        uv = [0.20029948; 0.31055768];
        payload.uv = uv;
        payload.cct_k = cct(uv);

    case 'metrics_delta_e_ab_1976_1d'
        xyz1 = [20.0 30.0 15.0];
        xyz2 = [18.0 27.0 16.5];
        white_point = [95.047 100.0 108.883];
        payload.xyz1 = xyz1;
        payload.xyz2 = xyz2;
        payload.white_point = white_point;
        payload.delta_e = deltaEab(xyz1, xyz2, white_point, '1976');

    case 'metrics_spd_angle_1d'
        wave = [500 510 520];
        spd1 = [1 0 0];
        spd2 = [0 1 0];
        payload.wave = wave;
        payload.spd1 = spd1;
        payload.spd2 = spd2;
        payload.angle = metricsSPD(spd1, spd2, 'metric', 'angle', 'wave', wave);

    case 'metrics_spd_cielab_1d'
        wave = (400:10:700)';
        spd1 = linspace(0.5, 1.7, numel(wave))';
        spd2 = linspace(1.6, 0.4, numel(wave))';
        [delta_e, params] = metricsSPD(spd1, spd2, 'metric', 'cielab', 'wave', wave);
        spd1_scaled = (spd1 / ieLuminanceFromEnergy(spd1, wave)) * 100;
        spd2_scaled = (spd2 / ieLuminanceFromEnergy(spd2, wave)) * 100;
        white_point = ieXYZFromEnergy(spd1_scaled', wave);
        white_point = (white_point / white_point(2)) * 100;
        payload.wave = wave;
        payload.spd1 = spd1;
        payload.spd2 = spd2;
        payload.delta_e = delta_e;
        payload.xyz1 = ieXYZFromEnergy(spd1_scaled', wave);
        payload.xyz2 = ieXYZFromEnergy(spd2_scaled', wave);
        payload.lab1 = params.lab1;
        payload.lab2 = params.lab2;
        payload.white_point = white_point;

    case 'metrics_spd_mired_1d'
        wave = (400:10:700)';
        spd1 = blackbody(wave, 6500, 'energy');
        spd2 = blackbody(wave, 5000, 'energy');
        [mired, params] = metricsSPD(spd1, spd2, 'metric', 'mired', 'wave', wave);
        payload.wave = wave;
        payload.spd1 = spd1;
        payload.spd2 = spd2;
        payload.mired = mired;
        payload.uv = params.uv;
        payload.cct_k = params.cTemps;

    case 'scene_illuminant_change'
        scene = sceneCreate();
        bb = blackbody(sceneGet(scene, 'wave'), 3000, 'energy');
        scene_preserve = sceneAdjustIlluminant(scene, bb, true);
        scene_no_preserve = sceneAdjustIlluminant(scene, bb, false);
        payload.preserve_mean = sceneGet(scene_preserve, 'mean luminance');
        payload.no_preserve_mean = sceneGet(scene_no_preserve, 'mean luminance');
        payload.preserve_photons = sceneGet(scene_preserve, 'photons');
        payload.no_preserve_photons = sceneGet(scene_no_preserve, 'photons');

    case 'display_create_lcd_example'
        d = displayCreate('lcdExample.mat');
        payload.wave = displayGet(d, 'wave');
        payload.spd = displayGet(d, 'spd');
        payload.gamma = displayGet(d, 'gamma');

    case 'oi_diffraction_limited_default'
        scene = sceneCreate();
        oi = oiCreate();
        oi = oiCompute(oi, scene, 'crop', true);
        payload.wave = oiGet(oi, 'wave');
        payload.photons = oiGet(oi, 'photons');

    case 'oi_wvf_small_scene'
        scene = sceneCreate('checkerboard', 8, 4);
        oi = oiCreate('wvf');
        oiStage = oi;
        optics = oiGet(oiStage, 'optics');
        oiStage = oiSet(oiStage, 'wangular', sceneGet(scene, 'wangular'));
        oiStage = oiSet(oiStage, 'wave', sceneGet(scene, 'wave'));
        oiStage = oiSet(oiStage, 'photons', oiCalculateIrradiance(scene, oiStage));
        offaxismethod = opticsGet(optics, 'offaxismethod');
        switch lower(offaxismethod)
            case {'skip', 'none', ''}
            case 'cos4th'
                oiStage = opticsCos4th(oiStage);
            otherwise
                oiStage = opticsCos4th(oiStage);
        end
        imSize = oiGet(oiStage, 'size');
        padSize = round(imSize / 8);
        padSize(3) = 0;
        sDist = sceneGet(scene, 'distance');
        oiStage = oiPadValue(oiStage, padSize, 'zero photons', sDist);
        payload.pre_psf_photons = oiGet(oiStage, 'photons');

        wavelist = oiGet(oiStage, 'wave');
        flength = oiGet(oiStage, 'focal length', 'mm');
        fnumber = oiGet(oiStage, 'f number');
        oiSize = max(oiGet(oiStage, 'size'));
        wvf = oiStage.optics.wvf;
        wvf = wvfSet(wvf, 'focal length', flength, 'mm');
        wvf = wvfSet(wvf, 'calc pupil diameter', flength / fnumber);
        wvf = wvfSet(wvf, 'wave', wavelist);
        wvf = wvfSet(wvf, 'spatial samples', oiSize);

        psf_spacing = oiGet(oiStage, 'sample spacing', 'mm');
        lambdaM = wvfGet(wvf, 'measured wl', 'm');
        lambdaUnit = 1000 * lambdaM;
        pupil_spacing = lambdaUnit * flength / (psf_spacing(1) * oiSize);
        wvf = wvfSet(wvf, 'field size mm', pupil_spacing * oiSize);
        wvf = wvfCompute(wvf);
        PSF = wvfGet(wvf, 'psf');
        if ~iscell(PSF)
            tmp = PSF; clear PSF; PSF{1} = tmp;
        end
        psf_stack = zeros(size(PSF{1}, 1), size(PSF{1}, 2), numel(PSF), 'single');
        for ii = 1:numel(PSF)
            psf_stack(:, :, ii) = PSF{ii};
        end
        payload.psf_stack = psf_stack;

        oi = oiCompute(oi, scene, 'crop', true);
        payload.wave = oiGet(oi, 'wave');
        payload.photons = oiGet(oi, 'photons');

    case 'sensor_bayer_noiseless'
        scene = sceneCreate();
        oi = oiCreate();
        oi = oiCompute(oi, scene, 'crop', true);
        sensor = sensorCreate();
        sensor = sensorSet(sensor, 'noise flag', 0);
        sensor = sensorCompute(sensor, oi, false);
        payload.volts = sensorGet(sensor, 'volts');
        payload.integration_time = sensorGet(sensor, 'integration time');

    case 'sensor_monochrome_noise_stats'
        scene = sceneCreate('uniform d65', 32);
        oi = oiCreate();
        oi = oiCompute(oi, scene, 'crop', true);
        sensor = sensorCreate('monochrome');
        sensor = sensorSet(sensor, 'noise flag', 2);
        rand('seed', 0);
        randn('seed', 0);
        sensor = sensorCompute(sensor, oi, false);
        volts = sensorGet(sensor, 'volts');
        payload.mean = mean(volts(:));
        payload.std = std(volts(:));
        payload.p05 = prctile(volts(:), 5);
        payload.p95 = prctile(volts(:), 95);

    case 'ip_default_pipeline'
        scene = sceneCreate();
        oi = oiCreate();
        oi = oiCompute(oi, scene, 'crop', true);
        sensor = sensorCreate();
        sensor = sensorSet(sensor, 'noise flag', 0);
        sensor = sensorCompute(sensor, oi, false);
        ip = ipCreate('default', sensor);
        ip = ipCompute(ip, sensor);
        payload.input = ipGet(ip, 'input');
        payload.sensorspace = ipGet(ip, 'sensorspace');
        payload.result = ipGet(ip, 'result');

    case 'camera_default_pipeline'
        scene = sceneCreate();
        camera = cameraCreate();
        camera = cameraSet(camera, 'sensor', sensorSet(cameraGet(camera, 'sensor'), 'noise flag', 0));
        camera = cameraCompute(camera, scene);
        payload.result = cameraGet(camera, 'ip').data.result;
        payload.sensor_volts = cameraGet(camera, 'sensor').data.volts;
        payload.oi_photons = cameraGet(camera, 'oi').data.photons;

    otherwise
        error('Unknown parity case: %s', case_name);
end

save('-mat7-binary', output_path, '-struct', 'payload');
end
