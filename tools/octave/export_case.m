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
