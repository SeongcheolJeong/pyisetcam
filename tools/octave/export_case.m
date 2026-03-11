function export_case(case_name, output_path, upstream_root)
% Export a curated ISETCam parity case from Octave into a MAT file.

if nargin < 3
    error('export_case requires case_name, output_path, and upstream_root');
end

addpath(genpath(upstream_root));
addpath(fileparts(mfilename('fullpath')));
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

    case 'scene_reflectance_chart_small'
        sFiles = {
            fullfile(upstream_root, 'data', 'surfaces', 'reflectances', 'MunsellSamples_Vhrel.mat')
            fullfile(upstream_root, 'data', 'surfaces', 'reflectances', 'Food_Vhrel.mat')
            fullfile(upstream_root, 'data', 'surfaces', 'reflectances', 'skin', 'HyspexSkinReflectance.mat')
        };
        sSamples = {
            [1 2]
            [1 2]
            [1]
        };
        scene = sceneCreate('reflectance chart', 8, sSamples, sFiles, [], true, 'without replacement');
        payload.wave = sceneGet(scene, 'wave');
        payload.photons = sceneGet(scene, 'photons');
        payload.mean_luminance = sceneGet(scene, 'mean luminance');
        chart_params = sceneGet(scene, 'chart parameters');
        payload.chart_rowcol = chart_params.rowcol;
        payload.chart_index_map = chart_params.rIdxMap;

    case 'scene_star_pattern_small'
        scene = sceneCreate('star pattern', 64, 'ee', 6);
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

    case 'oi_psf_default_small'
        scene = sceneCreate('checkerboard', 8, 4);
        oi = oiCreate('psf');
        oi = oiCompute(oi, scene, 'crop', true);
        payload.wave = oiGet(oi, 'wave');
        payload.photons = oiGet(oi, 'photons');

    case 'oi_psf550_diffraction_small'
        oi = oiCreate('diffraction limited');
        optics = oiGet(oi, 'optics');
        optics = opticsSet(optics, 'flength', 0.017);
        optics = opticsSet(optics, 'fnumber', 17/3);
        oi = oiSet(oi, 'optics', optics);
        psfData = opticsGet(oi.optics, 'psf data', 550, 'um');
        payload.x = psfData.xy(:, :, 1);
        payload.y = psfData.xy(:, :, 2);
        payload.psf = psfData.psf;

    case 'oi_si_lorentzian_small'
        scene = sceneCreate('grid lines', [64 64], 16, 'ee', 2);
        scene = sceneSet(scene, 'fov', 2.0);
        oi = oiCreate('psf');
        gamma = logspace(0, 1, oiGet(oi, 'nwave'));
        optics = siSynthetic('lorentzian', oi, gamma);
        oi = oiSet(oi, 'optics', optics);
        oi = oiCompute(oi, scene, 'crop', true);
        payload.wave = oiGet(oi, 'wave');
        payload.photons = oiGet(oi, 'photons');

    case 'oi_si_custom_file_small'
        scene = sceneCreate('grid lines', [64 64], 16, 'ee', 2);
        scene = sceneSet(scene, 'fov', 2.0);
        oi = oiCreate('shift invariant');
        wave = oiGet(oi, 'wave');
        samples = ((1:129) - 65);
        [xx, yy] = meshgrid(samples, samples);
        psf = zeros(129, 129, numel(wave));
        for ii = 1:numel(wave)
            sigma = 1.2 + 0.01 * ((wave(ii) - wave(1)) / 10);
            plane = exp(-0.5 * ((xx ./ sigma).^2 + (yy ./ sigma).^2));
            psf(:, :, ii) = plane ./ sum(plane(:));
        end
        fName = fullfile(tempdir, 'custom_si_psf_octave.mat');
        ieSaveSIDataFile(psf, wave, [0.25 0.25], fName);
        optics = siSynthetic('custom', oi, fName);
        oi = oiSet(oi, 'optics', optics);
        oi = oiSet(oi, 'compute method', 'opticsotf');
        oi = oiCompute(oi, scene, 'crop', true);
        payload.wave = oiGet(oi, 'wave');
        payload.photons = oiGet(oi, 'photons');
        payload.input_psf_mid_row_550 = squeeze(psf(65, :, 16));
        if exist(fName, 'file'), delete(fName); end

    case 'oi_custom_otf_flare_small'
        scene = sceneCreate('point array', 64, 16);
        scene = sceneSet(scene, 'hfov', 40);
        fname = fullfile(isetRootPath, 'data', 'optics', 'flare', 'flare1.png');
        OTF = opticsPSF2OTF(fname, 1.2e-6, 400:10:700);
        oi = oiCreate('shift invariant');
        oi = oiSet(oi, 'optics otfstruct', OTF);
        oi = oiSet(oi, 'compute method', 'opticsotf');
        oi = oiSet(oi, 'wangular', sceneGet(scene, 'wangular'));
        oi = oiSet(oi, 'wave', sceneGet(scene, 'wave'));
        paddedOI = oiSet(oi, 'photons', oiCalculateIrradiance(scene, oi));
        paddedOI = oiPadValue(paddedOI, [round(64/8), round(64/8), 0], 'zero photons', sceneGet(scene, 'distance'));
        interpOTF = oiCalculateOTF(paddedOI, oiGet(paddedOI, 'wave'), 'mm');
        oi = oiCompute(oi, scene, 'crop', true);
        payload.wave = oiGet(oi, 'wave');
        payload.photons = double(oi.data.photons);
        payload.fx = OTF.fx;
        payload.fy = OTF.fy;
        payload.otf_abs550 = abs(OTF.OTF(:,:,16));
        payload.interp_otf_abs550 = abs(interpOTF(:,:,16));

    case 'optics_psf_to_otf_flare_small'
        fname = fullfile(isetRootPath, 'data', 'optics', 'flare', 'flare1.png');
        OTF = opticsPSF2OTF(fname, 1.2e-6, 400:10:700);
        otf550 = fftshift(abs(OTF.OTF(:, :, 16)));
        midRow = floor(size(otf550, 1) / 2) + 1;
        payload.fx = OTF.fx;
        payload.fy = OTF.fy;
        payload.otf_abs550_row = otf550(midRow, :);
        payload.otf_abs550_center = getMiddleMatrix(otf550, 32);

    case 'oi_ideal_otf_small'
        params = FOTParams;
        params.blockSize = 16;
        params.angles = [0, pi/4, pi/2];
        params.freqs = [1, 2, 4];
        params.contrast = 1.0;
        scene = sceneCreate('freq orient', params);
        scene = sceneSet(scene, 'fov', 3);
        oi = oiCreate('shift invariant');
        oi = oiCompute(oi, scene, 'crop', true);
        OTF = oiGet(oi, 'optics OTF');
        oi = oiSet(oi, 'optics OTF', ones(size(OTF)));
        oi = oiCompute(oi, scene, 'crop', true);
        payload.wave = oiGet(oi, 'wave');
        payload.photons = oiGet(oi, 'photons');

    case 'oi_wvf_defocus_small'
        params = FOTParams;
        params.blockSize = 16;
        params.angles = [0, pi/4, pi/2];
        params.freqs = [1, 2, 4];
        params.contrast = 1.0;
        scene = sceneCreate('freq orient', params);
        scene = sceneSet(scene, 'fov', 5);
        wvf = wvfCreate('wave', sceneGet(scene, 'wave'));
        wvf = wvfSet(wvf, 'zcoeffs', [2, 0.5], {'defocus', 'vertical_astigmatism'});
        oi = oiCompute(wvf, scene, 'crop', true);
        payload.wave = oiGet(oi, 'wave');
        payload.photons = oiGet(oi, 'photons');
        payload.defocus = wvfGet(wvf, 'zcoeffs', 'defocus');
        payload.vertical_astigmatism = wvfGet(wvf, 'zcoeffs', 'vertical_astigmatism');

    case 'oi_wvf_script_defocus_small'
        params = FOTParams;
        params.blockSize = 16;
        params.angles = [0, pi/4, pi/2];
        params.freqs = [1, 2, 4];
        params.contrast = 1.0;
        scene = sceneCreate('freq orient', params);
        scene = sceneSet(scene, 'fov', 5);
        wvf = wvfCreate('wave', sceneGet(scene, 'wave'));
        wvf = wvfSet(wvf, 'focal length', 8, 'mm');
        wvf = wvfSet(wvf, 'pupil diameter', 3, 'mm');
        wvf = wvfSet(wvf, 'zcoeffs', 1.5, 'defocus');
        wvf = wvfCompute(wvf);
        oi = wvf2oi(wvf);
        oi = oiCompute(oi, scene, 'crop', true);
        payload.wave = oiGet(oi, 'wave');
        payload.photons = oiGet(oi, 'photons');
        payload.defocus_zcoeff = wvfGet(wvf, 'zcoeffs', 'defocus');
        payload.pupil_diameter_mm = wvfGet(wvf, 'pupil diameter', 'mm');
        payload.f_number = oiGet(oi, 'f number');

    case 'wvf_spatial_sampling_small'
        wvf = wvfCreate('wave', 550);
        thisWave = wvfGet(wvf, 'wave');
        focalLengthM = 7e-3;
        fNumber = 4.0;
        wvf = wvfSet(wvf, 'calc pupil diameter', (focalLengthM * 1e3) / fNumber);
        wvf = wvfSet(wvf, 'focal length', focalLengthM);
        wvf = wvfCompute(wvf);
        psfX = wvfGet(wvf, 'psf xaxis', 'um', thisWave);
        pupilAmp = wvfGet(wvf, 'pupil function amplitude', thisWave);
        pupilPhase = wvfGet(wvf, 'pupil function phase', thisWave);
        middleRow = floor(size(pupilAmp, 1) / 2) + 1;
        payload.wave = wvfGet(wvf, 'wave');
        payload.npixels = wvfGet(wvf, 'npixels');
        payload.calc_nwave = wvfGet(wvf, 'calc nwave');
        payload.psf_sample_spacing_arcmin = wvfGet(wvf, 'psf sample spacing');
        payload.ref_psf_sample_interval_arcmin = wvfGet(wvf, 'ref psf sample interval');
        payload.um_per_degree = wvfGet(wvf, 'um per degree');
        payload.pupil_plane_size_mm = wvfGet(wvf, 'pupil plane size', 'mm', thisWave);
        payload.pupil_sample_spacing_mm = wvfGet(wvf, 'pupil sample spacing', 'mm', thisWave);
        payload.pupil_positions_mm = wvfGet(wvf, 'pupil positions', thisWave, 'mm');
        payload.psf_xaxis_um = psfX.samp;
        payload.psf_xaxis_data = psfX.data;
        payload.pupil_amp_row = pupilAmp(middleRow, :);
        payload.pupil_phase_row = pupilPhase(middleRow, :);

    case 'wvf_compute_psf_small'
        wvf = wvfCreate('wave', 550);
        wvf = wvfSet(wvf, 'spatial samples', 101);
        wvf = wvfPupilFunction(wvf);
        wvf = wvfComputePSF(wvf, 'compute pupil func', false);
        thisWave = wvfGet(wvf, 'wave');
        psf = wvfGet(wvf, 'psf', thisWave);
        pupilAmp = wvfGet(wvf, 'pupil function amplitude', thisWave);
        pupilPhase = wvfGet(wvf, 'pupil function phase', thisWave);
        middleRow = floor(size(psf, 1) / 2) + 1;
        payload.wave = wvfGet(wvf, 'wave');
        payload.npixels = wvfGet(wvf, 'npixels');
        payload.psf_sum = sum(psf(:));
        payload.psf_mid_row = psf(middleRow, :);
        payload.pupil_amp_row = pupilAmp(middleRow, :);
        payload.pupil_phase_row = pupilPhase(middleRow, :);

    case 'wvf_aperture_polygon_clean_small'
        wvf = wvfCreate('wave', 550);
        wvf = wvfSet(wvf, 'spatial samples', 101);
        [aperture, params] = wvfAperture(wvf, ...
            'n sides', 8, ...
            'dot mean', 0, ...
            'dot sd', 0, ...
            'line mean', 0, ...
            'line sd', 0, ...
            'image rotate', 0);
        middleRow = floor(size(aperture, 1) / 2) + 1;
        payload.image = aperture;
        payload.mid_row = aperture(middleRow, :);
        payload.image_sum = sum(aperture(:));
        payload.nsides = params.nsides;

    case 'wvf_compute_aperture_polygon_small'
        wvf = wvfCreate('wave', 550);
        wvf = wvfSet(wvf, 'spatial samples', 101);
        [aperture, params] = wvfAperture(wvf, ...
            'n sides', 8, ...
            'dot mean', 0, ...
            'dot sd', 0, ...
            'line mean', 0, ...
            'line sd', 0, ...
            'image rotate', 0);
        wvf = wvfCompute(wvf, 'aperture', aperture);
        thisWave = wvfGet(wvf, 'wave');
        psf = wvfGet(wvf, 'psf', thisWave);
        pupilAmp = wvfGet(wvf, 'pupil function amplitude', thisWave);
        middleRow = floor(size(psf, 1) / 2) + 1;
        payload.psf_sum = sum(psf(:));
        payload.psf_mid_row = psf(middleRow, :);
        payload.pupil_amp_row = pupilAmp(middleRow, :);
        payload.nsides = params.nsides;

    case 'oi_lswavelength_diffraction_small'
        oi = oiCreate('diffraction limited');
        optics = oiGet(oi, 'optics');
        optics = opticsSet(optics, 'flength', 0.017);
        optics = opticsSet(optics, 'fnumber', 17/3);
        oi = oiSet(oi, 'optics', optics);
        units = 'um';
        wavelength = opticsGet(optics, 'wavelength');
        inCutoff = opticsGet(optics, 'inCutoff', units);
        peakF = 3 * max(inCutoff);
        middleSamps = 40;
        deltaSpace = 1 / (2 * peakF);
        nSamp = 100;
        fSamp = (-nSamp:(nSamp - 1)) / nSamp;
        [fX, fY] = meshgrid(fSamp, fSamp);
        fSupport(:, :, 1) = fX * peakF;
        fSupport(:, :, 2) = fY * peakF;
        otf = dlMTF(oi, fSupport, wavelength, units);
        tmp = otf(1, :, 1);
        lsf = fftshift(ifft(tmp));
        lsTemplate = getMiddleMatrix(lsf, middleSamps);
        lsWave = zeros(numel(wavelength), numel(lsTemplate));
        lsWave(1, :) = lsTemplate;
        for ii = 2:numel(wavelength)
            tmp = otf(1, :, ii);
            lsf = fftshift(ifft(tmp));
            lsWave(ii, :) = getMiddleMatrix(lsf, middleSamps);
        end
        lsWave = abs(lsWave);
        X = (-nSamp:(nSamp - 1)) * deltaSpace;
        X = getMiddleMatrix(X, middleSamps);
        payload.x = X;
        payload.wavelength = wavelength;
        payload.lsWave = lsWave;

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
