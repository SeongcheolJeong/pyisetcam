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

    case 'wvf_load_thibos_virtual_eyes_small'
        [sample_mean, sample_cov, subject_coeffs] = wvfLoadThibosVirtualEyes(6.0);
        payload.pupil_diameter_mm = 6.0;
        payload.sample_mean = sample_mean;
        payload.sample_cov = sample_cov;
        payload.left_eye = subject_coeffs.leftEye;
        payload.right_eye = subject_coeffs.rightEye;
        payload.both_eyes = subject_coeffs.bothEyes;

    case 'wvf_pupil_size_human_small'
        measuredPupilMM = 7.5;
        calcPupilMM = 3.0;
        wave = 520;
        zCoefs = wvfLoadThibosVirtualEyes(measuredPupilMM);
        wvf = wvfCreate('calc wavelengths', wave, ...
            'zcoeffs', zCoefs, ...
            'measured pupil size', measuredPupilMM, ...
            'calc pupil size', calcPupilMM, ...
            'name', '7-pupil');
        wvf = wvfSet(wvf, 'lcaMethod', 'human');
        wvf = wvfCompute(wvf);
        psf = wvfGet(wvf, 'psf', wave);
        middleRow = floor(size(psf, 1)/2) + 1;
        measuredWavelengthNM = wvfGet(wvf, 'measured wavelength', 'nm');
        lcaDiopters = wvfLCAFromWavelengthDifference(measuredWavelengthNM, wave);
        payload.measured_pupil_mm = measuredPupilMM;
        payload.calc_pupil_mm = calcPupilMM;
        payload.measured_wavelength_nm = measuredWavelengthNM;
        payload.wave = wvfGet(wvf, 'wave');
        payload.f_number = wvfGet(wvf, 'fnumber');
        payload.lca_diopters = lcaDiopters;
        payload.lca_microns = wvfDefocusDioptersToMicrons(-lcaDiopters, measuredPupilMM);
        payload.psf_sum = sum(psf(:));
        payload.psf_mid_row = psf(middleRow, :);

    case 'wvf_pupil_size_measured_compare_small'
        measuredPupilMM = [7.5 6 4.5 3];
        calcPupilMM = 3.0;
        wave = (400:10:700)';
        index550 = find(wave == 550);
        psfSums = zeros(size(measuredPupilMM));
        psfPeaks = zeros(size(measuredPupilMM));
        maxAbsDiffs = zeros(size(measuredPupilMM));
        psfMidRows = [];
        referencePSF = [];
        for ii = 1:numel(measuredPupilMM)
            zCoefs = wvfLoadThibosVirtualEyes(measuredPupilMM(ii));
            wvf = wvfCreate('calc wavelengths', wave, ...
                'zcoeffs', zCoefs, ...
                'measured pupil size', measuredPupilMM(ii), ...
                'calc pupil size', calcPupilMM, ...
                'name', sprintf('%g-pupil', measuredPupilMM(ii)));
            wvf = wvfSet(wvf, 'lcaMethod', 'human');
            wvf = wvfCompute(wvf);
            psf = wvfGet(wvf, 'psf');
            psf550 = psf{index550};
            middleRow = floor(size(psf550, 1)/2) + 1;
            psfMidRows(ii, :) = psf550(middleRow, :);
            psfSums(ii) = sum(psf550(:));
            psfPeaks(ii) = max(psf550(:));
            if isempty(referencePSF)
                referencePSF = psf550;
                maxAbsDiffs(ii) = 0;
            else
                maxAbsDiffs(ii) = max(abs(referencePSF(:) - psf550(:)));
            end
        end
        payload.measured_pupil_mm = measuredPupilMM;
        payload.calc_pupil_mm = calcPupilMM;
        payload.wave = wave;
        payload.wavelength_nm = 550;
        payload.psf_sum = psfSums;
        payload.psf_peak_550 = psfPeaks;
        payload.max_abs_diff_vs_first_550 = maxAbsDiffs;
        payload.psf_mid_row_550 = psfMidRows;

    case 'wvf_psf_spacing_small'
        lambdaNM = 550;
        fnumber = 4;
        focallengthMM = 4;
        nPixels = 1024;
        psf_spacingMM = 1e-3;

        wvf = wvfCreate;
        wvf = wvfSet(wvf, 'wave', lambdaNM);
        wvf = wvfSet(wvf, 'focal length', focallengthMM, 'mm');
        wvf = wvfSet(wvf, 'calc pupil diameter', focallengthMM/fnumber);
        wvf = wvfSet(wvf, 'spatial samples', nPixels);
        wvf = wvfSet(wvf, 'psf sample spacing', psf_spacingMM);

        payload.wavelength_nm = lambdaNM;
        payload.focal_length_mm = focallengthMM;
        payload.calc_pupil_diameter_mm = wvfGet(wvf, 'calc pupil size', 'mm');
        payload.npixels = wvfGet(wvf, 'npixels');
        payload.field_size_mm = wvfGet(wvf, 'field size mm', 'mm');
        payload.pupil_sample_spacing_mm = wvfGet(wvf, 'pupil sample spacing', 'mm', lambdaNM);
        payload.psf_sample_spacing_arcmin = wvfGet(wvf, 'psf sample spacing');
        payload.ref_psf_sample_interval_arcmin = wvfGet(wvf, 'ref psf sample interval');

    case 'wvf_osa_index_conversion_small'
        indices = [0 1 2 5 15 20 35];
        [n, m] = wvfOSAIndexToZernikeNM(indices);
        roundtrip = wvfZernikeNMToOSAIndex(n, m);
        [scalarN, scalarM] = wvfOSAIndexToZernikeNM(15);
        scalarRoundtrip = wvfZernikeNMToOSAIndex(scalarN, scalarM);
        payload.indices = indices(:);
        payload.n = n(:);
        payload.m = m(:);
        payload.roundtrip_indices = roundtrip(:);
        payload.scalar_index = 15;
        payload.scalar_n = scalarN;
        payload.scalar_m = scalarM;
        payload.scalar_roundtrip_index = scalarRoundtrip;

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

    case 'optics_airy_disk_small'
        [radius_um, img] = airyDisk(550, 3, 'units', 'um');
        payload.radius_um = radius_um;
        payload.diameter_um = airyDisk(550, 3, 'units', 'um', 'diameter', true);
        payload.radius_mm = airyDisk(550, 3, 'units', 'mm');
        payload.radius_deg = airyDisk(700, [], 'units', 'deg', 'pupil diameter', 1e-3);
        payload.radius_rad = airyDisk(700, [], 'units', 'rad', 'pupil diameter', 1e-3);
        payload.image_rows = size(img.data, 1);
        payload.image_cols = size(img.data, 2);

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

    case 'oi_psfxaxis_diffraction_small'
        oi = oiCreate('diffraction limited');
        optics = oiGet(oi, 'optics');
        optics = opticsSet(optics, 'flength', 0.017);
        optics = opticsSet(optics, 'fnumber', 17/3);
        oi = oiSet(oi, 'optics', optics);
        uData = oiGet(oi, 'optics psf xaxis', 550, 'um');
        payload.samp = uData.samp;
        payload.data = uData.data;
        payload.wave = 550;

    case 'oi_psfyaxis_diffraction_small'
        oi = oiCreate('diffraction limited');
        optics = oiGet(oi, 'optics');
        optics = opticsSet(optics, 'flength', 0.017);
        optics = opticsSet(optics, 'fnumber', 17/3);
        oi = oiSet(oi, 'optics', optics);
        uData = oiGet(oi, 'optics psf yaxis', 550, 'um');
        payload.samp = uData.samp;
        payload.data = uData.data;
        payload.wave = 550;

    case 'oi_psfxaxis_wvf_small'
        wvf = wvfCreate('wave', 550);
        thisWave = 550;
        wvf = wvfCompute(wvf);
        oi = wvf2oi(wvf);
        oiLine = oiGet(oi, 'optics psf xaxis', thisWave, 'um');
        wvfLine = wvfPlot(wvf, 'psf xaxis', 'unit', 'um', 'wave', thisWave, 'window', false);
        payload.wave = thisWave;
        payload.oi_samp = oiLine.samp;
        payload.oi_data = oiLine.data;
        payload.wvf_samp = wvfLine.samp;
        payload.wvf_data = wvfLine.data;

    case 'oi_psfyaxis_wvf_small'
        wvf = wvfCreate('wave', 550);
        thisWave = 550;
        wvf = wvfCompute(wvf);
        oi = wvf2oi(wvf);
        oiLine = oiGet(oi, 'optics psf yaxis', thisWave, 'um');
        wvfLine = wvfGet(wvf, 'psf yaxis', 'um', thisWave);
        payload.wave = thisWave;
        payload.oi_samp = oiLine.samp;
        payload.oi_data = oiLine.data;
        payload.wvf_samp = wvfLine.samp;
        payload.wvf_data = wvfLine.data;

    case 'oi_psf550_wvf_small'
        wvf = wvfCreate('wave', 550);
        wvf = wvfSet(wvf, 'focal length', 8, 'mm');
        wvf = wvfSet(wvf, 'pupil diameter', 3, 'mm');
        wvf = wvfCompute(wvf);
        oi = wvf2oi(wvf);
        psfData = oiGet(oi, 'optics psf data', 550, 'um');
        payload.x = psfData.xy(:, :, 1);
        payload.y = psfData.xy(:, :, 2);
        payload.psf = psfData.psf;

    case 'oi_wvf_otf_compare_small'
        wvf = wvfCreate('wave', 550);
        wvf = wvfSet(wvf, 'focal length', 8, 'mm');
        wvf = wvfSet(wvf, 'pupil diameter', 3, 'mm');
        wvf = wvfCompute(wvf);
        oi = wvf2oi(wvf);
        wvfOTF = wvfGet(wvf, 'otf', 550);
        oiOTF = oiGet(oi, 'optics otf');
        if ndims(oiOTF) == 3
            oiOTF = oiOTF(:, :, 1);
        end
        payload.oi_otf_abs = abs(oiOTF);
        payload.wvf_otf_abs_shifted = abs(ifftshift(wvfOTF));

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

    case 'oi_otfwavelength_si_lorentzian_small'
        oi = oiCreate('psf');
        gamma = logspace(0, 1, oiGet(oi, 'nwave'));
        optics = siSynthetic('lorentzian', oi, gamma);
        oi = oiSet(oi, 'optics', optics);
        wavelength = oiGet(oi, 'wavelength');
        optics = oiGet(oi, 'optics');
        fSupport = opticsGet(optics, 'otf support matrix');
        fx = fSupport(1, :, 1);
        nWave = numel(wavelength);
        otfWave = zeros(numel(fx), nWave);
        for ii = 1:nWave
            otf = abs(opticsGet(optics, 'otfdata', wavelength(ii)));
            otfWave(:, ii) = fftshift(otf(1, :))';
        end
        payload.fSupport = fx;
        payload.wavelength = wavelength;
        payload.otf = otfWave;

    case 'oi_psf550_si_lorentzian_small'
        oi = oiCreate('psf');
        gamma = logspace(0, 1, oiGet(oi, 'nwave'));
        optics = siSynthetic('lorentzian', oi, gamma);
        oi = oiSet(oi, 'optics', optics);
        psfData = opticsGet(oiGet(oi, 'optics'), 'psf data', 550, 'um');
        payload.x = psfData.xy(:, :, 1);
        payload.y = psfData.xy(:, :, 2);
        payload.psf = psfData.psf;

    case 'oi_si_pillbox_small'
        scene = sceneCreate('grid lines', [256 256], 64, 'ee', 3);
        scene = sceneSet(scene, 'fov', 2.0);
        oi = oiCreate('psf');
        patchSize = airyDisk(700, oiGet(oi, 'optics fnumber'), 'units', 'mm');
        optics = siSynthetic('pillbox', oi, patchSize);
        oi = oiSet(oi, 'optics', optics);
        oi = oiCompute(oi, scene, 'crop', true);
        psfData = opticsGet(optics, 'psf data');
        payload.wave = oiGet(oi, 'wave');
        payload.input_psf_mid_row_550 = squeeze(psfData.psf(floor(size(psfData.psf, 1) / 2) + 1, :, 16));

    case 'oi_si_gaussian_small'
        scene = sceneCreate('grid lines', [64 64], 16, 'ee', 2);
        scene = sceneSet(scene, 'fov', 2.0);
        oi = oiCreate('psf');
        wave = oiGet(oi, 'wave');
        waveSpread = 0.5 * (wave / wave(1)).^3;
        xyRatio = ones(1, numel(wave));
        optics = siSynthetic('gaussian', oi, double(waveSpread), xyRatio);
        oi = oiSet(oi, 'optics', optics);
        oi = oiCompute(oi, scene, 'crop', true);
        psfData = opticsGet(optics, 'psf data');
        payload.wave = oiGet(oi, 'wave');
        payload.input_psf_mid_row_550 = squeeze(psfData.psf(floor(size(psfData.psf, 1) / 2) + 1, :, 16));

    case 'oi_si_gaussian_ratio_small'
        scene = sceneCreate('grid lines', [64 64], 16, 'ee', 2);
        scene = sceneSet(scene, 'fov', 2.0);
        oi = oiCreate('psf');
        wave = oiGet(oi, 'wave');
        waveSpread = 0.5 * (wave / wave(1)).^3;
        xyRatio = 2 * ones(1, numel(wave));
        optics = siSynthetic('gaussian', oi, double(waveSpread), xyRatio);
        oi = oiSet(oi, 'optics', optics);
        oi = oiCompute(oi, scene, 'crop', true);
        psfData = opticsGet(optics, 'psf data');
        payload.wave = oiGet(oi, 'wave');
        payload.input_psf_mid_row_550 = squeeze(psfData.psf(floor(size(psfData.psf, 1) / 2) + 1, :, 16));

    case 'oi_psf550_si_gaussian_ratio_small'
        oi = oiCreate('psf');
        wave = oiGet(oi, 'wave');
        waveSpread = 0.5 * (wave / wave(1)).^3;
        xyRatio = 2 * ones(1, numel(wave));
        optics = siSynthetic('gaussian', oi, double(waveSpread), xyRatio);
        oi = oiSet(oi, 'optics', optics);
        psfData = opticsGet(oiGet(oi, 'optics'), 'psf data', 550, 'um');
        payload.x = psfData.xy(:, :, 1);
        payload.y = psfData.xy(:, :, 2);
        payload.psf = psfData.psf;

    case 'oi_illuminance_lines_si_gaussian_ratio_small'
        scene = sceneCreate('grid lines', [64 64], 16, 'ee', 2);
        scene = sceneSet(scene, 'fov', 2.0);
        oi = oiCreate('psf');
        wave = oiGet(oi, 'wave');
        waveSpread = 0.5 * (wave / wave(1)).^3;
        xyRatio = 2 * ones(1, numel(wave));
        optics = siSynthetic('gaussian', oi, double(waveSpread), xyRatio);
        oi = oiSet(oi, 'optics', optics);
        oi = oiCompute(oi, scene, 'crop', true);
        sz = oiGet(oi, 'size');
        xyMiddle = [ceil(sz(2) / 2), ceil(sz(1) / 2)];
        illum = oiGet(oi, 'illuminance');
        posMicrons = oiSpatialSupport(oi, 'um');
        payload.xy_middle = xyMiddle(:);
        payload.v_pos = posMicrons.y(:);
        payload.v_data = illum(:, xyMiddle(1));
        payload.h_pos = posMicrons.x(:);
        payload.h_data = illum(xyMiddle(2), :)';

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

    case 'wvf_spatial_controls_small'
        thisWave = 550;
        focalLengthM = 7e-3;
        focalLengthMM = focalLengthM * 1e3;
        fNumber = 4.0;

        wvf = wvfCreate('wave', thisWave);
        wvf = wvfSet(wvf, 'calc pupil diameter', focalLengthMM / fNumber);
        wvf = wvfSet(wvf, 'focal length', focalLengthM);
        wvf = wvfCompute(wvf);
        base.npixels = wvfGet(wvf, 'npixels');
        base.psf_sample_spacing_arcmin = wvfGet(wvf, 'psf sample spacing');
        base.um_per_degree = wvfGet(wvf, 'um per degree');
        base.pupil_plane_size_mm = wvfGet(wvf, 'pupil plane size', 'mm', thisWave);
        base.pupil_sample_spacing_mm = wvfGet(wvf, 'pupil sample spacing', 'mm', thisWave);
        base.focal_length_mm = wvfGet(wvf, 'focal length', 'mm');

        reducedPixels = wvfCreate('wave', thisWave);
        reducedPixels = wvfSet(reducedPixels, 'calc pupil diameter', focalLengthMM / fNumber);
        reducedPixels = wvfSet(reducedPixels, 'focal length', focalLengthM);
        reducedPixels = wvfSet(reducedPixels, 'npixels', round(base.npixels / 4));
        reducedPixels = wvfCompute(reducedPixels);
        reducedPixels.npixels = wvfGet(reducedPixels, 'npixels');
        reducedPixels.psf_sample_spacing_arcmin = wvfGet(reducedPixels, 'psf sample spacing');
        reducedPixels.um_per_degree = wvfGet(reducedPixels, 'um per degree');

        enlargedPupilPlane = wvfCreate('wave', thisWave);
        enlargedPupilPlane = wvfSet(enlargedPupilPlane, 'calc pupil diameter', focalLengthMM / fNumber);
        enlargedPupilPlane = wvfSet(enlargedPupilPlane, 'focal length', focalLengthM);
        enlargedPupilPlane = wvfSet(enlargedPupilPlane, 'pupil plane size', base.pupil_plane_size_mm * 4, 'mm');
        enlargedPupilPlane = wvfCompute(enlargedPupilPlane);
        enlargedPupilPlane.psf_sample_spacing_arcmin = wvfGet(enlargedPupilPlane, 'psf sample spacing');
        enlargedPupilPlane.um_per_degree = wvfGet(enlargedPupilPlane, 'um per degree');
        enlargedPupilPlane.pupil_plane_size_mm = wvfGet(enlargedPupilPlane, 'pupil plane size', 'mm', thisWave);

        reducedPupilPlane = wvfCreate('wave', thisWave);
        reducedPupilPlane = wvfSet(reducedPupilPlane, 'calc pupil diameter', focalLengthMM / fNumber);
        reducedPupilPlane = wvfSet(reducedPupilPlane, 'focal length', focalLengthM);
        reducedPupilPlane = wvfSet(reducedPupilPlane, 'pupil plane size', base.pupil_plane_size_mm / 4, 'mm');
        reducedPupilPlane = wvfCompute(reducedPupilPlane);
        reducedPupilPlane.psf_sample_spacing_arcmin = wvfGet(reducedPupilPlane, 'psf sample spacing');
        reducedPupilPlane.um_per_degree = wvfGet(reducedPupilPlane, 'um per degree');
        reducedPupilPlane.pupil_plane_size_mm = wvfGet(reducedPupilPlane, 'pupil plane size', 'mm', thisWave);

        focalHalf = wvfCreate('wave', thisWave);
        focalHalf = wvfSet(focalHalf, 'calc pupil diameter', focalLengthMM / fNumber);
        focalHalf = wvfSet(focalHalf, 'focal length', focalLengthM);
        focalHalf = wvfSet(focalHalf, 'focal length', focalLengthM / 2);
        focalHalf = wvfCompute(focalHalf);
        focalHalf.focal_length_m = wvfGet(focalHalf, 'focal length', 'm');
        focalHalf.psf_sample_spacing_arcmin = wvfGet(focalHalf, 'psf sample spacing');
        focalHalf.um_per_degree = wvfGet(focalHalf, 'um per degree');

        focalDouble = wvfCreate('wave', thisWave);
        focalDouble = wvfSet(focalDouble, 'calc pupil diameter', focalLengthMM / fNumber);
        focalDouble = wvfSet(focalDouble, 'focal length', focalLengthM);
        focalDouble = wvfSet(focalDouble, 'focal length', focalLengthM * 2);
        focalDouble = wvfCompute(focalDouble);
        focalDouble.focal_length_m = wvfGet(focalDouble, 'focal length', 'm');
        focalDouble.psf_sample_spacing_arcmin = wvfGet(focalDouble, 'psf sample spacing');
        focalDouble.um_per_degree = wvfGet(focalDouble, 'um per degree');

        payload.wave = thisWave;
        payload.base_npixels = base.npixels;
        payload.base_psf_sample_spacing_arcmin = base.psf_sample_spacing_arcmin;
        payload.base_um_per_degree = base.um_per_degree;
        payload.base_pupil_plane_size_mm = base.pupil_plane_size_mm;
        payload.reduced_pixels_npixels = reducedPixels.npixels;
        payload.reduced_pixels_psf_sample_spacing_arcmin = reducedPixels.psf_sample_spacing_arcmin;
        payload.reduced_pixels_um_per_degree = reducedPixels.um_per_degree;
        payload.pupil_plane_x4_psf_sample_spacing_arcmin = enlargedPupilPlane.psf_sample_spacing_arcmin;
        payload.pupil_plane_x4_um_per_degree = enlargedPupilPlane.um_per_degree;
        payload.pupil_plane_x4_size_mm = enlargedPupilPlane.pupil_plane_size_mm;
        payload.pupil_plane_div4_psf_sample_spacing_arcmin = reducedPupilPlane.psf_sample_spacing_arcmin;
        payload.pupil_plane_div4_um_per_degree = reducedPupilPlane.um_per_degree;
        payload.pupil_plane_div4_size_mm = reducedPupilPlane.pupil_plane_size_mm;
        payload.focal_length_half_m = focalHalf.focal_length_m;
        payload.focal_length_half_psf_sample_spacing_arcmin = focalHalf.psf_sample_spacing_arcmin;
        payload.focal_length_half_um_per_degree = focalHalf.um_per_degree;
        payload.focal_length_double_m = focalDouble.focal_length_m;
        payload.focal_length_double_psf_sample_spacing_arcmin = focalDouble.psf_sample_spacing_arcmin;
        payload.focal_length_double_um_per_degree = focalDouble.um_per_degree;

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

    case 'wvf_plot_otf_small'
        wvf = wvfCreate('wave', 550);
        wvf = wvfSet(wvf, 'spatial samples', 401);
        wvf = wvfCompute(wvf);
        uData = wvfPlot(wvf, '2d otf', 'unit', 'mm', 'wave', 550, 'plot range', 300, 'window', false);
        otf = uData.otf;
        middleRow = floor(size(otf, 1) / 2) + 1;
        payload.fx = uData.fx(:)';
        payload.otf_mid_row = otf(middleRow, :);
        payload.otf_center = otf(middleRow, floor(size(otf, 2) / 2) + 1);

    case 'wvf_plot_otf_normalized_small'
        wvf = wvfCreate('wave', 550);
        wvf = wvfSet(wvf, 'spatial samples', 401);
        wvf = wvfCompute(wvf);
        % Upstream wvfPlot.m computes normalizeFlag but does not accept the
        % exact normalized OTF pType in its switch block. Export the stable
        % underlying numerical contract directly instead.
        psf = wvfGet(wvf, 'psf', 550);
        psf = psf / max(psf(:));
        freq = wvfGet(wvf, 'otf support', 'mm', 550);
        otf = fftshift(fft2(ifftshift(psf)));
        index = abs(freq) < 300;
        freq = freq(index);
        otf = otf(index, index);
        otf = abs(otf);
        middleRow = floor(size(otf, 1) / 2) + 1;
        payload.fx = freq(:)';
        payload.otf_mid_row = otf(middleRow, :);
        payload.otf_center = otf(middleRow, floor(size(otf, 2) / 2) + 1);

    case 'wvf_plot_1d_otf_angle_small'
        wvf = wvfCreate('wave', 550);
        wvf = wvfSet(wvf, 'spatial samples', 401);
        wvf = wvfCompute(wvf);
        uData = wvfPlot(wvf, '1d otf angle', 'unit', 'deg', 'wave', 550, 'plot range', 10, 'window', false);
        otf = uData.otf;
        middleRow = floor(size(otf, 1) / 2) + 1;
        payload.fx = uData.fx(:)';
        payload.otf_mid_row = otf(middleRow, :);
        payload.otf_center = otf(middleRow, floor(size(otf, 2) / 2) + 1);

    case 'wvf_plot_1d_otf_angle_normalized_small'
        wvf = wvfCreate('wave', 550);
        wvf = wvfSet(wvf, 'spatial samples', 401);
        wvf = wvfCompute(wvf);
        % Upstream wvfPlot.m computes normalizeFlag but does not accept the
        % exact normalized OTF pType in its switch block. Export the stable
        % underlying numerical contract directly instead.
        psf = wvfGet(wvf, 'psf', 550);
        psf = psf / max(psf(:));
        freq = wvfGet(wvf, 'otf support', 'deg', 550);
        otf = fftshift(fft2(ifftshift(psf)));
        index = abs(freq) < 10;
        freq = freq(index);
        otf = otf(index, index);
        otf = abs(otf);
        middleRow = floor(size(otf, 1) / 2) + 1;
        payload.fx = freq(:)';
        payload.otf_mid_row = otf(middleRow, :);
        payload.otf_center = otf(middleRow, floor(size(otf, 2) / 2) + 1);

    case 'wvf_plot_1d_otf_small'
        wvf = wvfCreate('wave', 550);
        wvf = wvfSet(wvf, 'spatial samples', 401);
        wvf = wvfCompute(wvf);
        uData = wvfPlot(wvf, '1d otf', 'unit', 'mm', 'wave', 550, 'plot range', 300, 'window', false);
        otf = uData.otf;
        middleRow = floor(size(otf, 1) / 2) + 1;
        payload.fx = uData.fx(:)';
        payload.otf_mid_row = otf(middleRow, :);
        payload.otf_center = otf(middleRow, floor(size(otf, 2) / 2) + 1);

    case 'wvf_plot_1d_otf_normalized_small'
        wvf = wvfCreate('wave', 550);
        wvf = wvfSet(wvf, 'spatial samples', 401);
        wvf = wvfCompute(wvf);
        % Upstream wvfPlot.m computes normalizeFlag but does not accept the
        % exact normalized OTF pType in its switch block. Export the stable
        % underlying numerical contract directly instead.
        psf = wvfGet(wvf, 'psf', 550);
        psf = psf / max(psf(:));
        freq = wvfGet(wvf, 'otf support', 'mm', 550);
        otf = fftshift(fft2(ifftshift(psf)));
        index = abs(freq) < 300;
        freq = freq(index);
        otf = otf(index, index);
        otf = abs(otf);
        middleRow = floor(size(otf, 1) / 2) + 1;
        payload.fx = freq(:)';
        payload.otf_mid_row = otf(middleRow, :);
        payload.otf_center = otf(middleRow, floor(size(otf, 2) / 2) + 1);

    case 'wvf_plot_pupil_amp_small'
        wvf = wvfCreate('wave', 550);
        wvf = wvfSet(wvf, 'spatial samples', 401);
        wvf = wvfCompute(wvf);
        uData = wvfPlot(wvf, 'image pupil amp', 'unit', 'mm', 'wave', 550, 'plot range', 2, 'window', false);
        amp = uData.z;
        middleRow = floor(size(amp, 1) / 2) + 1;
        payload.x = uData.x(:)';
        payload.amp_mid_row = amp(middleRow, :);
        payload.amp_center = amp(middleRow, floor(size(amp, 2) / 2) + 1);

    case 'wvf_plot_2d_pupil_amplitude_space_small'
        wvf = wvfCreate('wave', 550);
        wvf = wvfSet(wvf, 'spatial samples', 401);
        wvf = wvfCompute(wvf);
        uData = wvfPlot(wvf, '2d pupil amplitude space', 'unit', 'mm', 'wave', 550, 'plot range', 2, 'window', false);
        amp = uData.z;
        middleRow = floor(size(amp, 1) / 2) + 1;
        payload.x = uData.x(:)';
        payload.amp_mid_row = amp(middleRow, :);
        payload.amp_center = amp(middleRow, floor(size(amp, 2) / 2) + 1);

    case 'wvf_plot_pupil_phase_small'
        wvf = wvfCreate('wave', 550);
        wvf = wvfSet(wvf, 'spatial samples', 401);
        wvf = wvfCompute(wvf);
        uData = wvfPlot(wvf, 'image pupil phase', 'unit', 'mm', 'wave', 550, 'plot range', 2, 'window', false);
        phase = uData.z;
        middleRow = floor(size(phase, 1) / 2) + 1;
        payload.x = uData.x(:)';
        payload.phase_mid_row = phase(middleRow, :);
        payload.phase_center = phase(middleRow, floor(size(phase, 2) / 2) + 1);

    case 'wvf_plot_2d_pupil_phase_space_small'
        wvf = wvfCreate('wave', 550);
        wvf = wvfSet(wvf, 'spatial samples', 401);
        wvf = wvfCompute(wvf);
        uData = wvfPlot(wvf, '2d pupil phase space', 'unit', 'mm', 'wave', 550, 'plot range', 2, 'window', false);
        phase = uData.z;
        middleRow = floor(size(phase, 1) / 2) + 1;
        payload.x = uData.x(:)';
        payload.phase_mid_row = phase(middleRow, :);
        payload.phase_center = phase(middleRow, floor(size(phase, 2) / 2) + 1);

    case 'wvf_plot_wavefront_aberrations_small'
        wvf = wvfCreate('wave', 550);
        wvf = wvfSet(wvf, 'zcoeffs', 0.12, 'defocus');
        wvf = wvfSet(wvf, 'spatial samples', 401);
        wvf = wvfCompute(wvf);
        uData = wvfPlot(wvf, 'image wavefront aberrations', 'unit', 'mm', 'wave', 550, 'plot range', 1.5, 'window', false);
        wavefront = uData.z;
        middleRow = floor(size(wavefront, 1) / 2) + 1;
        payload.x = uData.x(:)';
        payload.wavefront_mid_row = wavefront(middleRow, :);
        payload.wavefront_center = wavefront(middleRow, floor(size(wavefront, 2) / 2) + 1);

    case 'wvf_plot_2d_wavefront_aberrations_space_small'
        wvf = wvfCreate('wave', 550);
        wvf = wvfSet(wvf, 'zcoeffs', 0.12, 'defocus');
        wvf = wvfSet(wvf, 'spatial samples', 401);
        wvf = wvfCompute(wvf);
        uData = wvfPlot(wvf, '2d wavefront aberrations space', 'unit', 'mm', 'wave', 550, 'plot range', 1.5, 'window', false);
        wavefront = uData.z;
        middleRow = floor(size(wavefront, 1) / 2) + 1;
        payload.x = uData.x(:)';
        payload.wavefront_mid_row = wavefront(middleRow, :);
        payload.wavefront_center = wavefront(middleRow, floor(size(wavefront, 2) / 2) + 1);

    case 'wvf_plot_image_psf_small'
        wvf = wvfCreate('wave', 550);
        wvf = wvfSet(wvf, 'spatial samples', 401);
        wvf = wvfCompute(wvf);
        uData = wvfPlot(wvf, 'image psf', 'unit', 'um', 'wave', 550, 'plot range', 20, 'window', false);
        psf = uData.z;
        middleRow = floor(size(psf, 1) / 2) + 1;
        payload.x = uData.x(:)';
        payload.psf_mid_row = psf(middleRow, :);
        payload.psf_center = psf(middleRow, floor(size(psf, 2) / 2) + 1);

    case 'wvf_plot_image_psf_airy_small'
        wvf = wvfCreate('wave', 550);
        wvf = wvfSet(wvf, 'spatial samples', 401);
        wvf = wvfCompute(wvf);
        uData = wvfPlot(wvf, 'image psf', 'unit', 'um', 'wave', 550, 'plot range', 20, 'airy disk', true, 'window', false);
        psf = uData.z;
        middleRow = floor(size(psf, 1) / 2) + 1;
        payload.x = uData.x(:)';
        payload.psf_mid_row = psf(middleRow, :);
        payload.psf_center = psf(middleRow, floor(size(psf, 2) / 2) + 1);
        payload.airy_disk_radius = airyDisk(550, wvfGet(wvf, 'fnumber'), 'units', 'um');

    case 'wvf_plot_image_psf_normalized_small'
        wvf = wvfCreate('wave', 550);
        wvf = wvfSet(wvf, 'spatial samples', 401);
        wvf = wvfCompute(wvf);
        uData = wvfPlot(wvf, 'image psf normalized', 'unit', 'um', 'wave', 550, 'plot range', 20, 'window', false);
        psf = uData.z;
        middleRow = floor(size(psf, 1) / 2) + 1;
        payload.x = uData.x(:)';
        payload.psf_mid_row = psf(middleRow, :);
        payload.psf_center = psf(middleRow, floor(size(psf, 2) / 2) + 1);

    case 'wvf_plot_psf_small'
        wvf = wvfCreate('wave', 550);
        wvf = wvfSet(wvf, 'spatial samples', 401);
        wvf = wvfCompute(wvf);
        uData = wvfPlot(wvf, 'psf', 'unit', 'mm', 'wave', 550, 'plot range', 0.05, 'window', false);
        psf = uData.z;
        middleRow = floor(size(psf, 1) / 2) + 1;
        payload.x = uData.x(:)';
        payload.psf_mid_row = psf(middleRow, :);
        payload.psf_center = psf(middleRow, floor(size(psf, 2) / 2) + 1);

    case 'wvf_plot_psf_normalized_small'
        wvf = wvfCreate('wave', 550);
        wvf = wvfSet(wvf, 'spatial samples', 401);
        wvf = wvfCompute(wvf);
        uData = wvfPlot(wvf, 'psf normalized', 'unit', 'mm', 'wave', 550, 'plot range', 0.05, 'window', false);
        psf = uData.z;
        middleRow = floor(size(psf, 1) / 2) + 1;
        payload.x = uData.x(:)';
        payload.psf_mid_row = psf(middleRow, :);
        payload.psf_center = psf(middleRow, floor(size(psf, 2) / 2) + 1);

    case 'wvf_plot_image_psf_angle_small'
        wvf = wvfCreate('wave', 460);
        wvf = wvfSet(wvf, 'spatial samples', 401);
        wvf = wvfCompute(wvf);
        uData = wvfPlot(wvf, 'image psf angle', 'unit', 'min', 'wave', 460, 'plot range', 1, 'window', false);
        psf = uData.z;
        middleRow = floor(size(psf, 1) / 2) + 1;
        payload.x = uData.x(:)';
        payload.psf_mid_row = psf(middleRow, :);
        payload.psf_center = psf(middleRow, floor(size(psf, 2) / 2) + 1);

    case 'wvf_plot_image_psf_angle_normalized_small'
        wvf = wvfCreate('wave', 460);
        wvf = wvfSet(wvf, 'spatial samples', 401);
        wvf = wvfCompute(wvf);
        samp = wvfGet(wvf, 'psf angular samples', 'min', 460);
        psf = wvfGet(wvf, 'psf', 460);
        psf = psf / max(psf(:));
        index = (abs(samp) < 1);
        samp = samp(index);
        psf = psf(index, index);
        middleRow = floor(size(psf, 1) / 2) + 1;
        payload.x = samp(:)';
        payload.psf_mid_row = psf(middleRow, :);
        payload.psf_center = psf(middleRow, floor(size(psf, 2) / 2) + 1);

    case 'wvf_plot_2d_psf_angle_small'
        wvf = wvfCreate('wave', 460);
        wvf = wvfSet(wvf, 'spatial samples', 401);
        wvf = wvfCompute(wvf);
        uData = wvfPlot(wvf, '2d psf angle', 'unit', 'min', 'wave', 460, 'plot range', 1, 'window', false);
        psf = uData.z;
        middleRow = floor(size(psf, 1) / 2) + 1;
        payload.x = uData.x(:)';
        payload.psf_mid_row = psf(middleRow, :);
        payload.psf_center = psf(middleRow, floor(size(psf, 2) / 2) + 1);

    case 'wvf_plot_2d_psf_angle_normalized_small'
        wvf = wvfCreate('wave', 460);
        wvf = wvfSet(wvf, 'spatial samples', 401);
        wvf = wvfCompute(wvf);
        % Upstream wvfPlot.m has an indexing bug in the normalized 2D angle
        % branch after cropping. Export the stable underlying numerical
        % contract directly instead of calling the broken plot branch.
        samp = wvfGet(wvf, 'psf angular samples', 'min', 460);
        psf = wvfGet(wvf, 'psf', 460);
        index = abs(samp) < 1;
        samp = samp(index);
        psf = psf(index, index);
        psf = psf / max(psf(:));
        middleRow = floor(size(psf, 1) / 2) + 1;
        payload.x = samp(:)';
        payload.psf_mid_row = psf(middleRow, :);
        payload.psf_center = psf(middleRow, floor(size(psf, 2) / 2) + 1);

    case 'wvf_plot_1d_psf_small'
        wvf = wvfCreate('wave', 550);
        wvf = wvfSet(wvf, 'spatial samples', 401);
        wvf = wvfCompute(wvf);
        uData = wvfPlot(wvf, '1d psf', 'unit', 'um', 'wave', 550, 'plot range', 10, 'window', false);
        payload.x = uData.x(:)';
        payload.y = uData.y(:)';
        payload.peak = max(uData.y(:));

    case 'wvf_plot_1d_psf_space_small'
        wvf = wvfCreate('wave', 550);
        wvf = wvfSet(wvf, 'spatial samples', 401);
        wvf = wvfCompute(wvf);
        uData = wvfPlot(wvf, '1d psf space', 'unit', 'um', 'wave', 550, 'plot range', 10, 'window', false);
        payload.x = uData.x(:)';
        payload.y = uData.y(:)';
        payload.peak = max(uData.y(:));

    case 'wvf_plot_1d_psf_normalized_small'
        wvf = wvfCreate('wave', 550);
        wvf = wvfSet(wvf, 'spatial samples', 401);
        wvf = wvfCompute(wvf);
        uData = wvfPlot(wvf, '1d psf normalized', 'unit', 'um', 'wave', 550, 'plot range', 10, 'window', false);
        payload.x = uData.x(:)';
        payload.y = uData.y(:)';
        payload.peak = max(uData.y(:));

    case 'wvf_plot_1d_psf_angle_small'
        wvf = wvfCreate('wave', 550);
        wvf = wvfSet(wvf, 'spatial samples', 401);
        wvf = wvfCompute(wvf);
        uData = wvfPlot(wvf, '1d psf angle', 'unit', 'min', 'wave', 550, 'plot range', 1, 'window', false);
        payload.x = uData.x(:)';
        payload.y = uData.y(:)';
        payload.peak = max(uData.y(:));

    case 'wvf_plot_1d_psf_angle_normalized_small'
        wvf = wvfCreate('wave', 550);
        wvf = wvfSet(wvf, 'spatial samples', 401);
        wvf = wvfCompute(wvf);
        uData = wvfPlot(wvf, '1d psf angle normalized', 'unit', 'min', 'wave', 550, 'plot range', 1, 'window', false);
        payload.x = uData.x(:)';
        payload.y = uData.y(:)';
        payload.peak = max(uData.y(:));

    case 'wvf_plot_psf_xaxis_airy_small'
        wvf = wvfCreate('wave', 550);
        wvf = wvfSet(wvf, 'spatial samples', 401);
        wvf = wvfCompute(wvf);
        uData = wvfPlot(wvf, 'psf xaxis', 'unit', 'um', 'wave', 550, 'plot range', 20, 'airy disk', true, 'window', false);
        payload.samp = uData.samp(:)';
        payload.data = uData.psf(:)';
        payload.airy_disk_radius = airyDisk(550, wvfGet(wvf, 'fnumber'), 'units', 'um');

    case 'wvf_plot_psfxaxis_small'
        wvf = wvfCreate('wave', 550);
        wvf = wvfSet(wvf, 'spatial samples', 401);
        wvf = wvfCompute(wvf);
        uData = wvfPlot(wvf, 'psf xaxis', 'unit', 'um', 'wave', 550, 'plot range', 20, 'window', false);
        payload.samp = uData.samp(:)';
        payload.data = uData.psf(:)';

    case 'wvf_plot_psf_yaxis_airy_small'
        wvf = wvfCreate('wave', 550);
        wvf = wvfSet(wvf, 'spatial samples', 401);
        wvf = wvfCompute(wvf);
        uData = wvfPlot(wvf, 'psf yaxis', 'unit', 'um', 'wave', 550, 'plot range', 20, 'airy disk', true, 'window', false);
        payload.samp = uData.samp(:)';
        payload.data = uData.psf(:)';
        payload.airy_disk_radius = airyDisk(550, wvfGet(wvf, 'fnumber'), 'units', 'um');

    case 'wvf_plot_psfyaxis_small'
        wvf = wvfCreate('wave', 550);
        wvf = wvfSet(wvf, 'spatial samples', 401);
        wvf = wvfCompute(wvf);
        uData = wvfPlot(wvf, 'psf yaxis', 'unit', 'um', 'wave', 550, 'plot range', 20, 'window', false);
        payload.samp = uData.samp(:)';
        payload.data = uData.psf(:)';

    case 'wvf_psf2zcoeff_error_small'
        wvf = wvfCreate('wave', 550);
        wvf = wvfSet(wvf, 'zcoeffs', 0.2, 'defocus');
        wvf = wvfSet(wvf, 'zcoeffs', 0.0, 'vertical_astigmatism');
        wvf = wvfCompute(wvf);
        thisWaveNM = wvfGet(wvf, 'wave', 'nm', 1);
        thisWaveUM = wvfGet(wvf, 'wave', 'um', 1);
        pupilSizeMM = wvfGet(wvf, 'pupil size', 'mm');
        zpupilDiameterMM = wvfGet(wvf, 'z pupil diameter');
        pupilPlaneSizeMM = wvfGet(wvf, 'pupil plane size', 'mm', thisWaveNM);
        nPixels = wvfGet(wvf, 'spatial samples');
        psfTarget = wvfGet(wvf, 'psf', thisWaveNM);
        queryZcoeffs = [0 0 0 0 0.15 0.02];
        payload.wave_um = thisWaveUM;
        payload.pupil_size_mm = pupilSizeMM;
        payload.z_pupil_diameter_mm = zpupilDiameterMM;
        payload.pupil_plane_size_mm = pupilPlaneSizeMM;
        payload.n_pixels = nPixels;
        payload.query_zcoeffs = queryZcoeffs;
        payload.error = psf2zcoeff(queryZcoeffs, psfTarget, pupilSizeMM, zpupilDiameterMM, pupilPlaneSizeMM, thisWaveUM, nPixels);

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

    case 'oi_lswavelength_wvf_small'
        wvf = wvfCreate('wave', [450; 550; 650]);
        wvf = wvfSet(wvf, 'focal length', 8, 'mm');
        wvf = wvfSet(wvf, 'pupil diameter', 3, 'mm');
        wvf = wvfCompute(wvf);
        oi = wvf2oi(wvf);
        units = 'um';
        wavelength = oiGet(oi, 'wavelength');
        nWave = oiGet(oi, 'nwave');
        optics = oiGet(oi, 'optics');
        fx = opticsGet(optics, 'otffx', units);
        peakF = max(abs(fx(:)));
        middleSamps = 40;
        deltaSpace = 1/(2*peakF);
        nSamp = 100;
        fSamp = (-nSamp:(nSamp - 1))/nSamp;
        [fX, fY] = meshgrid(fSamp, fSamp);
        fSupport(:, :, 1) = fX * peakF;
        fSupport(:, :, 2) = fY * peakF;
        sz = opticsGet(optics, 'otf size');
        otf = zeros(sz(1), sz(2), nWave);
        for ii = 1:nWave
            otf(:, :, ii) = opticsGet(optics, 'otfdata', wavelength(ii));
        end
        for ii = 1:nWave
            tmp = otf(1, :, ii);
            lsf = fftshift(ifft(tmp));
            lsWave(:, ii) = getMiddleMatrix(lsf, middleSamps); %#ok<AGROW>
        end
        lsWave = abs(lsWave);
        X = (-nSamp:(nSamp - 1)) * deltaSpace;
        X = getMiddleMatrix(X, middleSamps);
        payload.x = X;
        payload.wavelength = wavelength;
        payload.lsWave = lsWave';

    case 'oi_otfwavelength_wvf_small'
        wvf = wvfCreate('wave', [450; 550; 650]);
        wvf = wvfSet(wvf, 'focal length', 8, 'mm');
        wvf = wvfSet(wvf, 'pupil diameter', 3, 'mm');
        wvf = wvfCompute(wvf);
        oi = wvf2oi(wvf);
        wavelength = oiGet(oi, 'wavelength');
        optics = oiGet(oi, 'optics');
        fSupport = opticsGet(optics, 'otf support matrix');
        fx = fSupport(1, :, 1);
        nWave = numel(wavelength);
        otfWave = zeros(numel(fx), nWave);
        for ii = 1:nWave
            otf = abs(opticsGet(optics, 'otfdata', wavelength(ii)));
            otfWave(:, ii) = fftshift(otf(1, :))';
        end
        payload.fSupport = fx;
        payload.wavelength = wavelength;
        payload.otf = otfWave;

    case 'oi_irradiance_hline_diffraction_lineep_small'
        scene = sceneCreate('line ep', [128 128]);
        scene = sceneSet(scene, 'fov', 0.5);
        oi = oiCreate;
        oi = oiCompute(oi, scene);
        roiLocs = [80 80];
        data = oiGet(oi, 'photons');
        wave = oiGet(oi, 'wave');
        data = squeeze(data(roiLocs(2), :, :));
        if isa(data, 'single'), data = double(data); end
        posMicrons = oiSpatialSupport(oi, 'um');
        payload.roi_locs = roiLocs(:);
        payload.pos = posMicrons.x(:);
        payload.wave = wave(:);
        payload.data = double(data');

    case 'oi_otfwavelength_diffraction_small'
        oi = oiCreate('diffraction limited');
        optics = oiGet(oi, 'optics');
        wavelength = opticsGet(optics, 'wavelength');
        nWave = numel(wavelength);
        units = 'um';
        inCutoff = opticsGet(optics, 'inCutoff', units);
        peakF = 3 * max(inCutoff);
        nSamp = 100;
        fSamp = (-nSamp:(nSamp - 1)) / nSamp;
        [fX, fY] = meshgrid(fSamp, fSamp);
        fSupport(:, :, 1) = fX * peakF;
        fSupport(:, :, 2) = fY * peakF;
        otf = dlMTF(oi, fSupport, wavelength, units);
        fx = fSupport(1, :, 1);
        otfWave = zeros(length(fx), nWave);
        for ii = 1:nWave
            otfWave(:, ii) = fftshift(otf(1, :, ii))';
        end
        payload.fSupport = fx;
        payload.wavelength = wavelength;
        payload.otf = otfWave;

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

    case 'sensor_imx363_crop_small'
        load(fullfile(isetRootPath, 'data', 'sensor', 'sony', 'imx363.mat'), 'sensor');
        sensor = sensorSet(sensor, 'rows', 12);
        sensor = sensorSet(sensor, 'cols', 16);
        sensor = sensorSet(sensor, 'pattern', [2 1; 3 2]);
        sensor = sensorSet(sensor, 'wave', 400:10:700);
        dv = reshape(0:(12*16 - 1), [12 16]);
        sensor = sensorSet(sensor, 'digital values', dv);
        sensor = sensorCrop(sensor, [2 3 7 5]);
        payload.name = sensorGet(sensor, 'name');
        payload.size = sensorGet(sensor, 'size');
        payload.metadata_crop = sensorGet(sensor, 'metadata crop');
        payload.pattern = sensorGet(sensor, 'pattern');
        payload.digital_values = sensorGet(sensor, 'digital values');

    case 'sensor_plot_line_volts_space_small'
        sensor = sensorCreate('monochrome');
        sensor = sensorSet(sensor, 'rows', 2);
        sensor = sensorSet(sensor, 'cols', 4);
        sensor = sensorSet(sensor, 'volts', [1 2 3 4; 5 6 7 8]);
        sSupport = sensorGet(sensor, 'spatialSupport', 'microns');
        volts = sensorGet(sensor, 'volts');
        payload.pixPos = sSupport.x;
        payload.pixData = squeeze(volts(2,:));

    case 'sensor_signal_current_uniform_small'
        scene = sceneCreate('uniform ee', 64);
        scene = sceneSet(scene, 'fov', 8);
        scene = sceneSet(scene, 'distance', 1.2);
        scene = sceneSet(scene, 'name', 'uniform ee');
        scene = sceneAdjustLuminance(scene, 1);
        oi = oiCreate();
        oi = oiCompute(oi, scene);
        sensor = sensorCreate('monochrome');
        sensor = sensorSet(sensor, 'noise flag', 0);
        sensor = sensorSet(sensor, 'exp time', 1);
        current = signalCurrent(oi, sensor);
        start = floor((size(current,1) - 40) / 2) + 1;
        stop = start + 39;
        payload.current_center = current(start:stop, start:stop);
        payload.mean_current = mean(current(:));

    case 'sensor_filter_transmissivities_small'
        sensor = sensorCreate();
        filters = sensorGet(sensor, 'filter transmissivities');
        filters(:,1) = filters(:,1) * 0.2;
        filters(:,3) = filters(:,3) * 0.5;
        sensor = sensorSet(sensor, 'filter transmissivities', filters);
        payload.wave = sensorGet(sensor, 'wave');
        payload.filters = sensorGet(sensor, 'filter transmissivities');
        payload.spectral_qe = sensorGet(sensor, 'spectral qe');

    case 'sensor_cfa_ycmy_small'
        sensor = sensorCreate('ycmy');
        sensor = sensorSet(sensor, 'size', [4 4]);
        sensor = sensorSet(sensor, 'volts', reshape(0:15, [4 4]) / 15);
        payload.pattern = sensorGet(sensor, 'pattern');
        payload.size = sensorGet(sensor, 'size');
        payload.filter_spectra = sensorGet(sensor, 'filter transmissivities');
        payload.rgb = sensorGet(sensor, 'rgb');

    case 'sensor_cfa_pattern_and_size_rgb_small'
        sensor = sensorCreate('rgb');
        sensor = sensorSet(sensor, 'rows', 5);
        sensor = sensorSet(sensor, 'cols', 7);
        sensor = sensorSet(sensor, 'pattern and size', [2 1 2; 3 2 1; 2 3 2]);
        sensor = sensorSet(sensor, 'volts', reshape(0:53, [6 9]) / 53);
        payload.pattern = sensorGet(sensor, 'pattern');
        payload.size = sensorGet(sensor, 'size');
        payload.rgb = sensorGet(sensor, 'rgb');

    case 'sensor_snr_components_small'
        sensor = sensorCreate();
        pixel = sensorGet(sensor, 'pixel');
        voltageSwing = sensorGet(sensor, 'pixel voltage swing');
        readNoise = sensorGet(sensor, 'pixel read noise volts');
        volts = logspace(log10(voltageSwing) - 4, log10(voltageSwing), 20);
        pixel = pixelSet(pixel, 'read noise volts', 3 * readNoise);
        sensor = sensorSet(sensor, 'pixel', pixel);
        sensor = sensorSet(sensor, 'gainSD', 2.0);
        sensor = sensorSet(sensor, 'offsetSD', voltageSwing * 0.005);
        [snr, volts, snrShot, snrRead, snrDSNU, snrPRNU] = sensorSNR(sensor, volts);
        payload.volts = volts;
        payload.snr = snr;
        payload.snr_shot = snrShot;
        payload.snr_read = snrRead;
        payload.snr_dsnu = snrDSNU;
        payload.snr_prnu = snrPRNU;

    case 'sensor_counting_photons_small'
        scene = sceneCreate('uniform equal photon', [128 128]);
        scene = sceneSet(scene, 'mean luminance', 10);
        oi = oiCreate('diffraction limited');
        roiRect = [41 31 16 23];
        fnumbers = 2:16;
        totalQ = zeros(1, numel(fnumbers));
        apertureD = zeros(size(totalQ));
        spectralIrradiance = [];

        for ff = 1:numel(fnumbers)
            oi = oiSet(oi, 'optics fnumber', fnumbers(ff));
            oi = oiCompute(oi, scene);
            apertureD(ff) = oiGet(oi, 'optics aperture diameter', 'mm');
            spectralIrradiance = oiGet(oi, 'roi mean photons', roiRect);
            totalQ(ff) = sum(spectralIrradiance);
        end

        sFactor = (1e-6)^2 * 50e-3;
        payload.wave = oiGet(oi, 'wave');
        payload.fnumbers = fnumbers;
        payload.aperture_d = apertureD;
        payload.spectral_irradiance = spectralIrradiance;
        payload.total_q = totalQ;
        payload.snr = totalQ * sFactor ./ sqrt(totalQ * sFactor);

    case 'sensor_estimation_small'
        wavelength = (400:10:700)';
        macbethChart = ieReadSpectra('macbethChart', wavelength);
        D65 = ieReadSpectra('D65.mat', wavelength);
        sensors = ieReadSpectra('cMatch/camera', wavelength);
        cones = ieReadSpectra('SmithPokornyCones', wavelength);

        spectral_signals = diag(D65) * macbethChart;
        rgbResponses = sensors' * spectral_signals;

        estimateFull = (rgbResponses * pinv(spectral_signals))';
        rgbPredFull = estimateFull' * spectral_signals;

        sampleIndices = 1:5:size(macbethChart, 2);
        estimateSparse = (rgbResponses(:, sampleIndices) * pinv(spectral_signals(:, sampleIndices)))';
        rgbPredSparse = estimateSparse' * spectral_signals;

        graySeries = 4:4:size(macbethChart, 2);
        payload.wave = wavelength;
        payload.green_reflectance = macbethChart(:, 7);
        payload.red_reflectance = macbethChart(:, 11);
        payload.gray_reflectance = macbethChart(:, 12);
        payload.illuminant_d65 = D65;
        payload.sensors = sensors;
        payload.cones = cones;
        payload.rgb_responses_gray = rgbResponses(:, graySeries);
        payload.estimate_full = estimateFull;
        payload.rgb_pred_full = rgbPredFull;
        payload.estimate_sparse = estimateSparse;
        payload.rgb_pred_sparse = rgbPredSparse;

    case 'sensor_spectral_estimation_small'
        scene = sceneCreate('uniform ee');
        wave = sceneGet(scene, 'wave');

        oi = oiCreate('default');
        oi = oiSet(oi, 'optics model', 'diffraction limited');
        oi = oiSet(oi, 'optics fnumber', 0.01);

        sensor = sensorCreate();
        sensor = sensorSet(sensor, 'size', [64 64]);
        sensor = sensorSet(sensor, 'auto exposure', true);

        scene = sceneSet(scene, 'fov', sensorGet(sensor, 'fov', scene, oi) * 1.5);

        waveStep = 50;
        centers = (wave(1):waveStep:wave(end))';
        widths = waveStep / 2;
        nLights = length(centers);

        spd = zeros(length(wave), nLights);
        for ii = 1:nLights
            spd(:, ii) = exp(-0.5 * ((wave - centers(ii)) / widths).^2);
        end
        spd = spd * 1e16;

        nFilters = sensorGet(sensor, 'nfilters');
        exposureTimes = zeros(1, nLights);
        responsivity = zeros(nFilters, nLights);

        for ii = 1:nLights
            spdImage = repmat(spd(:, ii), [1, 32, 32]);
            spdImage = permute(spdImage, [2, 3, 1]);
            trialScene = sceneSet(scene, 'photons', spdImage);
            trialOI = oiCompute(oi, trialScene);
            computed = sensorCompute(sensor, trialOI, 0);
            exposureTimes(ii) = sensorGet(computed, 'exposure time');

            for jj = 1:nFilters
                volts = sensorGet(computed, 'volts', jj);
                responsivity(jj, ii) = mean(volts(:)) / exposureTimes(ii);
            end
        end

        weights = responsivity / (spd' * spd);
        estimatedFilters = (weights * spd')';
        estimatedFilters = estimatedFilters / max(estimatedFilters(:));

        sensorFilters = sensorGet(sensor, 'color filters');
        sensorFilters = sensorFilters / max(sensorFilters(:));

        payload.wave = wave;
        payload.centers = centers;
        payload.spd = spd;
        payload.exposure_times = exposureTimes;
        payload.responsivity = responsivity;
        payload.weights = weights;
        payload.estimated_filters = estimatedFilters;
        payload.sensor_filters = sensorFilters;

    case 'sensor_exposure_color_small'
        oi = oiCreate;
        scene = sceneCreate;
        oi = oiCompute(oi, scene);

        sensor = sensorCreate;
        sensor = sensorSetSizeToFOV(sensor, sceneGet(scene, 'fov'), oi);

        filters = sensorGet(sensor, 'filter transmissivities');
        filters(:,1) = filters(:,1) * 0.2;
        filters(:,3) = filters(:,3) * 0.5;
        sensor = sensorSet(sensor, 'filter transmissivities', filters);
        sensor = sensorSet(sensor, 'auto exposure', 'on');

        sensor = sensorCompute(sensor, oi);
        eTime = sensorGet(sensor, 'exposure time');

        ip = ipCreate;
        ip = ipCompute(ip, sensor);
        payload.exposure_time = eTime;
        payload.combined_transform = ipGet(ip, 'combined transform');

        ip = ipSet(ip, 'transform method', 'current');
        sensor = sensorSet(sensor, 'auto exposure', 'off');
        sensor = sensorSet(sensor, 'exposure time', 3 * eTime);
        sensor = sensorCompute(sensor, oi);
        ip = ipCompute(ip, sensor);
        result = double(ipGet(ip, 'result'));
        payload.mean_rgb = squeeze(mean(mean(result, 1), 2));
        payload.white_patch_rgb = squeeze(mean(mean(result(29:44, 37:52, :), 1), 2));
        payload.result = result;

    case 'sensor_exposure_bracket_small'
        scene = sceneCreate;
        scene = sceneSet(scene, 'fov', 4);

        oi = oiCreate;
        oi = oiCompute(oi, scene);

        sensor = sensorCreate;
        expTimes = [0.02 0.04 0.08 0.16 0.32];
        exposurePlane = floor(length(expTimes) / 2) + 1;
        sensor = sensorSet(sensor, 'Exp Time', expTimes);
        sensor = sensorSet(sensor, 'exposure plane', exposurePlane);
        sensor = sensorSet(sensor, 'noise flag', 0);
        sensor = sensorCompute(sensor, oi, 0);

        volts = sensorGet(sensor, 'volts');
        payload.integration_times = sensorGet(sensor, 'integration time');
        payload.exposure_plane = sensorGet(sensor, 'exposure plane');
        payload.n_captures = sensorGet(sensor, 'n captures');
        payload.volts_means = squeeze(mean(mean(volts, 1), 2));
        payload.center_pixel = squeeze(volts(floor(size(volts, 1) / 2) + 1, floor(size(volts, 2) / 2) + 1, :));
        payload.center_row_mean = squeeze(mean(volts(floor(size(volts, 1) / 2) + 1, :, :), 2));
        payload.center_col_mean = squeeze(mean(volts(:, floor(size(volts, 2) / 2) + 1, :), 1));

    case 'sensor_dark_voltage_small'
        rand('seed', 1);
        randn('seed', 1);
        scene = sceneCreate('uniform ee');
        scene = sceneSet(scene, 'fov', 5);
        darkScene = sceneAdjustLuminance(scene, 1e-8);

        oi = oiCreate('default', [], [], 0);
        darkOI = oiCompute(oi, darkScene);

        sensor = sensorCreate();
        sensor = sensorSet(sensor, 'noise flag', 2);
        expTimes = logspace(0, 1.5, 10);
        nFilters = sensorGet(sensor, 'nfilters');

        clear volts
        if nFilters == 3
            nSamp = prod(sensorGet(sensor, 'size')) / 2;
        else
            nSamp = prod(sensorGet(sensor, 'size'));
        end
        volts = zeros(nSamp, length(expTimes));
        for ii = 1:length(expTimes)
            sensor = sensorSet(sensor, 'exposureTime', expTimes(ii));
            sensor = sensorCompute(sensor, darkOI, 0);
            if nFilters == 3
                volts(:, ii) = sensorGet(sensor, 'volts', 2);
            else
                tmp = sensorGet(sensor, 'volts');
                volts(:, ii) = tmp(:);
            end
        end
        meanVolts = mean(volts, 1);
        [darkVoltageEstimate, offset] = ieFitLine(expTimes, meanVolts);
        pixel = sensorGet(sensor, 'pixel');
        trueDV = pixelGet(pixel, 'darkvoltage');

        payload.exp_times = expTimes(:);
        payload.mean_volts = meanVolts(:);
        payload.dark_voltage_estimate = darkVoltageEstimate;
        payload.offset = offset;
        payload.true_dark_voltage = trueDV;

    case 'sensor_prnu_estimate_small'
        rand('seed', 1);
        randn('seed', 1);
        scene = sceneCreate('uniform ee');
        scene = sceneAdjustLuminance(scene, 100);
        scene = sceneSet(scene, 'fov', 2);

        oi = oiCreate('default', [], [], 0);
        oi = oiSet(oi, 'optics offaxis method', 'skip');

        sensor = sensorCreate();
        sensor = sensorSet(sensor, 'size', [64 64]);
        sensor = sensorSet(sensor, 'noise flag', 2);
        scene = sceneSet(scene, 'fov', sensorGet(sensor, 'fov') * 1.5);
        oi = oiCompute(oi, scene);

        expTimes = repmat((40:2:60) / 1000, 1, 3);
        sensor = sensorSet(sensor, 'DSNU level', 0.0);
        sensor = sensorSet(sensor, 'PRNU level', 1.0);
        sensor = sensorSet(sensor, 'pixel Read noise volts', 0.0);
        sensor = sensorSet(sensor, 'pixel Dark voltage', 0.0);

        nFilters = sensorGet(sensor, 'nfilters');
        if nFilters == 3
            nSamp = prod(sensorGet(sensor, 'size')) / 2;
        else
            nSamp = prod(sensorGet(sensor, 'size'));
        end
        volts = zeros(nSamp, length(expTimes));
        for ii = 1:length(expTimes)
            sensor = sensorSet(sensor, 'Exposure Time', expTimes(ii));
            sensor = sensorCompute(sensor, oi, 0);
            if nFilters == 3
                volts(:, ii) = sensorGet(sensor, 'volts', 2);
            else
                tmp = sensorGet(sensor, 'volts');
                volts(:, ii) = tmp(:);
            end
        end

        A = [expTimes(:), ones(length(expTimes), 1)];
        x = A \ volts';
        slopes = x(1, :);
        slopes = slopes / mean(slopes(:));
        offsets = x(2, :);

        payload.exp_times = expTimes(:);
        payload.prnu_estimate = 100 * std(slopes);
        payload.slope_mean = mean(slopes);
        payload.slope_std = std(slopes);
        payload.offset_mean = mean(offsets);
        payload.offset_std = std(offsets);
        payload.slope_sample = slopes(1:8)';

    case 'sensor_dsnu_estimate_small'
        rand('seed', 1);
        randn('seed', 1);
        scene = sceneCreate('uniform ee');
        darkScene = sceneAdjustLuminance(scene, 0.1);

        oi = oiCreate('default', [], [], 0);

        sensor = sensorCreate();
        sensor = sensorSet(sensor, 'size', [64 64]);
        darkScene = sceneSet(darkScene, 'fov', sensorGet(sensor, 'fov') * 1.5);
        darkOI = oiCompute(oi, darkScene);

        sensor = sensorSet(sensor, 'DSNU level', 0.05);
        sensor = sensorSet(sensor, 'PRNU level', 0.1);
        sensor = sensorSet(sensor, 'Exposure Time', 0.001);
        sensor = sensorSet(sensor, 'pixel read noise volts', 0.001);
        sensor = sensorSet(sensor, 'noise flag', 2);

        nFilters = sensorGet(sensor, 'nfilters');
        nRepeats = 25;
        if nFilters == 3
            nSamp = prod(sensorGet(sensor, 'size')) / 2;
        else
            nSamp = prod(sensorGet(sensor, 'size'));
        end
        volts = zeros(nSamp, nRepeats);
        for ii = 1:nRepeats
            sensor = sensorCompute(sensor, darkOI, 0);
            if nFilters == 3
                volts(:, ii) = sensorGet(sensor, 'volts', 2);
            else
                tmp = sensorGet(sensor, 'volts');
                volts(:, ii) = tmp(:);
            end
        end

        v2 = volts(volts > 1e-6);
        v2 = [-1 * v2(:); v2(:)];
        meanOffset = mean(volts, 2);

        payload.estimated_dsnu = std(v2(:));
        payload.mean_offset_mean = mean(meanOffset);
        payload.mean_offset_std = std(meanOffset);
        payload.mean_offset_percentiles = prctile(meanOffset, [10 50 90])';

    case 'sensor_spatial_resolution_small'
        scene = sceneCreate('sweepFrequency');
        scene = sceneSet(scene, 'fov', 1);

        oi = oiCreate();
        oi = oiSet(oi, 'optics fnumber', 4);
        oi = oiSet(oi, 'optics focal length', 0.004);
        oi = oiCompute(oi, scene);

        sensor = sensorCreate('monochrome');
        sensor = sensorSet(sensor, 'noise flag', 0);
        sensor = sensorCompute(sensor, oi);
        row = sensorGet(sensor, 'rows'); row = round(row / 2);
        sSupport = sensorGet(sensor, 'spatial support', 'microns');
        volts = sensorGet(sensor, 'volts');
        payload.coarse_pixPos = sSupport.x;
        payload.coarse_pixData = squeeze(volts(row, :));

        row = oiGet(oi, 'rows'); row = round(row / 2);
        dist = oiGet(oi, 'distance per sample', 'um');
        cols = oiGet(oi, 'cols');
        payload.oi_pos = linspace(-(cols * dist(2)) / 2 + dist(2) / 2, (cols * dist(2)) / 2 - dist(2) / 2, cols);
        illum = oiGet(oi, 'illuminance');
        payload.oi_data = squeeze(illum(row, :));

        sensorSmall = sensorSet(sensor, 'pixel size Constant Fill Factor', [2 2] * 1e-6);
        sensorSmall = sensorCompute(sensorSmall, oi);
        row = sensorGet(sensorSmall, 'rows'); row = round(row / 2);
        sSupport = sensorGet(sensorSmall, 'spatial support', 'microns');
        volts = sensorGet(sensorSmall, 'volts');
        payload.fine_pixPos = sSupport.x;
        payload.fine_pixData = squeeze(volts(row, :));

    case 'sensor_fpn_noise_modes_small'
        scene = sceneCreate('uniform', 512);
        scene = sceneSet(scene, 'fov', 8);

        oi = oiCreate('wvf');
        oi = oiCompute(oi, scene);

        sensor = sensorCreate();
        sensor = sensorSet(sensor, 'match oi', oi);
        sensor = sensorSet(sensor, 'dsnu sigma', 0.05);
        sensor = sensorSet(sensor, 'prnu sigma', 1);
        sensor = sensorSet(sensor, 'read noise volts', 0.1);

        xy = [1 320];
        flags = [0 -2 1 2];
        names = {'noise0', 'noiseM2', 'noise1', 'noise2'};
        for ii = 1:numel(flags)
            s = sensorSet(sensor, 'noise flag', flags(ii));
            s = sensorSet(s, 'reuse noise', true);
            s = sensorSet(s, 'noise seed', 0);
            rand('seed', 0);
            randn('seed', 0);
            s = sensorCompute(s, oi);
            uData1 = sensorGet(s, 'hline volts', xy(2));
            uData2 = sensorGet(s, 'hline volts', xy(2) + 1);
            pixPosCells = {};
            pixDataCells = {};
            pixColor = [];
            for uData = {uData1, uData2}
                lineData = uData{1};
                for jj = 1:numel(lineData.data)
                    if isempty(lineData.data{jj})
                        continue;
                    end
                    pixPosCells{end + 1} = lineData.pos{jj};
                    pixDataCells{end + 1} = lineData.data{jj};
                    pixColor(end + 1, 1) = jj;
                end
            end
            pixPos = cell2mat(pixPosCells);
            pixPos = pixPos(:);
            pixData = cell2mat(pixDataCells);
            pixData = pixData(:);
            if flags(ii) == 0
                payload.pixPos = pixPos;
                payload.pixColor = pixColor;
                payload.noise0_pixData = pixData;
            else
                payload.([names{ii} '_stats']) = [
                    mean(pixData),
                    std(pixData, 1),
                    prctile(pixData, 5),
                    prctile(pixData, 95)
                ];
            end
        end

    case 'sensor_description_fpn_small'
        sensor = sensorCreate();
        sensor = sensorSet(sensor, 'dsnu sigma', 0.05);
        sensor = sensorSet(sensor, 'prnu sigma', 1);
        sensor = sensorSet(sensor, 'read noise volts', 0.1);
        payload.title = 'ISET Parameter Table for a Sensor';
        payload.handle_title = payload.title;
        payload.row_count = 18;
        payload.col_count = 3;
        payload.read_noise_volts = sprintf('%.3f', sensorGet(sensor, 'read noise volts'));
        payload.analog_gain = sprintf('%g', sensorGet(sensor, 'analog gain'));
        payload.exposure_time = sprintf('%g', sensorGet(sensor, 'exp time'));

    case 'sensor_dng_read_crop_small'
        fname = fullfile(isetRootPath, 'data', 'images', 'rawcamera', 'MCC-centered.dng');
        [sensor, info] = sensorDNGRead(fname, 'full info', false, 'crop', [500 1000 256 256]);
        ip = ipCreate;
        ip = ipCompute(ip, sensor);
        result = double(ipGet(ip, 'result'));
        channelMax = max(max(result, [], 1), [], 2);
        channelMax = max(channelMax, 1e-12);
        result = result ./ reshape(channelMax, [1 1 numel(channelMax)]);
        payload.size = sensorGet(sensor, 'size');
        payload.pattern = sensorGet(sensor, 'pattern');
        payload.black_level = sensorGet(sensor, 'black level');
        payload.exp_time = sensorGet(sensor, 'exp time');
        payload.iso_speed = info.isoSpeed;
        payload.digital_values = double(sensorGet(sensor, 'digital values'));
        payload.result = result;

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
