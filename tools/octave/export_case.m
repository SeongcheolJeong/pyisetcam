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

    case 'wvf_thibos_model_small'
        measuredPupilMM = 4.5;
        calcPupilMM = 3.0;
        measuredWavelengthNM = 550;
        calcWaves = (450:100:650)';
        [sampleMean, sampleCov] = wvfLoadThibosVirtualEyes(measuredPupilMM);
        exampleCount = 10;
        exampleSubjectIndices = 1:3:exampleCount;
        standardNormal = local_deterministic_normal_samples(exampleCount, numel(sampleMean));
        exampleCoeffs = local_ie_mvnrnd(sampleMean, sampleCov, standardNormal);

        z = zeros(65, 1);
        z(1:13) = sampleMean(1:13);
        thisGuy = wvfCreate;
        thisGuy = wvfSet(thisGuy, 'zcoeffs', z);
        thisGuy = wvfSet(thisGuy, 'measured pupil', measuredPupilMM);
        thisGuy = wvfSet(thisGuy, 'calculated pupil', calcPupilMM);
        thisGuy = wvfSet(thisGuy, 'measured wavelength', measuredWavelengthNM);
        thisGuy = wvfSet(thisGuy, 'calc wave', calcWaves);
        thisGuy = wvfCompute(thisGuy);

        meanSubjectRows = [];
        meanSubjectPeaks = zeros(numel(calcWaves), 1);
        for ii = 1:numel(calcWaves)
            psf = wvfGet(thisGuy, 'psf', calcWaves(ii));
            psf = double(psf);
            meanSubjectRows(ii, :) = psf(floor(size(psf, 1) / 2) + 1, :);
            meanSubjectPeaks(ii) = max(psf(:));
        end

        subject = wvfCreate;
        subject = wvfSet(subject, 'measured pupil', measuredPupilMM);
        subject = wvfSet(subject, 'calculated pupil', calcPupilMM);
        subject = wvfSet(subject, 'measured wavelength', measuredWavelengthNM);
        subjectRows450 = [];
        subjectRows550 = [];
        subjectPeaks450 = zeros(numel(exampleSubjectIndices), 1);
        subjectPeaks550 = zeros(numel(exampleSubjectIndices), 1);
        selectedCoeffs = [];
        for ii = 1:numel(exampleSubjectIndices)
            z = zeros(65, 1);
            z(1:13) = exampleCoeffs(exampleSubjectIndices(ii), 1:13)';
            selectedCoeffs(ii, :) = z(1:13)';
            subject = wvfSet(subject, 'zcoeffs', z);

            subject = wvfSet(subject, 'calc wave', 450);
            subject = wvfCompute(subject);
            psf450 = double(wvfGet(subject, 'psf', 450));
            subjectRows450(ii, :) = psf450(floor(size(psf450, 1) / 2) + 1, :);
            subjectPeaks450(ii) = max(psf450(:));

            subject = wvfSet(subject, 'calc wave', 550);
            subject = wvfCompute(subject);
            psf550 = double(wvfGet(subject, 'psf', 550));
            subjectRows550(ii, :) = psf550(floor(size(psf550, 1) / 2) + 1, :);
            subjectPeaks550(ii) = max(psf550(:));
        end

        payload.measured_pupil_mm = measuredPupilMM;
        payload.calc_pupil_mm = calcPupilMM;
        payload.measured_wavelength_nm = measuredWavelengthNM;
        payload.calc_waves_nm = double(calcWaves);
        payload.example_coeffs = double(exampleCoeffs);
        payload.mean_subject_psf_mid_rows = double(meanSubjectRows);
        payload.mean_subject_psf_peaks = double(meanSubjectPeaks);
        payload.example_subject_indices = double(exampleSubjectIndices(:));
        payload.example_subject_coeffs = double(selectedCoeffs);
        payload.example_subject_psf_mid_rows_450 = double(subjectRows450);
        payload.example_subject_psf_mid_rows_550 = double(subjectRows550);
        payload.example_subject_psf_peaks_450 = double(subjectPeaks450);
        payload.example_subject_psf_peaks_550 = double(subjectPeaks550);

    case 'wvf_zernike_set_small'
        params = FOTParams;
        params.blockSize = 64;
        params.angles = [0, pi/4, pi/2];
        scene = sceneCreate('freq orient', params);
        scene = sceneSet(scene, 'fov', 5);

        astigmatismValues = [-1, 0, 1];
        defocusMicrons = 2;
        currentWvf = wvfCreate('wave', sceneGet(scene, 'wave'));
        psfMidRows = [];
        oiCenterRows550 = [];
        oiDefocus = zeros(numel(astigmatismValues), 1);
        oiAstigmatism = zeros(numel(astigmatismValues), 1);
        oiPeakPhotons550 = zeros(numel(astigmatismValues), 1);
        psfSupport = [];
        wave = sceneGet(scene, 'wave');
        wave550 = find(wave == 550, 1);

        for ii = 1:numel(astigmatismValues)
            currentWvf = wvfSet(currentWvf, 'zcoeffs', [defocusMicrons, astigmatismValues(ii)], {'defocus', 'vertical_astigmatism'});
            currentWvf = wvfCompute(currentWvf);
            uData = wvfPlot(currentWvf, 'psf', 'unit', 'um', 'wave', 550, 'plot range', 40, 'window', false);
            psf = double(uData.z);
            if isempty(psfSupport)
                psfSupport = double(uData.x(:));
            end
            psfMidRows(ii, :) = local_channel_normalize(psf(floor(size(psf, 1) / 2) + 1, :));

            oi = oiCompute(currentWvf, scene);
            photons = double(oiGet(oi, 'photons'));
            centerRow = squeeze(photons(floor(size(photons, 1) / 2) + 1, :, wave550));
            oiCenterRows550(ii, :) = local_channel_normalize(centerRow);
            oiDefocus(ii) = oiGet(oi, 'wvf', 'zcoeffs', 'defocus');
            oiAstigmatism(ii) = oiGet(oi, 'wvf', 'zcoeffs', 'vertical_astigmatism');
            oiPeakPhotons550(ii) = max(max(photons(:, :, wave550)));
        end

        payload.wave = double(wave(:));
        payload.astigmatism_values = double(astigmatismValues(:));
        payload.defocus_microns = defocusMicrons;
        payload.psf_support_um = psfSupport;
        payload.psf_mid_rows = double(psfMidRows);
        payload.oi_defocus_coeffs = double(oiDefocus(:));
        payload.oi_astigmatism_coeffs = double(oiAstigmatism(:));
        payload.oi_center_rows_550 = double(oiCenterRows550);
        payload.oi_peak_photons_550 = double(oiPeakPhotons550(:));

    case 'wvf_wavefronts_small'
        indices = 1:16;
        nVals = zeros(numel(indices), 1);
        mVals = zeros(numel(indices), 1);
        wavefrontMidRows = [];
        wavefrontMidCols = [];
        wavefrontPeakAbs = zeros(numel(indices), 1);
        xSupport = [];

        for ii = 1:numel(indices)
            wvf = wvfCreate;
            wvf = wvfSet(wvf, 'npixels', 801);
            wvf = wvfSet(wvf, 'measured pupil size', 2);
            wvf = wvfSet(wvf, 'calc pupil size', 2);
            wvf = wvfSet(wvf, 'zcoeff', 1, indices(ii));
            wvf = wvfCompute(wvf);
            [nVals(ii), mVals(ii)] = wvfOSAIndexToZernikeNM(indices(ii));
            uData = wvfPlot(wvf, 'image wavefront aberrations', 'unit', 'mm', 'wave', 550, 'plot range', 1, 'window', false);
            wavefront = double(uData.z);
            if isempty(xSupport)
                xSupport = double(uData.x(:)');
            end
            peakAbs = max(abs(wavefront(:)));
            if peakAbs <= 0
                peakAbs = 1;
            end
            normalizedWavefront = wavefront / peakAbs;
            middleRow = floor(size(wavefront, 1) / 2) + 1;
            middleCol = floor(size(wavefront, 2) / 2) + 1;
            wavefrontMidRows(ii, :) = normalizedWavefront(middleRow, :);
            wavefrontMidCols(ii, :) = normalizedWavefront(:, middleCol);
            wavefrontPeakAbs(ii) = peakAbs;
        end

        payload.indices = double(indices(:));
        payload.n = double(nVals(:));
        payload.m = double(mVals(:));
        payload.x = xSupport;
        payload.wavefront_mid_rows_norm = double(wavefrontMidRows);
        payload.wavefront_mid_cols_norm = double(wavefrontMidCols);
        payload.wavefront_peak_abs = double(wavefrontPeakAbs(:));
        payload.npixels = 801;
        payload.measured_pupil_mm = 2;
        payload.calc_pupil_mm = 2;

    case 'zernike_interpolation_small'
        raw = load(fullfile(isetRootPath, 'data', 'optics', 'zernike_doubleGauss.mat'), 'data');
        data = raw.data;
        wavelengths = double(data.wavelengths(:));
        imageHeights = double(data.image_heights(:));
        zCoeffs = data.zernikeCoefficients;

        imageHeightIndices = 1:4:21;
        thisWaveIndex = 3;
        testIndex = 6;
        wavelengthNM = wavelengths(thisWaveIndex);
        imageHeightsTest = imageHeights(imageHeightIndices);
        zernikeCoeffMatrix = zeros(numel(imageHeightIndices), numel(zCoeffs.(sprintf('wave_%d_field_%d', thisWaveIndex, imageHeightIndices(1)))));
        for ii = 1:numel(imageHeightIndices)
            zernikeCoeffMatrix(ii, :) = double(zCoeffs.(sprintf('wave_%d_field_%d', thisWaveIndex, imageHeightIndices(ii))))(:)';
        end

        nearestIndices = local_find_nearest_two(imageHeightIndices, testIndex);
        testHeight = imageHeights(testIndex);
        zernikeGT = double(zCoeffs.(sprintf('wave_%d_field_%d', thisWaveIndex, testIndex))(:));
        zernikeInterpolated = interp1(imageHeightsTest, zernikeCoeffMatrix, testHeight, 'linear')';
        validation = zernikeInterpolated - zernikeGT;

        [psfInterpolated, ~] = local_generate_fringe_psf(zernikeInterpolated);
        [psfGT, ~] = local_generate_fringe_psf(zernikeGT);
        [psf1, ~] = local_generate_fringe_psf(double(zCoeffs.(sprintf('wave_%d_field_%d', thisWaveIndex, nearestIndices(1)))));
        [psf2, ~] = local_generate_fringe_psf(double(zCoeffs.(sprintf('wave_%d_field_%d', thisWaveIndex, nearestIndices(2)))));
        psfInterpSpace = ...
            psf1 * (testHeight - imageHeights(nearestIndices(1))) / (nearestIndices(2) - nearestIndices(1)) + ...
            psf2 * (nearestIndices(2) - testHeight) / (nearestIndices(2) - nearestIndices(1));

        middleRow = floor(size(psfGT, 1) / 2) + 1;
        payload.this_wave_index = thisWaveIndex;
        payload.wavelength_nm = wavelengthNM;
        payload.image_height_indices = double(imageHeightIndices(:));
        payload.image_heights_test = double(imageHeightsTest(:));
        payload.test_index = testIndex;
        payload.test_height = testHeight;
        payload.nearest_indices = double(nearestIndices(:));
        payload.zernike_gt = double(zernikeGT(:));
        payload.zernike_interpolated = double(zernikeInterpolated(:));
        payload.validation = double(validation(:));
        payload.validation_rmse = sqrt(mean(validation(:) .^ 2));
        payload.psf_interpolated_mid_row_norm = local_channel_normalize(psfInterpolated(middleRow, :));
        payload.psf_gt_mid_row_norm = local_channel_normalize(psfGT(middleRow, :));
        payload.psf_interp_space_mid_row_norm = local_channel_normalize(psfInterpSpace(middleRow, :));
        payload.psf_interpolated_peak = max(psfInterpolated(:));
        payload.psf_gt_peak = max(psfGT(:));
        payload.psf_interp_space_peak = max(psfInterpSpace(:));

    case 'wvf_plot_script_sequence_small'
        wave550 = 550;
        wave460 = 460;
        wvf = wvfCreate;
        wvf = wvfSet(wvf, 'wave', wave550);
        wvf = wvfSet(wvf, 'spatial samples', 401);
        wvf = wvfCompute(wvf);

        uData550UM = wvfPlot(wvf, '1d psf', 'unit', 'um', 'wave', wave550, 'window', false);
        uData550MM = wvfPlot(wvf, '1d psf', 'unit', 'mm', 'wave', wave550, 'window', false);
        uData550Norm = wvfPlot(wvf, '1d psf normalized', 'unit', 'mm', 'wave', wave550, 'window', false);

        wvf = wvfSet(wvf, 'wave', wave460);
        wvf = wvfCompute(wvf);

        uData460Angle = wvfPlot(wvf, 'image psf angle', 'unit', 'min', 'wave', wave460, 'plot range', 1, 'window', false);
        psfAngle = double(uData460Angle.z);
        uData460Phase = wvfPlot(wvf, 'image pupil phase', 'unit', 'mm', 'wave', wave460, 'plot range', 2, 'window', false);
        pupilPhase = double(uData460Phase.z);

        middleRowAngle = floor(size(psfAngle, 1) / 2) + 1;
        middleRowPhase = floor(size(pupilPhase, 1) / 2) + 1;
        payload.wave_550_nm = wave550;
        payload.wave_460_nm = wave460;
        payload.line_550_um_x = double(uData550UM.x(:));
        payload.line_550_um_y_norm = local_channel_normalize(uData550UM.y(:));
        payload.line_550_mm_x = double(uData550MM.x(:));
        payload.line_550_mm_y_norm = local_channel_normalize(uData550MM.y(:));
        payload.line_550_mm_norm_y = double(uData550Norm.y(:));
        payload.psf_angle_460_x = double(uData460Angle.x(:));
        payload.psf_angle_460_mid_row_norm = local_channel_normalize(psfAngle(middleRowAngle, :));
        payload.psf_angle_460_center = psfAngle(middleRowAngle, floor(size(psfAngle, 2) / 2) + 1);
        payload.pupil_phase_460_x = double(uData460Phase.x(:));
        payload.pupil_phase_460_mid_row = double(pupilPhase(middleRowPhase, :));
        payload.pupil_phase_460_center = pupilPhase(middleRowPhase, floor(size(pupilPhase, 2) / 2) + 1);

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

    case 'wvf_astigmatism_small'
        maxUM = 20;
        wvfP = wvfCreate;
        wvfP = wvfSet(wvfP, 'lcaMethod', 'human');
        wvfParams = wvfCompute(wvfP);

        z4 = -0.5:0.5:0.5;
        z5 = -0.5:0.5:0.5;
        [Z4, Z5] = meshgrid(z4, z5);
        Zvals = [Z4(:), Z5(:)];

        xSupport = [];
        rowProfiles = [];
        colProfiles = [];
        centers = zeros(size(Zvals, 1), 1);
        for ii = 1:size(Zvals, 1)
            wvfParams = wvfSet(wvfParams, 'zcoeffs', Zvals(ii, :), {'defocus' 'vertical_astigmatism'});
            wvfParams = wvfSet(wvfParams, 'lcaMethod', 'human');
            wvfParams = wvfCompute(wvfParams);
            uData = wvfPlot(wvfParams, 'psf normalized', 'unit', 'um', 'wave', 550, 'plot range', maxUM, 'window', false);
            psf = double(uData.z);
            middleRow = floor(size(psf, 1) / 2) + 1;
            middleCol = floor(size(psf, 2) / 2) + 1;
            if isempty(xSupport)
                xSupport = double(uData.x(:));
                rowProfiles = zeros(size(Zvals, 1), numel(xSupport));
                colProfiles = zeros(size(Zvals, 1), numel(xSupport));
            end
            rowProfiles(ii, :) = psf(middleRow, :);
            colProfiles(ii, :) = psf(:, middleCol)';
            centers(ii) = psf(middleRow, middleCol);
        end

        payload.zvals = double(Zvals);
        payload.x = xSupport(:);
        payload.psf_mid_rows = rowProfiles;
        payload.psf_mid_cols = colProfiles;
        payload.psf_centers = centers(:);

    case 'wvf_diffraction_small'
        flengthMM = 6;
        flengthM = flengthMM*1e-3;
        fNumber = 3;
        thisWave = 550;

        wvf = wvfCreate;
        wvf = wvfSet(wvf, 'calc pupil diameter', flengthMM/fNumber);
        wvf = wvfSet(wvf, 'focal length', flengthM);
        wvf = wvfCompute(wvf);
        baseWvf = wvf;

        wvfPlot(wvf, 'psf', 'unit', 'um', 'wave', thisWave, 'plot range', 10, 'airy disk', true, 'window', false);

        oi = wvf2oi(wvf);
        oiPsfx = oiGet(oi, 'optics psf xaxis', thisWave, 'um');

        pupilMM = linspace(1.5, 8, 4);
        pupil550Airy = zeros(size(pupilMM));
        for ii = 1:numel(pupilMM)
            wvf = wvfSet(wvf, 'calc pupil diameter', pupilMM(ii));
            wvf = wvfCompute(wvf);
            uData = wvfPlot(wvf, 'image psf', 'unit', 'um', 'wave', thisWave, 'plot range', 5, 'airy disk', true, 'window', false);
            pupil550Airy(ii) = airyDisk(thisWave, flengthMM / pupilMM(ii), 'units', 'um', 'diameter', true);
        end

        thisWave = 400;
        wvf = wvfSet(wvf, 'calc wave', thisWave);
        pupil400Airy = zeros(size(pupilMM));
        for ii = 1:numel(pupilMM)
            wvf = wvfSet(wvf, 'calc pupil diameter', pupilMM(ii));
            wvf = wvfCompute(wvf);
            uData = wvfPlot(wvf, 'image psf', 'unit', 'um', 'wave', thisWave, 'plot range', 5, 'airy disk', true, 'window', false);
            pupil400Airy(ii) = airyDisk(thisWave, flengthMM / pupilMM(ii), 'units', 'um', 'diameter', true);
        end

        wvf = wvfSet(wvf, 'calc pupil diameter', 3);
        wvf = wvfSet(wvf, 'calc wave', 550);
        wList = linspace(400,700,4);
        lcaRows = [];
        lcaAiry = zeros(size(wList));
        for ii = 1:numel(wList)
            ww = wList(ii);
            wvf = wvfSet(wvf, 'calc wave', ww);
            wvf = wvfSet(wvf, 'lcaMethod', 'human');
            wvf = wvfCompute(wvf);
            uData = wvfPlot(wvf, 'image psf', 'unit', 'um', 'wave', ww, 'plot range', 20, 'airy disk', true, 'window', false);
            z = double(uData.z);
            lcaRows(ii, :) = z(floor(size(z, 1) / 2) + 1, :);
            lcaAiry(ii) = airyDisk(ww, flengthMM / 3, 'units', 'um', 'diameter', true);
        end

        wvf = wvfCreate;
        baseFLengthM = 7e-3;
        baseFNumber = 4;
        wvf = wvfSet(wvf, 'calc pupil diameter', (baseFLengthM*1e3)/baseFNumber);
        wvf = wvfSet(wvf, 'focal length', baseFLengthM);
        wvf = wvfCompute(wvf);

        focalSweepM = [baseFLengthM/2, baseFLengthM, baseFLengthM*2];
        focalUmPerDegree = focalSweepM * 1e6 * (2 * tan(pi/360));

        payload.base_fnumber_ratio_oi_wvf = oiGet(oi, 'optics fnumber') / max(wvfGet(baseWvf, 'fnumber'), 1e-12);
        payload.base_airy_diameter_um = airyDisk(550, fNumber, 'units', 'um', 'diameter', true);
        payload.base_oi_psfx_data = double(oiPsfx.data(:));
        payload.pupil_mm = double(pupilMM(:));
        payload.pupil_550_airy_diameter_um = double(pupil550Airy(:));
        payload.pupil_400_airy_diameter_um = double(pupil400Airy(:));
        payload.lca_wavelength_nm = double(wList(:));
        payload.lca_airy_diameter_um = double(lcaAiry(:));
        payload.lca_mid_rows = local_canonical_profile(lcaRows, 41);
        payload.focal_length_sweep_mm = double((focalSweepM(:))*1e3);
        payload.focal_length_um_per_degree = double(focalUmPerDegree(:));

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

    case 'optics_coc_small'
        object_distances = [0.5 3.0];
        f_numbers = [2.0 8.0];
        focal_length_m = 0.050;
        optics = opticsCreate;
        optics = opticsSet(optics, 'focal length', focal_length_m);
        optics = opticsSet(optics, 'fnumber', f_numbers(1));
        [circ_f2_focus_0_5, x_dist_focus_0_5] = opticsCoC(optics, object_distances(1), 'unit', 'mm', 'n samples', 50);
        [circ_f2_focus_3, x_dist_focus_3] = opticsCoC(optics, object_distances(2), 'unit', 'mm', 'n samples', 50);
        optics = opticsSet(optics, 'fnumber', f_numbers(2));
        circ_f8_focus_0_5 = opticsCoC(optics, object_distances(1), 'unit', 'mm', 'n samples', 50);
        circ_f8_focus_3 = opticsCoC(optics, object_distances(2), 'unit', 'mm', 'n samples', 50);
        payload.object_distances_m = object_distances;
        payload.f_numbers = f_numbers;
        payload.focal_length_m = focal_length_m;
        payload.x_dist_focus_0_5_m = x_dist_focus_0_5;
        payload.circ_f2_focus_0_5_mm = circ_f2_focus_0_5;
        payload.circ_f8_focus_0_5_mm = circ_f8_focus_0_5;
        payload.x_dist_focus_3_m = x_dist_focus_3;
        payload.circ_f2_focus_3_mm = circ_f2_focus_3;
        payload.circ_f8_focus_3_mm = circ_f8_focus_3;

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

    case 'oi_cos4th_small'
        scene = sceneCreate('uniform d65', 512);
        scene = sceneSet(scene, 'fov', 80);

        oi = oiCreate('shift invariant');
        focal_length_default_m = oiGet(oi, 'optics focal length');
        oi = oiCompute(oi, scene);
        size_default = oiGet(oi, 'size');
        pos_default = oiSpatialSupport(oi, 'um');
        illuminance_default = real(oiGet(oi, 'illuminance'));
        center_row = round(size_default(2) / 2);

        oi = oiSet(oi, 'optics focal length', 4 * focal_length_default_m);
        focal_length_long_m = oiGet(oi, 'optics focal length');
        oi = oiCompute(oi, scene);
        size_long = oiGet(oi, 'size');
        pos_long = oiSpatialSupport(oi, 'um');
        illuminance_long = real(oiGet(oi, 'illuminance'));

        payload.focal_length_default_m = focal_length_default_m;
        payload.focal_length_long_m = focal_length_long_m;
        payload.size_default = size_default;
        payload.size_long = size_long;
        payload.center_row = center_row;
        payload.edge_row = 20;
        payload.pos_default_um = pos_default.x;
        payload.center_line_default_lux = real(illuminance_default(center_row, :));
        payload.mean_illuminance_default_lux = real(mean(illuminance_default(:)));
        payload.pos_long_um = pos_long.x;
        payload.center_line_long_lux = real(illuminance_long(center_row, :));
        payload.edge_line_long_lux = real(illuminance_long(20, :));
        payload.mean_illuminance_long_lux = real(mean(illuminance_long(:)));

    case 'oi_pad_crop_small'
        scene = sceneCreate('sweep frequency');
        oi = oiCreate;
        oi = oiCompute(oi, scene);
        paddedSize = oiGet(oi, 'size');
        originalSize = paddedSize / 1.25;
        offset = (paddedSize - originalSize) / 2;
        rect = [offset(2) + 1, offset(1) + 1, originalSize(2) - 1, originalSize(1) - 1];
        oiCropped = oiCrop(oi, rect);

        sensorScene = sensorCreate;
        sensorScene = sensorSet(sensorScene, 'noise flag', 0);
        sensorScene = sensorSet(sensorScene, 'fov', sceneGet(scene, 'fov'), oi);
        sensorFromPadded = sensorCompute(sensorScene, oi);
        sensorFromCropped = sensorCompute(sensorScene, oiCropped);
        paddedVolts = sensorGet(sensorFromPadded, 'volts');
        croppedVolts = sensorGet(sensorFromCropped, 'volts');
        rowIndex = floor(size(paddedVolts, 1) / 2) + 1;
        sensorSupport = sensorGet(sensorScene, 'spatial support', 'um');

        sensorPadded = sensorSetSizeToFOV(sensorScene, oiGet(oi, 'fov'), oi);
        sensorPadded = sensorCompute(sensorPadded, oi);
        paddedSupport = sensorGet(sensorPadded, 'spatial support', 'um');
        paddedSensorVolts = sensorGet(sensorPadded, 'volts');
        paddedRowIndex = floor(size(paddedSensorVolts, 1) / 2) + 1;

        payload.scene_size = sceneGet(scene, 'size');
        payload.oi_padded_size = paddedSize(:);
        payload.crop_rect = round(rect(:));
        payload.oi_cropped_size = oiGet(oiCropped, 'size')(:);
        payload.scene_fov_deg = sceneGet(scene, 'fov');
        payload.oi_padded_fov_deg = oiGet(oi, 'fov');
        payload.oi_cropped_fov_deg = oiGet(oiCropped, 'fov');
        payload.sensor_scene_fov_size = sensorGet(sensorScene, 'size');
        payload.sensor_scene_fov_pos_um = sensorSupport.x(:);
        payload.sensor_scene_fov_padded_row = paddedVolts(rowIndex, :)';
        payload.sensor_scene_fov_cropped_row = croppedVolts(rowIndex, :)';
        payload.sensor_scene_fov_normalized_mae = mean(abs(paddedVolts(:) - croppedVolts(:))) / max(mean(abs(croppedVolts(:))), 1e-12);
        payload.sensor_padded_size = sensorGet(sensorPadded, 'size');
        payload.sensor_padded_pos_um = paddedSupport.x(:);
        payload.sensor_padded_row = paddedSensorVolts(paddedRowIndex, :)';

    case 'optics_microlens_small'
        oi = oiCreate;
        sensor = sensorCreate;
        sensor = sensorSet(sensor, 'fov', 30, oi);

        ml = mlensCreate;
        payload.name = mlensGet(ml, 'name');
        payload.type = mlensGet(ml, 'type');
        payload.source_fnumber = mlensGet(ml, 'source fnumber');
        payload.source_diameter_m = mlensGet(ml, 'source diameter', 'meters');
        payload.source_diameter_um = mlensGet(ml, 'source diameter', 'microns');
        payload.ml_fnumber = mlensGet(ml, 'ml fnumber');
        payload.ml_diameter_m = mlensGet(ml, 'ml diameter', 'meters');
        payload.ml_diameter_um = mlensGet(ml, 'ml diameter', 'microns');
        payload.chief_ray_angle_default_deg = mlensGet(ml, 'chief ray angle');
        ml = mlensSet(ml, 'chief ray angle', 10);
        payload.chief_ray_angle_set_deg = mlensGet(ml, 'chief ray angle');
        payload.sensor_fov_deg = 30;

        radianceML = mlensCreate;
        radianceML = mlRadiance(radianceML);
        sourceIrradiance = double(mlensGet(radianceML, 'source irradiance'));
        pixelIrradiance = double(mlensGet(radianceML, 'pixel irradiance'));
        xCoordinate = double(mlensGet(radianceML, 'x coordinate'));
        sourceCenterRow = floor(size(sourceIrradiance, 1) / 2) + 1;
        pixelCenterRow = floor(size(pixelIrradiance, 1) / 2) + 1;

        payload.x_coordinate_um = xCoordinate(:);
        payload.source_center_row = sourceIrradiance(sourceCenterRow, :)';
        payload.pixel_center_row = pixelIrradiance(pixelCenterRow, :)';
        payload.source_irradiance_stats = local_stats_vector(sourceIrradiance);
        payload.pixel_irradiance_stats = local_stats_vector(pixelIrradiance);
        payload.etendue = mlensGet(radianceML, 'etendue');

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

    case 'oi_psf_plot_diffraction_small'
        oi = oiCreate('diffraction limited');
        optics = oiGet(oi, 'optics');
        optics = opticsSet(optics, 'fnumber', 12);
        oi = oiSet(oi, 'optics', optics);
        psfData = opticsGet(oiGet(oi, 'optics'), 'psf data', 600, 'um', 100);
        payload.x = psfData.xy(:, :, 1);
        payload.y = psfData.xy(:, :, 2);
        payload.psf = psfData.psf;
        payload.airy_disk_radius_um = airyDisk(600, oiGet(oi, 'fnumber'), 'units', 'um');

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

    case 'oi_gaussian_psf_point_array_small'
        wave = 450:100:650;
        nWaves = numel(wave);
        scene = sceneCreate('pointArray', 128, 32);
        scene = sceneInterpolateW(scene, wave);
        scene = sceneSet(scene, 'hfov', 1);
        scene = sceneSet(scene, 'name', 'psfPointArray');
        oi = oiCreate;
        oi = oiSet(oi, 'wave', sceneGet(scene, 'wave'));
        xyRatio = 3 * ones(1, nWaves);
        waveSpread = wave / wave(1);
        optics = siSynthetic('gaussian', oi, double(waveSpread), xyRatio);
        oi = oiSet(oi, 'optics', optics);
        oi = oiCompute(oi, scene);
        photons = oiGet(oi, 'photons');
        peakPerWave = squeeze(max(max(photons, [], 1), [], 2));
        photonsNormalized = photons ./ reshape(max(peakPerWave, 1e-12), 1, 1, []);
        centerRow = floor(size(photonsNormalized, 1) / 2) + 1;
        centerCol = floor(size(photonsNormalized, 2) / 2) + 1;
        payload.wave = oiGet(oi, 'wave');
        payload.scene_wave = sceneGet(scene, 'wave');
        payload.scene_name = sceneGet(scene, 'name');
        payload.scene_size = sceneGet(scene, 'size')(:);
        payload.oi_size = oiGet(oi, 'size')(:);
        payload.center_row_normalized = squeeze(photonsNormalized(centerRow, :, :));
        payload.center_col_normalized = squeeze(photonsNormalized(:, centerCol, :));

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

    case 'optics_defocus_small'
        scene = sceneCreate('disk array', 256, 32, [2, 2]);
        scene = sceneSet(scene, 'fov', 0.5);

        wvf = wvfCreate('wave', sceneGet(scene, 'wave'));
        oi = oiCreate('wvf', wvf);
        oi = oiCompute(oi, scene);

        wave = double(oiGet(oi, 'wave')(:));
        [~, waveIndex550] = min(abs(wave - 550));

        basePhotons = double(oiGet(oi, 'photons'));
        baseMean = real(mean(real(basePhotons(:))));
        payload.wave = wave;
        payload.base_center_row_550_norm = local_channel_normalize(squeeze(basePhotons(floor(size(basePhotons, 1) / 2) + 1, :, waveIndex550)));

        oi = oiSet(oi, 'wvf zcoeffs', 2.5, 'defocus');
        oi = oiCompute(oi, scene);
        defocusPhotons = double(oiGet(oi, 'photons'));
        payload.defocus_center_row_550_norm = local_channel_normalize(squeeze(defocusPhotons(floor(size(defocusPhotons, 1) / 2) + 1, :, waveIndex550)));
        payload.defocus_coeff = oiGet(oi, 'wvf', 'zcoeffs', 'defocus');

        oi = oiSet(oi, 'wvf zcoeffs', 1, 'vertical_astigmatism');
        oi = oiCompute(oi, scene);
        astigPhotons = double(oiGet(oi, 'photons'));
        payload.astig_center_row_550_norm = local_channel_normalize(squeeze(astigPhotons(floor(size(astigPhotons, 1) / 2) + 1, :, waveIndex550)));
        payload.vertical_astigmatism_coeff = oiGet(oi, 'wvf', 'zcoeffs', 'vertical_astigmatism');

        oi = oiSet(oi, 'wvf zcoeffs', 0, 'vertical_astigmatism');
        oi = oiCompute(oi, scene);

        oi = oiSet(oi, 'wvf zcoeffs', 0, 'defocus');
        oi = oiCompute(oi, scene);
        resetPhotons = double(oiGet(oi, 'photons'));
        resetMean = real(mean(real(resetPhotons(:))));
        payload.reset_center_row_550_norm = local_channel_normalize(squeeze(resetPhotons(floor(size(resetPhotons, 1) / 2) + 1, :, waveIndex550)));

        currentWvf = oiGet(oi, 'wvf');
        pupilDiameterMM = wvfGet(currentWvf, 'calc pupil diameter', 'mm');
        currentWvf = wvfSet(currentWvf, 'calc pupil diameter', 2 * pupilDiameterMM, 'mm');
        oi = oiSet(oi, 'optics wvf', currentWvf);
        oi = oiCompute(oi, scene);
        largePupilPhotons = double(oiGet(oi, 'photons'));
        payload.large_pupil_center_row_550_norm = local_channel_normalize(squeeze(largePupilPhotons(floor(size(largePupilPhotons, 1) / 2) + 1, :, waveIndex550)));
        payload.pupil_diameter_mm = pupilDiameterMM;
        payload.doubled_pupil_diameter_mm = 2 * pupilDiameterMM;

        currentWvf = oiGet(oi, 'wvf');
        currentWvf = wvfSet(currentWvf, 'calc pupil diameter', pupilDiameterMM, 'mm');
        oi = oiSet(oi, 'optics wvf', currentWvf);
        oi = oiCompute(oi, scene);
        finalPhotons = double(oiGet(oi, 'photons'));
        finalMean = real(mean(real(finalPhotons(:))));
        payload.final_center_row_550_norm = local_channel_normalize(squeeze(finalPhotons(floor(size(finalPhotons, 1) / 2) + 1, :, waveIndex550)));
        payload.final_defocus_coeff = oiGet(oi, 'wvf', 'zcoeffs', 'defocus');
        payload.final_vertical_astigmatism_coeff = oiGet(oi, 'wvf', 'zcoeffs', 'vertical_astigmatism');
        payload.initial_reset_ratio = baseMean / max(resetMean, 1e-12);
        payload.initial_final_ratio = baseMean / max(finalMean, 1e-12);

    case 'optics_defocus_displacement_small'
        baseD = 50:100:350;
        deltaD = 1:15;
        displacementCurves = zeros(numel(baseD), numel(deltaD));
        for ii = 1:numel(baseD)
            displacementCurves(ii, :) = (1 / baseD(ii)) - (1 ./ (baseD(ii) + deltaD));
        end

        ratioBaseD = 50:50:300;
        ratioDeltaD = ratioBaseD / 10;
        ratioDisplacement = (1 ./ ratioBaseD) - (1 ./ (ratioBaseD + ratioDeltaD));
        ratioScaled = ratioDisplacement .* ratioBaseD;

        payload.base_diopters = double(baseD(:));
        payload.delta_diopters = double(deltaD(:));
        payload.displacement_curves_m = double(displacementCurves);
        payload.ratio_base_diopters = double(ratioBaseD(:));
        payload.ratio_delta_diopters = double(ratioDeltaD(:));
        payload.ratio_displacement_m = double(ratioDisplacement(:));
        payload.displacement_to_focal_length_ratio = double(ratioScaled(:));

    case 'optics_dof_small'
        fN = 2;
        fL = 0.100;
        oDist = 2;
        cocDiam = 50e-6;

        optics = opticsCreate;
        optics = opticsSet(optics, 'fnumber', fN);
        optics = opticsSet(optics, 'focal length', fL);

        dofFormula = opticsDoF(optics, oDist, cocDiam);
        [coc, xDist] = opticsCoC(optics, oDist, 'nsamples', 200);
        [~, idx1] = min(abs(coc(1:100) - cocDiam));
        [~, idx2] = min(abs(coc(101:end) - cocDiam));
        idx2 = idx2 + 100;
        cocDOF = xDist(idx2) - xDist(idx1);

        oDistSweep = 0.5:0.25:20;
        fnumberSweep = 2:0.25:12;
        sweepCoC = 20e-6;
        dofSweep = zeros(numel(oDistSweep), numel(fnumberSweep));
        for ii = 1:numel(oDistSweep)
            for jj = 1:numel(fnumberSweep)
                optics = opticsSet(optics, 'fnumber', fnumberSweep(jj));
                dofSweep(ii, jj) = opticsDoF(optics, oDistSweep(ii), sweepCoC);
            end
        end

        payload.f_number = double(fN);
        payload.focal_length_m = double(fL);
        payload.object_distance_m = double(oDist);
        payload.coc_diameter_m = double(cocDiam);
        payload.dof_formula_m = double(dofFormula);
        payload.coc_xdist_m = double(xDist(:));
        payload.coc_curve_m = double(coc(:));
        payload.coc_idx1 = double(idx1 - 1);
        payload.coc_idx2 = double(idx2 - 1);
        payload.coc_dof_m = double(cocDOF);
        payload.object_distances_m = double(oDistSweep(:));
        payload.f_numbers = double(fnumberSweep(:));
        payload.sweep_coc_diameter_m = double(sweepCoC);
        payload.dof_surface_m = double(dofSweep);

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

    case 'sensor_split_pixel_ovt_saturated_small'
        scene = sceneCreate('uniform ee', [32 48]);
        scene = sceneSet(scene, 'fov', 8);
        photons = sceneGet(scene, 'photons');
        levels = [1 10 100 1000];
        bandWidth = floor(size(photons, 2) / numel(levels));
        for ii = 1:numel(levels)
            startCol = (ii - 1) * bandWidth + 1;
            if ii == numel(levels)
                stopCol = size(photons, 2);
            else
                stopCol = ii * bandWidth;
            end
            photons(:, startCol:stopCol, :) = photons(:, startCol:stopCol, :) * levels(ii);
        end
        scene = sceneSet(scene, 'photons', photons);
        oi = oiCreate('wvf');
        oi = oiCompute(oi, scene, 'crop', true);

        sensorArray = local_sensor_create_array_ovt(0.1, [32 48], 0, upstream_root);
        [combined, captures] = sensorComputeArray(sensorArray, oi, 'method', 'saturated');

        payload.combined_volts = sensorGet(combined, 'volts');
        payload.sensor_max_volts = zeros(numel(captures), 1);
        payload.sensor_names = cell(numel(captures), 1);
        for ii = 1:numel(captures)
            payload.sensor_max_volts(ii) = max(sensorGet(captures(ii), 'volts')(:));
            payload.sensor_names{ii} = sensorGet(captures(ii), 'name');
        end
        payload.saturated_counts = squeeze(sum(sum(combined.metadata.saturated, 1), 2));

    case 'sensor_stacked_pixels_foveon_small'
        horizontalFOV = 8;
        meanLuminance = 100;
        patchSize = 32;

        scene = sceneCreate('macbeth d65', patchSize);
        scene = sceneAdjustLuminance(scene, meanLuminance);
        scene = sceneSet(scene, 'hfov', horizontalFOV);

        oi = oiCreate;
        oi = oiSet(oi, 'optics fnumber', 4);
        oi = oiSet(oi, 'optics focal length', 3e-3);
        oi = oiCompute(oi, scene);

        wave = sceneGet(scene, 'wave');
        fSpectra = ieReadSpectra('Foveon', wave);

        clear sensorMonochrome
        for ii = 1:3
            sensorMonochrome(ii) = sensorCreate('monochrome');
            sensorMonochrome(ii) = sensorSet(sensorMonochrome(ii), 'pixel size constant fill factor', [1.4 1.4] * 1e-6);
            sensorMonochrome(ii) = sensorSet(sensorMonochrome(ii), 'exp time', 0.1);
            sensorMonochrome(ii) = sensorSet(sensorMonochrome(ii), 'filterspectra', fSpectra(:, ii));
            sensorMonochrome(ii) = sensorSet(sensorMonochrome(ii), 'name', sprintf('Channel-%.0f', ii));
            sensorMonochrome(ii) = sensorSetSizeToFOV(sensorMonochrome(ii), sceneGet(scene, 'fov'), oi);
            sensorMonochrome(ii) = sensorSet(sensorMonochrome(ii), 'wave', wave);
        end
        sensorMonochrome = sensorCompute(sensorMonochrome, oi);

        sz = sensorGet(sensorMonochrome(1), 'size');
        stackedVolts = zeros(sz(1), sz(2), 3);
        for ii = 1:3
            stackedVolts(:, :, ii) = sensorGet(sensorMonochrome(ii), 'volts');
        end

        sensorFoveon = sensorCreate;
        sensorFoveon = sensorSet(sensorFoveon, 'name', 'foveon');
        sensorFoveon = sensorSet(sensorFoveon, 'pixel size constant fill factor', [1.4 1.4] * 1e-6);
        sensorFoveon = sensorSet(sensorFoveon, 'autoexp', 1);
        sensorFoveon = sensorSetSizeToFOV(sensorFoveon, sceneGet(scene, 'fov'), oi);
        sensorFoveon = sensorSet(sensorFoveon, 'wave', wave);
        sensorFoveon = sensorSet(sensorFoveon, 'filter spectra', fSpectra);
        sensorFoveon = sensorSet(sensorFoveon, 'pattern', [2]);
        sensorFoveon = sensorSet(sensorFoveon, 'volts', stackedVolts);

        ipFoveon = ipCreate;
        ipFoveon = ipCompute(ipFoveon, sensorFoveon);
        foveonResult = ipGet(ipFoveon, 'result');
        lineRow = min(120, size(foveonResult, 1));
        foveonLine = squeeze(foveonResult(lineRow, :, :));

        bayerSpectra = ieReadSpectra('NikonD1', wave);
        sensorBayer = sensorCreate;
        sensorBayer = sensorSet(sensorBayer, 'filterspectra', bayerSpectra);
        sensorBayer = sensorSet(sensorBayer, 'pixel size constant fill factor', [1.4 1.4] * 1e-6);
        sensorBayer = sensorSet(sensorBayer, 'autoexp', 1);
        sensorBayer = sensorSetSizeToFOV(sensorBayer, sceneGet(scene, 'fov'), oi);
        sensorBayer = sensorCompute(sensorBayer, oi);

        ipBayer = ipCreate;
        ipBayer = ipCompute(ipBayer, sensorBayer);
        bayerResult = ipGet(ipBayer, 'result');
        bayerLine = squeeze(bayerResult(lineRow, :, :));

        rowStart = floor((size(stackedVolts, 1) - 24) / 2) + 1;
        colStart = floor((size(stackedVolts, 2) - 24) / 2) + 1;
        rowStop = rowStart + 23;
        colStop = colStart + 23;

        stackedPatch = stackedVolts(rowStart:rowStop, colStart:colStop, :);
        payload.stacked_center_patch_mean = squeeze(mean(mean(stackedPatch, 1), 2));
        payload.stacked_center_patch_std = squeeze(std(reshape(stackedPatch, [], size(stackedPatch, 3)), 0, 1))';
        payload.stacked_center_patch_p90 = squeeze(prctile(reshape(stackedPatch, [], size(stackedPatch, 3)), 90, 1))';
        payload.stacked_mean_volts = squeeze(mean(mean(stackedVolts, 1), 2));
        payload.stacked_std_volts = squeeze(std(reshape(stackedVolts, [], size(stackedVolts, 3)), 0, 1))';
        payload.line_row = lineRow;
        payload.bayer_line_mean = squeeze(mean(bayerLine, 1));
        payload.bayer_line_std = squeeze(std(bayerLine, 0, 1));
        payload.bayer_line_p90 = squeeze(prctile(bayerLine, 90, 1));

    case 'sensor_microlens_etendue_small'
        oi = oiCreate;
        sensor = sensorCreate;
        ml = mlensCreate(sensor, oi);
        sensor = sensorSet(sensor, 'microlens', ml);
        sensor = sensorSetSizeToFOV(sensor, 4, oi);
        ieAddObject(oi);
        ieAddObject(sensor);

        sensorNoML = mlAnalyzeArrayEtendue(sensor, 'no microlens');
        sensorCentered = mlAnalyzeArrayEtendue(sensor, 'centered');
        sensorOptimal = mlAnalyzeArrayEtendue(sensor, 'optimal');

        cra = sensorGet(sensorOptimal, 'cra degrees');
        rayAngles = linspace(0, max(cra(:)), 10);
        optimalML = sensorGet(sensorOptimal, 'microlens');
        offsetCurve = zeros(size(rayAngles));
        for ii = 1:numel(rayAngles)
            workingML = mlensSet(optimalML, 'chief ray angle', rayAngles(ii));
            offsetCurve(ii) = local_ml_optimal_offset(workingML, sensorOptimal, 'microns');
        end

        halfFNumberML = mlensSet(optimalML, 'ml fnumber', 0.5 * mlensGet(optimalML, 'ml fnumber'));
        sourceF4ML = mlensSet(optimalML, 'source fnumber', 4);
        sourceF16ML = mlensSet(optimalML, 'source fnumber', 16);

        payload.no_microlens_etendue = sensorGet(sensorNoML, 'etendue');
        payload.centered_etendue = sensorGet(sensorCentered, 'etendue');
        payload.optimal_etendue = sensorGet(sensorOptimal, 'etendue');
        payload.ray_angles_deg = rayAngles(:)';
        payload.optimal_offset_curve_um = offsetCurve(:)';
        payload.optimal_offsets_default_um = local_ml_optimal_offsets(optimalML, sensorOptimal);
        payload.optimal_offsets_half_fnumber_um = local_ml_optimal_offsets(halfFNumberML, sensorOptimal);
        payload.optimal_offsets_source_f4_um = local_ml_optimal_offsets(sourceF4ML, sensorOptimal);
        payload.optimal_offsets_source_f16_um = local_ml_optimal_offsets(sourceF16ML, sensorOptimal);

        displaySensor = sensorCreate;
        displayML = mlensCreate(displaySensor, oi);
        displayML = mlensSet(displayML, 'ml fnumber', 8);
        chiefRays = [-10 0 10];
        for ii = 1:numel(chiefRays)
            displayML = mlensSet(displayML, 'chief ray angle', chiefRays(ii));
            displayML = mlRadiance(displayML, displaySensor);
            irradiance = mlensGet(displayML, 'pixel irradiance');
            centerRow = floor(size(irradiance, 1) / 2) + 1;
            switch ii
                case 1
                    payload.radiance_midline_neg10 = irradiance(centerRow, :);
                case 2
                    payload.radiance_midline_0 = irradiance(centerRow, :);
                otherwise
                    payload.radiance_midline_10 = irradiance(centerRow, :);
            end
        end

    case 'sensor_comparison_small'
        patchSize = 24;
        sceneC = sceneCreate('macbethD65', patchSize);
        sz = sceneGet(sceneC, 'size');
        sceneC = sceneSet(sceneC, 'resize', round([sz(1), sz(2) / 2]));
        sceneS = sceneCreate('sweep frequency', sz(1), sz(1) / 16);
        scene = sceneCombine(sceneC, sceneS, 'direction', 'horizontal');
        scene = sceneSet(scene, 'fov', 20);
        sceneVFOV = sceneGet(scene, 'v fov');

        oi = oiCreate;
        oi = oiSet(oi, 'optics fnumber', 1.2);
        oi = oiCompute(oi, scene);

        sensorList = {'imx363', 'mt9v024', 'cyym'};
        ipList = {'imx363', 'mt9v024'};
        payload.scene_size = sceneGet(scene, 'size');
        payload.scene_vfov = sceneVFOV;
        payload.oi_size = oiGet(oi, 'size');
        payload.small_sensor_sizes = zeros(numel(sensorList), 2);
        smallSensorMeanVolts = zeros(numel(sensorList), 1);
        smallSensorP90Volts = zeros(numel(sensorList), 1);
        payload.large_sensor_sizes = zeros(numel(sensorList), 2);
        largeSensorMeanVolts = zeros(numel(sensorList), 1);
        largeSensorP90Volts = zeros(numel(sensorList), 1);
        payload.small_ip_sizes = zeros(numel(ipList), 3);
        payload.large_ip_sizes = zeros(numel(ipList), 3);

        for ii = 1:numel(sensorList)
            sensorType = sensorList{ii};
            if isequal(sensorType, 'imx363')
                load(fullfile(isetRootPath, 'data', 'sensor', 'sony', 'imx363.mat'), 'sensor');
            elseif isequal(sensorType, 'mt9v024')
                sensor = sensorCreate(sensorType, [], 'rccc');
            else
                sensor = sensorCreate(sensorType);
            end
            sensor = sensorSet(sensor, 'pixel size', 1.5e-6);
            sensor = sensorSet(sensor, 'hfov', 20, oi);
            sensor = sensorSet(sensor, 'vfov', sceneVFOV);
            sensor = sensorSet(sensor, 'auto exposure', true);
            sensor = sensorCompute(sensor, oi);
            volts = sensorGet(sensor, 'volts');
            payload.small_sensor_sizes(ii, :) = sensorGet(sensor, 'size');
            smallSensorMeanVolts(ii) = mean(volts(:));
            smallSensorP90Volts(ii) = prctile(volts(:), 90);

            ipIndex = find(strcmp(ipList, sensorType), 1);
            if ~isempty(ipIndex)
                if isequal(sensorType, 'imx363')
                    ip = ipCreate('imx363 RGB', sensor);
                else
                    ip = ipCreate('mt9v024 RCCC', sensor);
                    ip = ipSet(ip, 'demosaic method', 'analog rccc');
                end
                ip = ipCompute(ip, sensor);
                result = ipGet(ip, 'result');
                payload.small_ip_sizes(ipIndex, :) = size(result);
            end

            sensor = sensorSet(sensor, 'pixel size constant fill factor', 6e-6);
            sensor = sensorSet(sensor, 'hfov', 20, oi);
            sensor = sensorSet(sensor, 'vfov', sceneVFOV);
            sensor = sensorSet(sensor, 'auto exposure', true);
            sensor = sensorCompute(sensor, oi);
            volts = sensorGet(sensor, 'volts');
            payload.large_sensor_sizes(ii, :) = sensorGet(sensor, 'size');
            largeSensorMeanVolts(ii) = mean(volts(:));
            largeSensorP90Volts(ii) = prctile(volts(:), 90);

            if ~isempty(ipIndex)
                if isequal(sensorType, 'imx363')
                    ip = ipCreate('imx363 RGB', sensor);
                else
                    ip = ipCreate('mt9v024 RCCC', sensor);
                    ip = ipSet(ip, 'demosaic method', 'analog rccc');
                end
                ip = ipCompute(ip, sensor);
                result = ipGet(ip, 'result');
                payload.large_ip_sizes(ipIndex, :) = size(result);
            end
        end
        payload.nonimx_small_sensor_mean_volts = smallSensorMeanVolts(2:end);
        payload.nonimx_small_sensor_p90_volts = smallSensorP90Volts(2:end);
        payload.nonimx_large_sensor_mean_volts = largeSensorMeanVolts(2:end);
        payload.nonimx_large_sensor_p90_volts = largeSensorP90Volts(2:end);
        payload.imx363_mean_ratio_large_small = largeSensorMeanVolts(1) / max(smallSensorMeanVolts(1), eps);
        payload.imx363_p90_ratio_large_small = largeSensorP90Volts(1) / max(smallSensorP90Volts(1), eps);

    case 'sensor_noise_samples_small'
        rand('seed', 7);
        randn('seed', 7);
        scene = sceneCreate('slanted bar', 128);
        scene = sceneSet(scene, 'fov', 4);
        oi = oiCreate;
        oi = oiCompute(oi, scene);

        sensor = sensorCreate;
        sensor = sensorSet(sensor, 'exp time', 0.050);
        sensor = sensorSet(sensor, 'noise flag', 0);
        sensorNF = sensorCompute(sensor, oi);
        voltsNF = sensorGet(sensorNF, 'volts');

        nSamp = 64;
        voltImages = sensorComputeSamples(sensorNF, nSamp, 2, 0);
        noiseImages = voltImages - repmat(voltsNF, [1 1 nSamp]);
        stdImage = std(voltImages, 0, 3);
        meanImage = mean(voltImages, 3);
        pairDiff = voltImages(:, :, 1) - voltImages(:, :, 2);
        meanResidual = meanImage - voltsNF;

        payload.sample_shape = size(voltImages);
        payload.noise_free_mean = mean(voltsNF(:));
        payload.noise_std_image_stats = [mean(stdImage(:)); prctile(stdImage(:), [10 50 90])'];
        payload.noise_distribution_stats = [std(noiseImages(:)); prctile(noiseImages(:), [5 50 95])'];
        payload.mean_residual_stats = [mean(abs(meanResidual(:))); prctile(abs(meanResidual(:)), 95)];
        payload.pair_diff_stats = [std(pairDiff(:)); prctile(pairDiff(:), [5 50 95])'];

    case 'sensor_mcc_small'
        fileName = fullfile(upstream_root, 'data', 'sensor', 'mccGBRGsensor.tif');
        mosaic = double(imread(fileName));

        sensor = sensorCreate('bayer (gbrg)');
        sensor = sensorSet(sensor, 'name', 'Sensor demo');

        mn = min(mosaic(:));
        mx = max(mosaic(:));
        vSwing = sensorGet(sensor, 'pixel voltage swing');
        volts = ((mosaic - mn) / max(mx - mn, eps)) * vSwing;

        sensor = sensorSet(sensor, 'size', size(volts));
        sensor = sensorSet(sensor, 'volts', volts);
        cp = [15 584; 782 584; 784 26; 23 19];
        sensor = sensorSet(sensor, 'chart corner points', cp);

        [~, mLocs, pSize] = chartRectangles(cp, 4, 6, 0.5);
        delta = round(pSize(1) * 0.5);
        rgb = chartRectsData(sensor, mLocs, delta, false, 'volts');
        idealRGB = macbethIdealColor('d65', 'lrgb');
        estimatedCCM = pinv(rgb) * idealRGB;

        ip = ipCreate;
        ip = ipSet(ip, 'name', 'No Correction');
        ip = ipSet(ip, 'scaledisplay', 1);
        ip = ipCompute(ip, sensor);
        uncorrected = ipGet(ip, 'result');

        fixedM = [
            0.9205   -0.1402   -0.1289
            -0.0148    0.8763   -0.0132
            -0.2516   -0.1567    0.6987
        ];
        ip = ipCreate;
        ip = ipSet(ip, 'name', 'CCM Correction');
        ip = ipSet(ip, 'scaledisplay', 1);
        ip = ipSet(ip, 'conversion transform sensor', fixedM);
        ip = ipSet(ip, 'correction transform illuminant', eye(3, 3));
        ip = ipSet(ip, 'ics2Display Transform', eye(3, 3));
        ip = ipSet(ip, 'conversion method sensor ', 'current matrix');
        ip = ipCompute(ip, sensor);
        corrected = ipGet(ip, 'result');

        payload.mosaic_size = size(volts);
        payload.volts_stats = [mean(volts(:)) std(volts(:)) prctile(volts(:), [5 95])];
        payload.estimated_ccm = estimatedCCM;
        payload.uncorrected_mean_rgb_norm = local_channel_normalize(squeeze(mean(mean(uncorrected, 1), 2)));
        payload.uncorrected_p95_rgb_norm = local_channel_normalize(prctile(reshape(uncorrected, [], size(uncorrected, 3)), 95, 1));
        payload.corrected_mean_rgb_norm = local_channel_normalize(squeeze(mean(mean(corrected, 1), 2)));
        payload.corrected_p95_rgb_norm = local_channel_normalize(prctile(reshape(corrected, [], size(corrected, 3)), 95, 1));

    case 'sensor_rolling_shutter_small'
        scene = sceneCreate('star pattern', 48, 'ee', 4);
        scene = sceneSet(scene, 'fov', 3);
        oi = oiCreate;

        sensor = sensorCreate();
        sensor = sensorSet(sensor, 'pixel size constant fill factor', [1.4 1.4] * 1e-6);
        sensor = sensorSet(sensor, 'fov', sceneGet(scene, 'fov') / 2, oi);
        sensor = sensorSet(sensor, 'exp time', 4e-5);
        sensor = sensorSet(sensor, 'noise flag', 0);

        sz = sensorGet(sensor, 'size');
        expTime = sensorGet(sensor, 'exp time');
        perRow = 10e-6;
        rate = 0.3;
        nFrames = sz(1) + round(expTime / perRow);
        cropWidth = sz(2) - 1;
        cropHeight = sz(1) - 1;

        v = zeros(sz(1), sz(2), nFrames);
        cropRects = zeros(nFrames, 4);
        temporalMeanVolts = zeros(nFrames, 1);
        currentSensor = sensor;
        for ii = 1:nFrames
            s = sceneRotate(scene, ii * rate);
            oiFrame = oiCompute(oi, s);
            cp = oiGet(oiFrame, 'center pixel');
            rect = round([cp(2) - cropWidth / 2, cp(1) - cropHeight / 2, cropWidth, cropHeight]);
            cropRects(ii, :) = rect;
            oiC = oiCrop(oiFrame, rect);
            currentSensor = sensorCompute(currentSensor, oiC, 0);
            volts = sensorGet(currentSensor, 'volts');
            v(:, :, ii) = volts;
            temporalMeanVolts(ii) = mean(volts(:));
        end

        slist = 1:round(expTime / perRow);
        final = zeros(sz);
        for rr = 1:sz(1)
            slist = slist + 1;
            z = zeros(nFrames, 1);
            z(slist) = 1;
            tmp = squeeze(v(rr, :, :));
            final(rr, :) = tmp * z;
        end

        sensorFinal = sensorSet(currentSensor, 'volts', final);
        ip = ipCreate;
        ip = ipCompute(ip, sensorFinal);
        result = ipGet(ip, 'result');

        sampleRows = [1, floor(sz(1) / 2) + 1, sz(1)];
        sampleCols = [1, floor(sz(2) / 2) + 1, sz(2)];

        payload.sensor_size = sz(:)';
        payload.n_frames = nFrames;
        payload.crop_size = [cropHeight + 1, cropWidth + 1];
        payload.first_crop_rect = cropRects(1, :);
        payload.last_crop_rect = cropRects(end, :);
        payload.temporal_mean_volts = temporalMeanVolts;
        payload.center_pixel_trace = squeeze(v(floor(sz(1) / 2) + 1, floor(sz(2) / 2) + 1, :));
        payload.final_stats = [mean(final(:)) std(final(:)) prctile(final(:), [5 95])];
        payload.sampled_rows = sampleRows;
        payload.sampled_cols = sampleCols;
        payload.sampled_row_stats = [
            mean(final(sampleRows(1), :)) std(final(sampleRows(1), :)) prctile(final(sampleRows(1), :), [5 95]);
            mean(final(sampleRows(2), :)) std(final(sampleRows(2), :)) prctile(final(sampleRows(2), :), [5 95]);
            mean(final(sampleRows(3), :)) std(final(sampleRows(3), :)) prctile(final(sampleRows(3), :), [5 95])
        ];
        payload.result_mean_rgb_norm = local_channel_normalize(squeeze(mean(mean(result, 1), 2)));
        payload.result_p95_rgb_norm = local_channel_normalize(prctile(reshape(result, [], size(result, 3)), 95, 1));

    case 'sensor_imx490_uniform_small'
        scene = sceneCreate('uniform', 256);
        oi = oiCreate;
        oi = oiCompute(oi, scene);
        oi = oiCrop(oi, 'border');
        oi = oiSpatialResample(oi, 3e-6);

        [sensor, metadata] = local_imx490_compute(oi, 'best snr', 0.1, 0, upstream_root);
        sArray = metadata.sensorArray;
        nCaptures = numel(sArray);

        captureNames = cell(nCaptures, 1);
        captureMeanElectrons = zeros(nCaptures, 1);
        captureMeanVolts = zeros(nCaptures, 1);
        captureMeanDV = zeros(nCaptures, 1);
        for ii = 1:nCaptures
            capture = sArray{ii};
            captureNames{ii} = sensorGet(capture, 'name');
            captureMeanElectrons(ii) = mean(sensorGet(capture, 'electrons')(:));
            captureMeanVolts(ii) = mean(sensorGet(capture, 'volts')(:));
            captureMeanDV(ii) = mean(sensorGet(capture, 'dv')(:));
        end

        bestPixel = sensor.metadata.bestPixel;
        bestPixelCounts = zeros(nCaptures, 1);
        for ii = 1:nCaptures
            bestPixelCounts(ii) = sum(bestPixel(:) == ii);
        end

        combinedVolts = sensorGet(sensor, 'volts');
        payload.oi_size = oiGet(oi, 'size');
        payload.capture_names = captureNames;
        payload.capture_mean_electrons = captureMeanElectrons;
        payload.capture_mean_volts = captureMeanVolts;
        payload.capture_mean_dv = captureMeanDV;
        payload.large_gain_ratio = captureMeanVolts(2) / max(captureMeanVolts(1), eps);
        payload.small_area_ratio = captureMeanElectrons(3) / max(captureMeanElectrons(1), eps);
        payload.combined_volts_stats = [
            mean(combinedVolts(:))
            std(combinedVolts(:))
            prctile(combinedVolts(:), 5)
            prctile(combinedVolts(:), 95)
        ];
        payload.best_pixel_counts = bestPixelCounts;

    case 'sensor_hdr_pixel_size_small'
        fileName = fullfile(upstream_root, 'data', 'images', 'multispectral', 'Feng_Office-hdrs.mat');
        scene = sceneFromFile(fileName, 'multispectral', 200);
        oi = oiCreate;
        oi = oiCompute(oi, scene);

        pSize = [1 2 4];
        dyeSizeMicrons = 512;
        baseSensor = sensorCreate('monochrome');
        baseSensor = sensorSet(baseSensor, 'exp time', 0.003);

        sensorSizes = zeros(numel(pSize), 2);
        meanVolts = zeros(numel(pSize), 1);
        p95Volts = zeros(numel(pSize), 1);
        meanElectrons = zeros(numel(pSize), 1);
        resultSizes = zeros(numel(pSize), 3);
        resultMeanGray = zeros(numel(pSize), 1);
        resultP95Gray = zeros(numel(pSize), 1);

        for ii = 1:numel(pSize)
            sensor = sensorSet(baseSensor, 'pixel size constant fill factor', [pSize(ii) pSize(ii)] * 1e-6);
            sensor = sensorSet(sensor, 'rows', round(dyeSizeMicrons / pSize(ii)));
            sensor = sensorSet(sensor, 'cols', round(dyeSizeMicrons / pSize(ii)));
            sensor = sensorCompute(sensor, oi);

            volts = sensorGet(sensor, 'volts');
            electrons = sensorGet(sensor, 'electrons');
            ip = ipCreate;
            ip = ipCompute(ip, sensor);
            result = ipGet(ip, 'result');
            gray = result(:, :, 1);

            sensorSizes(ii, :) = sensorGet(sensor, 'size');
            meanVolts(ii) = mean(volts(:));
            p95Volts(ii) = prctile(volts(:), 95);
            meanElectrons(ii) = mean(electrons(:));
            resultSizes(ii, :) = [size(result, 1), size(result, 2), size(result, 3)];
            resultMeanGray(ii) = mean(gray(:));
            resultP95Gray(ii) = prctile(gray(:), 95);
        end

        payload.scene_size = sceneGet(scene, 'size');
        payload.oi_size = oiGet(oi, 'size');
        payload.wave = sceneGet(scene, 'wave');
        payload.pixel_sizes_um = pSize(:);
        payload.sensor_sizes = sensorSizes;
        payload.mean_volts = meanVolts;
        payload.p95_volts = p95Volts;
        payload.mean_electrons = meanElectrons;
        payload.result_sizes = resultSizes;
        payload.result_mean_gray = resultMeanGray;
        payload.result_p95_gray = resultP95Gray;

    case 'sensor_log_ar0132at_small'
        dynamicRange = 2^16;
        scene = sceneCreate('exponential intensity ramp', 256, dynamicRange);
        scene = sceneSet(scene, 'fov', 60);

        oi = oiCreate;
        oi = oiSet(oi, 'optics fnumber', 2.8);
        oi = oiCompute(oi, scene);

        sensor = sensorCreate;
        sensor = sensorSet(sensor, 'response type', 'log');
        sensor = sensorSet(sensor, 'size', [960 1280]);
        sensor = sensorSet(sensor, 'pixel size same fill factor', 3.751e-6);

        colorFilterFile = fullfile(upstream_root, 'data', 'sensor', 'colorfilters', 'auto', 'ar0132at.mat');
        wave = sceneGet(scene, 'wave');
        [filterSpectra, filterNames] = ieReadColorFilter(wave, colorFilterFile);
        sensor = sensorSet(sensor, 'filter spectra', filterSpectra);
        sensor = sensorSet(sensor, 'filter names', filterNames);
        sensor = sensorSet(sensor, 'pixel read noise volts', 1e-3);
        sensor = sensorSet(sensor, 'pixel voltage swing', 2.8);
        sensor = sensorSet(sensor, 'pixel dark voltage', 1e-3);
        sensor = sensorSet(sensor, 'pixel conversion gain', 110e-6);
        sensor = sensorSet(sensor, 'exp time', 0.003);
        sensor = sensorSet(sensor, 'noise flag', 0);
        sensor = sensorCompute(sensor, oi);

        % Upstream sensorCompute currently leaves the deprecated log branch dormant,
        % so export the intended script contract explicitly here.
        volts = sensorGet(sensor, 'volts');
        readNoise = sensorGet(sensor, 'pixel read noise volts');
        if readNoise == 0
            readNoise = sensorGet(sensor, 'pixel voltage swing') / (2^16);
        end
        volts = log10(max(volts, 0) + readNoise) - log10(readNoise);

        sampledCols = round(linspace(1, size(volts, 2), 33));
        row15 = volts(15, :);
        row114 = volts(114, :);

        payload.scene_size = sceneGet(scene, 'size');
        payload.oi_size = oiGet(oi, 'size');
        payload.sensor_size = sensorGet(sensor, 'size');
        payload.wave = wave(:);
        payload.dr_at_1s = sensorDR(sensor, 1);
        payload.volts_stats = [mean(volts(:)) std(volts(:)) prctile(volts(:), [5 95])];
        payload.sampled_cols = sampledCols(:);
        payload.row15_stats = [mean(row15(:)) std(row15(:)) prctile(row15(:), [5 95])];
        payload.row114_stats = [mean(row114(:)) std(row114(:)) prctile(row114(:), [5 95])];
        payload.row15_profile_norm = local_channel_normalize(row15(sampledCols));
        payload.row114_profile_norm = local_channel_normalize(row114(sampledCols));

    case 'sensor_aliasing_small'
        fov = 5;
        sweepScene = sceneCreate('sweep frequency', 768, 30);
        sweepScene = sceneSet(sweepScene, 'fov', fov);

        oi = oiCreate('diffraction limited');
        oi = oiSet(oi, 'optics fnumber', 2);
        oi = oiCompute(oi, sweepScene);

        sensor = sensorCreate('monochrome');
        sensor = sensorSetSizeToFOV(sensor, fov, oi);
        sensor = sensorSet(sensor, 'noise flag', 0);

        sensorSmall = sensorSet(sensor, 'pixel size constant fill factor', 2e-6);
        sensorSmall = sensorCompute(sensorSmall, oi);
        smallLine = sensorGet(sensorSmall, 'hline electrons', 1);
        smallData = smallLine.data{1};
        smallPos = smallLine.pos{1};

        sensorLarge = sensorSet(sensor, 'pixel size constant fill factor', 6e-6);
        sensorLarge = sensorSetSizeToFOV(sensorLarge, fov, oi);
        sensorLarge = sensorCompute(sensorLarge, oi);
        largeLine = sensorGet(sensorLarge, 'hline electrons', 1);
        largeData = largeLine.data{1};
        largePos = largeLine.pos{1};

        oiBlur = oiSet(oi, 'optics fnumber', 12);
        oiBlur = oiCompute(oiBlur, sweepScene);
        sensorBlur = sensorCompute(sensorLarge, oiBlur);
        blurLine = sensorGet(sensorBlur, 'hline electrons', 1);
        blurData = blurLine.data{1};
        blurPos = blurLine.pos{1};

        slantedScene = sceneCreate('slanted bar', 1024);
        slantedScene = sceneSet(slantedScene, 'fov', fov);

        oiSlantedSharp = oiSet(oiBlur, 'optics fnumber', 2);
        oiSlantedSharp = oiCompute(oiSlantedSharp, slantedScene);
        sensorSlanted = sensorSet(sensorLarge, 'pixel size constant fill factor', 6e-6);
        sensorSlanted = sensorSetSizeToFOV(sensorSlanted, fov, oiSlantedSharp);
        sensorSlanted = sensorCompute(sensorSlanted, oiSlantedSharp);
        slantedSharp = sensorGet(sensorSlanted, 'electrons');

        oiSlantedBlur = oiSet(oiSlantedSharp, 'optics fnumber', 12);
        oiSlantedBlur = oiCompute(oiSlantedBlur, slantedScene);
        sensorSlantedBlur = sensorCompute(sensorSlanted, oiSlantedBlur);
        slantedBlur = sensorGet(sensorSlantedBlur, 'electrons');

        payload.fov_deg = fov;
        payload.sweep_scene_size = sceneGet(sweepScene, 'size');
        payload.sweep_oi_size = oiGet(oi, 'size');
        payload.small_sensor_size = sensorGet(sensorSmall, 'size');
        payload.large_sensor_size = sensorGet(sensorLarge, 'size');
        payload.small_line_pos = smallPos(:);
        payload.small_line_data_norm = local_channel_normalize(smallData(:));
        payload.large_line_pos = largePos(:);
        payload.large_line_data_norm = local_channel_normalize(largeData(:));
        payload.blur_line_pos = blurPos(:);
        payload.blur_line_data_norm = local_channel_normalize(blurData(:));
        payload.small_line_stats = [mean(smallData(:)) std(smallData(:)) prctile(smallData(:), [5 95])];
        payload.large_line_stats = [mean(largeData(:)) std(largeData(:)) prctile(largeData(:), [5 95])];
        payload.blur_line_stats = [mean(blurData(:)) std(blurData(:)) prctile(blurData(:), [5 95])];
        payload.slanted_scene_size = sceneGet(slantedScene, 'size');
        payload.slanted_sensor_size = sensorGet(sensorSlanted, 'size');
        payload.slanted_sharp_norm = slantedSharp / max(slantedSharp(:));
        payload.slanted_blur_norm = slantedBlur / max(slantedBlur(:));
        payload.slanted_sharp_stats = [mean(slantedSharp(:)) std(slantedSharp(:)) prctile(slantedSharp(:), [5 95])];
        payload.slanted_blur_stats = [mean(slantedBlur(:)) std(slantedBlur(:)) prctile(slantedBlur(:), [5 95])];

    case 'sensor_external_analysis_small'
        dut = sensorCreate;
        dut = sensorSet(dut, 'name', 'My Sensor');

        wave = 400:10:700;
        dut = sensorSet(dut, 'wave', wave);
        dut = sensorSet(dut, 'colorFilters', ieReadSpectra(fullfile(upstream_root, 'data', 'sensor', 'colorfilters', 'RGB.mat'), wave));
        dut = sensorSet(dut, 'irFilter', ieReadSpectra(fullfile(upstream_root, 'data', 'sensor', 'irfilters', 'infrared2.mat'), wave));
        dut = sensorSet(dut, 'cfapattern', [2 1; 3 2]);
        dut = sensorSet(dut, 'size', [144 176]);
        dut = sensorSet(dut, 'pixel name', 'My Pixel');
        dut = sensorSet(dut, 'pixel size constant fill factor', [2e-6 2e-6]);
        dut = sensorSet(dut, 'pixel spectral qe', ieReadSpectra(fullfile(upstream_root, 'data', 'sensor', 'photodetectors', 'photodetector.mat'), wave));
        dut = sensorSet(dut, 'pixel voltage swing', 1.5);

        tmp = load(fullfile(upstream_root, 'scripts', 'sensor', 'dutData.mat'), 'volts');
        dut = sensorSet(dut, 'volts', tmp.volts);
        volts = sensorGet(dut, 'volts');
        pixel = sensorGet(dut, 'pixel');

        payload.sensor_name = sensorGet(dut, 'name');
        payload.wave = sensorGet(dut, 'wave');
        payload.filter_spectra = sensorGet(dut, 'filter spectra');
        payload.ir_filter = sensorGet(dut, 'ir filter');
        payload.cfa_pattern = sensorGet(dut, 'cfapattern');
        payload.sensor_size = sensorGet(dut, 'size');
        payload.pixel_name = pixel.name;
        payload.pixel_size_m = sensorGet(dut, 'pixel size');
        payload.pixel_qe = sensorGet(dut, 'pixel spectral qe');
        payload.pixel_voltage_swing = sensorGet(dut, 'pixel voltage swing');
        payload.volts = volts;
        payload.volts_stats = [mean(volts(:)) std(volts(:)) prctile(volts(:), [5 95])];

    case 'sensor_filter_transmissivities_small'
        sensor = sensorCreate();
        filters = sensorGet(sensor, 'filter transmissivities');
        filters(:,1) = filters(:,1) * 0.2;
        filters(:,3) = filters(:,3) * 0.5;
        sensor = sensorSet(sensor, 'filter transmissivities', filters);
        payload.wave = sensorGet(sensor, 'wave');
        payload.filters = sensorGet(sensor, 'filter transmissivities');
        payload.spectral_qe = sensorGet(sensor, 'spectral qe');

    case 'sensor_color_filter_gaussian_roundtrip_small'
        wavelength = 400:10:700;
        cPos = 400:40:700;
        widths = ones(size(cPos)) * 30;
        cFilters = sensorColorFilter('gaussian', wavelength, cPos, widths);
        d.data = cFilters;
        d.wavelength = wavelength;
        d.filterNames = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'};
        d.comment = 'Gaussian filters created by parity sensorColorFilter';
        d.peakWavelengths = cPos;
        savedFile = fullfile(fileparts(output_path), 'gFiltersDeleteMe.mat');
        ieSaveColorFilter(d, savedFile);
        newFilters = ieReadColorFilter(wavelength, savedFile);
        savedStruct = load(savedFile);
        payload.wave = wavelength(:);
        payload.created_filters = cFilters;
        payload.read_filters = newFilters;
        payload.filter_names = savedStruct.filterNames;
        payload.comment = savedStruct.comment;
        payload.peak_wavelengths = savedStruct.peakWavelengths(:);

    case 'sensor_color_filter_asset_nikond100_small'
        wavelength = (400:1000)';
        data = ieReadColorFilter(wavelength, 'NikonD100');
        savedStruct = load(fullfile(isetRootPath, 'data', 'sensor', 'colorfilters', 'nikon', 'NikonD100.mat'));
        payload.wave = wavelength;
        payload.filters = data;
        payload.filter_names = savedStruct.filterNames;
        payload.comment = savedStruct.comment;

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

    case 'sensor_poisson_noise_small'
        rand('seed', 1);
        randn('seed', 1);
        scene = sceneCreate('macbeth');
        scene = sceneSet(scene, 'fov', 10);

        oi = oiCreate('diffraction limited');
        oi = oiCompute(oi, scene);

        load(fullfile(isetRootPath, 'data', 'sensor', 'sony', 'imx363.mat'), 'sensor');
        sensor = sensorSet(sensor, 'row', 256);
        sensor = sensorSet(sensor, 'col', 256);
        sensor = sensorSet(sensor, 'exp time', 0.016);
        sensor = sensorCompute(sensor, oi);

        rect = [96 156 24 28];
        sensor = sensorSet(sensor, 'roi', rect);
        dv = sensorGet(sensor, 'roi dv', rect);
        finiteDV = dv(isfinite(dv));
        sensorDV = sensorGet(sensor, 'dv');
        finiteSensorDV = sensorDV(isfinite(sensorDV));

        payload.rect = rect(:);
        payload.roi_mean_dv = mean(finiteDV(:));
        payload.roi_std_dv = std(finiteDV(:), 0);
        payload.roi_percentiles = prctile(finiteDV(:), [10 50 90])';
        payload.sqrt_mean_dv = sqrt(payload.roi_mean_dv);
        payload.sensor_mean_dv = mean(finiteSensorDV(:));

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

    case 'sensor_macbeth_daylight_estimate_small'
        wave = (400:10:700)';
        reflectance = macbethReadReflectance(wave);

        sensor = sensorCreate;
        sensorFilters = sensorGet(sensor, 'spectral qe');

        dayBasisEnergy = ieReadSpectra('cieDaylightBasis.mat', wave);
        dayBasisQuanta = Energy2Quanta(wave, dayBasisEnergy);

        trueWeights = [1 0 0]';
        illuminantPhotons = dayBasisQuanta * trueWeights;
        cameraData = sensorFilters' * diag(illuminantPhotons) * reflectance;

        X1 = sensorFilters' * diag(dayBasisQuanta(:, 1)) * reflectance;
        X2 = sensorFilters' * diag(dayBasisQuanta(:, 2)) * reflectance;
        X3 = sensorFilters' * diag(dayBasisQuanta(:, 3)) * reflectance;
        designMatrix = [X1(:), X2(:), X3(:)];
        cameraStacked = cameraData(:);
        normalMatrix = designMatrix' * designMatrix;
        rhs = designMatrix' * cameraStacked;
        solvedWeights = normalMatrix \ rhs;
        estimatedWeights = solvedWeights / solvedWeights(1);
        estimatedIlluminant = dayBasisQuanta * estimatedWeights;

        payload.wave = wave;
        payload.reflectance = reflectance;
        payload.sensor_filters = sensorFilters;
        payload.day_basis_quanta = dayBasisQuanta;
        payload.true_weights = trueWeights';
        payload.illuminant_photons = illuminantPhotons;
        payload.camera_data = cameraData;
        payload.design_matrix = designMatrix;
        payload.camera_stacked = cameraStacked;
        payload.normal_matrix = normalMatrix;
        payload.rhs = rhs;
        payload.estimated_weights = estimatedWeights';
        payload.estimated_illuminant = estimatedIlluminant;

    case 'sensor_spectral_radiometer_small'
        scene = sceneCreate('uniformD65');
        oi = oiCreate;
        oi = oiCompute(oi, scene);

        wave = (400:700)';
        [data, filterNames] = ieReadColorFilter(wave, 'radiometer');
        nFilters = size(data, 2);
        filterOrder = 1:nFilters;
        filterFile = fullfile(upstream_root, 'data', 'sensor', 'colorfilters', 'radiometer.mat');
        wSamples = zeros(1, numel(filterNames));
        for ii = 1:numel(filterNames)
            wSamples(ii) = str2double(filterNames{ii});
        end

        pixel = pixelCreate('default', wave);
        pixel = pixelSet(pixel, 'fill factor', 1);
        pixel = pixelSet(pixel, 'size same fill factor', [1.5 1.5] * 1e-6);

        sensorRadiometer = sensorCreate('custom', pixel, filterOrder, filterFile, [], wave);
        sensorRadiometer = sensorSet(sensorRadiometer, 'size', [10 nFilters]);

        sensorRadiometer = sensorSet(sensorRadiometer, 'exposure time', 1/100);
        sensorRadiometer = sensorSet(sensorRadiometer, 'noise flag', -2);
        sensorRadiometer = sensorCompute(sensorRadiometer, oi);
        electrons = sensorGet(sensorRadiometer, 'electrons');

        sensorRadiometer = sensorSet(sensorRadiometer, 'noise flag', -1);
        sensorRadiometer = sensorCompute(sensorRadiometer, oi);
        electronsNoNoise = sensorGet(sensorRadiometer, 'electrons');

        noisyLine = electrons(5, :);
        noiseFreeLine = electronsNoNoise(5, :);

        payload.wave = wave;
        payload.w_samples = wSamples;
        payload.filter_pattern = sensorGet(sensorRadiometer, 'pattern');
        payload.sensor_size = sensorGet(sensorRadiometer, 'size');
        payload.filter_spectra = sensorGet(sensorRadiometer, 'filter spectra');
        payload.noise_free_line = noiseFreeLine;
        payload.shot_sd_line = sqrt(max(noiseFreeLine, 0));
        payload.noisy_line_stats = [mean(noisyLine(:)) std(noisyLine(:)) prctile(noisyLine(:), [5 95])];
        payload.noisy_full_stats = [mean(electrons(:)) std(electrons(:)) prctile(electrons(:), [5 95])];
        payload.noise_free_full_stats = [mean(electronsNoNoise(:)) std(electronsNoNoise(:)) prctile(electronsNoNoise(:), [5 95])];

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

    case 'sensor_exposure_cfa_small'
        scene = sceneCreate;
        scene = sceneSet(scene, 'fov', 4);

        oi = oiCreate;
        oi = oiCompute(oi, scene);

        sensor = sensorCreate;
        sensor = sensorSet(sensor, 'noise flag', 0);
        bluish = [0.04 0.03; 0.30 0.02];
        sensorB = sensorSet(sensor, 'exposure duration', bluish);
        sensorB = sensorCompute(sensorB, oi, 0);
        reddish = [0.04 0.70; 0.03 0.02];
        sensorR = sensorSet(sensor, 'exposure duration', reddish);
        sensorR = sensorCompute(sensorR, oi, 0);

        ip = ipCreate;
        ip = ipCompute(ip, sensorR);

        camera = cameraCreate;
        camera = cameraSet(camera, 'sensor noise flag', 0);
        camera = cameraSet(camera, 'sensor exposure duration', reddish);
        camera = cameraCompute(camera, scene);
        cameraSensor = cameraGet(camera, 'sensor');

        bluishVolts = sensorGet(sensorB, 'volts');
        reddishVolts = sensorGet(sensorR, 'volts');
        payload.bluish_mean_volts = mean(bluishVolts(:));
        payload.reddish_mean_volts = mean(reddishVolts(:));
        voltsR = sensorGet(sensorR, 'volts');
        payload.reddish_center_pixel = voltsR(floor(size(voltsR, 1) / 2) + 1, floor(size(voltsR, 2) / 2) + 1);
        cameraVolts = sensorGet(cameraSensor, 'volts');
        payload.camera_mean_volts = mean(cameraVolts(:));
        payload.camera_center_pixel = cameraVolts(floor(size(cameraVolts, 1) / 2) + 1, floor(size(cameraVolts, 2) / 2) + 1);

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

function means = local_nanmean_channels(data)
data = double(data);
n_channels = size(data, 3);
means = zeros(1, n_channels);
for ii = 1:n_channels
    channel = data(:, :, ii);
    mask = ~isnan(channel);
    if any(mask(:))
        means(ii) = mean(channel(mask));
    else
        means(ii) = NaN;
    end
end
end

function means = local_sensor_filter_means(sensor)
n_filters = sensorGet(sensor, 'nfilters');
means = zeros(1, n_filters);
for ii = 1:n_filters
    channel = double(sensorGet(sensor, 'volts', ii));
    channel = channel(~isnan(channel));
    if isempty(channel)
        means(ii) = NaN;
    else
        means(ii) = mean(channel);
    end
end
end

function sensorArray = local_sensor_create_array_ovt(expTime, sz, noiseFlag, upstream_root)
[lpdLCG, lpdHCG] = local_sensor_create_ovt_large_pair(upstream_root);
spdLCG = local_sensor_create_ovt_small(upstream_root);
sensorArray = [lpdLCG lpdHCG spdLCG];
for ii = 1:numel(sensorArray)
    sensorArray(ii) = sensorSet(sensorArray(ii), 'exp time', expTime);
    sensorArray(ii) = sensorSet(sensorArray(ii), 'size', sz);
    sensorArray(ii) = sensorSet(sensorArray(ii), 'noise flag', noiseFlag);
end
end

function [sensorLCG, sensorHCG] = local_sensor_create_ovt_large_pair(upstream_root)
sensorLCG = local_sensor_create_ovt_base(upstream_root);
sensorLCG = sensorSet(sensorLCG, 'size', [968 1288]);
sensorLCG = sensorSet(sensorLCG, 'pixel size same fill factor', 2.8e-6);
sensorLCG = sensorSet(sensorLCG, 'pixel voltage swing', 22000 * 49e-6);
sensorLCG = sensorSet(sensorLCG, 'pixel conversion gain', 49e-6);
sensorLCG = sensorSet(sensorLCG, 'pixel fill factor', 1);
sensorLCG = sensorSet(sensorLCG, 'pixel read noise electrons', 3.05);
sensorLCG = sensorSet(sensorLCG, 'pixel dark voltage', 25.6 * 49e-6);
sensorLCG = sensorSet(sensorLCG, 'analog gain', 1);
sensorLCG = sensorSet(sensorLCG, 'quantization', '12 bit');
sensorLCG = sensorSet(sensorLCG, 'name', 'ovt-LPDLCG');

sensorHCG = sensorLCG;
sensorHCG = sensorSet(sensorHCG, 'pixel read noise electrons', 0.83);
sensorHCG = sensorSet(sensorHCG, 'analog gain', 49 / 200);
sensorHCG = sensorSet(sensorHCG, 'name', 'ovt-LPDHCG');
end

function sensor = local_sensor_create_ovt_small(upstream_root)
sensor = local_sensor_create_ovt_base(upstream_root);
sensor = sensorSet(sensor, 'size', [968 1288]);
sensor = sensorSet(sensor, 'pixel size same fill factor', 2.8e-6);
sensor = sensorSet(sensor, 'pixel voltage swing', 7900 * 49e-6);
sensor = sensorSet(sensor, 'pixel conversion gain', 49e-6);
sensor = sensorSet(sensor, 'pixel fill factor', 1e-2);
sensor = sensorSet(sensor, 'pixel read noise electrons', 0.83);
sensor = sensorSet(sensor, 'pixel dark voltage', 4.2 * 49e-6);
sensor = sensorSet(sensor, 'quantization', '12 bit');
sensor = sensorSet(sensor, 'name', 'ovt-SPDLCG');
end

function sensor = local_sensor_create_ovt_base(upstream_root)
sensor = sensorCreate();
wave = sensorGet(sensor, 'wave');
[filterData, filterNames] = local_read_ovt_color_filters(wave, upstream_root);
sensor = sensorSet(sensor, 'filter spectra', filterData);
sensor = sensorSet(sensor, 'filter names', filterNames);
end

function [filterData, filterNames] = local_read_ovt_color_filters(wave, upstream_root)
fileName = fullfile(upstream_root, 'data', 'sensor', 'colorfilters', 'OVT', 'ovt-large.mat');
raw = load(fileName, 'wavelength', 'data');
rawWave = double(raw.wavelength(:));
rawData = double(raw.data);
if size(rawData, 1) ~= numel(rawWave) && size(rawData, 2) == numel(rawWave)
    rawData = rawData';
end
filterData = interp1(rawWave, rawData, wave(:), 'linear');
if size(filterData, 1) ~= numel(wave)
    filterData = filterData';
end
filterData = max(filterData, 0);
if max(filterData(:)) > 1
    filterData = filterData ./ max(filterData(:));
end
filterNames = {'r', 'g', 'b'};
end

function [sensorCombined, metadata] = local_imx490_compute(oi, method, expTime, noiseFlag, upstream_root)
gains = [1 4 1 4];
isetGains = 1 ./ gains;
method = ieParamFormat(method);

sensorLarge = local_sensor_create_imx490_variant('large', upstream_root);
sensorSmall = local_sensor_create_imx490_variant('small', upstream_root);

sensorLarge = sensorSet(sensorLarge, 'match oi', oi);
sensorSmall = sensorSet(sensorSmall, 'match oi', oi);
sensorLarge = sensorSet(sensorLarge, 'noise flag', noiseFlag);
sensorSmall = sensorSet(sensorSmall, 'noise flag', noiseFlag);
sensorLarge = sensorSet(sensorLarge, 'exp time', expTime);
sensorSmall = sensorSet(sensorSmall, 'exp time', expTime);

oiSpacing = oiGet(oi, 'spatial resolution', 'um');
sensorSpacing = sensorGet(sensorLarge, 'pixel size', 'um');
assert(max(abs(oiSpacing(:) - sensorSpacing(:))) < 1e-3);

oiSize = oiGet(oi, 'size');
sensorLarge = sensorSet(sensorLarge, 'size', oiSize);
sensorSmall = sensorSet(sensorSmall, 'size', oiSize);

sensorArray = cell(1, 4);

sensorArray{1} = sensorSet(sensorLarge, 'analog gain', isetGains(1));
sensorArray{1} = sensorSet(sensorArray{1}, 'name', sprintf('large-%1dx', gains(1)));
sensorArray{1} = sensorCompute(sensorArray{1}, oi);

sensorArray{2} = sensorSet(sensorLarge, 'analog gain', isetGains(2));
sensorArray{2} = sensorSet(sensorArray{2}, 'name', sprintf('large-%1dx', gains(2)));
sensorArray{2} = sensorCompute(sensorArray{2}, oi);

sensorArray{3} = sensorSet(sensorSmall, 'analog gain', isetGains(3));
sensorArray{3} = sensorSet(sensorArray{3}, 'name', sprintf('small-%1dx', gains(3)));
sensorArray{3} = sensorCompute(sensorArray{3}, oi);

sensorArray{4} = sensorSet(sensorSmall, 'analog gain', isetGains(4));
sensorArray{4} = sensorSet(sensorArray{4}, 'name', sprintf('small-%1dx', gains(4)));
sensorArray{4} = sensorCompute(sensorArray{4}, oi);

sensorCombined = sensorLarge;

switch method
    case 'average'
        v1 = sensorGet(sensorArray{1}, 'volts');
        v2 = sensorGet(sensorArray{2}, 'volts');
        v3 = sensorGet(sensorArray{3}, 'volts');
        v4 = sensorGet(sensorArray{4}, 'volts');

        vSwingL = sensorGet(sensorLarge, 'pixel voltage swing');
        vSwingS = sensorGet(sensorSmall, 'pixel voltage swing');

        idx1 = (v1 < vSwingL);
        idx2 = (v2 < vSwingL);
        idx3 = (v3 < vSwingS);
        idx4 = (v4 < vSwingS);
        N = idx1 + idx2 + idx3 + idx4;

        in1 = sensorGet(sensorArray{1}, 'electrons per area', 'um');
        in2 = sensorGet(sensorArray{2}, 'electrons per area', 'um');
        in3 = sensorGet(sensorArray{3}, 'electrons per area', 'um');
        in4 = sensorGet(sensorArray{4}, 'electrons per area', 'um');

        cg = sensorGet(sensorLarge, 'pixel conversion gain');
        volts = zeros(size(in1));
        valid = (N > 0);
        volts(valid) = cg .* ((in1(valid) + in2(valid) + in3(valid) + in4(valid)) ./ N(valid));
        volts(~valid) = 1;

        vSwing = sensorGet(sensorLarge, 'pixel voltage swing');
        volts = vSwing * local_scale_to_peak(volts);

        sensorCombined = sensorSet(sensorCombined, 'volts', volts);
        sensorCombined = sensorSet(sensorCombined, 'analog gain', 1);
        sensorCombined = sensorSet(sensorCombined, 'analog offset', 0);
        sensorCombined.metadata.npixels = N;

    case 'bestsnr'
        e1 = sensorGet(sensorArray{1}, 'electrons');
        e2 = sensorGet(sensorArray{2}, 'electrons');
        e3 = sensorGet(sensorArray{3}, 'electrons');
        e4 = sensorGet(sensorArray{4}, 'electrons');

        wcL = sensorGet(sensorLarge, 'pixel well capacity');
        wcS = sensorGet(sensorSmall, 'pixel well capacity');
        idx1 = (e1 < wcL);
        idx2 = (e2 < wcL);
        idx3 = (e3 < wcS);
        idx4 = (e4 < wcS);
        e1(~idx1) = 0;
        e2(~idx2) = 0;
        e3(~idx3) = 0;
        e4(~idx4) = 0;

        [val, bestPixel] = max([e1(:), e2(:), e3(:), e4(:)], [], 2);
        val = reshape(val, size(e1));
        bestPixel = reshape(bestPixel, size(e1));

        cg = sensorGet(sensorLarge, 'pixel conversion gain');
        volts = val .* cg;
        sensorCombined = sensorSet(sensorCombined, 'volts', volts);
        sensorCombined.metadata.bestPixel = bestPixel;

    otherwise
        error('Unknown IMX490 method %s', method);
end

nbits = sensorGet(sensorCombined, 'nbits');
if isempty(nbits)
    nbits = 12;
end
dv = (2 ^ nbits) * local_scale_to_peak(volts);
sensorCombined = sensorSet(sensorCombined, 'dv', dv);
sensorCombined = sensorSet(sensorCombined, 'name', sprintf('Combined-%s', method));

metadata.sensorArray = sensorArray;
metadata.method = method;
end

function sensor = local_sensor_create_imx490_variant(variant, upstream_root)
variant = ieParamFormat(variant);
isLarge = strcmp(variant, 'large');
wave = (390:10:710)';
sensor = sensorCreate('bayer-rggb');
sensor = sensorSet(sensor, 'wave', wave);
sensor = sensorSet(sensor, 'size', [600 800]);
sensor = sensorSet(sensor, 'pixel size same fill factor', 3.0e-6);

voltageSwing = 4096 * 0.25e-3;
if isLarge
    wellCapacity = 120000;
    fillFactor = 0.9;
    sensorName = 'imx490-large';
else
    wellCapacity = 60000;
    fillFactor = 0.1;
    sensorName = 'imx490-small';
end

sensor = sensorSet(sensor, 'pixel conversion gain', voltageSwing / wellCapacity);
sensor = sensorSet(sensor, 'pixel voltage swing', voltageSwing);
sensor = sensorSet(sensor, 'pixel dark voltage', 0);
sensor = sensorSet(sensor, 'pixel read noise electrons', 1);
sensor = sensorSet(sensor, 'pixel fill factor', fillFactor);
sensor = sensorSet(sensor, 'dsnu level', 0);
sensor = sensorSet(sensor, 'prnu level', 0.7);
sensor = sensorSet(sensor, 'analog gain', 1);
sensor = sensorSet(sensor, 'analog offset', 0);
sensor = sensorSet(sensor, 'exp time', 1 / 60);
sensor = sensorSet(sensor, 'black level', 0);
sensor = sensorSet(sensor, 'quantization', '12 bit');

[filterData, filterNames] = local_read_imx490_color_filters(wave, upstream_root);
sensor = sensorSet(sensor, 'filter spectra', filterData);
sensor = sensorSet(sensor, 'filter names', filterNames);

irFilter = local_read_interpolated_spectrum( ...
    fullfile(upstream_root, 'data', 'sensor', 'irfilters', 'ircf_public.mat'), ...
    wave);
sensor = sensorSet(sensor, 'ir filter', irFilter);
sensor = sensorSet(sensor, 'name', sensorName);
end

function [filterData, filterNames] = local_read_imx490_color_filters(wave, upstream_root)
fileName = fullfile(upstream_root, 'data', 'sensor', 'colorfilters', 'auto', 'SONY', 'cf_imx490.mat');
raw = load(fileName, 'wavelength', 'data');
rawWave = double(raw.wavelength(:));
rawData = double(raw.data);
if size(rawData, 1) ~= numel(rawWave) && size(rawData, 2) == numel(rawWave)
    rawData = rawData';
end
filterData = interp1(rawWave, rawData, wave(:), 'linear');
if size(filterData, 1) ~= numel(wave)
    filterData = filterData';
end
filterData(isnan(filterData)) = 0;
filterData = max(filterData, 0);
if max(filterData(:)) > 1
    filterData = filterData ./ max(filterData(:));
end
filterNames = {'a', 'b', 'c'};
end

function data = local_read_interpolated_spectrum(fileName, wave)
raw = load(fileName, 'wavelength', 'data');
rawWave = double(raw.wavelength(:));
rawData = double(raw.data);
if size(rawData, 1) ~= numel(rawWave) && size(rawData, 2) == numel(rawWave)
    rawData = rawData';
end
data = interp1(rawWave, rawData, wave(:), 'linear');
if size(data, 1) ~= numel(wave)
    data = data';
end
data(isnan(data)) = 0;
end

function values = local_scale_to_peak(values)
peak = max(values(:));
if peak > 0
    values = values ./ peak;
else
    values = zeros(size(values));
end
end

function value = local_ml_optimal_offset(ml, sensor, unitName)
cra = mlensGet(ml, 'chief ray angle radians');
zStack = sensorGet(sensor, 'pixel layer thicknesses', unitName);
nStack = sensorGet(sensor, 'pixel refractive indices');
if numel(nStack) >= 3
    nStack = nStack(2:(end - 1));
end
value = 0;
for ii = 1:length(zStack)
    value = value + zStack(ii) * tan(asin(sin(cra) / nStack(ii)));
end
end

function nearestTwo = local_find_nearest_two(array, number)
differences = abs(double(array(:)) - double(number));
[~, sortedIndices] = sort(differences);
nearestTwo = double(array(sortedIndices(1:2)));
end

function mask = local_create_circle_mask(radius, imgSize)
[X, Y] = meshgrid(1:imgSize(2), 1:imgSize(1));
centerX = imgSize(2) / 2;
centerY = imgSize(1) / 2;
distFromCenter = sqrt((X - centerX) .^ 2 + (Y - centerY) .^ 2);
mask = distFromCenter <= radius;
end

function [psf, wavefront] = local_generate_fringe_psf(zernikeCoeffs)
gridSize = 512;
[x, y] = meshgrid(linspace(-1, 1, gridSize));
rho = sqrt(x .^ 2 + y .^ 2);
theta = atan2(y, x);
zernikeCoeffs = double(zernikeCoeffs(:));

zernikeTerms = {
    @(rho, theta) 1, ...
    @(rho, theta) rho .* cos(theta), ...
    @(rho, theta) rho .* sin(theta), ...
    @(rho, theta) -1 + 2 * rho .^ 2, ...
    @(rho, theta) rho .^ 2 .* cos(2 * theta), ...
    @(rho, theta) rho .^ 2 .* sin(2 * theta), ...
    @(rho, theta) (-2 * rho + 3 * rho .^ 3) .* cos(theta), ...
    @(rho, theta) (-2 * rho + 3 * rho .^ 3) .* sin(theta), ...
    @(rho, theta) 1 - 6 * rho .^ 2 + 6 * rho .^ 4, ...
    @(rho, theta) rho .^ 3 .* cos(3 * theta), ...
    @(rho, theta) rho .^ 3 .* sin(3 * theta), ...
    @(rho, theta) (-3 * rho .^ 2 + 4 * rho .^ 4) .* cos(2 * theta), ...
    @(rho, theta) (-3 * rho .^ 2 + 4 * rho .^ 4) .* sin(2 * theta), ...
    @(rho, theta) (3 * rho - 12 * rho .^ 3 + 10 * rho .^ 5) .* cos(theta), ...
    @(rho, theta) (3 * rho - 12 * rho .^ 3 + 10 * rho .^ 5) .* sin(theta)
};

wavefront = zeros(size(rho));
for ii = 1:numel(zernikeCoeffs)
    wavefront = wavefront + zernikeCoeffs(ii) .* zernikeTerms{ii}(rho, theta);
end

apertureMask = double(local_create_circle_mask(round(gridSize / 2), [gridSize, gridSize]));
pupilFuncPhase = exp(-1i * 2 * pi * wavefront);
amp = fftshift(fft2(ifftshift(pupilFuncPhase .* apertureMask)));
inten = amp .* conj(amp);
psf = real(inten);
end

function optimalOffsets = local_ml_optimal_offsets(ml, sensor)
n2 = mlensGet(ml, 'ml refractive index');
mlFL = mlensGet(ml, 'ml focal length', 'microns');
sourceFL = mlensGet(ml, 'source focal length', 'microns');
pixWidth = mlensGet(ml, 'diameter', 'meters');

sensorAdjusted = sensorSet(sensor, 'pixel width', pixWidth);
sensorAdjusted = sensorSet(sensorAdjusted, 'pixel height', pixWidth);
sSupport = sensorGet(sensorAdjusted, 'spatial support', 'um');
[X, Y] = meshgrid(sSupport.x, sSupport.y);
cra = atan(sqrt(X .^ 2 + Y .^ 2) / sourceFL);
optimalOffsets = mlFL * tan(asin(sin(cra) / n2));
end

function stats = local_stats_vector(values)
values = double(values(:));
stats = [
    mean(values);
    std(values);
    prctile(values, 5);
    prctile(values, 95);
]';
end

function values = local_channel_normalize(values)
values = double(values(:))';
values = values / max(max(abs(values)), eps);
end

function samples = local_deterministic_normal_samples(nRows, nCols)
indices = reshape(1:(nRows*nCols), nRows, nCols);
u1 = mod(indices * 0.7548776662466927, 1);
u2 = mod(indices * 0.5698402909980532, 1);
u1 = min(max(u1, 1e-6), 1 - 1e-6);
samples = sqrt(-2 .* log(u1)) .* cos(2 .* pi .* u2);
end

function s = local_ie_mvnrnd(mu, Sigma, standardNormal)
mu = double(mu);
Sigma = double(Sigma);
if isscalar(mu)
    mu = reshape(mu, 1, 1);
elseif isvector(mu)
    mu = reshape(mu, 1, []);
else
    mu = double(mu);
end

if size(mu, 2) == 1 && ~isscalar(Sigma)
    mu = mu';
end

[n, d] = size(mu);
if isscalar(Sigma)
    Sigma = double(Sigma) * eye(d);
end

if any(size(Sigma) ~= [d, d])
    error('Sigma must have dimensions d x d where mu is n x d.');
end

if nargin < 3 || isempty(standardNormal)
    standardNormal = randn(n, d);
else
    standardNormal = double(standardNormal);
    if n == 1 && size(standardNormal, 2) == d && size(standardNormal, 1) > 1
        mu = repmat(mu, size(standardNormal, 1), 1);
        n = size(mu, 1);
    end
    if any(size(standardNormal) ~= [n, d])
        error('standardNormal must match the expanded mu shape.');
    end
end

try
    U = chol(Sigma);
catch
    [E, Lambda] = eig(Sigma);
    if min(diag(Lambda)) < -1e-10
        error('Sigma must be positive semi-definite.');
    end
    U = sqrt(max(Lambda, 0)) * E';
end

s = standardNormal * U + mu;
end

function values = local_canonical_profile(values, nSamples)
if nargin < 2
    nSamples = 41;
end

values = double(values);
query = linspace(-1, 1, nSamples);
if isvector(values)
    row = double(values(:))';
    support = linspace(-1, 1, numel(row));
    values = interp1(support, row, query, 'linear')';
    return;
end

out = zeros(size(values, 1), nSamples);
for ii = 1:size(values, 1)
    row = double(values(ii, :));
    support = linspace(-1, 1, numel(row));
    out(ii, :) = interp1(support, row, query, 'linear');
end
values = out;
end
