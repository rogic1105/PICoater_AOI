"""Synthetic-log tests for settings, Mura, and camera-parameter validators."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parents[1]
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from flow_checks.core import CheckStatus, FlowLine, FlowSession
from flow_checks.contract import GlobalContractValidator
from flow_checks.data import DataFlowValidator
from flow_checks.live import LiveFlowValidator
from flow_checks.mura import MuraFlowValidator
from flow_checks.parameter import ParameterFlowValidator
from flow_checks.registry import PENDING_DOMAINS
from flow_checks.review import ReviewFlowValidator
from flow_checks.settings import SettingsFlowValidator


def session(*messages: str) -> FlowSession:
    lines = [
        FlowLine(
            elapsed=float(index),
            timestamp=f"00:00:{index:02d}.000",
            thread=1,
            message=message,
        )
        for index, message in enumerate(messages)
    ]
    return FlowSession(Path("synthetic.log"), lines)


def result(report, rule: str):
    return next(item for item in report.results if item.rule == rule)


class GlobalContractValidatorTests(unittest.TestCase):
    def test_shutdown_light_off_before_release_passes(self):
        report = GlobalContractValidator().validate(
            session(
                "ui:關閉程式",
                "shutdown light off result=sent",
                "shutdown resources released",
            )
        )
        self.assertEqual(CheckStatus.PASS, result(report, "G2.light-off").status)

    def test_shutdown_light_command_failure_fails(self):
        report = GlobalContractValidator().validate(
            session(
                "ui:關閉程式",
                "shutdown light off result=failed",
                "shutdown resources released",
            )
        )
        self.assertEqual(CheckStatus.FAIL, result(report, "G2.light-off").status)

    def test_legacy_shutdown_without_light_evidence_is_not_covered(self):
        report = GlobalContractValidator().validate(
            session(
                "ui:關閉程式",
                "shutdown resources released",
            )
        )
        self.assertEqual(
            CheckStatus.NOT_COVERED,
            result(report, "G2.light-off").status,
        )

    def test_overlay_restore_and_synchronized_change_pass(self):
        report = GlobalContractValidator().validate(
            session(
                "canvas overlay restore mode=Coordinates sync=live+review",
                "ui:canvas overlay mode=CoordinateFrames sync=live+review persisted=true",
            )
        )
        self.assertEqual(CheckStatus.PASS, result(report, "G3.overlay").status)

    def test_overlay_requires_one_restore_line(self):
        report = GlobalContractValidator().validate(
            session(
                "ui:canvas overlay mode=CoordinateFrames sync=live+review persisted=true",
            )
        )
        self.assertEqual(CheckStatus.FAIL, result(report, "G3.overlay").status)


class SettingsFlowValidatorTests(unittest.TestCase):
    def test_live_inspection_settings_scale_and_light_stimulus_pass(self):
        report = SettingsFlowValidator().validate(
            session(
                "ui:設定[dc_HessianMaxFactorV]=0.5",
                "setting route dc_HessianMaxFactorV owner=DataStats "
                "effects=InspectionService+CapturePolicy+ReviewCurves+LiveInspectionCurves",
                "live inspection apply setting=dc_HessianMaxFactorV "
                "hm=0.5000/1.0000 thresholdC=0.2000/0.6000 "
                "thresholdR=0.2000/0.6000 mode=Both direction=Both "
                "action=normalization-latest",
                "ui:閮剖?[dd_HessianMaxFactorH]=1.0",
                "setting route dd_HessianMaxFactorH owner=DataStats "
                "effects=ReviewCurves+LiveInspectionCurves",
                "live inspection apply setting=dd_HessianMaxFactorH "
                "hm=0.5000/1.0000 thresholdC=0.2000/0.6000 "
                "thresholdR=0.2000/0.6000 mode=Both direction=Both "
                "action=normalization-latest",
                "live row normalize captureHm=0.5000 rowHm=1.0000 ratio=0.5000",
                "DT curve refresh 260804-135456 "
                "reason=setting-dd_HessianMaxFactorH "
                "column=False row=True source=memory preserveRange=True "
                "rangeDelta=0.0000",
                "live inspection stimulus brightness=100 direction=col "
                "mean=0.1000 max=0.3000 threshold=0.2000/0.6000 "
                "mode=Both verdict=O source=light-surrogate-not-mura",
                "live inspection stimulus brightness=100 direction=row "
                "mean=0.1200 max=0.3200 threshold=0.2000/0.6000 "
                "mode=Both verdict=O source=light-surrogate-not-mura",
                "live inspection stimulus brightness=255 direction=col "
                "mean=0.1001 max=0.3001 threshold=0.2000/0.6000 "
                "mode=Both verdict=O source=light-surrogate-not-mura",
                "live inspection stimulus brightness=255 direction=row "
                "mean=0.1201 max=0.3201 threshold=0.2000/0.6000 "
                "mode=Both verdict=O source=light-surrogate-not-mura",
            )
        )

        self.assertEqual(CheckStatus.PASS, result(report, "S1.live-apply").status)
        self.assertEqual(CheckStatus.PASS, result(report, "S1.row-normalize").status)
        self.assertEqual(
            CheckStatus.PASS,
            result(report, "S1.report-preserve-range").status,
        )
        self.assertEqual(CheckStatus.PASS, result(report, "S1.light-stimulus").status)

    def test_live_inspection_stimulus_rejects_wrong_verdict(self):
        report = SettingsFlowValidator().validate(
            session(
                "ui:設定[ec_ErrorValueMeanV]=0.2",
                "setting route ec_ErrorValueMeanV owner=DataStats "
                "effects=InspectionService+ColumnThresholds+ReviewCurves+LiveInspectionCurves",
                "live inspection apply setting=ec_ErrorValueMeanV "
                "hm=0.5000/1.0000 thresholdC=0.2000/0.6000 "
                "thresholdR=0.2000/0.6000 mode=Mean direction=Both action=refresh",
                "live inspection stimulus brightness=100 direction=col "
                "mean=0.1000 max=0.3000 threshold=0.2000/0.6000 "
                "mode=Mean verdict=O source=light-surrogate-not-mura",
                "live inspection stimulus brightness=100 direction=row "
                "mean=0.1000 max=0.3000 threshold=0.2000/0.6000 "
                "mode=Both verdict=O source=light-surrogate-not-mura",
                "live inspection stimulus brightness=255 direction=col "
                "mean=0.3000 max=0.3000 threshold=0.2000/0.6000 "
                "mode=Mean verdict=O source=light-surrogate-not-mura",
                "live inspection stimulus brightness=255 direction=row "
                "mean=0.3000 max=0.7000 threshold=0.2000/0.6000 "
                "mode=Both verdict=X source=light-surrogate-not-mura",
            )
        )

        self.assertEqual(CheckStatus.FAIL, result(report, "S1.light-stimulus").status)

    def test_live_normalization_output_accepts_latest_curve_and_pixel_ratio(self):
        report = SettingsFlowValidator().validate(
            session(
                "set:[dc_HessianMaxFactorV]=0.8",
                "setting route dc_HessianMaxFactorV owner=DataStats "
                "effects=InspectionService+CapturePolicy+ReviewCurves+LiveInspectionCurves",
                "live inspection apply setting=dc_HessianMaxFactorV "
                "hm=0.8000/0.5000 thresholdC=0.2000/0.6000 "
                "thresholdR=0.2000/0.6000 mode=Both direction=Both "
                "action=normalization-latest",
                "set:[dc_HessianMaxFactorV]=1.0",
                "setting route dc_HessianMaxFactorV owner=DataStats "
                "effects=InspectionService+CapturePolicy+ReviewCurves+LiveInspectionCurves",
                "live inspection apply setting=dc_HessianMaxFactorV "
                "hm=1.0000/0.5000 thresholdC=0.2000/0.6000 "
                "thresholdR=0.2000/0.6000 mode=Both direction=Both "
                "action=normalization-latest",
                "live curve applied setting=dc_HessianMaxFactorV generation=2 "
                "hm=1.0000/0.5000 colMeanPeak=0.2000 colMaxPeak=0.7000 "
                "rowMeanPeak=0.1000 rowMaxPeak=0.3000 "
                "rowAction=rescale-current rowWrite=3000->3000",
                "live image scale source=adaptive-standard-half captureHm=0.5000 "
                "currentHm=1.0000/0.5000 scale=1.0000/0.5000",
                "RV normalization queued generation=1 setting=dc_HessianMaxFactorV",
                "RV normalization queued generation=2 setting=dc_HessianMaxFactorV",
                "RV normalization settle generation=2 setting=dc_HessianMaxFactorV "
                "hm=1.0000/0.5000",
            )
        )

        self.assertEqual(
            CheckStatus.PASS,
            result(report, "S1.live-normalization-output").status,
        )

    def test_live_normalization_output_rejects_stale_curve_and_wrong_pixel_ratio(self):
        report = SettingsFlowValidator().validate(
            session(
                "set:[dd_HessianMaxFactorH]=1.0",
                "setting route dd_HessianMaxFactorH owner=DataStats "
                "effects=ReviewCurves+LiveInspectionCurves",
                "live inspection apply setting=dd_HessianMaxFactorH "
                "hm=0.5000/1.0000 thresholdC=0.2000/0.6000 "
                "thresholdR=0.2000/0.6000 mode=Both direction=Both "
                "action=normalization-latest",
                "live curve applied setting=dd_HessianMaxFactorH generation=4 "
                "hm=0.5000/0.5000 colMeanPeak=0.2000 colMaxPeak=0.7000 "
                "rowMeanPeak=0.1000 rowMaxPeak=0.3000 "
                "rowAction=rescale-current rowWrite=3000->3000",
                "live image scale source=adaptive-standard-half captureHm=0.5000 currentHm=0.5000/1.0000 "
                "scale=1.0000/1.0000",
                "RV normalization queued generation=4 setting=dd_HessianMaxFactorH",
                "RV normalization settle generation=3 setting=dd_HessianMaxFactorH "
                "hm=0.5000/1.0000",
            )
        )

        self.assertEqual(
            CheckStatus.FAIL,
            result(report, "S1.live-normalization-output").status,
        )

    def test_live_normalization_output_rejects_waterfall_append(self):
        report = SettingsFlowValidator().validate(
            session(
                "set:[dd_HessianMaxFactorH]=1.0",
                "setting route dd_HessianMaxFactorH owner=DataStats "
                "effects=ReviewCurves+LiveInspectionCurves",
                "live inspection apply setting=dd_HessianMaxFactorH "
                "hm=0.5000/1.0000 thresholdC=0.2000/0.6000 "
                "thresholdR=0.2000/0.6000 mode=Both direction=Both "
                "action=normalization-latest",
                "live curve applied setting=dd_HessianMaxFactorH generation=4 "
                "hm=0.5000/1.0000 colMeanPeak=0.2000 colMaxPeak=0.7000 "
                "rowMeanPeak=0.1000 rowMaxPeak=0.3000 "
                "rowAction=rescale-current rowWrite=3000->6000",
            )
        )

        self.assertEqual(
            CheckStatus.FAIL,
            result(report, "S1.live-normalization-output").status,
        )

    def test_live_pixel_curve_probe_accepts_same_frame_peaks(self):
        report = SettingsFlowValidator().validate(
            session(
                "set:[dc_HessianMaxFactorV]=1.0",
                "live pixel-curve probe cam1 tick=123 axis=C "
                "captureHm=0.5000 currentHm=1.0000 sourceGain=1.2500 imagePeak=0.8000 "
                "curveMeanPeak=0.2000 curveMaxPeak=0.8030 delta=0.0030 "
                "sourceImageMax=102.0000 sourceCurveMax=102.3825 "
                "verdict=match reason=none",
                "live pixel-curve probe cam1 tick=123 axis=R "
                "captureHm=0.5000 currentHm=0.5000 sourceGain=1.2500 imagePeak=0.4000 "
                "curveMeanPeak=0.1000 curveMaxPeak=0.4000 delta=0.0000 "
                "sourceImageMax=102.0000 sourceCurveMax=102.0000 "
                "verdict=match reason=none",
            )
        )

        self.assertEqual(
            CheckStatus.PASS,
            result(report, "S1.live-pixel-curve").status,
        )

    def test_live_pixel_curve_probe_rejects_diverged_image(self):
        report = SettingsFlowValidator().validate(
            session(
                "set:[dc_HessianMaxFactorV]=1.0",
                "live pixel-curve probe cam1 tick=123 axis=C "
                "captureHm=0.5000 currentHm=1.0000 sourceGain=1.2500 imagePeak=0.2000 "
                "curveMeanPeak=0.1000 curveMaxPeak=0.8000 delta=0.6000 "
                "sourceImageMax=25.5000 sourceCurveMax=102.0000 "
                "verdict=mismatch reason=max-delta",
                "live pixel-curve probe cam1 tick=123 axis=R "
                "captureHm=0.5000 currentHm=0.5000 sourceGain=1.2500 imagePeak=0.4000 "
                "curveMeanPeak=0.1000 curveMaxPeak=0.4000 delta=0.0000 "
                "sourceImageMax=102.0000 sourceCurveMax=102.0000 "
                "verdict=match reason=none",
            )
        )

        self.assertEqual(
            CheckStatus.FAIL,
            result(report, "S1.live-pixel-curve").status,
        )

    def test_live_pixel_curve_probe_rejects_quantized_zero_image(self):
        report = SettingsFlowValidator().validate(
            session(
                "set:[dc_HessianMaxFactorV]=10.3",
                "live pixel-curve probe cam1 tick=123 axis=C "
                "captureHm=10.3000 currentHm=10.3000 sourceGain=100.0000 imagePeak=0.0000 "
                "curveMeanPeak=0.0008 curveMaxPeak=0.0029 delta=0.0029 "
                "sourceImageMax=0.0000 sourceCurveMax=0.7395 "
                "verdict=mismatch reason=quantized-zero",
                "live pixel-curve probe cam1 tick=123 axis=R "
                "captureHm=10.3000 currentHm=41.5000 sourceGain=100.0000 imagePeak=0.0000 "
                "curveMeanPeak=0.0004 curveMaxPeak=0.0011 delta=0.0011 "
                "sourceImageMax=0.0000 sourceCurveMax=0.0689 "
                "verdict=mismatch reason=quantized-zero",
            )
        )

        self.assertEqual(
            CheckStatus.FAIL,
            result(report, "S1.live-pixel-curve").status,
        )

    def test_report_refresh_rejects_physical_range_jump(self):
        report = SettingsFlowValidator().validate(
            session(
                "ui:設定[dd_HessianMaxFactorH]=1.0",
                "setting route dd_HessianMaxFactorH owner=DataStats "
                "effects=ReviewCurves+LiveInspectionCurves",
                "DT curve refresh 260804-135456 "
                "reason=setting-dd_HessianMaxFactorH "
                "column=False row=True source=memory preserveRange=True "
                "rangeDelta=12.5000",
            )
        )

        self.assertEqual(
            CheckStatus.FAIL,
            result(report, "S1.report-preserve-range").status,
        )

    def test_hessian_standard_image_brightness_follows_gain(self):
        report = SettingsFlowValidator().validate(
            session(
                "ui:設定[dd_HessianMaxFactorH]=0.5",
                "setting route dd_HessianMaxFactorH owner=DataStats "
                "effects=ReviewCurves+LiveInspectionCurves",
                "live inspection apply setting=dd_HessianMaxFactorH "
                "hm=1.0000/0.5000 thresholdC=0.2000/0.6000 "
                "thresholdR=0.2000/0.6000 mode=Both direction=Both "
                "action=normalization-latest",
                "RV hessian standard 260804-135456 dir=R gain=0.5 scale=25 "
                "sampleMin=0 sampleMax=128 sampleMean=12.500",
                "ui:設定[dd_HessianMaxFactorH]=1.0",
                "setting route dd_HessianMaxFactorH owner=DataStats "
                "effects=ReviewCurves+LiveInspectionCurves",
                "live inspection apply setting=dd_HessianMaxFactorH "
                "hm=1.0000/1.0000 thresholdC=0.2000/0.6000 "
                "thresholdR=0.2000/0.6000 mode=Both direction=Both "
                "action=normalization-latest",
                "RV hessian standard 260804-135456 dir=R gain=1 scale=25 "
                "sampleMin=0 sampleMax=255 sampleMean=25.000",
            )
        )

        self.assertEqual(
            CheckStatus.PASS,
            result(report, "S1.hessian-image-gain").status,
        )

    def test_hessian_standard_image_rejects_inverse_gain(self):
        report = SettingsFlowValidator().validate(
            session(
                "ui:設定[dc_HessianMaxFactorV]=0.5",
                "setting route dc_HessianMaxFactorV owner=DataStats "
                "effects=InspectionService+CapturePolicy+ReviewCurves+LiveInspectionCurves",
                "live inspection apply setting=dc_HessianMaxFactorV "
                "hm=0.5000/1.0000 thresholdC=0.2000/0.6000 "
                "thresholdR=0.2000/0.6000 mode=Both direction=Both "
                "action=normalization-latest",
                "RV hessian standard 260804-135456 dir=C gain=0.5 scale=25 "
                "sampleMin=0 sampleMax=255 sampleMean=30.000",
                "ui:設定[dc_HessianMaxFactorV]=1.0",
                "setting route dc_HessianMaxFactorV owner=DataStats "
                "effects=InspectionService+CapturePolicy+ReviewCurves+LiveInspectionCurves",
                "live inspection apply setting=dc_HessianMaxFactorV "
                "hm=1.0000/1.0000 thresholdC=0.2000/0.6000 "
                "thresholdR=0.2000/0.6000 mode=Both direction=Both "
                "action=normalization-latest",
                "RV hessian standard 260804-135456 dir=C gain=1 scale=25 "
                "sampleMin=0 sampleMax=128 sampleMean=15.000",
            )
        )

        self.assertEqual(
            CheckStatus.FAIL,
            result(report, "S1.hessian-image-gain").status,
        )

    def test_direction_and_review_enhance_are_followed_by_required_updates(self):
        report = SettingsFlowValidator().validate(
            session(
                "RV loadGrab done 260720-120000（20ms）",
                "ui:設定[hd_EnableReviewEnhance]=True",
                "setting route hd_EnableReviewEnhance owner=Enhance effects=None",
                "RV loadGrab begin 260720-120000（proc=True）",
                "RV loadGrab curves=keep source=display 260720-120000",
                "RV pushFrames 7/7（merge=True, feedScale=1, chartView=keep）",
                "RV variantView keep beforeX=0.00~10.00 beforeY=20.00~0.00 "
                "afterX=0.00~10.00 afterY=20.00~0.00 maxDelta=0.000",
                "RV loadGrab done 260720-120000（21ms）",
                "ui:設定[hee_VerticalDirection]=TopToBottom",
                "setting route hee_VerticalDirection owner=LiveLayout effects=None",
                "LC row rowView dir=TopToBottom n=-1 total=100mm view 0~100",
            )
        )
        self.assertEqual(CheckStatus.PASS, result(report, "S0.format").status)
        self.assertEqual(CheckStatus.PASS, result(report, "S0.route").status)
        self.assertEqual(
            CheckStatus.PASS, result(report, "S2.review-enhance").status
        )
        self.assertEqual(CheckStatus.PASS, result(report, "S3.direction").status)

    def test_live_enhance_applies_persistently_to_all_cameras(self):
        report = SettingsFlowValidator().validate(
            session(
                "ui:設定[hc_EnableMuraEnhance]=True",
                "setting route hc_EnableMuraEnhance owner=Enhance effects=None",
                "WF layer raw->column writeRow=3000 history=preserved",
                "live enhance enabled=True direction=column cams=7 "
                "scope=all-cameras waterfallHistory=preserved",
            )
        )
        self.assertEqual(CheckStatus.PASS, result(report, "S4.live-enhance").status)

    def test_live_enhance_requires_history_policy(self):
        report = SettingsFlowValidator().validate(
            session(
                "ui:設定[hc_EnableMuraEnhance]=True",
                "setting route hc_EnableMuraEnhance owner=Enhance effects=None",
                "live enhance enabled=True direction=column cams=7 scope=all-cameras",
            )
        )
        self.assertEqual(CheckStatus.FAIL, result(report, "S4.live-enhance").status)

    def test_enhance_heatmap_is_main_only_and_does_not_reload_data(self):
        report = SettingsFlowValidator().validate(
            session(
                "ui:設定[hda_EnhanceHeatmap]=BlueYellowRed",
                "setting route hda_EnhanceHeatmap owner=Enhance effects=None",
                "enhance heatmap mode=BlueYellowRed live=blue-yellow-red review=gray "
                "scope=main-only data=unchanged",
            )
        )
        self.assertEqual(
            CheckStatus.PASS, result(report, "S5.enhance-heatmap").status
        )

    def test_green_heatmap_is_accepted(self):
        report = SettingsFlowValidator().validate(
            session(
                "ui:設定[hda_EnhanceHeatmap]=Green",
                "setting route hda_EnhanceHeatmap owner=Enhance effects=None",
                "enhance heatmap mode=Green live=green review=gray "
                "scope=main-only data=unchanged",
            )
        )
        self.assertEqual(
            CheckStatus.PASS, result(report, "S5.enhance-heatmap").status
        )

    def test_enhance_heatmap_rejects_missing_immediate_state_line(self):
        report = SettingsFlowValidator().validate(
            session(
                "ui:設定[hda_EnhanceHeatmap]=Off",
                "setting route hda_EnhanceHeatmap owner=Enhance effects=None",
                "RV loadGrab begin 260720-120000（proc=True）",
                "enhance heatmap mode=Off live=gray review=gray "
                "scope=main-only data=unchanged",
            )
        )
        self.assertEqual(
            CheckStatus.FAIL, result(report, "S5.enhance-heatmap").status
        )

    def test_display_crop_is_display_only_and_reflects_setting(self):
        report = SettingsFlowValidator().validate(
            session(
                "ui:設定[cb_CropHead]=25",
                "setting route cb_CropHead owner=LiveLayout effects=None",
                "displayCrop applied head=25.00 tail=10.00 mode=WF "
                "content=9000x30000 zoom=0.050000 fit=True frames=dynamic",
                "displayCrop head=25.00 tail=10.00 "
                "scope=main+column-chart data=unchanged "
                "waterfallHistory=preserved",
            )
        )
        self.assertEqual(
            CheckStatus.PASS, result(report, "S6.display-crop").status
        )

    def test_display_crop_rejects_stale_value(self):
        report = SettingsFlowValidator().validate(
            session(
                "ui:設定[cc_CropTail]=30",
                "setting route cc_CropTail owner=LiveLayout effects=None",
                "displayCrop head=25.00 tail=10.00 "
                "scope=main+column-chart data=unchanged "
                "waterfallHistory=preserved",
            )
        )
        self.assertEqual(
            CheckStatus.FAIL, result(report, "S6.display-crop").status
        )

    def test_display_crop_during_grab_uses_last_value_at_stop(self):
        report = SettingsFlowValidator().validate(
            session(
                "ui:設定[cb_CropHead]=25",
                "setting route cb_CropHead owner=LiveLayout effects=None",
                "capture layout pending grab=260728-120000 "
                "setting=cb_CropHead apply=display-now+stop-final",
                "displayCrop applied head=25.00 tail=10.00 mode=WF "
                "content=8500x30000 zoom=0.050000 fit=True frames=dynamic",
                "displayCrop head=25.00 tail=10.00 "
                "scope=main+column-chart data=unchanged "
                "waterfallHistory=preserved",
                "ui:設定[cb_CropHead]=50",
                "setting route cb_CropHead owner=LiveLayout effects=None",
                "capture layout pending grab=260728-120000 "
                "setting=cb_CropHead apply=display-now+stop-final",
                "displayCrop applied head=50.00 tail=10.00 mode=WF "
                "content=8000x30000 zoom=0.050000 fit=True frames=dynamic",
                "displayCrop head=50.00 tail=10.00 "
                "scope=main+column-chart data=unchanged "
                "waterfallHistory=preserved",
                "StopGrab",
                "capture layout final grab=260728-120000 "
                "ops=1|1|1|1|1|1|1 start=0|1|2|3|4|5|6 "
                "speed=40 head=50 tail=10 path=x",
                "capture layout applied grab=260728-120000 timing=stop "
                "ops=1|1|1|1|1|1|1 start=0|1|2|3|4|5|6 "
                "speed=40 head=50 tail=10 "
                "render=already-applied source=unchanged",
            )
        )
        self.assertEqual(
            CheckStatus.PASS, result(report, "S6.display-crop").status
        )

    def test_display_crop_during_grab_rejects_missing_actual_apply(self):
        report = SettingsFlowValidator().validate(
            session(
                "ui:設定[cc_CropTail]=30",
                "setting route cc_CropTail owner=LiveLayout effects=None",
                "capture layout pending grab=260728-120000 "
                "setting=cc_CropTail apply=display-now+stop-final",
                "displayCrop head=0.00 tail=30.00 "
                "scope=main+column-chart data=unchanged "
                "waterfallHistory=preserved",
                "StopGrab",
                "capture layout final grab=260728-120000 "
                "ops=1|1|1|1|1|1|1 start=0|1|2|3|4|5|6 "
                "speed=40 head=0 tail=30 path=x",
                "capture layout applied grab=260728-120000 timing=stop "
                "ops=1|1|1|1|1|1|1 start=0|1|2|3|4|5|6 "
                "speed=40 head=0 tail=30 "
                "render=already-applied source=unchanged",
            )
        )
        self.assertEqual(
            CheckStatus.FAIL, result(report, "S6.display-crop").status
        )

    def test_enhance_heatmap_rejects_palette_different_from_selected_mode(self):
        report = SettingsFlowValidator().validate(
            session(
                "ui:設定[hda_EnhanceHeatmap]=Warm",
                "setting route hda_EnhanceHeatmap owner=Enhance effects=None",
                "enhance heatmap mode=Warm live=cold review=gray "
                "scope=main-only data=unchanged",
            )
        )
        self.assertEqual(
            CheckStatus.FAIL, result(report, "S5.enhance-heatmap").status
        )

    def test_review_enhance_reloads_current_period_mode(self):
        report = SettingsFlowValidator().validate(
            session(
                "RV period load 2026-07-21 08:00:00.000 images=7/7 proc=False cfg=yes",
                "ui:設定[hd_EnableReviewEnhance]=True",
                "setting route hd_EnableReviewEnhance owner=Enhance effects=None",
                "RV period load 2026-07-21 08:00:00.000 images=7/7 proc=True cfg=yes",
                "RV pushFrames 7/7（merge=True, feedScale=1, chartView=keep）",
                "RV variantView keep beforeX=0.00~10.00 beforeY=20.00~0.00 "
                "afterX=0.00~10.00 afterY=20.00~0.00 maxDelta=0.000",
                "RV period curves=keep source=display",
            )
        )
        self.assertEqual(CheckStatus.PASS, result(report, "S2.review-enhance").status)

    def test_review_enhance_curve_reload_or_prefit_fails(self):
        report = SettingsFlowValidator().validate(
            session(
                "RV loadGrab done 260720-120000（20ms）",
                "ui:設定[hd_EnableReviewEnhance]=True",
                "setting route hd_EnableReviewEnhance owner=Enhance effects=None",
                "RV loadGrab begin 260720-120000（proc=True）",
                "RV prefit 260720-120000 content=100x100 viewport=50x50 viewX=0~1 viewY=0~1",
                "RV loadGrab curves=load source=bin 260720-120000",
                "RV loadGrab done 260720-120000（21ms）",
            )
        )
        self.assertEqual(
            CheckStatus.FAIL, result(report, "S2.review-enhance").status
        )

    def test_review_enhance_physical_viewport_drift_fails(self):
        report = SettingsFlowValidator().validate(
            session(
                "RV loadGrab done 260720-120000（20ms）",
                "ui:設定[hd_EnableReviewEnhance]=True",
                "setting route hd_EnableReviewEnhance owner=Enhance effects=None",
                "RV loadGrab begin 260720-120000（proc=True）",
                "RV loadGrab curves=keep source=display 260720-120000",
                "RV pushFrames 7/7（merge=True, feedScale=25, chartView=keep）",
                "RV variantView keep beforeX=0.00~10.00 beforeY=20.00~0.00 "
                "afterX=0.00~10.00 afterY=100.00~0.00 maxDelta=80.000",
                "RV loadGrab done 260720-120000（21ms）",
            )
        )
        self.assertEqual(
            CheckStatus.FAIL, result(report, "S2.review-enhance").status
        )

    def test_review_enhance_ignores_later_user_view_changes(self):
        report = SettingsFlowValidator().validate(
            session(
                "RV loadGrab done 260720-120000（20ms）",
                "ui:設定[hd_EnableReviewEnhance]=True",
                "setting route hd_EnableReviewEnhance owner=Enhance effects=None",
                "RV loadGrab begin 260720-120000（proc=True）",
                "RV loadGrab curves=keep source=display 260720-120000",
                "RV pushFrames 7/7（merge=True, feedScale=25, chartView=keep）",
                "RV variantView keep beforeX=0.00~10.00 beforeY=20.00~0.00 "
                "afterX=0.00~10.00 afterY=20.00~0.00 maxDelta=0.000",
                "RV loadGrab done 260720-120000（21ms）",
                "RV mainRange 260720-120000 viewX=2~8 viewY=18~4",
            )
        )

        self.assertEqual(
            CheckStatus.PASS, result(report, "S2.review-enhance").status
        )

    def test_image_variant_only_does_not_require_a_new_prefit(self):
        report = DataFlowValidator().validate(
            session(
                "DT curve load policy latest-only shared-loader entries=512 maxMB=256 scale=merged-only minCycleMs=33",
                "RV prefit 260720-120000 content=100x100 viewport=50x50 viewX=0~1 viewY=0~1",
                "RV mainRange 260720-120000 viewX=0~1 viewY=0~1",
                "RV chartRange 260720-120000 chart=col axis=0~1/view=0~1",
                "RV loadGrab begin 260720-120000（proc=True）",
                "RV loadGrab curves=keep source=display 260720-120000",
                "RV pushFrames 7/7（merge=True, feedScale=1, chartView=keep）",
                "RV loadGrab done 260720-120000（21ms）",
            )
        )
        self.assertEqual(CheckStatus.PASS, result(report, "D3.fit").status)

    def test_direction_without_row_refresh_fails(self):
        report = SettingsFlowValidator().validate(
            session(
                "ui:設定[hee_VerticalDirection]=BottomToTop",
                "setting route hee_VerticalDirection owner=LiveLayout effects=None",
            )
        )
        self.assertEqual(CheckStatus.FAIL, result(report, "S3.direction").status)

    def test_setting_without_route_fails(self):
        report = SettingsFlowValidator().validate(
            session("ui:設定[IoIp]=127.0.0.1")
        )
        self.assertEqual(CheckStatus.FAIL, result(report, "S0.route").status)

    def test_unrelated_setting_with_capture_policy_fails(self):
        report = SettingsFlowValidator().validate(
            session(
                "ui:設定[IoIp]=127.0.0.1",
                "setting route IoIp owner=Io effects=CapturePolicy",
            )
        )
        self.assertEqual(CheckStatus.FAIL, result(report, "S0.route").status)


class ReviewFlowValidatorTests(unittest.TestCase):
    def test_reload_latest_compares_against_current_root_listing(self):
        report = ReviewFlowValidator().validate(
            session(
                "ui:【讀取資料】鈕（Review）",
                "DT list reload range=260724-080000~260724-120000 rows=2 ms=1 source=index",
                "RV loadGrab begin 260724-120000（proc=True）",
                "ui:【讀取資料】鈕（Review）",
                "RV folder selected root=D:\\Anilox\\Captures",
                "DT list reload range=251117-111919~260721-210928 rows=100 ms=2 source=index",
                "RV loadGrab begin 260721-210928（proc=True）",
            )
        )
        self.assertEqual(
            CheckStatus.PASS, result(report, "R1.reload-latest").status
        )

    def test_reload_latest_rejects_non_latest_from_current_root(self):
        report = ReviewFlowValidator().validate(
            session(
                "ui:【讀取資料】鈕（Review）",
                "DT list reload range=260724-080000~260724-120000 rows=2 ms=1 source=index",
                "RV loadGrab begin 260724-120000（proc=True）",
                "ui:【讀取資料】鈕（Review）",
                "DT list reload range=251117-111919~260721-210928 rows=100 ms=2 source=index",
                "RV loadGrab begin 260721-200000（proc=True）",
            )
        )
        self.assertEqual(
            CheckStatus.FAIL, result(report, "R1.reload-latest").status
        )

    def test_review_thumbnail_started_frame_can_finish_before_latest_pending(self):
        report = ReviewFlowValidator().validate(
            session(
                "ui:【單片序號】→ 260723-080000",
                "RV thumbnail begin 260723-080000",
                "ui:【單片序號】→ 260723-080001",
                "RV thumbnail done 260723-080000 total=8ms decode=5ms "
                "images=7 ratio=6.4 source=atlas atlas=1920x1080",
                "RV thumbnail coalesced 260723-080001 skipped=1 minCycleMs=33",
                "RV thumbnail begin 260723-080001",
                "RV thumbnail done 260723-080001 total=7ms decode=5ms "
                "images=7 ratio=6.4 source=atlas atlas=1920x1080",
            )
        )
        self.assertEqual(CheckStatus.PASS, result(report, "R2.thumbnail").status)

    def test_review_curve_backpressure_accepts_80ms_policy_and_coalescing(self):
        report = ReviewFlowValidator().validate(
            session(
                "ui:【單片序號】→ 260804-080000",
                "RV curve load policy latest-only minCycleMs=80",
                "RV curves paths 260804-080000 root=D:\\Anilox\\Captures "
                "images=20 cams=2 cfg=yes align=summary source=summary coalesced=3",
                "RV curves 260804-080000（12ms） presentation=progressive",
                "RV curves paths 260804-080010 root=D:\\Anilox\\Captures "
                "images=20 cams=2 cfg=yes align=summary source=summary coalesced=0",
                "RV curves 260804-080010（10ms） presentation=latest",
            )
        )
        self.assertEqual(CheckStatus.PASS, result(report, "R2.backpressure").status)

    def test_review_curve_backpressure_rejects_unthrottled_policy(self):
        report = ReviewFlowValidator().validate(
            session(
                "ui:【單片序號】→ 260804-080000",
                "RV curve load policy latest-only minCycleMs=0",
                "RV curves paths 260804-080000 root=D:\\Anilox\\Captures "
                "images=20 cams=2 cfg=yes align=summary source=summary coalesced=0",
                "RV curves 260804-080000（12ms） presentation=latest",
            )
        )
        self.assertEqual(CheckStatus.FAIL, result(report, "R2.backpressure").status)

    def test_review_curve_backpressure_rejects_coalescing_without_progressive_display(self):
        report = ReviewFlowValidator().validate(
            session(
                "ui:【單片序號】→ 260804-080000",
                "RV curve load policy latest-only minCycleMs=80",
                "RV curves paths 260804-080000 root=D:\\Anilox\\Captures "
                "images=20 cams=2 cfg=yes align=summary source=summary coalesced=3",
                "RV curves 260804-080000（12ms） presentation=latest",
            )
        )
        self.assertEqual(CheckStatus.FAIL, result(report, "R2.backpressure").status)

    def test_review_thumbnail_begin_for_non_latest_selection_fails(self):
        report = ReviewFlowValidator().validate(
            session(
                "ui:【單片序號】→ 260723-080000",
                "ui:【單片序號】→ 260723-080001",
                "RV thumbnail begin 260723-080000",
                "RV thumbnail done 260723-080000 total=7ms decode=5ms "
                "images=7 ratio=6.4 source=atlas atlas=1920x1080",
            )
        )
        self.assertEqual(CheckStatus.FAIL, result(report, "R2.thumbnail").status)

    def test_review_thumbnail_cannot_publish_after_full_load_begins(self):
        report = ReviewFlowValidator().validate(
            session(
                "ui:【單片序號】→ 260723-080000",
                "RV thumbnail begin 260723-080000",
                "RV loadGrab begin 260723-080000",
                "RV thumbnail done 260723-080000 total=7ms decode=5ms "
                "images=7 ratio=6.4 source=atlas atlas=1920x1080",
            )
        )
        self.assertEqual(CheckStatus.FAIL, result(report, "R2.thumbnail").status)

    def test_review_adjacent_prefetch_accepts_announced_neighbor_hit(self):
        report = ReviewFlowValidator().validate(
            session(
                "RV prefetch begin center=260804-090000 "
                "neighbors=260804-085959|260804-090001",
                "RV prefetch ready center=260804-090000 "
                "neighbor=260804-085959 thumbnail=cold total=40ms",
                "RV thumbnail begin 260804-085959",
                "RV thumbnail done 260804-085959 total=0ms decode=6ms "
                "images=7 ratio=5.0 source=atlas cache=hit atlas=1920x1080",
            )
        )
        self.assertEqual(CheckStatus.PASS, result(report, "R2.prefetch").status)

    def test_review_adjacent_prefetch_rejects_unannounced_ready(self):
        report = ReviewFlowValidator().validate(
            session(
                "RV prefetch begin center=260804-090000 "
                "neighbors=260804-085959",
                "RV prefetch ready center=260804-090000 "
                "neighbor=260804-090001 thumbnail=cold total=40ms",
            )
        )
        self.assertEqual(CheckStatus.FAIL, result(report, "R2.prefetch").status)

    def test_review_assets_accept_archive_source(self):
        report = ReviewFlowValidator().validate(
            session(
                "RV loadGrab paths 260722-154554 root=D:\\Anilox\\Captures "
                "images=20 cams=2 cfg=yes align=tick source=acap"
            )
        )
        self.assertEqual(CheckStatus.PASS, result(report, "R2.assets").status)

    def test_review_assets_reject_empty_selection(self):
        report = ReviewFlowValidator().validate(
            session(
                "RV loadGrab paths 260722-154128 root=D:\\Anilox\\Captures "
                "images=0 cams=0 cfg=yes align=filename source=legacy"
            )
        )
        self.assertEqual(CheckStatus.FAIL, result(report, "R2.assets").status)

    def test_initial_empty_review_tab_does_not_require_visible_content(self):
        report = ReviewFlowValidator().validate(
            session(
                "ui:tab → 回顧",
                "RV tabVisible repaint view=False",
                "ui:tab → 報表",
            )
        )
        self.assertEqual(CheckStatus.PASS, result(report, "R0.tab-visible").status)

    def test_review_with_preloaded_data_requires_visible_content(self):
        report = ReviewFlowValidator().validate(
            session(
                "DT curve share 260721-080001 target=Review",
                "ui:tab → 回顧",
                "RV tabVisible repaint view=True",
                "ui:tab → 報表",
            )
        )
        self.assertEqual(CheckStatus.FAIL, result(report, "R0.tab-visible").status)


class DataFlowValidatorTests(unittest.TestCase):
    def test_column_chart_peak_matches_single_record_verdict(self):
        report = DataFlowValidator().validate(
            session(
                "DT curve display 260804-135456 mode=mean "
                "mean=0.7000/0.8000 max=0.7200/2.0000 scale=1.0000 points=128",
                "DT verdict 260804-135456 cam=1 mode=mean "
                "mean=0.7000/0.8000 enabled=1 max=0.7200/2.0000 enabled=0 "
                "result=pass cause=none source=visible-merged-curve",
            )
        )
        self.assertEqual(
            CheckStatus.PASS, result(report, "D1.chart-verdict").status
        )

    def test_column_chart_peak_rejects_hidden_verdict_peak(self):
        report = DataFlowValidator().validate(
            session(
                "DT curve display 260804-135456 mode=mean "
                "mean=0.6900/0.8000 max=0.7000/2.0000 scale=2.0000 points=500",
                "DT verdict 260804-135456 cam=1 mode=mean "
                "mean=1.3821/0.8000 enabled=1 max=1.4080/2.0000 enabled=0 "
                "result=fail cause=mean source=visible-merged-curve",
            )
        )
        self.assertEqual(
            CheckStatus.FAIL, result(report, "D1.chart-verdict").status
        )

    def test_column_verdict_uses_mean_and_max_thresholds_independently(self):
        report = DataFlowValidator().validate(
            session(
                "DT verdict 260804-135533 cam=1 mean=0.3341/0.2000 "
                "max=0.3366/0.5000 result=fail cause=mean source=visible-merged-curve",
                "DT verdict 260804-135533 cam=1 mean=0.3341/0.5000 "
                "max=0.3366/0.5000 result=pass cause=none source=merged-curve",
                "DT verdict 260804-135533 cam=1 mean=0.3341/0.5000 "
                "max=0.3366/0.3000 result=fail cause=max source=merged-curve",
                "DT verdict 260804-135533 cam=1 mean=0.3341/0.3000 "
                "max=0.3366/0.3000 result=fail cause=both source=merged-curve",
            )
        )
        self.assertEqual(CheckStatus.PASS, result(report, "D1.verdict").status)

    def test_column_verdict_rejects_curve_max_compared_to_mean_threshold(self):
        report = DataFlowValidator().validate(
            session(
                "DT verdict 260804-135533 cam=1 mean=0.1000/0.2000 "
                "max=0.4000/0.6000 result=fail cause=max source=merged-curve"
            )
        )
        self.assertEqual(CheckStatus.FAIL, result(report, "D1.verdict").status)

    def test_column_verdict_click_mean_mode_ignores_maximum(self):
        report = DataFlowValidator().validate(
            session(
                "DT verdict click 260804-135344 cam=1 mode=mean "
                "mean=0.3000/0.4000 enabled=1 max=0.9000/0.5000 enabled=0 "
                "result=pass cause=none list=pass source=curve-index",
                "DT verdict click done 260804-135344 cams=1",
            )
        )
        self.assertEqual(
            CheckStatus.PASS, result(report, "D1.verdict-click").status
        )

    def test_column_verdict_click_max_mode_ignores_mean(self):
        report = DataFlowValidator().validate(
            session(
                "DT verdict click 260804-135344 cam=1 mode=max "
                "mean=1.5000/1.2000 enabled=0 max=1.8000/2.0000 enabled=1 "
                "result=pass cause=none list=pass source=curve-index",
                "DT verdict click done 260804-135344 cams=1",
            )
        )
        self.assertEqual(
            CheckStatus.PASS, result(report, "D1.verdict-click").status
        )

    def test_column_verdict_click_rejects_list_result_mismatch(self):
        report = DataFlowValidator().validate(
            session(
                "DT verdict click 260804-135344 cam=1 mode=both "
                "mean=0.3000/0.4000 enabled=1 max=0.4500/0.5000 enabled=1 "
                "result=pass cause=none list=fail source=curve-index",
                "DT verdict click done 260804-135344 cams=1",
            )
        )
        self.assertEqual(
            CheckStatus.FAIL, result(report, "D1.verdict-click").status
        )

    def test_column_verdict_index_accounts_for_summary_bins_and_missing(self):
        report = DataFlowValidator().validate(
            session(
                "DT verdict index apply=ok gen=1 summaries=1388 bins=590 "
                "missing=5/1983 cams=3956 verdicts=3956 ms=12000"
            )
        )
        self.assertEqual(
            CheckStatus.PASS, result(report, "D1.verdict-index").status
        )

    def test_column_verdict_index_rejects_unaccounted_grabs(self):
        report = DataFlowValidator().validate(
            session(
                "DT verdict index apply=ok gen=1 summaries=1388 bins=500 "
                "missing=5/1983 cams=3900 verdicts=3900 ms=12000"
            )
        )
        self.assertEqual(
            CheckStatus.FAIL, result(report, "D1.verdict-index").status
        )

    def test_range_preview_finishes_running_then_jumps_to_latest(self):
        report = DataFlowValidator().validate(
            session(
                "DT range policy listMs=33 curveMs=80 settleMs=150 curveMode=monotonic "
                "curveSamples=50 curveCacheEntries=2048 curveCacheMB=256",
                "ui:【序號範圍-起始】變更",
                "DT range list preview gen=1 range=260721-080000~260721-080010 rows=11 ms=1 source=index",
                "ui:【序號範圍-結束】變更",
                "DT range list preview gen=2 range=260721-080000~260721-080020 rows=21 ms=1 source=index",
                "DT range preview apply gen=1 latest=2 range=260721-080000~260721-080010 loadMs=10 drawMs=2 "
                "meanRows=11 maxRows=11 method=top-maxcmean coverage=11/11 rankedCams=7/7 "
                "index=1/0 cache=0/100 hmCoverage=11/11 hmCurrent=0.3000 sampleLimit=50",
                "DT range preview apply gen=2 latest=2 range=260721-080000~260721-080020 loadMs=10 drawMs=2 "
                "meanRows=21 maxRows=21 method=top-maxcmean coverage=21/21 rankedCams=7/7 "
                "index=1/0 cache=100/0 hmCoverage=21/21 hmCurrent=0.3000 sampleLimit=50",
                "DT range settle → refresh",
            )
        )
        self.assertEqual(CheckStatus.PASS, result(report, "D3.range-policy").status)
        self.assertEqual(CheckStatus.PASS, result(report, "D3.range-preview").status)

    def test_range_preview_requires_final_generation_to_catch_up(self):
        report = DataFlowValidator().validate(
            session(
                "DT range policy listMs=33 curveMs=80 settleMs=150 curveMode=monotonic "
                "curveSamples=50 curveCacheEntries=2048 curveCacheMB=256",
                "ui:【序號範圍-結束】變更",
                "DT range list preview gen=4 range=260721-080000~260721-080020 rows=21 ms=1 source=index",
                "DT range preview apply gen=3 latest=4 range=260721-080000~260721-080010 loadMs=10 drawMs=2 "
                "meanRows=50 maxRows=50 method=top-maxcmean coverage=11/11 rankedCams=7/7 "
                "index=1/0 cache=0/100 hmCoverage=11/11 hmCurrent=0.3000 sampleLimit=50",
                "DT range settle → refresh",
            )
        )
        self.assertEqual(CheckStatus.FAIL, result(report, "D3.range-preview").status)

    def test_sustained_range_scroll_requires_visible_jump_and_final_catch_up(self):
        lines = [
            "DT range policy listMs=33 curveMs=80 settleMs=150 curveMode=monotonic "
            "curveSamples=50 curveCacheEntries=2048 curveCacheMB=256",
        ]
        for generation in range(1, 101):
            lines.append("ui:【序號範圍-結束】變更")
            lines.append(
                "DT range list preview gen={} range=260721-080000~260721-080020 "
                "rows=21 ms=1 source=index".format(generation)
            )
        lines.extend([
            "DT range preview apply gen=20 latest=80 range=260721-080000~260721-080010 "
            "loadMs=10 drawMs=2 meanRows=21 maxRows=21 method=top-maxcmean "
            "coverage=21/21 rankedCams=7/7 index=1/0 cache=0/100 "
            "hmCoverage=21/21 hmCurrent=0.3000 sampleLimit=50",
            "DT range preview apply gen=100 latest=100 range=260721-080000~260721-080020 "
            "loadMs=10 drawMs=2 meanRows=21 maxRows=21 method=top-maxcmean "
            "coverage=21/21 rankedCams=7/7 index=1/0 cache=100/0 "
            "hmCoverage=21/21 hmCurrent=0.3000 sampleLimit=50",
            "DT range settle → refresh",
        ])
        report = DataFlowValidator().validate(session(*lines))
        self.assertEqual(CheckStatus.PASS, result(report, "D3.range-preview").status)

    def test_sustained_range_scroll_rejects_final_only_curve(self):
        lines = [
            "DT range policy listMs=33 curveMs=80 settleMs=150 curveMode=monotonic "
            "curveSamples=50 curveCacheEntries=2048 curveCacheMB=256",
        ]
        for generation in range(1, 101):
            lines.append("ui:【序號範圍-結束】變更")
            lines.append(
                "DT range list preview gen={} range=260721-080000~260721-080020 "
                "rows=21 ms=1 source=index".format(generation)
            )
        lines.extend([
            "DT range preview apply gen=100 latest=100 range=260721-080000~260721-080020 "
            "loadMs=10 drawMs=2 meanRows=21 maxRows=21 method=top-maxcmean "
            "coverage=21/21 rankedCams=7/7 index=1/0 cache=100/0 "
            "hmCoverage=21/21 hmCurrent=0.3000 sampleLimit=50",
            "DT range settle → refresh",
        ])
        report = DataFlowValidator().validate(session(*lines))
        self.assertEqual(CheckStatus.FAIL, result(report, "D3.range-preview").status)

    def test_virtual_list_fallback_is_contract_failure(self):
        report = DataFlowValidator().validate(
            session("DT list virtual fallback index=42 rows=0 native=43099 reason=stale-index")
        )
        self.assertEqual(
            CheckStatus.FAIL, result(report, "D2.virtual-list").status
        )

    def test_fail_filter_requires_range_option_evidence(self):
        report = DataFlowValidator().validate(
            session(
                "ui:【篩選異常】→ 只顯示異常 dataOptions=2 rangeOptions=2 "
                "selected=260721-080001 range=260721-080001~260721-080003",
                "ui:【篩選異常】→ 顯示全部 dataOptions=4 rangeOptions=4 "
                "selected=260721-080001 range=260721-080000~260721-080003",
            )
        )
        self.assertEqual(CheckStatus.PASS, result(report, "D5.fail-filter").status)

    def test_fail_filter_rejects_data_and_range_option_mismatch(self):
        report = DataFlowValidator().validate(
            session(
                "ui:【篩選異常】→ 只顯示異常 dataOptions=3 rangeOptions=2 "
                "selected=260721-080001 range=260721-080001~260721-080003"
            )
        )
        self.assertEqual(CheckStatus.FAIL, result(report, "D5.fail-filter").status)

    def test_fail_filter_rejects_stale_list_only_log(self):
        report = DataFlowValidator().validate(
            session("ui:【篩選異常】→ 只顯示異常")
        )
        self.assertEqual(CheckStatus.FAIL, result(report, "D5.fail-filter").status)

    def test_report_to_review_reuses_presented_curves(self):
        report = DataFlowValidator().validate(
            session(
                "ui:【報表序號】→ 260721-080001",
                "DT selected 260721-080001 stats=cache list=keep ms=1",
                "DT curve share 260721-080001 target=Review",
                "DT curve load 260721-080001 captures=7 source=shared storage=summary configMs=1 waitMs=2 pathMs=0 mergeMs=0 summaryMs=1 points=100 drawMs=3 totalMs=5",
                "DT review sync apply 260721-080001",
                "RV loadGrab begin 260721-080001（proc=False）",
                "RV loadGrab curves=reuse source=Data 260721-080001",
                "RV loadGrab done 260721-080001（50ms）",
            )
        )
        self.assertEqual(CheckStatus.PASS, result(report, "D3.review-reuse").status)

    def test_report_to_review_curve_bin_reload_fails(self):
        report = DataFlowValidator().validate(
            session(
                "ui:【報表序號】→ 260721-080001",
                "DT selected 260721-080001 stats=cache list=keep ms=1",
                "DT curve share 260721-080001 target=Review",
                "DT review sync apply 260721-080001",
                "RV loadGrab begin 260721-080001（proc=False）",
                "RV loadGrab curves=load source=bin 260721-080001",
                "RV curves paths 260721-080001 root=D:\\Anilox images=7 cams=7 cfg=yes align=tick source=bins",
                "RV loadGrab done 260721-080001（50ms）",
            )
        )
        self.assertEqual(CheckStatus.FAIL, result(report, "D3.review-reuse").status)

    def test_timer_starvation_without_ping_or_stack_is_not_a_hard_ui_block(self):
        lines = [
            FlowLine(0.0, "00:00:00.000", 1, "ui:【報表序號】→ 260721-080000"),
            FlowLine(1.5, "00:00:01.500", 1, "[UiStall] 1500ms（GC0+2 GC1+1 GC2+0）"),
            FlowLine(1.51, "00:00:01.510", 1, "DT selected 260721-080000 stats=cache list=keep ms=1"),
        ]
        report = DataFlowValidator().validate(FlowSession(Path("synthetic.log"), lines))
        responsiveness = result(report, "U.stall")
        self.assertEqual(CheckStatus.PASS, responsiveness.status)
        self.assertIn("計時器飢餓=1", responsiveness.detail)

    def test_correlated_ping_and_stack_are_a_hard_ui_block(self):
        lines = [
            FlowLine(0.0, "00:00:00.000", 1, "ui:【報表序號】→ 260721-080000"),
            FlowLine(0.8, "00:00:00.800", 3, "[UiStack] BlockingCall.Wait ←"),
            FlowLine(1.5, "00:00:01.500", 1, "[UiPing] 1400ms"),
            FlowLine(1.51, "00:00:01.510", 1, "[UiStall] 1500ms（GC0+0 GC1+0 GC2+0）"),
            FlowLine(1.52, "00:00:01.520", 1, "DT selected 260721-080000 stats=cache list=keep ms=1"),
        ]
        report = DataFlowValidator().validate(FlowSession(Path("synthetic.log"), lines))
        responsiveness = result(report, "U.stall")
        self.assertEqual(CheckStatus.FAIL, responsiveness.status)
        self.assertIn("真阻塞=1", responsiveness.detail)

    def test_single_curve_latest_only_allows_stale_intermediate_and_requires_final(self):
        report = DataFlowValidator().validate(
            session(
                "DT curve load policy latest-only shared-loader entries=512 maxMB=256 scale=merged-only minCycleMs=33",
                "ui:【報表序號】→ 260721-080000",
                "DT selected 260721-080000 stats=cache list=keep ms=1",
                "ui:【報表序號】→ 260721-080001",
                "DT selected 260721-080001 stats=cache list=keep ms=1",
                "DT curve stale-drop 260721-080000",
                "DT row curve load 260721-080001 source=shared storage=summary points=100 pitch=0.010000mm",
                "DT curve load 260721-080001 captures=7 source=shared storage=summary configMs=1 waitMs=2 pathMs=0 mergeMs=0 summaryMs=1 points=100 drawMs=3 totalMs=5",
            )
        )
        self.assertEqual(CheckStatus.PASS, result(report, "D3.curve-policy").status)
        self.assertEqual(CheckStatus.PASS, result(report, "D3.curve").status)
        self.assertEqual(CheckStatus.PASS, result(report, "D3.row-curve").status)

    def test_single_curve_latest_only_fails_when_final_selection_never_applies(self):
        report = DataFlowValidator().validate(
            session(
                "DT curve load policy latest-only shared-loader entries=512 maxMB=256 scale=merged-only minCycleMs=33",
                "ui:【報表序號】→ 260721-080001",
                "DT selected 260721-080001 stats=cache list=keep ms=1",
                "DT curve stale-drop 260721-080001",
            )
        )
        self.assertEqual(CheckStatus.FAIL, result(report, "D3.curve").status)
        self.assertEqual(CheckStatus.FAIL, result(report, "D3.row-curve").status)

    def test_single_fit_matches_later_review_lod_geometry(self):
        report = DataFlowValidator().validate(
            session(
                "ui:【報表序號】→ 260721-080001",
                "DT selected 260721-080001 stats=cache list=keep ms=1",
                "DT prefit 260721-080001 content=20236x15000 viewX=-952~3422 viewY=17105~-439 source=main-geometry",
                "RV loadGrab begin 260721-080001（proc=False）",
                "RV prefit 260721-080001 content=20236x15000 viewport=1353x596 viewX=-952~3422 viewY=17105~-439",
                "RV prefitPaint 260721-080001 chart=col after=0ms axis=-952~3422/view=-900~3350",
                "RV chartRange 260721-080001 chart=col axis=-952~3422/view=-900~3350",
                "RV prefitPaint 260721-080001 chart=row after=1ms axis=-439~17105/view=-400~17000",
                "RV chartRange 260721-080001 chart=row axis=-439~17105/view=-400~17000",
                "RV prefitApply 260721-080001 after=1ms visible=True col=axis=-952~3422/view=-900~3350 row=axis=-439~17105/view=-400~17000",
                "RV mainRange 260721-080001 viewX=-952.00~3422.00 viewY=17105.00~-439.00",
                "DT chartRange 260721-080001 chart=col axis=-952~3422/view=-900~3350",
                "DT chartRange 260721-080001 chart=row axis=-439~17105/view=-400~17000",
                "RV lodRebind merge 20236x15000（fit reset）",
                "RV loadGrab done 260721-080001（100ms）",
            )
        )
        self.assertEqual(CheckStatus.PASS, result(report, "D3.fit").status)

    def test_single_fit_accepts_equivalent_hessian_standard_map_geometry(self):
        report = DataFlowValidator().validate(
            session(
                "ui:?銵典??? 260721-080001",
                "DT selected 260721-080001 stats=cache list=keep ms=1",
                "DT prefit 260721-080001 content=20000x15000 viewX=0~2000 viewY=1500~0 source=main-geometry",
                "RV loadGrab begin 260721-080001",
                "RV prefit 260721-080001 content=20000x15000 viewport=1000x600 viewX=0~2000 viewY=1500~0",
                "RV prefitPaint 260721-080001 chart=col after=0ms axis=0~2000/view=0~2000",
                "RV chartRange 260721-080001 chart=col axis=0~2000/view=0~2000",
                "RV prefitPaint 260721-080001 chart=row after=0ms axis=0~1500/view=0~1500",
                "RV chartRange 260721-080001 chart=row axis=0~1500/view=0~1500",
                "RV prefitApply 260721-080001 after=0ms visible=True col=axis=0~2000/view=0~2000 row=axis=0~1500/view=0~1500",
                "RV mainRange 260721-080001 viewX=0~2000 viewY=1500~0",
                "DT chartRange 260721-080001 chart=col axis=0~2000/view=0~2000",
                "DT chartRange 260721-080001 chart=row axis=0~1500/view=0~1500",
                "RV lodRebind merge 4000x3000",
                "RV pushFrames 2/7 merge=True, feedScale=25, chartView=publish",
                "RV loadGrab done 260721-080001",
            )
        )
        self.assertEqual(CheckStatus.PASS, result(report, "D3.fit").status)

    def test_single_fit_fails_when_review_lod_geometry_differs(self):
        report = DataFlowValidator().validate(
            session(
                "ui:【報表序號】→ 260721-080001",
                "DT selected 260721-080001 stats=cache list=keep ms=1",
                "DT prefit 260721-080001 content=20236x15000 viewX=-952~3422 viewY=17105~-439 source=main-geometry",
                "RV loadGrab begin 260721-080001（proc=False）",
                "RV prefit 260721-080001 content=20233x15000 viewport=1353x596 viewX=-952~3422 viewY=17105~-439",
                "RV prefitPaint 260721-080001 chart=col after=0ms axis=-952~3422/view=-900~3350",
                "RV chartRange 260721-080001 chart=col axis=-952~3422/view=-900~3350",
                "RV prefitPaint 260721-080001 chart=row after=1ms axis=-439~17105/view=-400~17000",
                "RV chartRange 260721-080001 chart=row axis=-439~17105/view=-400~17000",
                "RV prefitApply 260721-080001 after=1ms visible=True col=axis=-952~3422/view=-900~3350 row=axis=-439~17105/view=-400~17000",
                "RV mainRange 260721-080001 viewX=-952.00~3422.00 viewY=17105.00~-439.00",
                "DT chartRange 260721-080001 chart=col axis=-952~3422/view=-900~3350",
                "DT chartRange 260721-080001 chart=row axis=-439~17105/view=-400~17000",
                "RV lodRebind merge 20236x15000（fit reset）",
                "RV loadGrab done 260721-080001（100ms）",
            )
        )
        self.assertEqual(CheckStatus.FAIL, result(report, "D3.fit").status)

    def test_single_fit_fails_when_prefit_runs_after_image_rebind(self):
        report = DataFlowValidator().validate(
            session(
                "ui:【報表序號】→ 260721-080001",
                "DT selected 260721-080001 stats=cache list=keep ms=1",
                "DT prefit 260721-080001 content=20236x15000 viewX=-952~3422 viewY=17105~-439 source=main-geometry",
                "RV loadGrab begin 260721-080001（proc=False）",
                "RV lodRebind merge 20236x15000（fit reset）",
                "RV prefit 260721-080001 content=20236x15000 viewport=1353x596 viewX=-952~3422 viewY=17105~-439",
                "RV prefitPaint 260721-080001 chart=col after=0ms axis=-952~3422/view=-900~3350",
                "RV chartRange 260721-080001 chart=col axis=-952~3422/view=-900~3350",
                "RV prefitPaint 260721-080001 chart=row after=1ms axis=-439~17105/view=-400~17000",
                "RV chartRange 260721-080001 chart=row axis=-439~17105/view=-400~17000",
                "RV prefitApply 260721-080001 after=1ms visible=True col=axis=-952~3422/view=-900~3350 row=axis=-439~17105/view=-400~17000",
                "RV mainRange 260721-080001 viewX=-952.00~3422.00 viewY=17105.00~-439.00",
                "DT chartRange 260721-080001 chart=col axis=-952~3422/view=-900~3350",
                "DT chartRange 260721-080001 chart=row axis=-439~17105/view=-400~17000",
                "RV loadGrab done 260721-080001（100ms）",
            )
        )
        self.assertEqual(CheckStatus.FAIL, result(report, "D3.fit").status)

    def test_single_fit_fails_when_chart_paints_after_image_rebind(self):
        report = DataFlowValidator().validate(
            session(
                "ui:【報表序號】→ 260721-080001",
                "DT selected 260721-080001 stats=cache list=keep ms=1",
                "DT prefit 260721-080001 content=20236x15000 viewX=-952~3422 viewY=17105~-439 source=main-geometry",
                "RV loadGrab begin 260721-080001（proc=False）",
                "RV prefit 260721-080001 content=20236x15000 viewport=1353x596 viewX=-952~3422 viewY=17105~-439",
                "RV chartRange 260721-080001 chart=col axis=-952~3422/view=-900~3350",
                "RV chartRange 260721-080001 chart=row axis=-439~17105/view=-400~17000",
                "RV prefitApply 260721-080001 after=0ms visible=True col=axis=-952~3422/view=-900~3350 row=axis=-439~17105/view=-400~17000",
                "RV mainRange 260721-080001 viewX=-952.00~3422.00 viewY=17105.00~-439.00",
                "DT chartRange 260721-080001 chart=col axis=-952~3422/view=-900~3350",
                "DT chartRange 260721-080001 chart=row axis=-439~17105/view=-400~17000",
                "RV lodRebind merge 20236x15000（fit reset）",
                "RV prefitPaint 260721-080001 chart=col after=350ms axis=-952~3422/view=-900~3350",
                "RV prefitPaint 260721-080001 chart=row after=350ms axis=-439~17105/view=-400~17000",
                "RV loadGrab done 260721-080001（400ms）",
            )
        )
        self.assertEqual(CheckStatus.FAIL, result(report, "D3.fit").status)

    def test_single_fit_fails_when_chart_view_changes_after_prefit(self):
        report = DataFlowValidator().validate(
            session(
                "ui:【報表序號】→ 260721-080001",
                "DT selected 260721-080001 stats=cache list=keep ms=1",
                "DT prefit 260721-080001 content=20236x15000 viewX=-952~3422 viewY=17105~-439 source=main-geometry",
                "RV loadGrab begin 260721-080001（proc=False）",
                "RV prefit 260721-080001 content=20236x15000 viewport=1353x596 viewX=-952~3422 viewY=17105~-439",
                "RV prefitPaint 260721-080001 chart=col after=0ms axis=-952~3422/view=-900~3350",
                "RV chartRange 260721-080001 chart=col axis=-952~3422/view=-900~3350",
                "RV prefitPaint 260721-080001 chart=row after=0ms axis=-439~17105/view=-400~17000",
                "RV chartRange 260721-080001 chart=row axis=-439~17105/view=-400~17000",
                "RV prefitApply 260721-080001 after=0ms visible=True col=axis=-952~3422/view=-900~3350 row=axis=-439~17105/view=-400~17000",
                "RV mainRange 260721-080001 viewX=-952.00~3422.00 viewY=17105.00~-439.00",
                "DT chartRange 260721-080001 chart=col axis=-952~3422/view=-900~3350",
                "DT chartRange 260721-080001 chart=row axis=-439~17105/view=-400~17000",
                "RV chartRange 260721-080001 chart=row axis=-439~17105/view=-350~16950",
                "RV lodRebind merge 20236x15000（fit reset）",
                "RV loadGrab done 260721-080001（100ms）",
            )
        )
        self.assertEqual(CheckStatus.FAIL, result(report, "D3.fit").status)

    def test_single_fit_fails_when_fast_curve_prefit_is_overwritten_before_image_load(self):
        messages = [
            "DT selected 260721-080001 stats=cache list=keep ms=1",
            "RV curves paths 260721-080001 root=D:\\Anilox images=7 cams=7 cfg=yes align=tick source=bins",
            "RV prefit 260721-080001 content=20236x15000 viewport=1353x596 viewX=-952~3422 viewY=17105~-439",
            "RV prefitPaint 260721-080001 chart=col after=0ms axis=-952~3422/view=-900~3350",
            "RV chartRange 260721-080001 chart=col axis=-952~3422/view=-900~3350",
            "RV prefitPaint 260721-080001 chart=row after=0ms axis=-439~17105/view=-400~17000",
            "RV chartRange 260721-080001 chart=row axis=-439~17105/view=-400~17000",
            "RV prefitApply 260721-080001 after=0ms visible=True col=axis=-952~3422/view=-900~3350 row=axis=-439~17105/view=-400~17000",
            "RV layout intent 260721-080001 images=7 cams=7 align=tick before=curves",
            "RV curves 260721-080001 (10ms)",
            "RV loadGrab begin 260721-080001 (proc=False)",
            "RV prefit 260721-080001 content=20236x15000 viewport=1353x596 viewX=-952~3422 viewY=17105~-439",
            "RV prefitPaint 260721-080001 chart=col after=0ms axis=-952~3422/view=-900~3350",
            "RV chartRange 260721-080001 chart=col axis=-952~3422/view=-900~3350",
            "RV prefitPaint 260721-080001 chart=row after=0ms axis=-439~17105/view=-400~17000",
            "RV chartRange 260721-080001 chart=row axis=-439~17105/view=-400~17000",
            "RV prefitApply 260721-080001 after=0ms visible=True col=axis=-952~3422/view=-900~3350 row=axis=-439~17105/view=-400~17000",
            "RV mainRange 260721-080001 viewX=-952.00~3422.00 viewY=17105.00~-439.00",
            "RV lodRebind merge 20236x15000 (fit reset)",
            "RV fit(record-change)",
            "RV pushFrames 7/7 (merge=True, feedScale=5, chartView=publish)",
            "RV loadGrab done 260721-080001 (100ms)",
        ]
        report = DataFlowValidator().validate(session(*messages))
        self.assertEqual(CheckStatus.PASS, result(report, "D3.fit").status)

        messages.insert(
            10,
            "RV chartRange 260721-080001 chart=row axis=-439~17105/view=-350~16950",
        )
        report = DataFlowValidator().validate(session(*messages))
        self.assertEqual(CheckStatus.FAIL, result(report, "D3.fit").status)

    def test_single_fit_fails_when_chart_axis_changes_but_view_stays_equal(self):
        report = DataFlowValidator().validate(
            session(
                "DT selected 260721-080001 stats=cache list=keep ms=1",
                "RV loadGrab begin 260721-080001（proc=False）",
                "RV prefit 260721-080001 content=20236x15000 viewport=1353x596 viewX=-952~3422 viewY=17105~-439",
                "RV prefitPaint 260721-080001 chart=col after=0ms axis=-952~3422/view=-900~3350",
                "RV chartRange 260721-080001 chart=col axis=-952~3422/view=-900~3350",
                "RV prefitPaint 260721-080001 chart=row after=0ms axis=-439~17105/view=-400~17000",
                "RV chartRange 260721-080001 chart=row axis=-439~17105/view=-400~17000",
                "RV prefitApply 260721-080001 after=0ms visible=True col=axis=-952~3422/view=-900~3350 row=axis=-439~17105/view=-400~17000",
                "RV mainRange 260721-080001 viewX=-952.00~3422.00 viewY=17105.00~-439.00",
                "RV chartRange 260721-080001 chart=row axis=0~22000/view=-400~17000",
                "RV lodRebind merge 20236x15000（fit reset）",
                "RV loadGrab done 260721-080001（100ms）",
            )
        )
        self.assertEqual(CheckStatus.FAIL, result(report, "D3.fit").status)

    def test_single_fit_requires_layout_before_successful_curve(self):
        messages = [
            "DT selected 260721-080001 stats=cache list=keep ms=1",
            "RV curves paths 260721-080001 root=D:\\Anilox images=7 cams=7 cfg=yes align=tick source=bins",
            "RV prefit 260721-080001 content=20236x15000 viewport=1353x596 viewX=-952~3422 viewY=17105~-439",
            "RV layout intent 260721-080001 images=7 cams=7 align=tick before=curves",
            "RV curves 260721-080001（10ms）",
            "RV loadGrab begin 260721-080001（proc=False）",
            "RV prefit 260721-080001 content=20236x15000 viewport=1353x596 viewX=-952~3422 viewY=17105~-439",
            "RV prefitPaint 260721-080001 chart=col after=0ms axis=-952~3422/view=-900~3350",
            "RV chartRange 260721-080001 chart=col axis=-952~3422/view=-900~3350",
            "RV prefitPaint 260721-080001 chart=row after=0ms axis=-439~17105/view=-400~17000",
            "RV chartRange 260721-080001 chart=row axis=-439~17105/view=-400~17000",
            "RV prefitApply 260721-080001 after=0ms visible=True col=axis=-952~3422/view=-900~3350 row=axis=-439~17105/view=-400~17000",
            "RV mainRange 260721-080001 viewX=-952.00~3422.00 viewY=17105.00~-439.00",
            "RV lodRebind merge 20236x15000（fit reset）",
            "RV loadGrab done 260721-080001（100ms）",
        ]
        report = DataFlowValidator().validate(session(*messages))
        self.assertEqual(CheckStatus.PASS, result(report, "D3.fit").status)

        messages[3:4] = []
        report = DataFlowValidator().validate(session(*messages))
        self.assertEqual(CheckStatus.FAIL, result(report, "D3.fit").status)

    def test_report_single_fit_precedes_curve_and_covers_both_charts(self):
        report = DataFlowValidator().validate(
            session(
                "ui:【報表序號】→ 260721-080001",
                "DT selected 260721-080001 stats=cache list=keep ms=1",
                "DT prefit 260721-080001 content=20236x15000 viewX=-952~3422 viewY=17105~-439 source=main-geometry",
                "DT chartRange 260721-080001 chart=col axis=-952~3422/view=-900~3350",
                "DT chartRange 260721-080001 chart=row axis=-439~17105/view=-400~17000",
                "DT curve load 260721-080001 captures=7 source=shared storage=summary configMs=1 waitMs=2 pathMs=0 mergeMs=0 summaryMs=1 points=100 drawMs=3 totalMs=5",
            )
        )
        self.assertEqual(CheckStatus.PASS, result(report, "D3.fit").status)

    def test_report_single_fit_stops_tracking_after_range_mode(self):
        report = DataFlowValidator().validate(
            session(
                "ui:【報表序號】→ 260721-080001",
                "DT prefit 260721-080001 content=20236x15000 viewX=-952~3422 viewY=17105~-439 source=main-geometry",
                "DT chartRange 260721-080001 chart=col axis=-952~3422/view=-900~3350",
                "DT chartRange 260721-080001 chart=row axis=-439~17105/view=-400~17000",
                "DT curve load 260721-080001 captures=7 source=shared storage=summary configMs=1 waitMs=2 pathMs=0 mergeMs=0 summaryMs=1 points=100 drawMs=3 totalMs=5",
                "ui:【期間-日】→ 範圍 260721-080001~260721-090001",
                "DT chartRange 260721-080001 chart=col axis=0~2470/view=0~2470",
            )
        )
        self.assertEqual(CheckStatus.PASS, result(report, "D3.fit").status)

    def test_report_single_fit_accepts_reentrant_paint_before_final_intent(self):
        report = DataFlowValidator().validate(
            session(
                "ui:【報表序號】→ 260721-080001",
                "ui:【報表序號】→ 260721-080002",
                "DT chartRange 260721-080003 chart=col axis=-952~3422/view=-900~3350",
                "ui:【報表序號】→ 260721-080003",
                "DT prefit 260721-080003 content=20236x15000 viewX=-952~3422 viewY=17105~-439 source=main-geometry",
                "DT chartRange 260721-080003 chart=row axis=-439~17105/view=-400~17000",
                "DT curve load 260721-080003 captures=7 source=shared storage=summary configMs=1 waitMs=2 pathMs=0 mergeMs=0 summaryMs=1 points=100 drawMs=3 totalMs=5",
            )
        )
        self.assertEqual(CheckStatus.PASS, result(report, "D3.fit").status)

    def test_review_image_load_ignores_newer_curve_prefit(self):
        report = DataFlowValidator().validate(
            session(
                "DT prefit 260721-080002 content=20236x3000 viewX=-65~2535 viewY=6880~-3546 source=main-geometry",
                "RV loadGrab begin 260721-080001（proc=False）",
                "RV prefit 260721-080002 content=20236x3000 viewport=1353x596 viewX=-65~2535 viewY=6880~-3546",
                "RV prefitPaint 260721-080002 chart=col after=0ms axis=-13~2365/view=-13~2365",
                "RV prefitApply 260721-080002 after=0ms visible=True col=axis=-13~2365/view=-13~2365 row=axis=-2503~6618/view=-2503~6618",
                "RV mainRange 260721-080002 viewX=-65~2535 viewY=6880~-3546",
                "RV chartRange 260721-080002 chart=col axis=-13~2365/view=-13~2365",
                "RV loadGrab stale-drop 260721-080001（20ms）",
            )
        )
        self.assertEqual(CheckStatus.PASS, result(report, "D3.fit").status)

    def test_report_single_fit_fails_when_prefit_follows_curve(self):
        report = DataFlowValidator().validate(
            session(
                "ui:【報表序號】→ 260721-080001",
                "DT selected 260721-080001 stats=cache list=keep ms=1",
                "DT chartRange 260721-080001 chart=col axis=-952~3422/view=-900~3350",
                "DT chartRange 260721-080001 chart=row axis=-439~17105/view=-400~17000",
                "DT curve load 260721-080001 captures=7 source=shared storage=summary configMs=1 waitMs=2 pathMs=0 mergeMs=0 summaryMs=1 points=100 drawMs=3 totalMs=5",
                "DT prefit 260721-080001 content=20236x15000 viewX=-952~3422 viewY=17105~-439 source=main-geometry",
            )
        )
        self.assertEqual(CheckStatus.FAIL, result(report, "D3.fit").status)

class MuraFlowValidatorTests(unittest.TestCase):
    def test_edges_health_and_pause_sequences_pass(self):
        report = MuraFlowValidator().validate(
            session(
                "StartGrab（cams=4）",
                "⚠ MURA 超標（v）mean=0.30/max=0.70（thr 0.20/0.60，IO已連線）",
                "[OutputHealth] raise code=MuraExceed.v severity=Critical message=檢測異常（欄）",
                "MURA 恢復（v）",
                "[OutputHealth] resolve code=MuraExceed.v message=檢測異常（欄）",
                "ui:【暫停Mura檢測】鈕",
                "set:[MuraDetectPaused]=True",
                "MURA 暫停 → 清除 DO1",
                "ui:【暫停Mura檢測】鈕",
                "set:[MuraDetectPaused]=False",
            )
        )
        self.assertEqual(CheckStatus.PASS, result(report, "M1.edges").status)
        self.assertEqual(CheckStatus.PASS, result(report, "M1.health").status)
        self.assertEqual(CheckStatus.PASS, result(report, "M1.pause").status)

    def test_duplicate_edge_fails_and_legacy_health_is_not_covered(self):
        report = MuraFlowValidator().validate(
            session(
                "StartGrab（cams=4）",
                "⚠ MURA 超標（h）mean=0.30/max=0.70（thr 0.20/0.60，IO未連線→僅畫面警告）",
                "⚠ MURA 超標（h）mean=0.31/max=0.71（thr 0.20/0.60，IO未連線→僅畫面警告）",
            )
        )
        self.assertEqual(CheckStatus.FAIL, result(report, "M1.edges").status)
        self.assertEqual(
            CheckStatus.NOT_COVERED, result(report, "M1.health").status
        )

    def test_missing_health_resolve_fails(self):
        report = MuraFlowValidator().validate(
            session(
                "StartGrab（cams=4）",
                "⚠ MURA 超標（v）mean=0.30/max=0.70（thr 0.20/0.60，IO已連線）",
                "[OutputHealth] raise code=MuraExceed.v severity=Critical message=檢測異常（欄）",
                "MURA 恢復（v）",
            )
        )
        self.assertEqual(CheckStatus.PASS, result(report, "M1.edges").status)
        self.assertEqual(CheckStatus.FAIL, result(report, "M1.health").status)


class ParameterFlowValidatorTests(unittest.TestCase):
    def test_user_adjustment_after_startup_passes(self):
        report = ParameterFlowValidator().validate(
            session(
                "AllocateCameras begin（expect 7 cams）",
                "AllocateCameras done（配置 4、在線 4/7）",
                "idle",
                "ui:【相機參數】cam2 Height=4000",
                "[UiStall] 200ms（GC0+0 GC1+0 GC2+0）",
            )
        )
        self.assertEqual(CheckStatus.PASS, result(report, "P1.startup").status)
        self.assertEqual(CheckStatus.PASS, result(report, "P1.intent").status)
        self.assertEqual(
            CheckStatus.PASS, result(report, "P1.responsiveness").status
        )

    def test_debounced_initialization_intent_fails_startup(self):
        lines = [
            FlowLine(0.0, "00:00:00.000", 1, "AllocateCameras begin（expect 7 cams）"),
            FlowLine(0.5, "00:00:00.500", 1, "AllocateCameras done（配置 4、在線 4/7）"),
            FlowLine(0.7, "00:00:00.700", 1, "ui:【相機參數】All HeightAll=3001"),
        ]
        report = ParameterFlowValidator().validate(
            FlowSession(Path("synthetic.log"), lines)
        )
        self.assertEqual(CheckStatus.FAIL, result(report, "P1.startup").status)

    def test_live_exposure_uses_fast_path_without_reconfiguration(self):
        report = ParameterFlowValidator().validate(
            session(
                "capture gate open cams=2 warm=True",
                "ui:【相機參數】cam2 Exp=4000",
                "exposure live apply begin scope=cam2 gate=open",
                "exposure live apply complete scope=cam2 gate=open elapsedMs=320",
            )
        )
        self.assertEqual(
            CheckStatus.PASS, result(report, "P1.synchronization").status
        )

    def test_live_exposure_reconfiguration_path_fails(self):
        report = ParameterFlowValidator().validate(
            session(
                "capture gate open cams=2 warm=True",
                "ui:【相機參數】All ExpAll=6000",
                "parameter reconfigure begin scope=All gate=closed targets=2",
            )
        )
        self.assertEqual(
            CheckStatus.FAIL, result(report, "P1.synchronization").status
        )

    def test_live_exposure_fast_path_can_finish_after_stop(self):
        report = ParameterFlowValidator().validate(
            session(
                "capture gate open cams=2 warm=True",
                "ui:【相機參數】All ExpAll=3000",
                "exposure live apply begin scope=All gate=open",
                "StopGrab",
                "capture gate closed standby=on",
                "exposure live apply complete scope=All gate=closed elapsedMs=450",
            )
        )
        self.assertEqual(
            CheckStatus.PASS, result(report, "P1.synchronization").status
        )

    def test_live_exposure_over_five_seconds_fails(self):
        report = ParameterFlowValidator().validate(
            session(
                "capture gate open cams=2 warm=True",
                "ui:【相機參數】All ExpAll=3000",
                "exposure live apply begin scope=All gate=open",
                "exposure live apply complete scope=All gate=open elapsedMs=5001",
            )
        )
        self.assertEqual(
            CheckStatus.FAIL, result(report, "P1.synchronization").status
        )

    def test_live_line_rate_or_height_intent_fails_policy(self):
        report = ParameterFlowValidator().validate(
            session(
                "capture gate open cams=2 warm=True",
                "ui:【相機參數】cam2 LineRate=6000",
            )
        )
        self.assertEqual(
            CheckStatus.FAIL, result(report, "P1.live-policy").status
        )
        self.assertEqual(
            CheckStatus.NOT_COVERED,
            result(report, "P1.synchronization").status,
        )

    def test_live_backend_block_is_valid_policy_evidence(self):
        report = ParameterFlowValidator().validate(
            session(
                "capture gate open cams=2 warm=True",
                "parameter change blocked scope=cam2 param=Height reason=GrabActive",
            )
        )
        self.assertEqual(
            CheckStatus.PASS, result(report, "P1.live-policy").status
        )

    def test_registry_has_no_pending_domains(self):
        self.assertEqual((), PENDING_DOMAINS)


class LiveStandbyFlowValidatorTests(unittest.TestCase):
    def test_standard_background_binding_and_first_frame_evidence_pass(self):
        report = LiveFlowValidator().validate(
            session(
                "background bind cam1 mode=standard source=bg1.bin "
                "status=ready width=16384 samples=16384 min=1 max=2 mean=1.5",
                "background bind cam2 mode=standard source=bg2.bin "
                "status=ready width=16384 samples=16384 min=1 max=2 mean=1.5",
                "capture plan grab=260728-150000 root=D:\\Anilox",
                "capture gate open cams=2 warm=True",
                "background apply cam1 grab=260728-150000 mode=standard "
                "source=precomputed width=16384",
                "background apply cam2 grab=260728-150000 mode=standard "
                "source=precomputed width=16384",
                "StopGrab",
            )
        )
        self.assertEqual(
            CheckStatus.PASS,
            result(report, "F8.background-subtraction").status,
        )

    def test_standard_background_silent_fallback_fails(self):
        report = LiveFlowValidator().validate(
            session(
                "background bind cam1 mode=standard source=bg1.bin "
                "status=ready width=16384 samples=16384 min=1 max=2 mean=1.5",
                "capture plan grab=260728-150000 root=D:\\Anilox",
                "capture gate open cams=1 warm=True",
                "background apply cam1 grab=260728-150000 mode=single "
                "source=per-frame width=16384",
                "StopGrab",
            )
        )
        self.assertEqual(
            CheckStatus.FAIL,
            result(report, "F8.background-subtraction").status,
        )

    def test_background_evidence_requires_every_connected_camera(self):
        report = LiveFlowValidator().validate(
            session(
                "background bind cam1 mode=single source=per-frame status=ready",
                "background bind cam2 mode=single source=per-frame status=ready",
                "capture plan grab=260728-150000 root=D:\\Anilox",
                "capture gate open cams=2 warm=True",
                "background apply cam1 grab=260728-150000 mode=single "
                "source=per-frame width=16384",
                "StopGrab",
            )
        )
        self.assertEqual(
            CheckStatus.FAIL,
            result(report, "F8.background-subtraction").status,
        )

    def test_standard_background_skips_offline_camera_without_failure(self):
        report = LiveFlowValidator().validate(
            session(
                "background bind cam1 mode=standard source=bg1.bin "
                "status=ready width=16384 samples=16384 min=1 max=2 mean=1.5",
                "background bind cam2 mode=standard source=none "
                "status=skipped reason=offline",
                "capture plan grab=260728-150000 root=D:\\Anilox",
                "capture gate open cams=1 warm=True",
                "background apply cam1 grab=260728-150000 mode=standard "
                "source=precomputed width=16384",
                "StopGrab",
            )
        )
        self.assertEqual(
            CheckStatus.PASS,
            result(report, "F8.background-subtraction").status,
        )

    def test_background_capture_disables_product_output(self):
        report = LiveFlowValidator().validate(
            session(
                "background capture begin output=disabled",
                "capture gate open cams=2 warm=True",
                "capture first-set ready path=verified-standby cams=1,2 aligned=True",
                "background capture sampling start duration=3s",
                "background apply cam1 grab=none mode=standard "
                "source=precomputed width=16384",
                "background apply cam2 grab=none mode=standard "
                "source=precomputed width=16384",
                "background capture sampling complete durationMs=3007 "
                "frames=cam1:30,cam2:30",
                "background capture end output=disabled result=ok",
            )
        )
        self.assertEqual(
            CheckStatus.PASS,
            result(report, "F8.background-capture").status,
        )

    def test_background_capture_short_timed_sample_fails(self):
        report = LiveFlowValidator().validate(
            session(
                "background capture begin output=disabled",
                "background capture sampling start duration=3s",
                "background capture sampling complete durationMs=1900 "
                "frames=cam1:19",
                "background capture end output=disabled result=ok",
            )
        )
        self.assertEqual(
            CheckStatus.FAIL,
            result(report, "F8.background-capture").status,
        )

    def test_background_capture_product_write_attempt_fails(self):
        report = LiveFlowValidator().validate(
            session(
                "background capture begin output=disabled",
                "background capture sampling start duration=3s",
                "background capture sampling complete durationMs=3000 "
                "frames=cam1:30",
                "background capture end output=disabled result=ok",
                "[OutputHealth] raise code=CaptureWriteFailure.CAM1 "
                "severity=OutputFault message=unexpected",
            )
        )
        self.assertEqual(
            CheckStatus.FAIL,
            result(report, "F8.background-capture").status,
        )

    def test_background_preview_clears_and_suppresses_row_chart(self):
        report = LiveFlowValidator().validate(
            session(
                "EnterBackgroundPreview（view=True merge=True mode=WF設定）",
                "background preview rowChart clear",
                "bgPreview push cam1 16384x3000（view=True）",
                "ExitBackgroundPreview",
            )
        )
        self.assertEqual(
            CheckStatus.PASS,
            result(report, "F8.background-preview-row").status,
        )

    def test_background_preview_row_curve_presentation_fails(self):
        report = LiveFlowValidator().validate(
            session(
                "EnterBackgroundPreview（view=True merge=True mode=WF設定）",
                "background preview rowChart clear",
                "bgPreview push cam1 16384x3000（view=True）",
                "rowCurve present after=mainImage cams=2 mode=WF",
            )
        )
        self.assertEqual(
            CheckStatus.FAIL,
            result(report, "F8.background-preview-row").status,
        )

    def test_time_stop_arms_only_after_aligned_first_set(self):
        report = LiveFlowValidator().validate(
            session(
                "grab stop waiting condition=Time configured=10s "
                "source=manual grab=260728-150000",
                "capture gate open cams=2 warm=True",
                "capture first-set ready path=verified-standby cams=1,2 aligned=True",
                "grab stop armed condition=Time limit=10s configured=10s "
                "grace=0s source=manual start=first-set grab=260728-150000",
                "StopGrab",
            )
        )
        self.assertEqual(
            CheckStatus.PASS,
            result(report, "F2.time-origin").status,
        )

    def test_time_stop_before_first_set_fails(self):
        report = LiveFlowValidator().validate(
            session(
                "grab stop waiting condition=Time configured=10s "
                "source=manual grab=260728-150000",
                "capture gate open cams=2 warm=True",
                "grab stop armed condition=Time limit=10s configured=10s "
                "grace=0s source=manual start=first-set grab=260728-150000",
                "capture first-set ready path=verified-standby cams=1,2 aligned=True",
                "StopGrab",
            )
        )
        self.assertEqual(
            CheckStatus.FAIL,
            result(report, "F2.time-origin").status,
        )

    def test_head_guard_drops_one_callback_per_camera_before_first_set(self):
        report = LiveFlowValidator().validate(
            session(
                "experiment build=mil-edge-coverage-v8",
                "capture gate open cams=2 warm=True path=verified-standby",
                "capture head frame dropped cam1 tick=100 reason=cross-boundary",
                "capture head frame dropped cam2 tick=102 reason=cross-boundary",
                "capture first-set ready path=verified-standby "
                "cams=1,2 aligned=True",
            )
        )
        self.assertEqual(
            CheckStatus.PASS, result(report, "F2.head-guard").status
        )

    def test_head_guard_missing_camera_before_first_set_fails(self):
        report = LiveFlowValidator().validate(
            session(
                "experiment build=mil-edge-coverage-v8",
                "capture gate open cams=2 warm=True path=verified-standby",
                "capture head frame dropped cam1 tick=100 reason=cross-boundary",
                "capture first-set ready path=verified-standby "
                "cams=1,2 aligned=True",
            )
        )
        self.assertEqual(
            CheckStatus.FAIL, result(report, "F2.head-guard").status
        )

    def test_head_guard_duplicate_camera_fails(self):
        report = LiveFlowValidator().validate(
            session(
                "experiment build=mil-edge-coverage-v8",
                "capture gate open cams=2 warm=True path=verified-standby",
                "capture head frame dropped cam1 tick=100 reason=cross-boundary",
                "capture head frame dropped cam1 tick=101 reason=cross-boundary",
                "capture head frame dropped cam2 tick=102 reason=cross-boundary",
                "capture first-set ready path=verified-standby "
                "cams=1,2 aligned=True",
            )
        )
        self.assertEqual(
            CheckStatus.FAIL, result(report, "F2.head-guard").status
        )

    def test_head_phase_guard_allows_first_set_only_after_aligned_probe(self):
        report = LiveFlowValidator().validate(
            session(
                "capture gate open cams=2 warm=True path=verified-standby",
                "capture head frame dropped cam1 tick=100 reason=cross-boundary",
                "capture head frame dropped cam2 tick=102 reason=cross-boundary",
                "capture head guard path=verified-standby cams=1,2 aligned=True",
                "capture first-set ready path=verified-standby "
                "cams=1,2 aligned=True",
            )
        )
        self.assertEqual(
            CheckStatus.PASS, result(report, "F2.head-guard").status
        )

    def test_head_phase_guard_rejection_stops_without_first_set(self):
        report = LiveFlowValidator().validate(
            session(
                "capture gate open cams=2 warm=True path=verified-standby",
                "capture head frame dropped cam1 tick=100 reason=cross-boundary",
                "capture head frame dropped cam2 tick=120 reason=cross-boundary",
                "capture head guard path=verified-standby cams=1,2 aligned=False",
                "StopGrab",
            )
        )
        self.assertEqual(
            CheckStatus.PASS, result(report, "F2.head-guard").status
        )

    def test_head_phase_guard_rejection_followed_by_first_set_fails(self):
        report = LiveFlowValidator().validate(
            session(
                "capture gate open cams=2 warm=True path=verified-standby",
                "capture head frame dropped cam1 tick=100 reason=cross-boundary",
                "capture head frame dropped cam2 tick=120 reason=cross-boundary",
                "capture head guard path=verified-standby cams=1,2 aligned=False",
                "capture first-set ready path=verified-standby "
                "cams=1,2 aligned=True",
            )
        )
        self.assertEqual(
            CheckStatus.FAIL, result(report, "F2.head-guard").status
        )

    def test_head_phase_guard_rejection_without_stop_fails(self):
        report = LiveFlowValidator().validate(
            session(
                "capture gate open cams=2 warm=True path=verified-standby",
                "capture head frame dropped cam1 tick=100 reason=cross-boundary",
                "capture head frame dropped cam2 tick=120 reason=cross-boundary",
                "capture head guard path=verified-standby cams=1,2 aligned=False",
            )
        )
        self.assertEqual(
            CheckStatus.FAIL, result(report, "F2.head-guard").status
        )

    def test_head_phase_guard_rejection_followed_by_product_output_fails(self):
        report = LiveFlowValidator().validate(
            session(
                "capture gate open cams=2 warm=True path=verified-standby",
                "capture head frame dropped cam1 tick=100 reason=cross-boundary",
                "capture head frame dropped cam2 tick=120 reason=cross-boundary",
                "capture head guard path=verified-standby cams=1,2 aligned=False",
                "firstFrame cam1 100x100 -> Waterfall",
                "StopGrab",
            )
        )
        self.assertEqual(
            CheckStatus.FAIL, result(report, "F2.head-guard").status
        )

    def test_verified_standby_io_edges_and_complete_tail_pass(self):
        report = LiveFlowValidator().validate(
            session(
                "acquisition parameters ready cam1 cl=True lineRate=3000",
                "acquisition parameters ready cam2 cl=True lineRate=3000",
                "acquisition standby start cam1",
                "acquisition standby start cam2",
                "acquisition standby ready cam1 tick=100",
                "acquisition standby ready cam2 tick=102",
                "acquisition phase verified reason=idle-sync cams=2",
                "acquisition start path=verified-standby cams=2",
                "StartGrab cams=2",
                "capture plan grab=260723-120000 root=D:\\Anilox",
                "capture gate open cams=2 warm=True path=verified-standby",
                "capture first-set phase system=0 cams=1,2 spreadTicks=2 "
                "spreadMs=2.000 limitMs=5.000 aligned=True",
                "capture first-set ready path=verified-standby cams=1,2 aligned=True",
                "firstFrame cam1 100x100 -> Waterfall",
                "firstFrame cam2 100x100 -> Waterfall",
                "capture tail begin cams=1,2 timeoutMs=2000",
                "capture tail accepted cam1 tick=200",
                "capture tail complete cam1 tick=200",
                "capture tail accepted cam2 tick=202",
                "capture tail complete cam2 tick=202",
                "capture tail complete pending=",
                "StopGrab",
                "capture gate closed standby=on",
            )
        )
        self.assertEqual(CheckStatus.PASS, result(report, "F2.standby").status)

    def test_verified_standby_without_phase_proof_fails(self):
        report = LiveFlowValidator().validate(
            session(
                "acquisition parameters ready cam1 cl=True lineRate=3000",
                "acquisition standby start cam1",
                "acquisition standby ready cam1 tick=100",
                "acquisition start path=verified-standby cams=1",
                "StartGrab cams=1",
                "capture plan grab=260723-120000 root=D:\\Anilox",
                "capture gate open cams=1 warm=True path=verified-standby",
            )
        )
        self.assertEqual(CheckStatus.FAIL, result(report, "F2.standby").status)

    def test_io_tail_timeout_fails(self):
        report = LiveFlowValidator().validate(
            session(
                "acquisition parameters ready cam1 cl=True lineRate=3000",
                "acquisition standby start cam1",
                "acquisition standby ready cam1 tick=100",
                "acquisition phase verified reason=idle-sync cams=1",
                "acquisition start path=verified-standby cams=1",
                "StartGrab cams=1",
                "capture plan grab=260723-120000 root=D:\\Anilox",
                "capture gate open cams=1 warm=True path=verified-standby",
                "capture tail begin cams=1 timeoutMs=2000",
                "capture tail timeout pending=1 elapsedMs=2000",
                "StopGrab",
                "capture gate closed standby=on",
            )
        )
        self.assertEqual(CheckStatus.FAIL, result(report, "F2.standby").status)

    def test_row_chart_waits_for_main_image_presentation(self):
        report = LiveFlowValidator().validate(
            session(
                "rowCurve present after=mainImage cams=2 mode=WF",
                "LC row rowChart dir=BottomToTop n=100 total=10mm view 0~10 "
                "dataPhys 0~5mm dataChart 0~5",
            )
        )
        self.assertEqual(
            CheckStatus.PASS, result(report, "F2.row-presentation").status
        )

    def test_content_aware_zoom_floor_passes(self):
        report = LiveFlowValidator().validate(
            session(
                "WF wheelZoom out → zoom=0.00002（fit=0.01 min=0.00002 "
                "content=100000x50000）",
            )
        )
        self.assertEqual(
            CheckStatus.PASS, result(report, "F6.zoom-floor").status
        )

    def test_every_capture_resets_charts_before_start(self):
        report = LiveFlowValidator().validate(
            session(
                "capture charts reset reason=start-grab",
                "StartGrab cams=2",
                "capture charts reset reason=start-grab",
                "StartGrab cams=2",
            )
        )
        self.assertEqual(
            CheckStatus.PASS, result(report, "F2.chart-reset").status
        )

    def test_capture_without_chart_reset_fails(self):
        report = LiveFlowValidator().validate(session("StartGrab cams=2"))
        self.assertEqual(
            CheckStatus.FAIL, result(report, "F2.chart-reset").status
        )

    def test_content_aware_zoom_floor_accepts_logged_rounding(self):
        report = LiveFlowValidator().validate(
            session(
                "RV wheelZoom out → zoom=0.00003（fit=0.01271 min=0.00003 "
                "content=101171x30000）",
            )
        )
        self.assertEqual(
            CheckStatus.PASS, result(report, "F6.zoom-floor").status
        )

    def test_legacy_fixed_zoom_floor_fails(self):
        report = LiveFlowValidator().validate(
            session(
                "WF wheelZoom out → zoom=0.01（fit=0.01）",
            )
        )
        self.assertEqual(
            CheckStatus.FAIL, result(report, "F6.zoom-floor").status
        )

    def test_capture_view_range_is_refired_before_gate_opens(self):
        report = LiveFlowValidator().validate(
            session(
                "StartGrab cams=2",
                "viewRange refire reason=capture-start mode=WF",
                "capture gate open cams=2 warm=True",
            )
        )
        self.assertEqual(
            CheckStatus.PASS, result(report, "F2.view-refire").status
        )

    def test_capture_gate_before_view_range_refire_fails(self):
        report = LiveFlowValidator().validate(
            session(
                "StartGrab cams=2",
                "capture gate open cams=2 warm=True",
            )
        )
        self.assertEqual(
            CheckStatus.FAIL, result(report, "F2.view-refire").status
        )

    def test_waterfall_bootstrap_period_precedes_each_start(self):
        report = LiveFlowValidator().validate(
            session(
                "ApplyMainDisplayMode → Waterfall",
                "WF bootstrap period cam1 periodMs=500.000 source=applied-hardware",
                "StartGrab cams=2",
            )
        )
        self.assertEqual(
            CheckStatus.PASS, result(report, "F2.waterfall-bootstrap").status
        )

    def test_waterfall_runtime_period_learning_fails_new_contract(self):
        report = LiveFlowValidator().validate(
            session(
                "ApplyMainDisplayMode → Waterfall",
                "WF bootstrap period unavailable; learn from runtime frames",
                "StartGrab cams=2",
            )
        )
        self.assertEqual(
            CheckStatus.FAIL, result(report, "F2.waterfall-bootstrap").status
        )

    def test_waterfall_first_band_requires_every_expected_camera(self):
        report = LiveFlowValidator().validate(
            session(
                "WF band first generation=16 seq=0 cams=2 expected=1,2 "
                "ticks=1002~1002 startRow=0 height=3000 reason=complete",
            )
        )
        self.assertEqual(
            CheckStatus.FAIL,
            result(report, "F2.waterfall-first-band").status,
        )

    def test_waterfall_first_band_complete_camera_set_passes(self):
        report = LiveFlowValidator().validate(
            session(
                "WF band first generation=16 seq=0 cams=1,2 expected=1,2 "
                "ticks=1000~1002 startRow=0 height=3000 reason=complete",
            )
        )
        self.assertEqual(
            CheckStatus.PASS,
            result(report, "F2.waterfall-first-band").status,
        )

    def test_row_chart_without_main_image_presentation_fails(self):
        report = LiveFlowValidator().validate(
            session(
                "LC row rowChart dir=BottomToTop n=100 total=10mm view 0~10 "
                "dataPhys 0~5mm dataChart 0~5",
            )
        )
        self.assertEqual(
            CheckStatus.FAIL, result(report, "F2.row-presentation").status
        )

    def test_initialization_ignores_later_height_reallocation_metrics(self):
        lines = [
            FlowLine(0.0, "00:00:00.000", 1, "AllocateCameras begin (expect 2 cams)"),
            FlowLine(1.0, "00:00:01.000", 1, "camera init cam=1 phase=acquisition ms=10 size=10x10 thread=1"),
            FlowLine(2.0, "00:00:02.000", 1, "camera init cam=2 phase=acquisition ms=10 size=10x10 thread=1"),
            FlowLine(3.0, "00:00:03.000", 1, "camera init phase=acquisition done cams=2 ms=20"),
            FlowLine(4.0, "00:00:04.000", 1, "camera init phase=processing begin cams=2"),
            FlowLine(5.0, "00:00:05.000", 15, "camera init cam=1 phase=processing ms=10 pinnedMB=1 allocCalls=2 thread=15"),
            FlowLine(6.0, "00:00:06.000", 15, "camera init cam=2 phase=processing ms=10 pinnedMB=1 allocCalls=2 thread=15"),
            FlowLine(7.0, "00:00:07.000", 1, "camera init phase=processing done cams=2 ms=20"),
            FlowLine(8.0, "00:00:08.000", 1, "camera init summary acquisition=20ms processing=20ms total=40ms"),
            FlowLine(9.0, "00:00:09.000", 29, "camera init cam=1 phase=processing ms=8 pinnedMB=1 allocCalls=2 thread=29"),
        ]
        report = LiveFlowValidator().validate(
            FlowSession(Path("synthetic.log"), lines)
        )
        self.assertEqual(CheckStatus.PASS, result(report, "F1.init").status)

    def test_warm_ready_gate_start_and_stop_pass(self):
        report = LiveFlowValidator().validate(
            session(
                "acquisition parameters ready cam1 cl=True lineRate=3001",
                "acquisition parameters ready cam2 cl=True lineRate=3001",
                "acquisition standby start cam1",
                "acquisition standby start cam2",
                "acquisition standby ready cam1 tick=100",
                "acquisition standby ready cam2 tick=102",
                "acquisition sync begin reason=start attempt=1 gate=closed cams=2",
                "acquisition sync paused reason=start attempt=1 cams=2",
                "acquisition sync resumed reason=start attempt=1 cams=2",
                "acquisition sync ready reason=start attempt=1 cam1 system=0 tick=100 freq=1000",
                "acquisition sync ready reason=start attempt=1 cam2 system=0 tick=102 freq=1000",
                "acquisition sync phase reason=start attempt=1 system=0 cams=1,2 spreadTicks=2 spreadMs=2.000 limitMs=5.000 measurable=True aligned=True sampleSource=warm-snapshot",
                "acquisition sync complete reason=start attempts=1 cams=2 phase=True",
                "StartGrab cams=4",
                "capture plan grab=260720-120000 root=D:\\Anilox",
                "capture gate open cams=2 warm=True",
                "firstFrame cam1 100x100 -> ImageDisplayView",
                "firstFrame cam2 100x100 -> ImageDisplayView",
                "StopGrab",
                "capture gate closed standby=on",
                "drop drainedFrame after StopGrab cam1",
            )
        )
        self.assertEqual(CheckStatus.PASS, result(report, "F2.standby").status)

    def test_warm_standby_without_physical_start_sync_fails(self):
        report = LiveFlowValidator().validate(
            session(
                "acquisition parameters ready cam1 cl=True lineRate=3000",
                "acquisition parameters ready cam2 cl=True lineRate=3000",
                "acquisition standby start cam1",
                "acquisition standby start cam2",
                "acquisition standby ready cam1 tick=100",
                "acquisition standby ready cam2 tick=200",
                "StartGrab cams=2",
                "capture plan grab=260720-120000 root=D:\\Anilox",
                "capture gate open cams=2 warm=True",
                "firstFrame cam1 100x100 -> Waterfall",
                "firstFrame cam2 100x100 -> Waterfall",
                "StopGrab",
                "capture gate closed standby=on",
            )
        )
        self.assertEqual(CheckStatus.FAIL, result(report, "F2.standby").status)

    def test_start_phase_retry_can_succeed(self):
        report = LiveFlowValidator().validate(
            session(
                "acquisition parameters ready cam1 cl=True lineRate=3000",
                "acquisition parameters ready cam2 cl=True lineRate=3000",
                "acquisition standby start cam1",
                "acquisition standby start cam2",
                "acquisition standby ready cam1 tick=100",
                "acquisition standby ready cam2 tick=200",
                "acquisition sync begin reason=start attempt=1 gate=closed cams=2",
                "acquisition sync phase reason=start attempt=1 system=0 cams=1,2 spreadTicks=20 spreadMs=20.000 limitMs=5.000 measurable=True aligned=False sampleSource=warm-snapshot",
                "acquisition sync retry reason=start attempt=1 error=PhaseOutOfRange",
                "acquisition sync begin reason=start attempt=2 gate=closed cams=2",
                "acquisition sync ready reason=start attempt=2 cam1 system=0 tick=200 freq=1000",
                "acquisition sync ready reason=start attempt=2 cam2 system=0 tick=202 freq=1000",
                "acquisition sync phase reason=start attempt=2 system=0 cams=1,2 spreadTicks=2 spreadMs=2.000 limitMs=5.000 measurable=True aligned=True sampleSource=warm-snapshot",
                "acquisition sync complete reason=start attempts=2 cams=2 phase=True",
                "StartGrab cams=2",
                "capture plan grab=260720-120000 root=D:\\Anilox",
                "capture gate open cams=2 warm=True",
                "firstFrame cam1 100x100 -> Waterfall",
                "firstFrame cam2 100x100 -> Waterfall",
                "StopGrab",
                "capture gate closed standby=on",
            )
        )
        self.assertEqual(CheckStatus.PASS, result(report, "F2.standby").status)

    def test_gate_before_all_cameras_are_ready_fails(self):
        report = LiveFlowValidator().validate(
            session(
                "acquisition parameters ready cam1 cl=True lineRate=3001",
                "acquisition parameters ready cam2 cl=True lineRate=3001",
                "acquisition standby start cam1",
                "acquisition standby start cam2",
                "acquisition standby ready cam1 tick=100",
                "StartGrab cams=4",
                "capture plan grab=260720-120000 root=D:\\Anilox",
                "capture gate open cams=2 warm=False",
            )
        )
        self.assertEqual(CheckStatus.FAIL, result(report, "F2.standby").status)

    def test_exposure_fast_path_keeps_capture_gate_open(self):
        report = LiveFlowValidator().validate(
            session(
                "acquisition parameters ready cam1 cl=True lineRate=3001",
                "acquisition standby start cam1",
                "acquisition standby ready cam1 tick=100",
                "acquisition sync begin reason=start attempt=1 gate=closed cams=1",
                "acquisition sync paused reason=start attempt=1 cams=1",
                "acquisition sync resumed reason=start attempt=1 cams=1",
                "acquisition sync ready reason=start attempt=1 cam1 system=0 tick=100 freq=1000",
                "acquisition sync phase reason=start attempt=1 system=0 cams=1 spreadTicks=0 spreadMs=0.000 limitMs=5.000 measurable=True aligned=True sampleSource=warm-snapshot",
                "acquisition sync complete reason=start attempts=1 cams=1 phase=True",
                "StartGrab cams=1",
                "capture plan grab=260720-120000 root=D:\\Anilox",
                "capture gate open cams=1 warm=True",
                "exposure live apply begin scope=cam1 gate=open",
                "exposure live apply complete scope=cam1 gate=open elapsedMs=300",
                "StopGrab",
                "capture gate closed standby=on",
            )
        )
        self.assertEqual(CheckStatus.PASS, result(report, "F2.standby").status)

    def test_gate_before_capture_plan_fails(self):
        report = LiveFlowValidator().validate(
            session(
                "acquisition parameters ready cam1 cl=True lineRate=3001",
                "acquisition standby ready cam1 tick=100",
                "StartGrab cams=4",
                "capture gate open cams=1 warm=True",
            )
        )
        self.assertEqual(CheckStatus.FAIL, result(report, "F2.standby").status)

    def test_standby_before_parameter_work_completes_fails(self):
        report = LiveFlowValidator().validate(
            session(
                "acquisition standby start cam1",
                "acquisition standby ready cam1 tick=100",
                "StartGrab cams=4",
                "capture plan grab=260720-120000 root=D:\\Anilox",
                "capture gate open cams=1 warm=True",
            )
        )
        self.assertEqual(CheckStatus.FAIL, result(report, "F2.standby").status)
