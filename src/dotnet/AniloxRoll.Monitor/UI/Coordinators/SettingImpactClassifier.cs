using System;
using System.Collections.Generic;
using AniloxRoll.Monitor.Core.Data;

namespace AniloxRoll.Monitor.UI.Coordinators
{
    internal enum SettingFeatureOwner
    {
        None,
        AppRole,
        LiveLayout,
        ChartScale,
        DataStats,
        Light,
        Io,
        Storage,
        Logging,
        Enhance,
        MuraPause,
        Background
    }

    [Flags]
    internal enum SettingImpact
    {
        None = 0,
        InspectionService = 1 << 0,
        CapturePolicy = 1 << 1,
        ColumnThresholds = 1 << 2,
        RowThresholds = 1 << 3,
        RowPitch = 1 << 4,
        ReviewCurves = 1 << 5
    }

    internal struct SettingRoute
    {
        public SettingFeatureOwner Owner { get; }
        public SettingImpact Impacts { get; }

        public SettingRoute(SettingFeatureOwner owner, SettingImpact impacts = SettingImpact.None)
        {
            Owner = owner;
            Impacts = impacts;
        }

        public string ToLogText()
        {
            string effects = Impacts == SettingImpact.None
                ? nameof(SettingImpact.None)
                : Impacts.ToString().Replace(", ", "+");
            return $"owner={Owner} effects={effects}";
        }
    }

    /// <summary>
    /// Single decision table for SettingsHub runtime side effects.
    /// A route names one feature owner plus any cross-feature impacts.
    /// </summary>
    internal static class SettingImpactClassifier
    {
        private static readonly Dictionary<string, SettingRoute> Routes = BuildRoutes();

        public static IEnumerable<string> KnownSettingNames => Routes.Keys;

        public static SettingRoute Classify(string name)
        {
            return name != null && Routes.TryGetValue(name, out SettingRoute route)
                ? route
                : new SettingRoute(SettingFeatureOwner.None);
        }

        private static Dictionary<string, SettingRoute> BuildRoutes()
        {
            var routes = new Dictionary<string, SettingRoute>(StringComparer.Ordinal);

            Add(routes, nameof(InspectionSettings.AppRole), SettingFeatureOwner.AppRole);

            AddMany(routes, SettingFeatureOwner.LiveLayout,
                nameof(InspectionSettings.ab_OpsCam1),
                nameof(InspectionSettings.ac_OpsCam2),
                nameof(InspectionSettings.ad_OpsCam3),
                nameof(InspectionSettings.ae_OpsCam4),
                nameof(InspectionSettings.af_OpsCam5),
                nameof(InspectionSettings.ag_OpsCam6),
                nameof(InspectionSettings.ah_OpsCam7),
                nameof(InspectionSettings.bb_StartCam1),
                nameof(InspectionSettings.bc_StartCam2),
                nameof(InspectionSettings.bd_StartCam3),
                nameof(InspectionSettings.be_StartCam4),
                nameof(InspectionSettings.bf_StartCam5),
                nameof(InspectionSettings.bg_StartCam6),
                nameof(InspectionSettings.bh_StartCam7),
                nameof(InspectionSettings.he_MainDisplay),
                nameof(InspectionSettings.hee_VerticalDirection),
                nameof(InspectionSettings.hf_LiveLod),
                nameof(InspectionSettings.hg_WaterfallTotalHeight),
                nameof(InspectionSettings.hh_WaterfallFullMode));

            Add(routes, nameof(InspectionSettings.ai_OpsSpeed), SettingFeatureOwner.None,
                SettingImpact.RowPitch);
            AddMany(routes, SettingFeatureOwner.None,
                nameof(InspectionSettings.cb_CropHead),
                nameof(InspectionSettings.cc_CropTail),
                nameof(InspectionSettings.fb_BackgroundSampleSeconds),
                nameof(InspectionSettings.fc_GrabLimitSeconds),
                nameof(InspectionSettings.RemotePath),
                nameof(InspectionSettings.LightWarmupMs));

            Add(routes, nameof(InspectionSettings.db_Algorithm), SettingFeatureOwner.Background);
            Add(routes, nameof(InspectionSettings.dc_HessianMaxFactorV), SettingFeatureOwner.DataStats,
                SettingImpact.InspectionService | SettingImpact.CapturePolicy | SettingImpact.ReviewCurves);
            Add(routes, nameof(InspectionSettings.dd_HessianMaxFactorH), SettingFeatureOwner.DataStats,
                SettingImpact.ReviewCurves);
            Add(routes, nameof(InspectionSettings.de_RidgeSigma), SettingFeatureOwner.None,
                SettingImpact.CapturePolicy);
            Add(routes, nameof(InspectionSettings.eb_RidgeDir), SettingFeatureOwner.DataStats,
                SettingImpact.InspectionService | SettingImpact.CapturePolicy | SettingImpact.ReviewCurves);
            Add(routes, nameof(InspectionSettings.ec_ErrorValueMeanV), SettingFeatureOwner.DataStats,
                SettingImpact.InspectionService | SettingImpact.ColumnThresholds | SettingImpact.ReviewCurves);
            Add(routes, nameof(InspectionSettings.ed_ErrorValueMaxV), SettingFeatureOwner.DataStats,
                SettingImpact.InspectionService | SettingImpact.ColumnThresholds | SettingImpact.ReviewCurves);
            Add(routes, nameof(InspectionSettings.ee_ErrorValueMeanH), SettingFeatureOwner.DataStats,
                SettingImpact.RowThresholds | SettingImpact.ReviewCurves);
            Add(routes, nameof(InspectionSettings.ef_ErrorValueMaxH), SettingFeatureOwner.DataStats,
                SettingImpact.RowThresholds | SettingImpact.ReviewCurves);

            AddMany(routes, SettingFeatureOwner.ChartScale,
                nameof(InspectionSettings.gb_ChartScaleMode),
                nameof(InspectionSettings.gc_YearlyYMax),
                nameof(InspectionSettings.gd_MonthlyYMax),
                nameof(InspectionSettings.ge_DailyYMax));
            AddMany(routes, SettingFeatureOwner.Enhance,
                nameof(InspectionSettings.hc_EnableMuraEnhance),
                nameof(InspectionSettings.hd_EnableReviewEnhance));

            Add(routes, nameof(InspectionSettings.AniloxRootPath), SettingFeatureOwner.None,
                SettingImpact.CapturePolicy);
            Add(routes, nameof(InspectionSettings.EnableAutoCapture), SettingFeatureOwner.None,
                SettingImpact.CapturePolicy);
            Add(routes, nameof(InspectionSettings.SaveOriginalBmp), SettingFeatureOwner.None,
                SettingImpact.CapturePolicy);
            AddMany(routes, SettingFeatureOwner.Storage,
                nameof(InspectionSettings.LocalMinFreeGB));
            AddMany(routes, SettingFeatureOwner.Logging,
                nameof(InspectionSettings.LogMode),
                nameof(InspectionSettings.LogRetentionHours));

            AddMany(routes, SettingFeatureOwner.Light,
                nameof(InspectionSettings.LightEnabled),
                nameof(InspectionSettings.LightComPort),
                nameof(InspectionSettings.LightChannel),
                nameof(InspectionSettings.LightBrightness));
            AddMany(routes, SettingFeatureOwner.Io,
                nameof(InspectionSettings.IoEnabled),
                nameof(InspectionSettings.IoIp),
                nameof(InspectionSettings.IoPort),
                nameof(InspectionSettings.IoModel));
            Add(routes, nameof(InspectionSettings.MuraDetectPaused), SettingFeatureOwner.MuraPause);

            return routes;
        }

        private static void Add(
            IDictionary<string, SettingRoute> routes,
            string name,
            SettingFeatureOwner owner,
            SettingImpact impacts = SettingImpact.None)
        {
            routes.Add(name, new SettingRoute(owner, impacts));
        }

        private static void AddMany(
            IDictionary<string, SettingRoute> routes,
            SettingFeatureOwner owner,
            params string[] names)
        {
            foreach (string name in names)
                Add(routes, name, owner);
        }
    }
}
