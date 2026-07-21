using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Windows.Forms;
using AniloxRoll.Monitor.Core.Services;
using AniloxRoll.Monitor.UI.Navigators;
using AniloxRoll.Monitor.UI.Presenters;
using NUnit.Framework;

namespace AniloxRoll.Monitor.Tests
{
    [TestFixture]
    [Apartment(ApartmentState.STA)]
    public class DataStatisticsPresenterTests
    {
        [Test]
        public void SelectFailRangeInfos_ReturnsOnlyCameraOrRowFailuresInOriginalOrder()
        {
            var infos = new List<GrabIdInfo>
            {
                new GrabIdInfo { GrabId = "260721-080003" },
                new GrabIdInfo { GrabId = "260721-080002" },
                new GrabIdInfo { GrabId = "260721-080001" },
                new GrabIdInfo { GrabId = "260721-080000" },
            };
            var cameraFail = new GrabDetail { GrabId = "260721-080003" };
            cameraFail.CamResult[4] = true;
            var pass = new GrabDetail { GrabId = "260721-080002", RowResult = false };
            pass.CamResult[0] = false;
            var rowFail = new GrabDetail { GrabId = "260721-080001", RowResult = true };
            var details = new Dictionary<string, GrabDetail>
            {
                [cameraFail.GrabId] = cameraFail,
                [pass.GrabId] = pass,
                [rowFail.GrabId] = rowFail,
            };

            List<GrabIdInfo> result =
                DataStatisticsPresenter.SelectFailRangeInfos(infos, details);

            CollectionAssert.AreEqual(
                new[] { "260721-080003", "260721-080001" },
                result.Select(info => info.GrabId).ToArray());
        }

        [Test]
        public void Navigator_UsesFilteredDataSelectorsAndKeepsReviewOptionsComplete()
        {
            var all = new List<GrabIdInfo>
            {
                new GrabIdInfo { GrabId = "260721-080003" },
                new GrabIdInfo { GrabId = "260721-080002" },
                new GrabIdInfo { GrabId = "260721-080001" },
            };
            var failures = new List<GrabIdInfo> { all[0], all[2] };
            var context = new DataStatisticsContext
            {
                CbGrabIdStart = new ComboBox(),
                CbGrabIdEnd = new ComboBox(),
                CbDataGrabId = new ComboBox(),
                CbReviewGrabId = new ComboBox(),
                GroupBoxGrabIdRange = new GroupBox(),
                GrpDataSingleSheet = new GroupBox(),
                GrpReviewGrabNav = new GroupBox(),
            };
            string selectedFromData = null;
            int selectedAllIndex = -1;
            var navigator = new DataDateGrabIdNavigator(
                context, () => all, () => failures,
                () => { }, () => { },
                (id, earliest, latest, index) =>
                {
                    selectedFromData = id;
                    selectedAllIndex = index;
                },
                (id, earliest, latest, index) => { },
                (box, active) => { }, (label, active) => { });

            navigator.WireEvents();
            navigator.PopulateAllGrabIdCombos();

            CollectionAssert.AreEqual(
                new[] { "260721-080003", "260721-080001" },
                context.CbDataGrabId.Items.Cast<string>().ToArray());
            Assert.That(context.CbReviewGrabId.Items.Count, Is.EqualTo(3));
            CollectionAssert.AreEqual(
                new[] { "260721-080003", "260721-080001" },
                context.CbGrabIdStart.Items.Cast<string>().ToArray());
            Assert.That(navigator.TryGetSelectedRange(out List<GrabIdInfo> selected), Is.True);
            CollectionAssert.AreEqual(
                new[] { "260721-080003", "260721-080001" },
                selected.Select(info => info.GrabId).ToArray());

            context.CbDataGrabId.SelectedIndex = 1;
            Assert.That(selectedFromData, Is.EqualTo("260721-080001"));
            Assert.That(selectedAllIndex, Is.EqualTo(2));

            navigator.SyncDataGrabIdFromReview(1, all[1]);
            Assert.That(context.CbDataGrabId.SelectedItem, Is.EqualTo("260721-080001"));
        }

        [Test]
        public void RefreshFilteredGrabIdCombos_SelectsNearestAndThenPreservesExactSelection()
        {
            var all = new List<GrabIdInfo>
            {
                new GrabIdInfo { GrabId = "260721-080003" },
                new GrabIdInfo { GrabId = "260721-080002" },
                new GrabIdInfo { GrabId = "260721-080001" },
                new GrabIdInfo { GrabId = "260721-080000" },
            };
            List<GrabIdInfo> visible = all;
            var context = new DataStatisticsContext
            {
                CbGrabIdStart = new ComboBox(),
                CbGrabIdEnd = new ComboBox(),
                CbDataGrabId = new ComboBox(),
                CbReviewGrabId = new ComboBox(),
                GroupBoxGrabIdRange = new GroupBox(),
                GrpDataSingleSheet = new GroupBox(),
            };
            var navigator = new DataDateGrabIdNavigator(
                context, () => all, () => visible,
                () => { }, () => { },
                (id, earliest, latest, index) => { },
                (id, earliest, latest, index) => { },
                (box, active) => { }, (label, active) => { });

            navigator.PopulateAllGrabIdCombos();
            context.CbDataGrabId.SelectedIndex = 2;
            string preferred = context.CbDataGrabId.SelectedItem.ToString();

            visible = new List<GrabIdInfo> { all[0], all[3] };
            navigator.RefreshFilteredGrabIdCombos(preferred);
            Assert.That(context.CbDataGrabId.SelectedItem, Is.EqualTo("260721-080000"));

            preferred = context.CbDataGrabId.SelectedItem.ToString();
            visible = all;
            navigator.RefreshFilteredGrabIdCombos(preferred);
            Assert.That(context.CbDataGrabId.SelectedItem, Is.EqualTo("260721-080000"));
        }
    }
}
