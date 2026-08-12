using System.Collections.Generic;
using System.Drawing;
using System.Threading;
using System.Windows.Forms;
using AniloxRoll.Monitor.Core.Services;
using AniloxRoll.Monitor.UI.Binders;
using NUnit.Framework;

namespace AniloxRoll.Monitor.Tests
{
    [TestFixture]
    [Apartment(ApartmentState.STA)]
    public class GrabDetailListBinderTests
    {
        [Test]
        public void SetItems_CopiesInputAndDisposeDrainsVirtualList()
        {
            using (var listView = new ListView())
            {
                var source = new List<GrabDetail>
                {
                    new GrabDetail { GrabId = "260805-080000" }
                };
                var binder = new GrabDetailListBinder(listView, 7);
                binder.Initialize();

                binder.SetItems(source);
                source.Clear();

                Assert.That(listView.VirtualListSize, Is.EqualTo(1));
                binder.Dispose();
                Assert.That(listView.VirtualListSize, Is.EqualTo(0));
            }
        }

        [Test]
        public void SetItems_MalformedCameraResultsStillPublishesVirtualRows()
        {
            using (var listView = new ListView())
            {
                var binder = new GrabDetailListBinder(listView, 7);
                binder.Initialize();
                binder.SetItems(new List<GrabDetail> { null });

                Assert.That(listView.VirtualListSize, Is.EqualTo(1));
                binder.Dispose();
            }
        }

        [Test]
        public void ResultColors_AreIndependentForColumnRowAndDisabledAxes()
        {
            Color pass = GrabDetailListBinder.GetResultBackColor(false);
            Color fail = GrabDetailListBinder.GetResultBackColor(true);
            Color disabled = GrabDetailListBinder.GetResultBackColor(null);

            Assert.That(pass, Is.Not.EqualTo(fail));
            Assert.That(disabled, Is.Not.EqualTo(pass));
            Assert.That(disabled, Is.Not.EqualTo(fail));
        }
    }
}
