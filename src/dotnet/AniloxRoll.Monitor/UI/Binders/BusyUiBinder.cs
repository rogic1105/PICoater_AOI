using System;
using System.Windows.Forms;

namespace AniloxRoll.Monitor.UI.Binders
{
    /// <summary>
    /// Applies one feature's busy state to its form cursor and command buttons.
    /// </summary>
    public sealed class BusyUiBinder
    {
        private readonly Form _form;
        private readonly Button[] _buttonsToLock;

        public BusyUiBinder(Form form, params Button[] buttonsToLock)
        {
            _form = form ?? throw new ArgumentNullException(nameof(form));
            _buttonsToLock = buttonsToLock ?? Array.Empty<Button>();
        }

        public void SetBusy(bool isBusy)
        {
            if (_form.IsDisposed || !_form.IsHandleCreated) return;
            if (_form.InvokeRequired)
            {
                try { _form.Invoke(new Action<bool>(SetBusy), isBusy); }
                catch (InvalidOperationException) { /* Form closing or already disposed. */ }
                return;
            }

            _form.Cursor = isBusy ? Cursors.WaitCursor : Cursors.Default;
            foreach (var button in _buttonsToLock)
                button.Enabled = !isBusy;
        }
    }
}
