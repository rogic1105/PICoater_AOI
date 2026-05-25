using System;
using System.Drawing;
using System.Threading;       // 用於 Thread.Sleep
using System.Threading.Tasks; // 用於 Task
using System.Windows.Forms;
using Matrox.MatroxImagingLibrary;

namespace AOI.SDK.MILGrab
{
    public partial class Form1 : Form
    {
        // =========================================================================
        // 設定與資源定義
        // =========================================================================
        private string DcfFilePath1 = @"C:\Users\User\Downloads\dcf\Radient eV-CL Dual-Base-Digitizer0_FreeRun.dcf";
        private string DcfFilePath2 = @"C:\Users\User\Downloads\dcf\Radient eV-CL Dual-Base-Digitizer1_FreeRun.dcf";

        private MIL_ID MilApplication = MIL.M_NULL;
        private MIL_ID MilSystem = MIL.M_NULL;

        private MIL_ID MilDigitizer1 = MIL.M_NULL;
        private MIL_ID MilDisplay1 = MIL.M_NULL;
        private MIL_ID MilImage1 = MIL.M_NULL;
        private bool isLive1 = false; // 代表目前是否「真正」在抓取

        private MIL_ID MilDigitizer2 = MIL.M_NULL;
        private MIL_ID MilDisplay2 = MIL.M_NULL;
        private MIL_ID MilImage2 = MIL.M_NULL;
        private bool isLive2 = false;

        private MIL_DIG_HOOK_FUNCTION_PTR CameraStatusHandlerDelegate;
        private System.Windows.Forms.Timer statusTimer;

        // 【新功能】總開關：記錄使用者是否按下了 Button2 (是否允許抓取)
        // 預設為 false，這樣 Button1 初始化後就不會自動跑
        private bool UserWantsGrab = false;

        // 【新功能】釋放鎖：當正在釋放資源時，禁止 Timer 去讀取相機，防止卡死
        private volatile bool isReleasing = false;

        public Form1()
        {
            InitializeComponent();
            CameraStatusHandlerDelegate = new MIL_DIG_HOOK_FUNCTION_PTR(CameraStatusHandler);

            statusTimer = new System.Windows.Forms.Timer();
            statusTimer.Interval = 500;
            statusTimer.Tick += StatusTimer_Tick;

            UpdateStatusLabel(1, false);
            UpdateStatusLabel(2, false);
        }

        // =========================================================================
        // Timer: 更新 UI (增加鎖定機制防止卡死)
        // =========================================================================
        private void StatusTimer_Tick(object sender, EventArgs e)
        {
            // 如果正在釋放資源，絕對不要去碰 MIL ID
            if (isReleasing) return;

            CheckCameraPresence(MilDigitizer1, 1);
            CheckCameraPresence(MilDigitizer2, 2);
        }

        private void CheckCameraPresence(MIL_ID dig, int id)
        {
            if (dig == MIL.M_NULL)
            {
                UpdateStatusLabel(id, false);
                return;
            }

            // 使用 try-catch 包覆，因為在斷線邊緣 MdigInquire 可能會報錯
            try
            {
                MIL_INT presence = 0;
                MIL.MdigInquire(dig, MIL.M_CAMERA_PRESENT, ref presence);
                bool isOnline = (presence == MIL.M_YES || presence == MIL.M_SUPPORTED);
                UpdateStatusLabel(id, isOnline);
            }
            catch
            {
                // 如果出錯，視為離線
                UpdateStatusLabel(id, false);
            }
        }

        // =========================================================================
        // Button 1: 初始化 (只準備，不取像)
        // =========================================================================
        private void button1_Click(object sender, EventArgs e)
        {
            if (MilApplication != MIL.M_NULL) return;

            isReleasing = false; // 重置釋放鎖
            UserWantsGrab = false; // 初始化時，預設不抓取

            MIL.MappAllocDefault(MIL.M_DEFAULT, ref MilApplication, ref MilSystem, MIL.M_NULL, MIL.M_NULL, MIL.M_NULL);
            MIL.MappControl(MIL.M_DEFAULT, MIL.M_ERROR, MIL.M_PRINT_DISABLE);

            InitCamera(MIL.M_DEV0, DcfFilePath1, ref MilDigitizer1, ref MilDisplay1, ref MilImage1, panel1.Handle, 1);
            InitCamera(MIL.M_DEV1, DcfFilePath2, ref MilDigitizer2, ref MilDisplay2, ref MilImage2, panel2.Handle, 2);

            statusTimer.Start();
        }

        private void InitCamera(MIL_INT devNum, string dcfPath, ref MIL_ID milDig, ref MIL_ID milDisp, ref MIL_ID milImg, IntPtr panelHandle, int id)
        {
            MIL.MdigAlloc(MilSystem, devNum, dcfPath, MIL.M_DEFAULT, ref milDig);
            if (milDig != MIL.M_NULL)
            {
                MIL.MdigControl(milDig, MIL.M_GRAB_TIMEOUT, 1000); // 延長一點 Timeout 給相機反應

                MIL.MdispAlloc(MilSystem, MIL.M_DEFAULT, "M_DEFAULT", MIL.M_DEFAULT, ref milDisp);
                MIL_INT sizeX = MIL.MdigInquire(milDig, MIL.M_SIZE_X, MIL.M_NULL);
                MIL_INT sizeY = MIL.MdigInquire(milDig, MIL.M_SIZE_Y, MIL.M_NULL);
                MIL.MbufAlloc2d(MilSystem, sizeX, sizeY, 8 + MIL.M_UNSIGNED, MIL.M_IMAGE + MIL.M_GRAB + MIL.M_DISP, ref milImg);
                MIL.MbufClear(milImg, 0);
                MIL.MdispSelectWindow(milDisp, milImg, panelHandle);
                MIL.MdispControl(milDisp, MIL.M_SCALE_DISPLAY, MIL.M_ENABLE);

                // 註冊 Hook
                MIL.MdigHookFunction(milDig, MIL.M_CAMERA_PRESENT, CameraStatusHandlerDelegate, (IntPtr)id);
            }
        }

        // =========================================================================
        // Button 2: 總開關 (控制 UserWantsGrab)
        // =========================================================================
        private void button2_Click(object sender, EventArgs e)
        {
            // 切換使用者的意圖：想看 or 不想看
            UserWantsGrab = !UserWantsGrab;

            ApplyGrabState(1);
            ApplyGrabState(2);
        }

        private void ApplyGrabState(int id)
        {
            MIL_ID dig = (id == 1) ? MilDigitizer1 : MilDigitizer2;
            MIL_ID img = (id == 1) ? MilImage1 : MilImage2;
            ref bool isLive = ref (id == 1 ? ref isLive1 : ref isLive2);

            if (dig == MIL.M_NULL) return;

            try
            {
                if (UserWantsGrab)
                {
                    // 使用者想看，且目前沒在跑 -> 啟動
                    if (!isLive)
                    {
                        // 先檢查相機是否真的在線上，避免報錯
                        MIL_INT presence = 0;
                        MIL.MdigInquire(dig, MIL.M_CAMERA_PRESENT, ref presence);
                        if (presence == MIL.M_YES || presence == MIL.M_SUPPORTED)
                        {
                            MIL.MdigHalt(dig); // 保險起見 Halt 一次
                            MIL.MdigGrabContinuous(dig, img);
                            isLive = true;
                        }
                    }
                }
                else
                {
                    // 使用者不想看 -> 停止
                    if (isLive)
                    {
                        MIL.MdigHalt(dig);
                        isLive = false;
                    }
                }
            }
            catch { /* 忽略錯誤 */ }
        }

        // =========================================================================
        // Hook: 自動重連處理 (含延遲功能)
        // =========================================================================
        private MIL_INT CameraStatusHandler(MIL_INT HookType, MIL_ID EventId, IntPtr UserPtr)
        {
            // 如果正在釋放資源，Hook 直接退出，不要搗亂
            if (isReleasing) return MIL.M_NULL;

            int cameraId = UserPtr.ToInt32();
            MIL_ID currentDig = (cameraId == 1) ? MilDigitizer1 : MilDigitizer2;
            MIL_ID currentImg = (cameraId == 1) ? MilImage1 : MilImage2;

            if (currentDig == MIL.M_NULL) return MIL.M_NULL;

            MIL_INT presence = 0;
            MIL.MdigInquire(currentDig, MIL.M_CAMERA_PRESENT, ref presence);

            if (presence == MIL.M_YES || presence == MIL.M_SUPPORTED)
            {
                // =============================================================
                // 解決條碼問題：相機剛上電需要時間 Sync
                // =============================================================
                MIL.MdigHalt(currentDig); // 先停，清除舊 Buffer

                // 讓 Hook 執行緒睡 1.5 秒，等待相機影像穩定
                // Hook 是跑在背景執行緒，所以這不會卡住你的 UI (視窗還可以動)
                Thread.Sleep(1500);

                // =============================================================
                // 解決自動亂跑問題：檢查 UserWantsGrab
                // =============================================================
                if (UserWantsGrab)
                {
                    MIL.MdigGrabContinuous(currentDig, currentImg);
                    if (cameraId == 1) isLive1 = true; else isLive2 = true;
                }
            }
            else
            {
                // 斷線
                MIL.MdigHalt(currentDig);
                if (cameraId == 1) isLive1 = false; else isLive2 = false;
            }

            return MIL.M_NULL;
        }

        private void UpdateStatusLabel(int cameraId, bool isConnected)
        {
            // UI 更新邏輯 (省略 Invoke 檢查，與之前相同)
            if (this.InvokeRequired)
            {
                this.BeginInvoke(new Action(() => UpdateStatusLabel(cameraId, isConnected)));
                return;
            }
            Label targetLabel = (cameraId == 1) ? label1 : label2;
            string statusText = isConnected ? $"Camera {cameraId}: Online" : $"Camera {cameraId}: Offline";
            Color backColor = isConnected ? Color.LightGreen : Color.Red;

            if (targetLabel.Text != statusText)
            {
                targetLabel.Text = statusText;
                targetLabel.BackColor = backColor;
                targetLabel.ForeColor = isConnected ? Color.Black : Color.White;
            }
        }

        // =========================================================================
        // Button 3: 釋放資源 (修復卡死問題)
        // =========================================================================
        private async void button3_Click(object sender, EventArgs e)
        {
            // 1. 【關鍵】立刻舉起「釋放中」旗標
            // 這會讓 Timer_Tick 立刻 return，不再去存取 MIL ID
            isReleasing = true;
            statusTimer.Stop();

            button1.Enabled = false;
            button2.Enabled = false;
            button3.Enabled = false;

            // 2. 到背景執行釋放，UI 不會卡住
            await Task.Run(() =>
            {
                FreeCameraResources(ref MilDigitizer1, ref MilImage1, ref MilDisplay1, 1);
                FreeCameraResources(ref MilDigitizer2, ref MilImage2, ref MilDisplay2, 2);

                if (MilApplication != MIL.M_NULL)
                {
                    MIL.MappFreeDefault(MilApplication, MilSystem, MIL.M_NULL, MIL.M_NULL, MIL.M_NULL);
                    MilSystem = MilApplication = MIL.M_NULL;
                }
            });

            UpdateStatusLabel(1, false);
            UpdateStatusLabel(2, false);
            button3.Enabled = true;
            button1.Enabled = true;
            button2.Enabled = true;

            isLive1 = false;
            isLive2 = false;
            UserWantsGrab = false; // 重置使用者意圖
        }

        private void FreeCameraResources(ref MIL_ID dig, ref MIL_ID img, ref MIL_ID disp, int id)
        {
            if (dig != MIL.M_NULL)
            {
                // 先解除 Hook
                MIL.MdigHookFunction(dig, MIL.M_CAMERA_PRESENT + MIL.M_UNHOOK, CameraStatusHandlerDelegate, (IntPtr)id);

                // 嘗試停止
                MIL.MdigHalt(dig);

                MIL.MbufFree(img);
                MIL.MdispFree(disp);
                MIL.MdigFree(dig);

                dig = MIL.M_NULL;
                img = MIL.M_NULL;
                disp = MIL.M_NULL;
            }
        }

        protected override void OnFormClosing(FormClosingEventArgs e)
        {
            isReleasing = true;
            statusTimer.Stop();
            FreeCameraResources(ref MilDigitizer1, ref MilImage1, ref MilDisplay1, 1);
            FreeCameraResources(ref MilDigitizer2, ref MilImage2, ref MilDisplay2, 2);
            if (MilApplication != MIL.M_NULL)
            {
                MIL.MappFreeDefault(MilApplication, MilSystem, MIL.M_NULL, MIL.M_NULL, MIL.M_NULL);
            }
            base.OnFormClosing(e);
        }
    }
}