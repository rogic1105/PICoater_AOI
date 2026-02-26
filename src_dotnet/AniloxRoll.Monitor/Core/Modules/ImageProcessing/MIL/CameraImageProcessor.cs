using Matrox.MatroxImagingLibrary;

namespace AniloxRoll.Monitor.Core.ImageProcessing.MIL
{
    public static class CameraImageProcessor
    {
        /// <param name="srcBuffer">來源影像 (8-bit)</param>
        /// <param name="destBuffer">結果影像 (8-bit)</param>
        /// <param name="sigma">平滑係數 (Python: 9.0)</param>
        /// <param name="fixedMaxVal">絕對強度上限 (Python: 1.0)</param>
        public static void ApplyColMeanSubtraction(MIL_ID srcBuffer, MIL_ID destBuffer)
        {
            if (srcBuffer == MIL.M_NULL || destBuffer == MIL.M_NULL) return;

            // 1. 取得影像資訊
            MIL_INT width = MIL.MbufInquire(srcBuffer, MIL.M_SIZE_X, MIL.M_NULL);
            MIL_INT height = MIL.MbufInquire(srcBuffer, MIL.M_SIZE_Y, MIL.M_NULL);

            // [修正1] 加上 (MIL_ID) 強制轉型，因為 MbufInquire 回傳的是 MIL_INT
            MIL_ID sysId = (MIL_ID)MIL.MbufInquire(srcBuffer, MIL.M_OWNER_SYSTEM, MIL.M_NULL);

            // 宣告暫存 Buffer ID
            MIL_ID meanLine1D = MIL.M_NULL;  // 存放每一行的平均值 (1 x Width)
            MIL_ID meanImage2D = MIL.M_NULL; // 將 1D 平均擴展成 2D 影像
            MIL_ID temp16Bit = MIL.M_NULL;   // 16-bit 用於計算避免負數截斷

            try
            {
                // ==========================================
                // Step A: 計算 Col Mean (投影到 X 軸)
                // ==========================================

                // 分配 1D Buffer (Size: Width x 1)
                MIL.MbufAlloc2d(sysId, width, 1, 8 + MIL.M_UNSIGNED, MIL.M_IMAGE + MIL.M_PROC, ref meanLine1D);

                // [修正2] 改用正確的函式名稱 MimProjection (統計投影)
                // M_0_DEGREE 表示投影到 X 軸 (針對每一個 Column 做計算)
                // M_MEAN 表示計算平均值
                MIL.MimProjection(srcBuffer, meanLine1D, MIL.M_0_DEGREE, MIL.M_MEAN, MIL.M_NULL);


                // ==========================================
                // Step B: 將 1D Mean 擴展為 2D 影像
                // ==========================================

                // 分配與原圖一樣大的 2D Buffer
                MIL.MbufAlloc2d(sysId, width, height, 8 + MIL.M_UNSIGNED, MIL.M_IMAGE + MIL.M_PROC, ref meanImage2D);

                // 使用 MimResize (Nearest Neighbor) 將 1D 線條拉伸填滿整個 2D 畫面
                // 這樣 meanImage2D 的每一列(Row)都會是一樣的數值
                MIL.MimResize(meanLine1D, meanImage2D, MIL.M_FILL_DESTINATION, MIL.M_DEFAULT, MIL.M_NEAREST_NEIGHBOR);


                // ==========================================
                // Step C: 執行運算 (Src - Mean + 127)
                // ==========================================

                // 分配 16-bit Signed Buffer (因為 Src - Mean 可能為負數)
                MIL.MbufAlloc2d(sysId, width, height, 16 + MIL.M_SIGNED, MIL.M_IMAGE + MIL.M_PROC, ref temp16Bit);

                // 1. 複製原圖到 16-bit Buffer (為了進行有號運算)
                MIL.MbufCopy(srcBuffer, temp16Bit);

                // 2. 減去平均值: Temp = Temp(Src) - Mean
                MIL.MimArith(temp16Bit, meanImage2D, temp16Bit, MIL.M_SUB);

                // 3. 加上 127: Temp = Temp + 127
                MIL.MimArith(temp16Bit, 127.0, temp16Bit, MIL.M_ADD_CONST);


                // ==========================================
                // Step D: 輸出結果
                // ==========================================

                // 將 16-bit 結果轉回 8-bit (MbufCopy 會自動處理飽和截斷：小於0變0，大於255變255)
                MIL.MbufCopy(temp16Bit, destBuffer);
            }
            finally
            {
                // ==========================================
                // Step E: 釋放暫存資源
                // ==========================================
                if (meanLine1D != MIL.M_NULL) MIL.MbufFree(meanLine1D);
                if (meanImage2D != MIL.M_NULL) MIL.MbufFree(meanImage2D);
                if (temp16Bit != MIL.M_NULL) MIL.MbufFree(temp16Bit);
            }
        }

        public static void ApplyHessianVerticalFixed(MIL_ID srcBuffer, MIL_ID destBuffer, double sigma, double fixedMaxVal)
        {
            if (srcBuffer == MIL.M_NULL || destBuffer == MIL.M_NULL) return;
            if (fixedMaxVal <= 0) fixedMaxVal = 1.0;

            // 1. 取得影像資訊
            MIL_INT width = MIL.MbufInquire(srcBuffer, MIL.M_SIZE_X, MIL.M_NULL);
            MIL_INT height = MIL.MbufInquire(srcBuffer, MIL.M_SIZE_Y, MIL.M_NULL);
            MIL_ID sysId = (MIL_ID)MIL.MbufInquire(srcBuffer, MIL.M_OWNER_SYSTEM, MIL.M_NULL);

            // 2. 宣告 MIL 物件
            MIL_ID edgeCtx = MIL.M_NULL;
            MIL_ID edgeRes = MIL.M_NULL;
            MIL_ID bufFloat = MIL.M_NULL; // 存放 Dxx (Float)

            try
            {
                // =========================================================
                // Step A: 分配 Edge Finder 資源 (M_CREST 模式)
                // =========================================================
                MIL.MedgeAlloc(sysId, MIL.M_CREST, MIL.M_DEFAULT, ref edgeCtx);
                MIL.MedgeAllocResult(sysId, MIL.M_DEFAULT, ref edgeRes);

                // =========================================================
                // Step B: 設定參數 (模擬 Python: GaussianBlur sigma=9.0)
                // =========================================================
                // 1. 啟用儲存導數，這樣才能把 Dxx 拿出來
                MIL.MedgeControl(edgeCtx, MIL.M_SAVE_DERIVATIVES, MIL.M_ENABLE);

                // 2. 啟用 Float 模式，確保精度
                MIL.MedgeControl(edgeCtx, MIL.M_FLOAT_MODE, MIL.M_ENABLE);

                // 3. 設定平滑度 (Smoothness)
                // 注意: MIL 的 Smoothness 是 0-100。
                // Python 的 Sigma=9.0 非常大。在 MIL 中建議設為較高的值 (例如 80-90) 來模擬大 Sigma。
                // 這裡暫時直接使用傳入的 sigma 數值，若效果不夠強，請在 UI 傳入較大的值 (如 85)。
                MIL.MedgeControl(edgeCtx, MIL.M_FILTER_SMOOTHNESS, sigma);

                // =========================================================
                // Step C: 計算 (Calculate)
                // =========================================================
                MIL.MedgeCalculate(edgeCtx, srcBuffer, MIL.M_NULL, MIL.M_NULL, MIL.M_NULL, edgeRes, MIL.M_DEFAULT);

                // =========================================================
                // Step D: 取出 Dxx (Vertical Ridge)
                // =========================================================
                long floatAttr = MIL.M_IMAGE + MIL.M_PROC;
                MIL.MbufAlloc2d(sysId, width, height, 32 + MIL.M_FLOAT, floatAttr, ref bufFloat);

                // M_DRAW_SECOND_DERIVATIVE_X 對應 Python 的 Sobel(dx=2, dy=0)
                // 這會取出 X 方向的二階導數 (偵測垂直線條)
                MIL.MedgeDraw(MIL.M_DEFAULT, edgeRes, bufFloat, MIL.M_DRAW_SECOND_DERIVATIVE_X, MIL.M_DEFAULT, MIL.M_DEFAULT);

                // =========================================================
                // Step E: 取絕對值 (np.abs)
                // =========================================================
                // bufFloat = |bufFloat|
                MIL.MimArith(bufFloat, MIL.M_NULL, bufFloat, MIL.M_ABS);

                // =========================================================
                // Step F: Fixed Scaling
                // =========================================================
                // Output = (Response / FixedMax) * 255
                double scaleFactor = 255.0 / fixedMaxVal;
                MIL.MimArith(bufFloat, scaleFactor, bufFloat, MIL.M_MULT_CONST);

                // =========================================================
                // Step G: Clip & Output
                // =========================================================
                // MbufCopy 自動處理 Clip (0-255)
                MIL.MbufCopy(bufFloat, destBuffer);
            }
            finally
            {
                if (edgeCtx != MIL.M_NULL) MIL.MedgeFree(edgeCtx);
                if (edgeRes != MIL.M_NULL) MIL.MedgeFree(edgeRes);
                if (bufFloat != MIL.M_NULL) MIL.MbufFree(bufFloat);
            }
        }

        public static void ApplyBinarize(MIL_ID srcBuffer, MIL_ID destBuffer, double threshold)
        {
            if (srcBuffer == MIL.M_NULL || destBuffer == MIL.M_NULL) return;
            MIL.MimBinarize(srcBuffer, destBuffer, MIL.M_FIXED + MIL.M_GREATER, threshold, MIL.M_NULL);
        }

        public static void CopyImage(MIL_ID srcBuffer, MIL_ID destBuffer)
        {
            if (srcBuffer == MIL.M_NULL || destBuffer == MIL.M_NULL) return;
            MIL.MbufCopy(srcBuffer, destBuffer);
        }
    }
}