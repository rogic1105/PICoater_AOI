using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;

using Matrox.MatroxImagingLibrary;


namespace Envision_MdigGrab
{
    public partial class Form1 : Form
    {
        MIL_ID MilApplication = MIL.M_NULL;         // Application identifier.
        MIL_ID MilSystem = MIL.M_NULL;              // System identifier.
        MIL_ID MilDisplay = MIL.M_NULL;             // Display identifier.
        MIL_ID MilDigitizer = MIL.M_NULL;           // Digitizer identifier.
        MIL_ID MilImage = MIL.M_NULL;               // Image buffer identifier.

        public Form1()
        {
            InitializeComponent();
        }

        private void button1_Click(object sender, EventArgs e)
        {
            MIL.MappAlloc(MIL.M_DEFAULT, ref MilApplication);   //Application Module 할당
            MIL.MsysAlloc("M_SYSTEM_RADIENTEVCL", MIL.M_DEV0, MIL.M_DEFAULT, ref MilSystem);   //System Module 할당
            MIL.MdigAlloc(MilSystem, MIL.M_DEV0, ".\\camera.dcf", MIL.M_DEFAULT, ref MilDigitizer);   //Digitizer Module 할당
            MIL.MdispAlloc(MilSystem, MIL.M_DEFAULT, "M_DEFAULT", MIL.M_DEFAULT, ref MilDisplay);   //Display Module 할당
            MIL_INT iSizeX = MIL.MdigInquire(MilDigitizer, MIL.M_SIZE_X, MIL.M_NULL);   //할당된 Digitizer로 부터 가로 해당도 받아오기
            MIL_INT iSizeY = MIL.MdigInquire(MilDigitizer, MIL.M_SIZE_X, MIL.M_NULL);   //할당된 Digitizer로 부터 세로 해당도 받아오기
            MIL.MbufAlloc2d(MilSystem, iSizeX, iSizeY, 8 + MIL.M_UNSIGNED, MIL.M_IMAGE + MIL.M_GRAB + MIL.M_DISP, ref MilImage);   //Buffer 할당
            MIL.MbufClear(MilImage, 0);   //버퍼 초기화
            MIL.MdispSelectWindow(MilDisplay, MilImage, panel1.Handle);   //디스플레이와 버퍼를 연결하여 Picture Control에 출력
            MIL.MdispControl(MilDisplay, MIL.M_FILL_DISPLAY, MIL.M_ENABLE); //디스플레이 화면에 맞춤
        }

        private void button2_Click(object sender, EventArgs e)
        {
            MIL.MdigGrab(MilDigitizer, MilImage);   //MilImage 버퍼에 획득한 영상이 들어감
        }

        private void button3_Click(object sender, EventArgs e)
        {
            MIL.MbufFree(MilImage);   //Buffer 해제
            MIL.MdispFree(MilDisplay);  //Display 해제
            MIL.MdigFree(MilDigitizer);  //Digitizer 해제
            MIL.MsysFree(MilSystem);  //System 해제
            MIL.MappFree(MilApplication); //Application 해제
        }

    }
}
