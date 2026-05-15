
// DSPDlg.cpp : 实现文件
//

#include "stdafx.h"
#include "DSP.h"
#include "DSPDlg.h"
#include "afxdialogex.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// 用于应用程序“关于”菜单项的 CAboutDlg 对话框

class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();

// 对话框数据
	enum { IDD = IDD_ABOUTBOX };

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 支持

// 实现
protected:
	DECLARE_MESSAGE_MAP()
};

CAboutDlg::CAboutDlg() : CDialogEx(CAboutDlg::IDD)
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialogEx)
END_MESSAGE_MAP()


// CDSPDlg 对话框


CDSPDlg::CDSPDlg(CWnd* pParent /*=NULL*/)
	: CDialogEx(CDSPDlg::IDD, pParent)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void CDSPDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Control(pDX, IDC_COMBO1, cmSerial);
	DDX_Control(pDX, IDC_COMBO4, cmBaudRate);
	DDX_Control(pDX, IDC_COMBO2, cmInitial);
	DDX_Control(pDX, IDC_SLIDER2, m_SliderChan1);
	DDX_Control(pDX, IDC_SLIDER18, m_SliderChan2);
	DDX_Control(pDX, IDC_SLIDER12, m_SliderChan3);
	DDX_Control(pDX, IDC_SLIDER13, m_SliderChan4);
	DDX_Control(pDX, IDC_EDIT5, tbCH1);
	DDX_Control(pDX, IDC_EDIT30, tbCH2);
	DDX_Control(pDX, IDC_EDIT31, tbCH3);
	DDX_Control(pDX, IDC_EDIT32, tbCH4);
	DDX_Control(pDX, IDC_SCROLLBAR3, scBC1);
	DDX_Control(pDX, IDC_SCROLLBAR19, scBC2);
	DDX_Control(pDX, IDC_SCROLLBAR20, scBC3);
	DDX_Control(pDX, IDC_SCROLLBAR9, scBC4);
	DDX_Control(pDX, IDC_LIST3, m_listbox);
	DDX_Control(pDX, IDC_BUTTON1, butComOpen);
	DDX_Control(pDX, IDC_LIST2, listHelp);
	DDX_Control(pDX, IDC_EDIT2, ListInstructions);
	DDX_Control(pDX, IDC_COMBO3, m_Channel);
	DDX_Control(pDX, IDC_EDIT6, tbCH);
	DDX_Control(pDX, IDC_EDIT1, tbAddIp);
}

BEGIN_MESSAGE_MAP(CDSPDlg, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED(IDC_RADIO3, &CDSPDlg::OnBnClickedRadio3)
	ON_BN_CLICKED(IDC_RADIO4, &CDSPDlg::OnBnClickedRadio4)
	ON_BN_CLICKED(IDC_RADIO1, &CDSPDlg::OnBnClickedRadio1)
	ON_BN_CLICKED(IDC_RADIO6, &CDSPDlg::OnBnClickedRadio6)
	ON_BN_CLICKED(IDC_RADIO8, &CDSPDlg::OnBnClickedRadio8)
	ON_BN_CLICKED(IDC_RADIO10, &CDSPDlg::OnBnClickedRadio10)
	ON_BN_CLICKED(IDC_RADIO5, &CDSPDlg::OnBnClickedRadio5)
	ON_NOTIFY(NM_RELEASEDCAPTURE, IDC_SLIDER2, &CDSPDlg::OnNMReleasedcaptureSlider1)
	ON_NOTIFY(NM_RELEASEDCAPTURE, IDC_SLIDER18, &CDSPDlg::OnNMReleasedcaptureSlider2)
	ON_NOTIFY(NM_RELEASEDCAPTURE, IDC_SLIDER12, &CDSPDlg::OnNMReleasedcaptureSlider3)
	ON_NOTIFY(NM_RELEASEDCAPTURE, IDC_SLIDER13, &CDSPDlg::OnNMReleasedcaptureSlider4)
	ON_BN_CLICKED(IDC_RADIO7, &CDSPDlg::OnBnClickedRadio7)
	ON_BN_CLICKED(IDC_RADIO9, &CDSPDlg::OnBnClickedRadio9)
	ON_BN_CLICKED(IDC_RADIO11, &CDSPDlg::OnBnClickedRadio11)
	ON_BN_CLICKED(IDC_BUTTON5, &CDSPDlg::OnBnClickedButton5)
	ON_BN_CLICKED(IDC_BUTTON3, &CDSPDlg::OnBnClickedButton3)
	ON_BN_CLICKED(IDC_BUTTON4, &CDSPDlg::OnBnClickedButton4)
	ON_BN_CLICKED(IDC_BUTTON6, &CDSPDlg::OnBnClickedButton6)
	ON_BN_CLICKED(IDC_BUTTON1, &CDSPDlg::OnBnClickedButton1)
	ON_WM_VSCROLL()
	ON_BN_CLICKED(IDC_BUTTON2, &CDSPDlg::OnBnClickedButton2)
	ON_BN_CLICKED(IDC_BUTTON7, &CDSPDlg::OnBnClickedButton7)
	ON_BN_CLICKED(IDC_BUTTON8, &CDSPDlg::OnBnClickedButton8)
	ON_BN_CLICKED(IDC_BUTTON9, &CDSPDlg::OnBnClickedButton9)
	ON_WM_RBUTTONDOWN()
	ON_BN_CLICKED(IDC_BUTTON10, &CDSPDlg::OnBnClickedButton10)
	ON_CBN_SELCHANGE(IDC_COMBO2, &CDSPDlg::OnCbnSelchangeCombo2)
	ON_LBN_SELCHANGE(IDC_LIST3, &CDSPDlg::OnLbnSelchangeList2)
END_MESSAGE_MAP()


// CDSPDlg 消息处理程序

BOOL CDSPDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// 将“关于...”菜单项添加到系统菜单中。

	// IDM_ABOUTBOX 必须在系统命令范围内。
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);

	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu != NULL)
	{
		BOOL bNameValid;
		CString strAboutMenu;
		bNameValid = strAboutMenu.LoadString(IDS_ABOUTBOX);
		ASSERT(bNameValid);
		if (!strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}

	// 设置此对话框的图标。当应用程序主窗口不是对话框时，框架将自动

	//  执行此操作
	SetIcon(m_hIcon, TRUE);			// 设置大图标
	SetIcon(m_hIcon, FALSE);		// 设置小图标

	isOpen = false;

	((CButton *)GetDlgItem(IDC_RADIO3))->SetCheck(TRUE);

	cmInitial.InsertString(0,_T("$"));
	cmInitial.InsertString(1,_T("#"));
	cmInitial.SetCurSel(0);

	FindCommPort(&cmSerial);
	if (cmSerial.GetCount() != 0)
	{
		cmSerial.SetCurSel(0);
	}

	cmBaudRate.InsertString(0,_T("2400"));
	cmBaudRate.InsertString(1,_T("4800"));
	cmBaudRate.InsertString(2,_T("9600"));
	cmBaudRate.InsertString(3,_T("14400"));
	cmBaudRate.InsertString(4,_T("19200"));
	cmBaudRate.InsertString(5,_T("38400"));
	cmBaudRate.InsertString(6,_T("56000"));
	cmBaudRate.InsertString(7,_T("57600"));
	cmBaudRate.InsertString(8,_T("115200"));
	cmBaudRate.SetCurSel(2);

	m_SliderChan1.SetRange(0, 255);
	m_SliderChan2.SetRange(0, 255);
	m_SliderChan3.SetRange(0, 255);
	m_SliderChan4.SetRange(0, 255);

	m_Channel.SetCurSel(0);

	return TRUE;  // 除非将焦点设置到控件，否则返回 TRUE
}

void CDSPDlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		CAboutDlg dlgAbout;
		dlgAbout.DoModal();
	}
	else
	{
		CDialogEx::OnSysCommand(nID, lParam);
	}
}

// 如果向对话框添加最小化按钮，则需要下面的代码
//  来绘制该图标。对于使用文档/视图模型的 MFC 应用程序，
//  这将由框架自动完成。

void CDSPDlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // 用于绘制的设备上下文

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// 使图标在工作区矩形中居中
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// 绘制图标
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
}

//当用户拖动最小化窗口时系统调用此函数取得光标
//显示。
HCURSOR CDSPDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}

int CDSPDlg::HelpText(CString IP)
{
	int count = listHelp.GetCount();
	listHelp.InsertString(count,IP);
	listHelp.SetTopIndex(count);
	return 0;
}

int __stdcall SearchCB(CString IP, long long nUserVal)
{
	CDSPDlg *pDlg = (CDSPDlg *)nUserVal;
	return (pDlg->CheckConnectCB(IP));
}

int CDSPDlg::CheckConnectCB(CString IP)
{
	CString str;
	int count = m_listbox.GetCount();
	if (count == 0)
	{
		m_listbox.AddString(IP);
		listbox_items[0] = 0;
	}
	else
	{
		int i;
		for (i = 0; i < count; i++)
		{
			m_listbox.GetText(i, str);
			if (IP == str)
				break;
		}
		listbox_items[i] = 0;
		if(i == count)
			m_listbox.AddString(IP);
	}

	return 0;
}

int __stdcall ConnectCB(CString IP, long long nUserVal)
{
	CDSPDlg *pDlg = (CDSPDlg *)nUserVal;
	return (pDlg->ConnectResultCB(IP));
}

int CDSPDlg::ConnectResultCB(CString IP)
{
	HelpText(IP + _T(":Connection successful"));
	return 0;
}

void CDSPDlg::FindCommPort(CComboBox *pComboBox)
{
	HKEY hKey;
	if (::RegOpenKeyEx(HKEY_LOCAL_MACHINE, _T("Hardware\\DeviceMap\\SerialComm"), NULL, KEY_READ, &hKey) == ERROR_SUCCESS)
	{
		int i = 0;
		wchar_t portName[256], commName[256];
		DWORD dwLong, dwSize;

		//CString strFilePath;
		//strFilePath=GetCurrentDirectory(256,portName);  
		//strFilePath.Format("%s//ESssDataSynchronous.ini",strBuff);
		while (1)
		{
			dwLong = dwSize = sizeof(portName);
			if (::RegEnumValue(hKey,
				i,
				portName,
				&dwLong,
				NULL,
				NULL,
				(PUCHAR)commName,
				&dwSize) == ERROR_NO_MORE_ITEMS) // 枚举串口
				break;
			pComboBox->AddString(commName); // commName就是串口名字
			i++;
		}
		if (pComboBox->GetCount() == 0)
		{
			//::AfxMessageBox("在HKEY_LOCAL_MACHINE:Hardware\\DeviceMap\\SerialComm里找不到串口!!!" );
		}
		RegCloseKey(hKey);
	}
}

void CDSPDlg::OnBnClickedRadio3()
{
    GetDlgItem(IDC_COMBO1)->ShowWindow(TRUE);
	GetDlgItem(IDC_BUTTON1)->ShowWindow(TRUE);
	GetDlgItem(IDC_STATIC19)->ShowWindow(TRUE);
	GetDlgItem(IDC_STATIC20)->ShowWindow(TRUE);
	GetDlgItem(IDC_COMBO4)->ShowWindow(TRUE);

	GetDlgItem(IDC_LIST3)->ShowWindow(FALSE);
	GetDlgItem(IDC_BUTTON4)->ShowWindow(FALSE);
	GetDlgItem(IDC_BUTTON5)->ShowWindow(FALSE);
	GetDlgItem(IDC_BUTTON6)->ShowWindow(FALSE);
	GetDlgItem(IDC_BUTTON10)->ShowWindow(FALSE);
	GetDlgItem(IDC_EDIT1)->ShowWindow(FALSE);
}


void CDSPDlg::OnBnClickedRadio4()
{
    GetDlgItem(IDC_COMBO1)->ShowWindow(FALSE);
	GetDlgItem(IDC_BUTTON1)->ShowWindow(FALSE);
    GetDlgItem(IDC_STATIC19)->ShowWindow(FALSE);
	GetDlgItem(IDC_STATIC20)->ShowWindow(FALSE);
	GetDlgItem(IDC_COMBO4)->ShowWindow(FALSE);

	GetDlgItem(IDC_LIST3)->ShowWindow(TRUE);
	GetDlgItem(IDC_BUTTON4)->ShowWindow(TRUE);
	GetDlgItem(IDC_BUTTON5)->ShowWindow(TRUE);
	GetDlgItem(IDC_BUTTON6)->ShowWindow(TRUE);
	GetDlgItem(IDC_BUTTON10)->ShowWindow(TRUE);
	GetDlgItem(IDC_EDIT1)->ShowWindow(TRUE);
}


void CDSPDlg::OnBnClickedRadio1()
{
	if(((CButton *)GetDlgItem(IDC_RADIO1))->GetCheck())
	{
		CString strValue,strIp;
		tbCH1.GetWindowTextW(strValue);
		int nValue = _ttoi(strValue);
		char* pCommand = Controller_TurnOnChannelLength(1, nValue);
		if(pCommand != NULL)
		{
			CString command(pCommand);
			ListInstructions.SetWindowTextW(command);
			HelpText(_T("OK"));
		}
		else
		{
			HelpText(_T("NG"));
		}
	}
}


void CDSPDlg::OnBnClickedRadio6()
{
	if(((CButton *)GetDlgItem(IDC_RADIO6))->GetCheck())
	{
		CString strValue;
		tbCH1.GetWindowTextW(strValue);
		int nValue = _ttoi(strValue);

		char* pCommand = Controller_TurnOnChannelLength(2, nValue);
		if(pCommand != NULL)
		{
			CString command(pCommand);

			ListInstructions.SetWindowTextW(command);
			HelpText(_T("OK"));
		}
		else
		{
			HelpText(_T("NG"));
		}
		
	}
}


void CDSPDlg::OnBnClickedRadio8()
{
	if(((CButton *)GetDlgItem(IDC_RADIO8))->GetCheck())
	{
		CString strValue;
		tbCH1.GetWindowTextW(strValue);
		int nValue = _ttoi(strValue);
		
		char* pCommand = Controller_TurnOnChannelLength(3, nValue);
		if(pCommand != NULL)
		{
			CString command(pCommand);

			ListInstructions.SetWindowTextW(command);
			HelpText(_T("OK"));
		}
		else
		{
			HelpText(_T("NG"));
		}
		
	}
}


void CDSPDlg::OnBnClickedRadio10()
{
	if(((CButton *)GetDlgItem(IDC_RADIO10))->GetCheck())
	{
		CString strValue;
		tbCH1.GetWindowTextW(strValue);
		int nValue = _ttoi(strValue);
		
		char* pCommand = Controller_TurnOnChannelLength(4, nValue);
		if(pCommand != NULL)
		{
			CString command(pCommand);

			ListInstructions.SetWindowTextW(command);
			HelpText(_T("OK"));
		}
		else
		{
			HelpText(_T("NG"));
		}
		
	}
}


void CDSPDlg::OnBnClickedRadio5()
{
	if(((CButton *)GetDlgItem(IDC_RADIO5))->GetCheck())
	{
		CString strValue;
		tbCH1.GetWindowTextW(strValue);
		int nValue = _ttoi(strValue);
		
		char* pCommand = Controller_TurnOffChannelLength(1, nValue);
		if(pCommand != NULL)
		{
			CString command(pCommand);

			ListInstructions.SetWindowTextW(command);
			HelpText(_T("OK"));
		}
		else
		{
			HelpText(_T("NG"));
		}
		
	}
}


void CDSPDlg::OnBnClickedRadio7()
{
	if(((CButton *)GetDlgItem(IDC_RADIO7))->GetCheck())
	{
		CString strValue;
		tbCH1.GetWindowTextW(strValue);
		int nValue = _ttoi(strValue);
		
		char* pCommand = Controller_TurnOffChannelLength(2, nValue);
		if(pCommand != NULL)
		{
			CString command(pCommand);

			ListInstructions.SetWindowTextW(command);
			HelpText(_T("OK"));
		}
		else
		{
			HelpText(_T("NG"));
		}
		
	}
}


void CDSPDlg::OnBnClickedRadio9()
{
	if(((CButton *)GetDlgItem(IDC_RADIO9))->GetCheck())
	{
		CString strValue;
		tbCH1.GetWindowTextW(strValue);
		int nValue = _ttoi(strValue);
		
		char* pCommand = Controller_TurnOffChannelLength(3, nValue);
		if(pCommand != NULL)
		{
			CString command(pCommand);

			ListInstructions.SetWindowTextW(command);
			HelpText(_T("OK"));
		}
		else
		{
			HelpText(_T("NG"));
		}
		
	}
}


void CDSPDlg::OnBnClickedRadio11()
{
	if(((CButton *)GetDlgItem(IDC_RADIO11))->GetCheck())
	{
		CString strValue;
		tbCH1.GetWindowTextW(strValue);
		int nValue = _ttoi(strValue);
		
		char* pCommand = Controller_TurnOffChannelLength(4, nValue);
		if(pCommand != NULL)
		{
			CString command(pCommand);

			ListInstructions.SetWindowTextW(command);
			HelpText(_T("OK"));
		}
		else
		{
			HelpText(_T("NG"));
		}
		
	}
}


void CDSPDlg::OnNMReleasedcaptureSlider1(NMHDR *pNMHDR, LRESULT *pResult)
{
	CString strValue;

	int nValue = this->m_SliderChan1.GetPos();

	strValue.Format(_T("%d"), nValue);

	tbCH1.SetWindowTextW(strValue);

	char* pData = Controller_SetIntensityLength(1, nValue);
	if(pData != NULL)
	{
		CString command(pData);

		ListInstructions.SetWindowTextW(command);
		HelpText(_T("OK"));
	}
	else
	{
		HelpText(_T("NG"));
	}
	
	*pResult = 0;
}


void CDSPDlg::OnNMReleasedcaptureSlider2(NMHDR *pNMHDR, LRESULT *pResult)
{
	CString strValue;

	int nValue = this->m_SliderChan2.GetPos();

	strValue.Format(_T("%d"), nValue);

	tbCH2.SetWindowTextW(strValue);

	char* pData = Controller_SetIntensityLength(2, nValue);
	if(pData != NULL)
	{
		CString command(pData);

		ListInstructions.SetWindowTextW(command);
		HelpText(_T("OK"));
	}
	else
	{
		HelpText(_T("NG"));
	}
	
	*pResult = 0;
}


void CDSPDlg::OnNMReleasedcaptureSlider3(NMHDR *pNMHDR, LRESULT *pResult)
{
	CString strValue;

	int nValue = this->m_SliderChan3.GetPos();

	strValue.Format(_T("%d"), nValue);

	tbCH3.SetWindowTextW(strValue);

	char* pData = Controller_SetIntensityLength(3, nValue);
	if(pData != NULL)
	{
		CString command(pData);

		ListInstructions.SetWindowTextW(command);
		HelpText(_T("OK"));
	}
	else
	{
		HelpText(_T("NG"));
	}
	
	*pResult = 0;
}


void CDSPDlg::OnNMReleasedcaptureSlider4(NMHDR *pNMHDR, LRESULT *pResult)
{
	CString strValue;

	int nValue = this->m_SliderChan4.GetPos();

	strValue.Format(_T("%d"), nValue);

	tbCH4.SetWindowTextW(strValue);

	char* pData = Controller_SetIntensityLength(4, nValue);
	if(pData != NULL)
	{
		CString command(pData);

		ListInstructions.SetWindowTextW(command);
		HelpText(_T("OK"));
	}
	else
	{
		HelpText(_T("NG"));
	}
	
	*pResult = 0;
}


void CDSPDlg::OnVScroll(UINT nSBCode, UINT nPos, CScrollBar* pScrollBar)
{
	if (nSBCode == SB_ENDSCROLL)
	{
		return;
	}

	if (pScrollBar == &scBC1)
	{
		CString strValue;

		tbCH1.GetWindowTextW(strValue);

		int nValue = _ttoi(strValue);

		if (nSBCode == SB_LINEDOWN)
		{
			if (nValue > 0)
			{
				nValue--;
			}
		}
		else if (nSBCode == SB_LINEUP)
		{
			if (nValue < 255)
			{
				nValue++;
			}
		}
		strValue.Format(_T("%d"), nValue);
		tbCH1.SetWindowTextW(strValue);
		m_SliderChan1.SetPos(nValue);

		char* pData = Controller_SetIntensityLength(1, nValue);
		if(pData != NULL)
		{
			CString command(pData);

			ListInstructions.SetWindowTextW(command);
			HelpText(_T("OK"));
		}
		else
		{
			HelpText(_T("NG"));
		}
	}

	if (pScrollBar == &scBC2)
	{
		CString strValue;
		CString strIp;

		tbCH2.GetWindowTextW(strValue);

		int nValue = _ttoi(strValue);

		if (nSBCode == SB_LINEDOWN)
		{
			if (nValue > 0)
			{
				nValue--;
			}
		}
		else if (nSBCode == SB_LINEUP)
		{
			if (nValue < 255)
			{
				nValue++;
			}
		}
		strValue.Format(_T("%d"), nValue);
		tbCH2.SetWindowTextW(strValue);
		m_SliderChan2.SetPos(nValue);

		char* pData = Controller_SetIntensityLength(2, nValue);
		if(pData != NULL)
		{
			CString command(pData);

			ListInstructions.SetWindowTextW(command);
			HelpText(_T("OK"));
		}
		else
		{
			HelpText(_T("NG"));
		}
	}

	if (pScrollBar == &scBC3)
	{
		CString strValue;
		CString strIp;

		tbCH3.GetWindowTextW(strValue);

		int nValue = _ttoi(strValue);

		if (nSBCode == SB_LINEDOWN)
		{
			if (nValue > 0)
			{
				nValue--;
			}
		}
		else if (nSBCode == SB_LINEUP)
		{
			if (nValue < 255)
			{
				nValue++;
			}
		}
		strValue.Format(_T("%d"), nValue);
		tbCH3.SetWindowTextW(strValue);
		m_SliderChan3.SetPos(nValue);

		char* pData = Controller_SetIntensityLength(3, nValue);
		if(pData != NULL)
		{
			CString command(pData);

			ListInstructions.SetWindowTextW(command);
			HelpText(_T("OK"));
		}
		else
		{
			HelpText(_T("NG"));
		}
	}

	if (pScrollBar == &scBC4)
	{
		CString strValue;
		CString strIp;

		tbCH4.GetWindowTextW(strValue);

		int nValue = _ttoi(strValue);

		if (nSBCode == SB_LINEDOWN)
		{
			if (nValue > 0)
			{
				nValue--;
			}
		}
		else if (nSBCode == SB_LINEUP)
		{
			if (nValue < 255)
			{
				nValue++;
			}
		}
		strValue.Format(_T("%d"), nValue);
		tbCH4.SetWindowTextW(strValue);
		m_SliderChan4.SetPos(nValue);

		char* pData = Controller_SetIntensityLength(4, nValue);
		if(pData != NULL)
		{
			CString command(pData);

			ListInstructions.SetWindowTextW(command);
			HelpText(_T("OK"));
		}
		else
		{
			HelpText(_T("NG"));
		}
	}

	CDialogEx::OnVScroll(nSBCode, nPos, pScrollBar);
}


void CDSPDlg::OnBnClickedButton5()
{
	m_listbox.ResetContent();
	Controller_SearchIP(SearchCB, (long long)this);
}


void CDSPDlg::OnBnClickedButton3()
{
	//CString *IP_Array;
	//IP_Array = new CString[m_listbox.GetCount()];
	//int j = 0;
	////给CString赋值
	//for (int i = 0; i < m_listbox.GetCount(); i++)
	//{
	//	m_listbox.GetText(i, IP_Array[i]);
	//}

	//GetDlgItem(IDC_BUTTON8)->EnableWindow(TRUE);
	//GetDlgItem(IDC_BUTTON9)->EnableWindow(TRUE);
	////GetDlgItem(IDC_BUTTON10)->EnableWindow(TRUE);
	////GetDlgItem(IDC_BUTTON11)->EnableWindow(TRUE);

	//long err = Controller_ConnectNetwork(IP_Array, m_listbox.GetCount(), 8234, ConnectCB, (long long)this);		//控制器默认端口号是8234
	//if (err)
	//{
	//	MessageBox(IP_Array[err - 1] + _T("连接失败"));
	//}
	//else
	//{
	//	GetDlgItem(IDC_BUTTON8)->EnableWindow(TRUE);
	//	GetDlgItem(IDC_BUTTON9)->EnableWindow(TRUE);
	//	//GetDlgItem(IDC_BUTTON10)->EnableWindow(TRUE);
	//	//GetDlgItem(IDC_BUTTON11)->EnableWindow(TRUE);
	//	return;
	//}
}


void CDSPDlg::OnBnClickedButton4()
{
	if(m_strIP == _T(""))
	{
		HelpText(_T("Please select IP"));
		return;
	}
	GetDlgItem(IDC_BUTTON8)->EnableWindow(TRUE);
	GetDlgItem(IDC_BUTTON9)->EnableWindow(TRUE);
	//GetDlgItem(IDC_BUTTON10)->EnableWindow(TRUE);
	//GetDlgItem(IDC_BUTTON11)->EnableWindow(TRUE);
	
	long err = Controller_InitNetwork(m_strIP, 8234, ConnectCB, (long long)this);		//控制器默认端口号是8234
	if (err)
	{
		HelpText(m_strIP + _T(":Connection Fail"));
	}
	else
	{
		GetDlgItem(IDC_BUTTON8)->EnableWindow(TRUE);
		GetDlgItem(IDC_BUTTON9)->EnableWindow(TRUE);
		//GetDlgItem(IDC_BUTTON10)->EnableWindow(TRUE);
		//GetDlgItem(IDC_BUTTON11)->EnableWindow(TRUE);
		return;
	}
}


void CDSPDlg::OnBnClickedButton6()
{		
	Controller_ReleaseNetwork();
	HelpText(_T("Close successful"));
}


void CDSPDlg::OnBnClickedButton1()
{
	if (((CButton *)GetDlgItem(IDC_RADIO3))->GetCheck())
	{
		if (!isOpen)
		{
			CString strCom,strBaudRate;
			cmSerial.GetWindowTextW(strCom);
			cmBaudRate.GetWindowTextW(strBaudRate);
			USES_CONVERSION;
			if (Controller_InitSerialPort(T2A(strCom),_ttoi(strBaudRate)) == 0)
			{
				CString strInit;

				USES_CONVERSION;

				isOpen = true;

				HelpText(_T("Succeed to initialize serial port"));
				butComOpen.SetWindowTextW(_T("Disconnect"));

				GetDlgItem(IDC_COMBO1)->EnableWindow(FALSE);
				GetDlgItem(IDC_RADIO3)->EnableWindow(FALSE);
				GetDlgItem(IDC_RADIO4)->EnableWindow(FALSE);
				GetDlgItem(IDC_BUTTON8)->EnableWindow(TRUE);
				GetDlgItem(IDC_BUTTON9)->EnableWindow(TRUE);
			}
			else
			{
				HelpText(_T("Failed to initialize serial port"));
			}
		}
		else
		{
			if(Controller_ReleaseSerialPort() == 0)
			{
				HelpText(_T("Succeed to release serial port"));
				butComOpen.SetWindowTextW(_T("Connect"));

				GetDlgItem(IDC_COMBO1)->EnableWindow(TRUE);
				GetDlgItem(IDC_RADIO3)->EnableWindow(TRUE);
				GetDlgItem(IDC_RADIO4)->EnableWindow(TRUE);
				GetDlgItem(IDC_BUTTON8)->EnableWindow(FALSE);
				GetDlgItem(IDC_BUTTON9)->EnableWindow(FALSE);
				isOpen = false;
			}
			else
			{
				HelpText(_T("Failed to release serial port"));
			}
		}
	}
}


void CDSPDlg::OnBnClickedButton2()
{
	CString strIp;
	CString strValue;
	int nChannel = m_Channel.GetCurSel();
	if(nChannel == 0) 
	{
		long nValue = Controller_ReadIntensityLength(1);
		if(nValue != -1)
		{
			strValue.Format(_T("%d"), nValue);
			tbCH1.SetWindowTextW(strValue);
			tbCH.SetWindowTextW(strValue);
			m_SliderChan1.SetPos(nValue);
			HelpText(_T("OK"));
		}
		else
		{
			HelpText(_T("NG"));
		}
	}
	else if(nChannel == 1) 
	{		
		long nValue = Controller_ReadIntensityLength(2);
		if(nValue != -1)
		{
			strValue.Format(_T("%d"), nValue);
			tbCH2.SetWindowTextW(strValue);
			tbCH.SetWindowTextW(strValue);
			m_SliderChan2.SetPos(nValue);
			HelpText(_T("OK"));
		}
		else
		{
			HelpText(_T("NG"));
		}
	}
	else if(nChannel == 2) 
	{
		long nValue = Controller_ReadIntensityLength(3);
		if(nValue != -1)
		{
			strValue.Format(_T("%d"), nValue);
			tbCH3.SetWindowTextW(strValue);
			tbCH.SetWindowTextW(strValue);
			m_SliderChan3.SetPos(nValue);
			HelpText(_T("OK"));
		}
		else
		{
			HelpText(_T("NG"));
		}
	}
	else if(nChannel == 3) 
	{
		long nValue = Controller_ReadIntensityLength(4);
		if(nValue != -1)
		{
			strValue.Format(_T("%d"), nValue);
			tbCH4.SetWindowTextW(strValue);
			tbCH.SetWindowTextW(strValue);
			m_SliderChan4.SetPos(nValue);
			HelpText(_T("OK"));
		}
		else
		{
			HelpText(_T("NG"));
		}
	}
    else if(nChannel == 4) 
	{
		char* nValue = Controller_ReadAlarm(1);
		if (nValue == NULL)
		{
			HelpText(_T("NG"));
		}
		else
		{
			strValue.Format(_T("%S"), nValue);
			tbCH.SetWindowTextW(strValue);
			HelpText(_T("OK"));
		}
	}
	else if(nChannel == 5) 
	{
		char* nValue = Controller_ReadAlarm(2);
		if (nValue == NULL)
		{
			HelpText(_T("NG"));
		}
		else
		{
			strValue.Format(_T("%S"), nValue);
			tbCH.SetWindowTextW(strValue);
			HelpText(_T("OK"));
		}
	}
}


void CDSPDlg::OnBnClickedButton7()
{
    if (Controller_RebootDevice() == 0)
    {
        HelpText(_T("OK"));
    }
    else
    {
		HelpText(_T("NG"));
    }
}


void CDSPDlg::OnBnClickedButton8()
{
	CString command;
	ListInstructions.GetWindowTextW(command);

    Controller_ReturnData ReturnData;

	USES_CONVERSION;
    long nReturn = Controller_WriteCommand(T2A(command), &ReturnData);
    if (nReturn == 0)
    {
        //bCon = true;
        if (ReturnData.nType == 1)
        {
            if (ReturnData.nChannel == 1)
            {
				((CButton *)GetDlgItem(IDC_RADIO1))->SetCheck(TRUE);
            }
            else if (ReturnData.nChannel == 2)
            {
                ((CButton *)GetDlgItem(IDC_RADIO6))->SetCheck(TRUE);
            }
            else if (ReturnData.nChannel == 3)
            {
                ((CButton *)GetDlgItem(IDC_RADIO8))->SetCheck(TRUE);
            }
            else if (ReturnData.nChannel == 4)
            {
               ((CButton *)GetDlgItem(IDC_RADIO10))->SetCheck(TRUE);
            }
        }
        else if (ReturnData.nType == 2)
        {
            if (ReturnData.nChannel == 1)
            {
                ((CButton *)GetDlgItem(IDC_RADIO5))->SetCheck(TRUE);
            }
            else if (ReturnData.nChannel == 2)
            {
                ((CButton *)GetDlgItem(IDC_RADIO7))->SetCheck(TRUE);
            }
            else if (ReturnData.nChannel == 3)
            {
                ((CButton *)GetDlgItem(IDC_RADIO9))->SetCheck(TRUE);
            }
            else if (ReturnData.nChannel == 4)
            {
                ((CButton *)GetDlgItem(IDC_RADIO11))->SetCheck(TRUE);
            }
        }
        else if (ReturnData.nType == 3)
        {	
			CString strValue;
            if(ReturnData.nChannel == 1) 
			{
				strValue.Format(_T("%d"), ReturnData.nValue);
				tbCH1.SetWindowTextW(strValue);
				tbCH.SetWindowTextW(strValue);
				m_SliderChan1.SetPos(ReturnData.nValue);
			}
			else if(ReturnData.nChannel == 2) 
			{		
				strValue.Format(_T("%d"), ReturnData.nValue);
				tbCH2.SetWindowTextW(strValue);
				tbCH.SetWindowTextW(strValue);
				m_SliderChan2.SetPos(ReturnData.nValue);
			}
			else if(ReturnData.nChannel == 3) 
			{
				strValue.Format(_T("%d"), ReturnData.nValue);
				tbCH3.SetWindowTextW(strValue);
				tbCH.SetWindowTextW(strValue);
				m_SliderChan3.SetPos(ReturnData.nValue);
			}
			else if(ReturnData.nChannel == 4) 
			{
				strValue.Format(_T("%d"), ReturnData.nValue);
				tbCH4.SetWindowTextW(strValue);
				tbCH.SetWindowTextW(strValue);
				m_SliderChan4.SetPos(ReturnData.nValue);
			}
        }
		HelpText(_T("OK"));
    }
    else
    {
       HelpText(_T("NG"));
    }
}


void CDSPDlg::OnBnClickedButton9()
{
	listHelp.ResetContent();
}


void CDSPDlg::OnBnClickedButton10()
{
	CString strIp;
	tbAddIp.GetWindowTextW(strIp);
	if(strIp != "")
		m_listbox.AddString(strIp);
}


void CDSPDlg::OnCbnSelchangeCombo2()
{
	CString str;
	GetDlgItem(IDC_COMBO2)->GetWindowTextW(str);

	char * pData = (LPSTR)(LPCTSTR)str;
	Controller_SetInitial(pData[0]);
}


void CDSPDlg::OnLbnSelchangeList2()
{
	int nIndex = m_listbox.GetCurSel();
	if (nIndex != -1)
	{
		m_listbox.GetText(nIndex,m_strIP);
	}
}
