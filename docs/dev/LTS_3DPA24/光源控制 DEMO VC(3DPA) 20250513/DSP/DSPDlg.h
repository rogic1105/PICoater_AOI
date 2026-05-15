
// DSPDlg.h : 头文件
//

#pragma once
#include "afxwin.h"
#include "afxcmn.h"
#include "..\Controller\\Controller.h"

// CDSPDlg 对话框
class CDSPDlg : public CDialogEx
{
// 构造
public:
	CDSPDlg(CWnd* pParent = NULL);	// 标准构造函数

// 对话框数据
	enum { IDD = IDD_DSP_DIALOG };

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV 支持


// 实现
protected:
	HICON m_hIcon;

	// 生成的消息映射函数
	virtual BOOL OnInitDialog();
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	DECLARE_MESSAGE_MAP()
public:
	bool isOpen;
	CString m_strIP;
	afx_msg void OnBnClickedRadio3();
	afx_msg void OnBnClickedRadio4();
	void FindCommPort(CComboBox *pComboBox);

	CComboBox cmSerial;
	CComboBox cmBaudRate;
	CComboBox m_valid_IP;
	CComboBox cmInitial;
	CComboBox m_Channel;

	CButton butComOpen;

	CListBox m_listbox;
	CListBox listHelp;

	CEdit ListInstructions;

	CSliderCtrl m_SliderChan1;
	CSliderCtrl m_SliderChan2;
	CSliderCtrl m_SliderChan3;
	CSliderCtrl m_SliderChan4;

	CEdit tbCH1;
	CEdit tbCH2;
	CEdit tbCH3;
	CEdit tbCH4;

	CScrollBar scBC1;
	CScrollBar scBC2;
	CScrollBar scBC3;
	CScrollBar scBC4;

	afx_msg void OnBnClickedRadio1();
	afx_msg void OnBnClickedRadio6();
	afx_msg void OnBnClickedRadio8();
	afx_msg void OnBnClickedRadio10();
	afx_msg void OnBnClickedRadio5();
	afx_msg void OnNMReleasedcaptureSlider1(NMHDR *pNMHDR, LRESULT *pResult);
	afx_msg void OnNMReleasedcaptureSlider2(NMHDR *pNMHDR, LRESULT *pResult);
	afx_msg void OnNMReleasedcaptureSlider3(NMHDR *pNMHDR, LRESULT *pResult);
	afx_msg void OnNMReleasedcaptureSlider4(NMHDR *pNMHDR, LRESULT *pResult);
	afx_msg void OnVScroll(UINT nSBCode, UINT nPos, CScrollBar* pScrollBar);

	afx_msg void OnBnClickedRadio7();
	afx_msg void OnBnClickedRadio9();
	afx_msg void OnBnClickedRadio11();
	afx_msg void OnBnClickedButton5();
	afx_msg void OnBnClickedButton3();
	afx_msg void OnBnClickedButton4();
	afx_msg void OnBnClickedButton6();
	afx_msg void OnBnClickedButton1();

	int ConnectResultCB(CString IP);
	int CheckConnectCB(CString IP);
	unsigned int listbox_items[100];
	int HelpText(CString IP);

	afx_msg void OnBnClickedButton2();
	afx_msg void OnBnClickedButton7();
	CEdit tbCH;
	afx_msg void OnBnClickedButton8();
	afx_msg void OnBnClickedButton9();
	afx_msg void OnBnClickedButton10();
	CEdit tbAddIp;
	afx_msg void OnCbnSelchangeCombo2();
	afx_msg void OnLbnSelchangeList2();
};
