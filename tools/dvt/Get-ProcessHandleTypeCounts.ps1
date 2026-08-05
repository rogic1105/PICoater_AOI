param(
    [Parameter(Mandatory = $true)]
    [int]$ProcessId
)

$ErrorActionPreference = "Stop"

if (-not ("DvtDiagnostics.ProcessHandleSnapshot" -as [type])) {
    Add-Type -TypeDefinition @"
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Runtime.InteropServices;

namespace DvtDiagnostics
{
    public static class ProcessHandleSnapshot
    {
        private const int SystemExtendedHandleInformation = 64;
        private const int ObjectTypeInformation = 2;
        private const uint ProcessDuplicateHandle = 0x0040;
        private const uint DuplicateSameAccess = 0x00000002;
        private const int StatusInfoLengthMismatch =
            unchecked((int)0xC0000004);

        [StructLayout(LayoutKind.Sequential)]
        private struct SystemHandleEntry
        {
            public IntPtr Object;
            public IntPtr UniqueProcessId;
            public IntPtr HandleValue;
            public uint GrantedAccess;
            public ushort CreatorBackTraceIndex;
            public ushort ObjectTypeIndex;
            public uint HandleAttributes;
            public uint Reserved;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct UnicodeString
        {
            public ushort Length;
            public ushort MaximumLength;
            public IntPtr Buffer;
        }

        private sealed class TypeCount
        {
            public int Count;
            public IntPtr SampleHandle;
        }

        [DllImport("ntdll.dll")]
        private static extern int NtQuerySystemInformation(
            int informationClass,
            IntPtr information,
            int informationLength,
            out int returnLength);

        [DllImport("ntdll.dll")]
        private static extern int NtQueryObject(
            IntPtr handle,
            int informationClass,
            IntPtr information,
            int informationLength,
            out int returnLength);

        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern IntPtr OpenProcess(
            uint desiredAccess,
            bool inheritHandle,
            int processId);

        [DllImport("kernel32.dll", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static extern bool DuplicateHandle(
            IntPtr sourceProcess,
            IntPtr sourceHandle,
            IntPtr targetProcess,
            out IntPtr targetHandle,
            uint desiredAccess,
            bool inheritHandle,
            uint options);

        [DllImport("kernel32.dll")]
        private static extern IntPtr GetCurrentProcess();

        [DllImport("kernel32.dll", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static extern bool CloseHandle(IntPtr handle);

        public static string[] Capture(int processId)
        {
            var counts = ReadCounts(processId);
            IntPtr process = OpenProcess(
                ProcessDuplicateHandle, false, processId);
            if (process == IntPtr.Zero)
                throw new Win32Exception(
                    Marshal.GetLastWin32Error(),
                    "OpenProcess(PROCESS_DUP_HANDLE) failed");

            try
            {
                var rows = new List<string>();
                foreach (KeyValuePair<ushort, TypeCount> pair in counts)
                {
                    string name = TryReadTypeName(
                        process, pair.Value.SampleHandle);
                    rows.Add(
                        pair.Key + "\t" + name + "\t" +
                        pair.Value.Count);
                }
                rows.Sort(StringComparer.OrdinalIgnoreCase);
                return rows.ToArray();
            }
            finally
            {
                CloseHandle(process);
            }
        }

        private static Dictionary<ushort, TypeCount> ReadCounts(
            int processId)
        {
            int size = 1024 * 1024;
            IntPtr buffer = IntPtr.Zero;
            try
            {
                while (true)
                {
                    if (buffer != IntPtr.Zero)
                        Marshal.FreeHGlobal(buffer);
                    buffer = Marshal.AllocHGlobal(size);
                    int required;
                    int status = NtQuerySystemInformation(
                        SystemExtendedHandleInformation,
                        buffer,
                        size,
                        out required);
                    if (status == 0)
                        break;
                    if (status != StatusInfoLengthMismatch)
                        throw new InvalidOperationException(
                            "NtQuerySystemInformation failed: 0x" +
                            status.ToString("X8"));
                    size = Math.Max(size * 2, required + 65536);
                }

                long total = Marshal.ReadIntPtr(buffer).ToInt64();
                int entrySize = Marshal.SizeOf(
                    typeof(SystemHandleEntry));
                long baseAddress =
                    buffer.ToInt64() + IntPtr.Size * 2L;
                var result = new Dictionary<ushort, TypeCount>();
                for (long i = 0; i < total; i++)
                {
                    IntPtr entryAddress = new IntPtr(
                        baseAddress + i * entrySize);
                    var entry = (SystemHandleEntry)
                        Marshal.PtrToStructure(
                            entryAddress,
                            typeof(SystemHandleEntry));
                    if (entry.UniqueProcessId.ToInt64() != processId)
                        continue;

                    TypeCount count;
                    if (!result.TryGetValue(
                        entry.ObjectTypeIndex, out count))
                    {
                        count = new TypeCount
                        {
                            SampleHandle = entry.HandleValue
                        };
                        result.Add(entry.ObjectTypeIndex, count);
                    }
                    count.Count++;
                }
                return result;
            }
            finally
            {
                if (buffer != IntPtr.Zero)
                    Marshal.FreeHGlobal(buffer);
            }
        }

        private static string TryReadTypeName(
            IntPtr sourceProcess,
            IntPtr sourceHandle)
        {
            IntPtr duplicate;
            if (!DuplicateHandle(
                sourceProcess,
                sourceHandle,
                GetCurrentProcess(),
                out duplicate,
                0,
                false,
                DuplicateSameAccess))
                return "<duplicate-failed>";

            IntPtr buffer = Marshal.AllocHGlobal(8192);
            try
            {
                int required;
                int status = NtQueryObject(
                    duplicate,
                    ObjectTypeInformation,
                    buffer,
                    8192,
                    out required);
                if (status != 0)
                    return "<query-0x" + status.ToString("X8") + ">";

                var value = (UnicodeString)
                    Marshal.PtrToStructure(
                        buffer,
                        typeof(UnicodeString));
                return value.Buffer == IntPtr.Zero
                    ? "<unnamed>"
                    : Marshal.PtrToStringUni(
                        value.Buffer,
                        value.Length / 2);
            }
            finally
            {
                Marshal.FreeHGlobal(buffer);
                CloseHandle(duplicate);
            }
        }
    }
}
"@
}

[DvtDiagnostics.ProcessHandleSnapshot]::Capture($ProcessId) |
    ForEach-Object {
        $parts = $_ -split "`t", 3
        [pscustomobject]@{
            TypeIndex = [int]$parts[0]
            Type = $parts[1]
            Count = [int]$parts[2]
        }
    } |
    Sort-Object Type, TypeIndex
