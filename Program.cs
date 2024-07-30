using System;
using System.Diagnostics;
using System.Text;
using System.IO;  // Add this line

class Program
{
    static void Main(string[] args)
    {
        string pythonScriptPath = @"/home/prakhargaming/FastSAM/hi.py";
        string pythonExecutablePath = @"/home/prakhargaming/miniconda3/envs/FastSAM/bin/python3";
        string output = ExecutePythonScript(pythonExecutablePath, pythonScriptPath);
        Console.WriteLine("Python script output:");
        Console.WriteLine(output);
    }

    static string ExecutePythonScript(string pythonPath, string scriptPath)
    {
        ProcessStartInfo start = new ProcessStartInfo
        {
            FileName = pythonPath,
            Arguments = scriptPath,
            UseShellExecute = false,
            RedirectStandardOutput = true,
            CreateNoWindow = true
        };

        using (Process process = Process.Start(start))
        {
            using (StreamReader reader = process.StandardOutput)
            {
                string result = reader.ReadToEnd();
                process.WaitForExit();
                return result;
            }
        }
    }
}