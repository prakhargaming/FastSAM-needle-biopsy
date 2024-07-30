using System;
using Python.Runtime;

class Program
{
    static void Main(string[] args)
    {
        var watch = System.Diagnostics.Stopwatch.StartNew();
        // Set environment variables for Python.NET to use the Conda environment
        string condaEnvPath = @"/home/prakhargaming/miniconda3/envs/FastSAM"; // Root of the Conda environment

        Environment.SetEnvironmentVariable("PYTHONHOME", condaEnvPath);
        string path = Environment.GetEnvironmentVariable("PATH");
        path = condaEnvPath + @"/bin:" + path;
        Environment.SetEnvironmentVariable("PATH", path);

        // Set the PythonDLL property to point to the Python shared library
        Runtime.PythonDLL = @"/home/prakhargaming/miniconda3/envs/FastSAM/lib/libpython3.10.so"; // Adjust the path and version accordingly

        // Initialize the Python runtime
        PythonEngine.Initialize();

        using (Py.GIL())
        {
            // Import necessary modules
            dynamic sys = Py.Import("sys");
            sys.path.append("/home/prakhargaming/FastSAM"); // Add the directory containing the script to the sys.path

            // Execute the Python script
            dynamic script = Py.Import("FastSAM_img_segmentation"); // Use the script name without extension
            dynamic argsModule = Py.Import("FastSAM_img_segmentation"); // Import the same module for args
            dynamic scriptArgs = argsModule.parse_args(); // Parse the arguments
            dynamic leVariable = script.img_segment(
                model_path: "/home/prakhargaming/FastSAM/weights/FastSAM-x.pt",
                img_path: "/home/prakhargaming/FastSAM/tissue/21548917.png",
                imgsz: (int)scriptArgs.imgsz,
                iou: (float)scriptArgs.iou,
                conf: (float)scriptArgs.conf,
                output: "/home/prakhargaming/FastSAM/output",
                point_prompt: (string)scriptArgs.point_prompt,
                point_label: (string)scriptArgs.point_label,
                box_prompt: (string)scriptArgs.box_prompt,
                better_quality: (bool)scriptArgs.better_quality,
                device: (string)scriptArgs.device,
                retina: (bool)scriptArgs.retina,
                withContours: (bool)scriptArgs.withContours,
                microDims: (string)scriptArgs.microDims,
                plot: (bool)scriptArgs.plot
            );
            Console.WriteLine(leVariable);
        }

        watch.Stop();
        Console.WriteLine(watch.ElapsedMilliseconds);

        // Shutdown the Python runtime, this takes a long time  
        PythonEngine.Shutdown();
    }
}
